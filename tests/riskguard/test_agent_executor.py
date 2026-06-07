"""Tests for the RiskGuard agent executor."""

from collections.abc import Callable

import pytest
from a2a.helpers import get_data_parts, new_data_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import Message, Role, TaskState
from a2a.types import SendMessageRequest as MessageSendParams
from a2a.types.a2a_pb2 import TaskArtifactUpdateEvent, TaskStatusUpdateEvent

from riskguard.agent_executor import RiskGuardAgentExecutor
from tests.conftest import get_executor_results


@pytest.fixture
def riskguard_message_factory(
    riskguard_input_data_factory,
) -> Callable[..., Message]:
    """Create a complete A2A Message for RiskGuard tests."""

    def _create_message(**kwargs) -> Message:
        input_data = riskguard_input_data_factory(**kwargs)
        return Message(
            message_id="test_message_id",
            role=Role.ROLE_USER,
            parts=[new_data_part(input_data.model_dump())],
        )

    return _create_message


@pytest.mark.asyncio
async def test_execute_success_approved(
    riskguard_message_factory,
    mock_runner_factory,
    event_queue,
    adk_mock_riskguard_generator,
) -> None:
    """Test the execute method for a successful approved trade."""
    mock_runner_instance = mock_runner_factory("riskguard.agent_executor")

    # Arrange
    request_message = riskguard_message_factory(
        trade_proposal={"quantity": 10},
        portfolio_state={"cash": 50000},
    )
    mock_runner_instance.run_async.return_value = adk_mock_riskguard_generator(
        result_name="risk_check_result",
        result_data={"approved": True, "reason": "Within risk parameters."},
    )

    # Act
    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    await executor.execute(
        context=RequestContext(
            ServerCallContext(),
            request=MessageSendParams(message=request_message),
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    enqueued_message, _events = await get_executor_results(event_queue)

    await event_queue.close()
    assert event_queue.is_closed()

    assert mock_runner_instance.run_async.call_count == 1
    assert isinstance(enqueued_message, TaskArtifactUpdateEvent)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    data_parts = get_data_parts(enqueued_message.artifact.parts)
    assert len(data_parts) == 1
    expected_data = {"approved": True, "reason": "Within risk parameters."}
    assert data_parts[0] == expected_data
    # Assert lifecycle events are emitted
    assert any(
        isinstance(e, TaskStatusUpdateEvent)
        and e.status.state == TaskState.TASK_STATE_COMPLETED
        for e in _events
    )


@pytest.mark.asyncio
async def test_execute_missing_trade_proposal(
    mock_runner_factory,
    adk_mock_riskguard_generator,
    event_queue,
) -> None:
    """Test the execute method with a missing trade proposal."""
    mock_runner_instance = mock_runner_factory("riskguard.agent_executor")

    # Arrange
    request_message = Message(
        message_id="test_message_id",
        role=Role.ROLE_USER,
        parts=[
            new_data_part(
                {
                    "portfolio_state": {
                        "cash": 10000.0,
                        "shares": 100,
                        "total_value": 20000.0,
                    },
                },
            ),
        ],
    )

    # Act
    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    await executor.execute(
        context=RequestContext(
            ServerCallContext(),
            request=MessageSendParams(message=request_message),
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    enqueued_message, _events = await get_executor_results(event_queue)

    await event_queue.close()
    assert event_queue.is_closed()

    assert isinstance(enqueued_message, TaskStatusUpdateEvent)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert enqueued_message.status.state == TaskState.TASK_STATE_FAILED
    assert (
        "Missing 'trade_proposal' or 'portfolio_state' in data payload"
        in enqueued_message.status.message.parts[0].text
    )
    mock_runner_instance.run_async.assert_not_called()


@pytest.mark.asyncio
async def test_execute_adk_runner_exception(
    riskguard_message_factory,
    mock_runner_factory,
    adk_mock_riskguard_generator,
    event_queue,
) -> None:
    """Test the execute method with an ADK runner exception."""
    mock_runner_instance = mock_runner_factory("riskguard.agent_executor")

    # Arrange
    mock_runner_instance.run_async.side_effect = Exception("ADK Borked")
    request_message = riskguard_message_factory()

    # Act
    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    await executor.execute(
        context=RequestContext(
            ServerCallContext(),
            request=MessageSendParams(message=request_message),
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    enqueued_message, _events = await get_executor_results(event_queue)

    await event_queue.close()
    assert event_queue.is_closed()

    assert isinstance(enqueued_message, TaskStatusUpdateEvent)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert enqueued_message.status.state == TaskState.TASK_STATE_FAILED
    assert "ADK Borked" in enqueued_message.status.message.parts[0].text


@pytest.mark.asyncio
async def test_execute_handles_adk_runner_exception(
    riskguard_message_factory,
    mock_runner_factory,
    event_queue,
) -> None:
    """Test that the executor handles an ADK runner failure gracefully."""
    # Arrange
    mock_runner = mock_runner_factory("riskguard.agent_executor")
    # Simulate an exception during the ADK agent's execution
    mock_runner.run_async.side_effect = Exception("ADK agent failed!")

    request_message = riskguard_message_factory()
    context = RequestContext(
        ServerCallContext(),
        request=MessageSendParams(message=request_message),
        context_id="test-context-456",
        task_id="test-task-123",
    )

    # Act
    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner  # Inject the mock runner

    await executor.execute(context, event_queue)

    # Dequeue and verify results
    enqueued_message, _events = await get_executor_results(event_queue)

    await event_queue.close()
    assert event_queue.is_closed()

    # The enqueued event is a TaskStatusUpdateEvent containing error details
    assert isinstance(enqueued_message, TaskStatusUpdateEvent)
    assert enqueued_message.status.state == TaskState.TASK_STATE_FAILED

    # The message part contains the error
    assert "ADK agent failed!" in enqueued_message.status.message.parts[0].text
