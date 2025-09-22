"""Tests for the RiskGuard agent executor."""

from typing import Callable

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.types import DataPart, Message, MessageSendParams, Part, Role

from riskguard.agent_executor import RiskGuardAgentExecutor


@pytest.fixture
def riskguard_message_factory(
    riskguard_input_data_factory,
) -> Callable[..., Message]:
    """Create a complete A2A Message for RiskGuard tests."""

    def _create_message(**kwargs) -> Message:
        input_data = riskguard_input_data_factory(**kwargs)
        return Message(
            message_id="test_message_id",
            role=Role.user,
            parts=[Part(root=DataPart(data=input_data.model_dump()))],
        )

    return _create_message


@pytest.mark.asyncio
async def test_execute_success_approved(
    riskguard_message_factory,
    mock_runner_factory,
    event_queue,
    adk_mock_riskguard_generator,
):
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
            request=MessageSendParams(message=request_message),
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    # Assert
    assert mock_runner_instance.run_async.call_count == 1
    enqueued_message = await event_queue.dequeue_event()

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)

    expected_data = {"approved": True, "reason": "Within risk parameters."}
    assert enqueued_message.parts[0].root.data == expected_data


@pytest.mark.asyncio
async def test_execute_missing_trade_proposal(
    mock_runner_factory,
    adk_mock_riskguard_generator,
    event_queue,
):
    """Test the execute method with a missing trade proposal."""
    mock_runner_instance = mock_runner_factory("riskguard.agent_executor")

    # Arrange
    request_message = Message(
        message_id="test_message_id",
        role=Role.user,
        parts=[
            Part(
                root=DataPart(
                    data={
                        "portfolio_state": {
                            "cash": 10000.0,
                            "shares": 100,
                            "total_value": 20000.0,
                        },
                    },
                ),
            ),
        ],
    )

    # Act
    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    await executor.execute(
        context=RequestContext(
            request=MessageSendParams(message=request_message),
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    # Assert
    enqueued_message = await event_queue.dequeue_event()

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)
    assert (
        "An internal error occurred: Missing 'trade_proposal' or 'portfolio_state' in data payload"
        in enqueued_message.parts[0].root.data["reason"]
    )
    mock_runner_instance.run_async.assert_not_called()


@pytest.mark.asyncio
async def test_execute_adk_runner_exception(
    riskguard_message_factory,
    mock_runner_factory,
    adk_mock_riskguard_generator,
    event_queue,
):
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
            request=MessageSendParams(message=request_message),
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    # Assert
    enqueued_message = await event_queue.dequeue_event()

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)
    assert (
        "An internal error occurred: ADK Borked"
        in enqueued_message.parts[0].root.data["reason"]
    )


@pytest.mark.asyncio
async def test_execute_handles_adk_runner_exception(
    riskguard_message_factory,
    mock_runner_factory,
    event_queue,
):
    """Test that the executor handles an ADK runner failure gracefully.

    If the ADK runner raises an exception, the executor should enqueue an
    error message and then close the event queue.
    """
    # Arrange
    mock_runner = mock_runner_factory("riskguard.agent_executor")
    # Simulate an exception during the ADK agent's execution
    mock_runner.run_async.side_effect = Exception("ADK agent failed!")

    request_message = riskguard_message_factory()
    context = RequestContext(
        request=MessageSendParams(message=request_message),
        context_id="test-context-456",
        task_id="test-task-123",
    )

    # Act
    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner  # Inject the mock runner

    await executor.execute(context, event_queue)

    # Assert
    # 1. An event was enqueued
    enqueued_message = await event_queue.dequeue_event()
    assert event_queue.is_closed()

    # 2. The enqueued event is a Message containing error details
    assert isinstance(enqueued_message, Message)

    # 3. The message part contains the error
    error_part = enqueued_message.parts[0].root
    assert isinstance(error_part, DataPart)
    assert error_part.data["approved"] is False
    assert "An internal error occurred: ADK agent failed!" in error_part.data["reason"]
