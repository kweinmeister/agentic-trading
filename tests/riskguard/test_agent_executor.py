"""Tests for the RiskGuard agent executor."""

from unittest.mock import AsyncMock

import pytest
from a2a.helpers import get_data_parts, new_data_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import Message, Role, TaskState
from a2a.types import SendMessageRequest as MessageSendParams
from a2a.types.a2a_pb2 import TaskArtifactUpdateEvent, TaskStatusUpdateEvent

from riskguard.agent_executor import RiskGuardAgentExecutor
from tests.conftest import get_executor_results


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
    assert (
        "An unexpected server error occurred."
        in enqueued_message.status.message.parts[0].text
    )


@pytest.mark.asyncio
async def test_execute_session_continuity(
    riskguard_message_factory,
    mock_runner_factory,
    event_queue,
    adk_session,
    adk_mock_riskguard_generator,
) -> None:
    """Test that RiskGuard reuses an existing session if present."""
    mock_runner_instance = mock_runner_factory("riskguard.agent_executor")
    request_message = riskguard_message_factory()

    mock_runner_instance.session_service.get_session = AsyncMock(
        return_value=adk_session,
    )
    mock_runner_instance.session_service.create_session = AsyncMock()
    mock_runner_instance.run_async.return_value = adk_mock_riskguard_generator(
        result_name="risk_check_result",
        result_data={"approved": True, "reason": "Session test"},
    )

    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner_instance
    await executor.execute(
        context=RequestContext(
            ServerCallContext(),
            request=MessageSendParams(message=request_message),
            context_id="test-existing-session-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    mock_runner_instance.session_service.get_session.assert_called_once()
    mock_runner_instance.session_service.create_session.assert_not_called()
    await get_executor_results(event_queue)
    await event_queue.close()


@pytest.mark.asyncio
async def test_execute_get_session_exception_fallback(
    riskguard_message_factory,
    mock_runner_factory,
    event_queue,
    adk_session,
    adk_mock_riskguard_generator,
) -> None:
    """Test that if get_session raises an exception, the executor falls back to create_session."""
    mock_runner_instance = mock_runner_factory("riskguard.agent_executor")
    request_message = riskguard_message_factory()

    mock_runner_instance.session_service.get_session = AsyncMock(
        side_effect=Exception("DB Connection Error"),
    )
    mock_runner_instance.session_service.create_session = AsyncMock(
        return_value=adk_session,
    )
    mock_runner_instance.run_async.return_value = adk_mock_riskguard_generator(
        result_name="risk_check_result",
        result_data={"approved": True, "reason": "Fallback test"},
    )

    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner_instance
    await executor.execute(
        context=RequestContext(
            ServerCallContext(),
            request=MessageSendParams(message=request_message),
            context_id="test-existing-session-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    mock_runner_instance.session_service.get_session.assert_called_once()
    mock_runner_instance.session_service.create_session.assert_called_once()
    await get_executor_results(event_queue)
    await event_queue.close()


@pytest.mark.asyncio
async def test_execute_missing_context_id_raises_value_error(
    event_queue,
) -> None:
    """Test that execute raises ValueError if context_id is missing/None."""
    executor = RiskGuardAgentExecutor()
    context = RequestContext(
        ServerCallContext(),
        context_id=None,
    )
    with pytest.raises(ValueError, match="Context ID is missing, cannot execute."):
        await executor.execute(context, event_queue)
    await event_queue.close()


@pytest.mark.asyncio
async def test_execute_cancel(
    event_queue,
) -> None:
    """Test that cancel() enqueues a cancel task status update."""
    context = RequestContext(
        ServerCallContext(),
        context_id="test-context-123",
        task_id="test-task-123",
    )

    executor = RiskGuardAgentExecutor()
    await executor.cancel(context, event_queue)

    _, _events = await get_executor_results(event_queue)
    await event_queue.close()

    assert any(
        isinstance(e, TaskStatusUpdateEvent)
        and e.status.state == TaskState.TASK_STATE_CANCELED
        for e in _events
    )


@pytest.mark.asyncio
async def test_execute_pydantic_validation_error_sanitization(
    riskguard_message_factory,
    mock_runner_factory,
    event_queue,
) -> None:
    """Test that execute catches Pydantic ValidationError and sanitizes it to generic error text."""
    from pydantic import BaseModel, ValidationError

    mock_runner_instance = mock_runner_factory("riskguard.agent_executor")
    request_message = riskguard_message_factory()

    class DummyModel(BaseModel):
        field: int

    try:
        DummyModel.model_validate({"field": "not an int"})
    except ValidationError as val_err:
        mock_runner_instance.run_async.side_effect = val_err

    executor = RiskGuardAgentExecutor()
    executor._adk_runner = mock_runner_instance
    await executor.execute(
        context=RequestContext(
            ServerCallContext(),
            request=MessageSendParams(message=request_message),
            context_id="test-context-111",
            task_id="test-task-222",
        ),
        event_queue=event_queue,
    )

    enqueued_message, _ = await get_executor_results(event_queue)
    await event_queue.close()

    assert isinstance(enqueued_message, TaskStatusUpdateEvent)
    assert enqueued_message.status.state == TaskState.TASK_STATE_FAILED
    assert enqueued_message.status.message.parts[0].text == "Input validation failed."
