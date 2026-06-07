"""Tests for the AlphaBot agent executor."""

from unittest.mock import AsyncMock

import pytest
from a2a.helpers import get_data_parts, new_data_part
from a2a.server.agent_execution import RequestContext
from a2a.server.context import ServerCallContext
from a2a.types import (
    Message,
    Role,
    TaskState,
)
from a2a.types import (
    SendMessageRequest as MessageSendParams,
)
from a2a.types.a2a_pb2 import TaskArtifactUpdateEvent, TaskStatusUpdateEvent
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from alphabot.agent_executor import AlphaBotAgentExecutor
from tests.conftest import get_executor_results


@pytest.mark.asyncio
async def test_execute_success_buy_decision(
    mock_runner_factory,
    event_queue,
    alphabot_message_factory,
    adk_session,
) -> None:
    """Test the execute method for a successful buy decision."""
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    request_message = alphabot_message_factory(
        historical_prices=[
            98,
            99,
            100,
            101,
            105,
            110,
        ],  # Creates a buy signal
        portfolio_state={"cash": 50000, "shares": 0, "total_value": 50000},
    )
    context = RequestContext(
        ServerCallContext(),
        request=MessageSendParams(message=request_message),
        context_id="test-context-456",
        task_id="test-task-123",
    )

    # Mock the session service methods to return a valid session
    mock_runner_instance.session_service.get_session = AsyncMock(return_value=None)
    mock_runner_instance.session_service.create_session = AsyncMock(
        return_value=adk_session,
    )

    # Use the adk_mock_alphabot_generator fixture to create the events
    async def mock_run_async_generator():
        # Yield an event with the state delta
        yield Event(
            author="test_author",
            actions=EventActions(
                state_delta={
                    "approved_trade": {
                        "action": "BUY",
                        "ticker": "TECH",
                        "quantity": 100,
                        "price": 110.0,
                    },
                },
            ),
        )
        # Yield a final event with the reason text
        yield Event(
            author="test_author",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text="Trade Approved (A2A): Within risk parameters.",
                    ),
                ],
            ),
            turn_complete=True,
        )

    # Configure the mock to return the generator object.
    mock_runner_instance.run_async.return_value = mock_run_async_generator()

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner

    # Run the executor
    await executor.execute(context, event_queue)

    # Dequeue and verify results
    enqueued_message, _events = await get_executor_results(event_queue)

    # Close the queue manually for testing
    await event_queue.close()
    assert event_queue.is_closed()

    # Verify the message content
    assert isinstance(enqueued_message, TaskArtifactUpdateEvent)
    assert enqueued_message.artifact.parts is not None
    data_parts = get_data_parts(enqueued_message.artifact.parts)
    assert len(data_parts) == 1

    expected_data = {
        "status": "APPROVED",
        "reason": "Trade Approved (A2A): Within risk parameters.",
        "trade_proposal": {
            "action": "BUY",
            "ticker": "TECH",
            "quantity": 100,
            "price": 110.0,
        },
    }
    assert data_parts[0] == expected_data
    # Assert lifecycle events are emitted
    assert any(
        isinstance(e, TaskStatusUpdateEvent)
        and e.status.state == TaskState.TASK_STATE_COMPLETED
        for e in _events
    )


@pytest.mark.asyncio
async def test_execute_missing_market_data(mock_runner_factory, event_queue) -> None:
    """Test the execute method with missing market data."""
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    request_message = Message(
        message_id="test_message_id",
        role=Role.ROLE_USER,
        parts=[new_data_part({"cash": 10000.0, "shares": 100})],
    )

    # Act
    executor = AlphaBotAgentExecutor()
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
    assert "validation error" in enqueued_message.status.message.parts[0].text


@pytest.mark.asyncio
async def test_execute_adk_runner_exception(
    alphabot_message_factory,
    mock_runner_factory,
    event_queue,
    adk_mock_alphabot_generator,
) -> None:
    """Test the execute method with an ADK runner exception."""
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    mock_runner_instance.run_async.side_effect = Exception("ADK Borked")
    request_message = alphabot_message_factory(
        historical_prices=[150.0, 151.0, 152.0],
        current_price=155.0,
    )

    # Act
    executor = AlphaBotAgentExecutor()
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
async def test_execute_returns_dict_not_string(
    alphabot_message_factory,
    mock_runner_factory,
    event_queue,
    adk_mock_alphabot_generator,
) -> None:
    """Ensure the final `DataPart` contains a dictionary, not a JSON string.

    This test verifies that the data in the final `DataPart` is a Python
    dictionary to prevent a `pydantic.ValidationError` at runtime.
    """
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    request_message = alphabot_message_factory(
        historical_prices=[150.0, 151.0, 152.0],
        current_price=155.0,
        day=1,
    )
    mock_runner_instance.run_async.return_value = adk_mock_alphabot_generator(
        final_state_delta={"approved_trade": {"action": "BUY", "quantity": 10}},
        final_reason="SMA crossover indicates buy signal.",
    )

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance
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

    assert isinstance(enqueued_message, TaskArtifactUpdateEvent)
    data_parts = get_data_parts(enqueued_message.artifact.parts)
    assert len(data_parts) == 1
    assert isinstance(data_parts[0], dict)
    # Assert lifecycle events are emitted
    assert any(
        isinstance(e, TaskStatusUpdateEvent)
        and e.status.state == TaskState.TASK_STATE_COMPLETED
        for e in _events
    )


@pytest.mark.asyncio
async def test_execute_session_continuity(
    mock_runner_factory,
    event_queue,
    alphabot_message_factory,
    adk_session,
) -> None:
    """Test that existing session is reused (session continuity)."""
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")
    request_message = alphabot_message_factory()
    context = RequestContext(
        ServerCallContext(),
        request=MessageSendParams(message=request_message),
        context_id="test-existing-session-123",
        task_id="test-task-123",
    )

    mock_runner_instance.session_service.get_session = AsyncMock(
        return_value=adk_session
    )
    mock_runner_instance.session_service.create_session = AsyncMock()

    async def mock_run_async_generator():
        yield Event(author="test", turn_complete=True)

    mock_runner_instance.run_async.return_value = mock_run_async_generator()

    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance
    await executor.execute(context, event_queue)

    mock_runner_instance.session_service.get_session.assert_called_once()
    mock_runner_instance.session_service.create_session.assert_not_called()
    await get_executor_results(event_queue)
    await event_queue.close()


@pytest.mark.asyncio
async def test_execute_no_action_path(
    mock_runner_factory,
    event_queue,
    alphabot_message_factory,
    adk_session,
) -> None:
    """Test executor when agent decides NO_ACTION."""
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")
    request_message = alphabot_message_factory()
    context = RequestContext(
        ServerCallContext(),
        request=MessageSendParams(message=request_message),
        context_id="test-context-123",
        task_id="test-task-123",
    )

    mock_runner_instance.session_service.get_session = AsyncMock(
        return_value=adk_session
    )

    async def mock_run_async_generator():
        yield Event(
            author="test",
            actions=EventActions(state_delta={}),  # No trades proposed
        )
        yield Event(
            author="test",
            content=genai_types.Content(
                parts=[genai_types.Part(text="No trades today.")]
            ),
            turn_complete=True,
        )

    mock_runner_instance.run_async.return_value = mock_run_async_generator()

    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance
    await executor.execute(context, event_queue)

    enqueued_message, _events = await get_executor_results(event_queue)
    await event_queue.close()

    assert isinstance(enqueued_message, TaskArtifactUpdateEvent)
    data_parts = get_data_parts(enqueued_message.artifact.parts)
    assert data_parts[0]["status"] == "NO_ACTION"
    assert data_parts[0]["reason"] == "No trades today."


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

    executor = AlphaBotAgentExecutor()
    await executor.cancel(context, event_queue)

    _, _events = await get_executor_results(event_queue)
    await event_queue.close()

    assert any(
        isinstance(e, TaskStatusUpdateEvent)
        and e.status.state == TaskState.TASK_STATE_CANCELED
        for e in _events
    )


@pytest.mark.asyncio
async def test_execute_event_queue_lifecycle_on_exception(
    mock_runner_factory,
    event_queue,
    alphabot_message_factory,
) -> None:
    """Verify event queue lifecycle is handled correctly even on exceptions."""
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")
    mock_runner_instance.run_async.side_effect = Exception("ADK agent failed!")
    request_message = alphabot_message_factory()
    context = RequestContext(
        ServerCallContext(),
        request=MessageSendParams(message=request_message),
        context_id="test-context-123",
        task_id="test-task-123",
    )

    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance

    try:
        await executor.execute(context, event_queue)
    finally:
        await get_executor_results(event_queue)
        await event_queue.close()
        assert event_queue.is_closed()


@pytest.mark.asyncio
async def test_execute_missing_context_id_raises_value_error(
    event_queue,
) -> None:
    """Test that execute raises ValueError if context_id is missing/None."""
    executor = AlphaBotAgentExecutor()
    context = RequestContext(
        ServerCallContext(),
        context_id=None,
    )
    with pytest.raises(ValueError, match="Context ID is missing, cannot execute."):
        await executor.execute(context, event_queue)
    await event_queue.close()
