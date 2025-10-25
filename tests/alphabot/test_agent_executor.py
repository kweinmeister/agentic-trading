"""Tests for the AlphaBot agent executor."""

import asyncio
from typing import Callable
from unittest.mock import AsyncMock

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.types import (
    DataPart,
    Message,
    MessageSendParams,
    Part,
    Role,
)
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from alphabot.agent_executor import AlphaBotAgentExecutor


@pytest.fixture
def alphabot_message_factory(
    alphabot_input_data_factory,
) -> Callable[..., Message]:
    """Create a complete A2A Message for AlphaBot tests."""

    def _create_message(**kwargs) -> Message:
        input_data = alphabot_input_data_factory(**kwargs)
        return Message(
            message_id="test_message_id",
            role=Role.user,
            parts=[Part(root=DataPart(data=input_data.model_dump()))],
            kind="message",
        )

    return _create_message


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
        request=MessageSendParams(message=request_message),
        context_id="test-context-456",
        task_id="test-task-123",
    )

    # Mock the session service methods to return a valid session
    mock_runner_instance.session_service.get_session = AsyncMock(return_value=None)
    mock_runner_instance.session_service.create_session = AsyncMock(
        return_value=adk_session
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
                    }
                }
            ),
        )
        # Yield a final event with the reason text
        yield Event(
            author="test_author",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text="Trade Approved (A2A): Within risk parameters."
                    )
                ]
            ),
            turn_complete=True,
        )

    # Configure the mock to return the generator object.
    mock_runner_instance.run_async.return_value = mock_run_async_generator()

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner

    # Run the executor in a background task to avoid deadlock
    execution_task = asyncio.create_task(executor.execute(context, event_queue))

    # Assert
    # Dequeue the message and mark it as done
    enqueued_message = await event_queue.dequeue_event()
    event_queue.task_done()

    # Wait for the executor task to finish (which includes closing the queue)
    await execution_task

    assert event_queue.is_closed()

    # Verify the message content
    assert enqueued_message.parts is not None
    assert len(enqueued_message.parts) == 1
    data_part = enqueued_message.parts[0].root
    assert isinstance(data_part, DataPart)

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
    assert data_part.data == expected_data


@pytest.mark.asyncio
async def test_execute_missing_market_data(mock_runner_factory, event_queue) -> None:
    """Test the execute method with missing market data."""
    mock_runner_instance = mock_runner_factory("alphabot.agent_executor")

    # Arrange
    request_message = Message(
        message_id="test_message_id",
        role=Role.user,
        parts=[Part(root=DataPart(data={"cash": 10000.0, "shares": 100}))],
        kind="message",
    )

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner_instance  # Inject the mock runner
    execution_task = asyncio.create_task(
        executor.execute(
            context=RequestContext(
                request=MessageSendParams(message=request_message),
                context_id="test-context-456",
                task_id="test-task-123",
            ),
            event_queue=event_queue,
        )
    )

    # Assert
    enqueued_message = await event_queue.dequeue_event()
    event_queue.task_done()

    await execution_task
    assert event_queue.is_closed()

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)
    assert "validation error" in enqueued_message.parts[0].root.data["reason"]


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
    execution_task = asyncio.create_task(
        executor.execute(
            context=RequestContext(
                request=MessageSendParams(message=request_message),
                context_id="test-context-456",
                task_id="test-task-123",
            ),
            event_queue=event_queue,
        )
    )

    # Assert
    enqueued_message = await event_queue.dequeue_event()
    event_queue.task_done()

    await execution_task
    assert event_queue.is_closed()

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)
    assert (
        "An unexpected server error occurred."
        in enqueued_message.parts[0].root.data["reason"]
    )


@pytest.mark.asyncio
async def test_execute_handles_adk_runner_exception(
    alphabot_message_factory,
    mock_runner_factory,
    event_queue,
) -> None:
    """Test that if the ADK runner fails, the executor enqueues an error message.

    and closes the queue.
    """
    # Arrange
    mock_runner = mock_runner_factory("alphabot.agent_executor")
    # Simulate an exception during the ADK agent's execution
    mock_runner.run_async.side_effect = Exception("ADK agent failed!")

    request_message = alphabot_message_factory()
    context = RequestContext(request=MessageSendParams(message=request_message))

    # Act
    executor = AlphaBotAgentExecutor()
    executor._adk_runner = mock_runner  # Inject the mock runner

    execution_task = asyncio.create_task(executor.execute(context, event_queue))

    # Assert
    # 1. An event was enqueued
    enqueued_message = await event_queue.dequeue_event()
    event_queue.task_done()

    await execution_task
    assert event_queue.is_closed()

    # 2. The enqueued event is a Message containing error details
    assert isinstance(enqueued_message, Message)

    # 3. The message part contains the error
    error_part = enqueued_message.parts[0].root
    assert isinstance(error_part, DataPart)
    assert error_part.data["status"] == "ERROR"
    assert "An unexpected server error occurred." in error_part.data["reason"]


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
    execution_task = asyncio.create_task(
        executor.execute(
            context=RequestContext(
                request=MessageSendParams(message=request_message),
                context_id="test-context-456",
                task_id="test-task-123",
            ),
            event_queue=event_queue,
        )
    )

    # Assert
    enqueued_message = await event_queue.dequeue_event()
    event_queue.task_done()

    await execution_task
    assert event_queue.is_closed()

    data_part = enqueued_message.parts[0].root
    assert isinstance(data_part.data, dict)
