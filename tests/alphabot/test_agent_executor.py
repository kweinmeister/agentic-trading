from typing import Callable

import pytest
from a2a.server.agent_execution import RequestContext
from a2a.types import DataPart, Message, MessageSendParams, Part, Role

from alphabot.agent_executor import AlphaBotAgentExecutor


@pytest.fixture
def alphabot_message_factory(
    alphabot_input_data_factory,
) -> Callable[..., Message]:
    """Factory to create a complete A2A Message for AlphaBot tests."""

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
    alphabot_message_factory,
    mock_runner_factory,
    event_queue,
    adk_mock_alphabot_generator,
):
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
    _, call_kwargs = mock_runner_instance.run_async.call_args
    assert call_kwargs["session_id"] == "test-context-456"

    # Verify a single message was enqueued
    enqueued_message = await event_queue.dequeue_event()

    assert isinstance(enqueued_message, Message)
    assert enqueued_message.context_id == "test-context-456"
    assert enqueued_message.task_id == "test-task-123"
    assert len(enqueued_message.parts) == 1
    assert isinstance(enqueued_message.parts[0].root, DataPart)

    expected_data = {
        "status": "APPROVED",
        "reason": "SMA crossover indicates buy signal.",
        "trade_proposal": {
            "action": "BUY",
            "quantity": 10,
            "ticker": "TEST",
            "price": 100.0,
        },
    }
    assert enqueued_message.parts[0].root.data == expected_data
    assert event_queue.is_closed()


@pytest.mark.asyncio
async def test_execute_missing_market_data(mock_runner_factory, event_queue):
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
    assert "validation error" in enqueued_message.parts[0].root.data["reason"]
    assert event_queue.is_closed()


@pytest.mark.asyncio
async def test_execute_adk_runner_exception(
    alphabot_message_factory,
    mock_runner_factory,
    event_queue,
    adk_mock_alphabot_generator,
):
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
        "An unexpected server error occurred."
        in enqueued_message.parts[0].root.data["reason"]
    )
    assert event_queue.is_closed()


@pytest.mark.asyncio
async def test_execute_handles_adk_runner_exception(
    alphabot_message_factory,
    mock_runner_factory,
    event_queue,
):
    """Tests that if the ADK runner fails, the executor enqueues an error message
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
    assert error_part.data["status"] == "ERROR"
    assert "An unexpected server error occurred." in error_part.data["reason"]
    assert event_queue.is_closed()


@pytest.mark.asyncio
async def test_execute_returns_dict_not_string(
    alphabot_message_factory,
    mock_runner_factory,
    event_queue,
    adk_mock_alphabot_generator,
):
    """Ensures the final `DataPart` contains a dictionary, not a JSON string.
    This prevents a `ValidationError` at runtime.
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
            request=MessageSendParams(message=request_message),
            context_id="test-context-456",
            task_id="test-task-123",
        ),
        event_queue=event_queue,
    )

    # Assert
    enqueued_message = await event_queue.dequeue_event()
    data_part = enqueued_message.parts[0].root
    assert isinstance(data_part.data, dict)
