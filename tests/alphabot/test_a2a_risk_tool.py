"""Tests for the A2A Risk Check Tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.client import (
    A2AClientError,
    A2AClientTimeoutError,
    AgentCardResolutionError,
)
from a2a.helpers import get_data_parts, new_data_part
from a2a.types import (
    AgentCard,
    Message,
    Role,
    SendMessageRequest,
)
from google.adk.agents.context import Context
from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions import Session
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools import ToolContext

from alphabot.a2a_risk_tool import A2ARiskCheckTool
from alphabot.agent import AlphaBotAgent
from common.models import RiskCheckResult
from tests.conftest import create_async_error_iterator


@pytest.fixture
def tool_context(adk_session: Session) -> Context:
    """Fixture to create a mock ToolContext."""
    return ToolContext(
        invocation_context=InvocationContext(
            invocation_id="test-invocation-id",
            session=adk_session,
            agent=AlphaBotAgent(),
            session_service=InMemorySessionService(),
        ),
    )


def create_success_response_message(result_data: dict) -> Message:
    """Create a successful A2A Message response for mocking."""
    risk_check_result = RiskCheckResult.model_validate(result_data)
    return Message(
        message_id="test-message-id",
        context_id="test-context-id",
        task_id="test-task-id",
        role=Role.ROLE_AGENT,
        parts=[new_data_part(risk_check_result.model_dump(mode="json"))],
    )


def _verify_a2a_payload(
    mock_a2a_client: MagicMock,
    args: dict,
) -> None:
    """Verify the payload sent to the A2AClient."""
    mock_a2a_client.send_message.assert_called_once()
    request: SendMessageRequest = mock_a2a_client.send_message.call_args[0][0]
    sent_message = request.message
    assert sent_message.parts
    data_parts = get_data_parts(sent_message.parts)
    assert len(data_parts) == 1
    sent_payload = data_parts[0]
    assert isinstance(sent_payload, dict)
    assert sent_payload["trade_proposal"] == args["trade_proposal"]
    assert sent_payload["portfolio_state"] == args["portfolio_state"]


@pytest.mark.asyncio
async def test_run_async_approved(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
    mock_alphabot_a2a,
    test_agent_card: AgentCard,
    mock_a2a_send_message_generator,
) -> None:
    """Test using shared fixtures for input data."""
    # Arrange
    args = {
        "trade_proposal": {
            "action": "BUY",
            "quantity": 10,
            "price": 100.0,
            "ticker": "TEST",
        },
        "portfolio_state": {"cash": 10000.0, "shares": 50, "total_value": 15000.0},
        "risk_params": {"risk_guard_url": "http://mock-riskguard.com"},
    }
    expected_result = {"approved": True, "reason": "Within limits"}

    # Use the mocked client from the global fixture
    mock_a2a_client = mock_alphabot_a2a["mock_a2a_client"]
    mock_resolver_instance = mock_alphabot_a2a["mock_resolver_instance"]
    mock_resolver_instance.get_agent_card.return_value = test_agent_card

    # Configure the mock client's send_message to be an async generator function.
    mock_send_message = mock_a2a_send_message_generator(
        create_success_response_message(expected_result),
    )

    # Replace the send_message method directly with our async generator spy
    mock_a2a_client.send_message = MagicMock(side_effect=mock_send_message)

    # Act
    event = await risk_check_tool.run_async(args=args, tool_context=tool_context)

    # Assert
    assert event.content is not None
    assert event.content.parts
    response_part = event.content.parts[0]
    assert response_part.function_response is not None
    response_data = response_part.function_response.response
    assert response_data is not None
    assert response_data == expected_result

    # Verify the payload sent to the mocked A2A client
    _verify_a2a_payload(mock_a2a_client, args)


@pytest.mark.asyncio
async def test_run_async_rejected(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
    mock_alphabot_a2a,
    test_agent_card: AgentCard,
    mock_a2a_send_message_generator,
) -> None:
    """Test the tool's run_async method for a rejected trade."""
    # Arrange
    args = {
        "trade_proposal": {
            "action": "SELL",
            "quantity": 20,
            "price": 100.0,
            "ticker": "TEST",
        },
        "portfolio_state": {"cash": 10000.0, "shares": 10, "total_value": 11000.0},
        "risk_params": {"risk_guard_url": "http://mock-riskguard.com"},
    }
    expected_result = {"approved": False, "reason": "Exceeds max position size"}
    mock_a2a_client = mock_alphabot_a2a["mock_a2a_client"]
    mock_resolver_instance = mock_alphabot_a2a["mock_resolver_instance"]
    mock_resolver_instance.get_agent_card.return_value = test_agent_card

    # Configure the mock client's send_message to be an async generator function.
    mock_send_message = mock_a2a_send_message_generator(
        create_success_response_message(expected_result),
    )

    # Replace the send_message method directly with our async generator spy
    mock_a2a_client.send_message = MagicMock(side_effect=mock_send_message)

    # Act
    event = await risk_check_tool.run_async(args=args, tool_context=tool_context)

    # Assert
    assert event.content is not None
    assert event.content.parts
    response_part = event.content.parts[0]
    assert response_part.function_response is not None
    response_data = response_part.function_response.response
    assert response_data is not None
    assert response_data == expected_result
    _verify_a2a_payload(mock_a2a_client, args)


@pytest.mark.asyncio
async def test_run_async_handles_malformed_message(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
    mock_alphabot_a2a,
    test_agent_card: AgentCard,
    mock_a2a_send_message_generator,
) -> None:
    """Tests that the tool gracefully handles a malformed A2A response."""
    # Arrange
    args = {
        "trade_proposal": {
            "action": "BUY",
            "quantity": 5,
            "price": 100.0,
            "ticker": "TEST",
        },
        "portfolio_state": {"cash": 10000.0, "shares": 5, "total_value": 10500.0},
        "risk_params": {"risk_guard_url": "http://mock-riskguard.com"},
    }
    malformed_message = Message(
        message_id="malformed-id",
        role=Role.ROLE_AGENT,
        parts=[new_data_part({"some": "data"})],
    )
    mock_a2a_client = mock_alphabot_a2a["mock_a2a_client"]
    mock_resolver_instance = mock_alphabot_a2a["mock_resolver_instance"]
    mock_resolver_instance.get_agent_card.return_value = test_agent_card

    # Configure the mock client's send_message to be an async generator function.
    mock_send_message = mock_a2a_send_message_generator(malformed_message)

    # Replace the send_message method directly with our async generator spy
    mock_a2a_client.send_message = MagicMock(side_effect=mock_send_message)

    # Act
    event = await risk_check_tool.run_async(args=args, tool_context=tool_context)

    # Assert
    assert event.content is not None
    assert event.content.parts
    response_part = event.content.parts[0]
    assert response_part.function_response is not None
    response_data = response_part.function_response.response
    assert response_data is not None
    assert response_data["approved"] is False
    assert "Malformed response from RiskGuard" in response_data["reason"]
    _verify_a2a_payload(mock_a2a_client, args)


@pytest.mark.asyncio
async def test_run_async_a2a_client_timeout(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
    mock_alphabot_a2a,
    test_agent_card: AgentCard,
) -> None:
    """Tests that the tool handles an A2AClientTimeoutError."""
    # Arrange
    args = {
        "trade_proposal": {
            "action": "BUY",
            "quantity": 10,
            "price": 100.0,
            "ticker": "TEST",
        },
        "portfolio_state": {"cash": 10000.0, "shares": 50, "total_value": 15000.0},
        "risk_params": {"risk_guard_url": "http://mock-riskguard.com"},
    }
    mock_resolver_instance = mock_alphabot_a2a["mock_resolver_instance"]
    mock_resolver_instance.get_agent_card.return_value = test_agent_card
    mock_a2a_client = mock_alphabot_a2a["mock_a2a_client"]

    # Replace the send_message method with our error iterator using the helper function
    mock_a2a_client.send_message = lambda *args, **kwargs: create_async_error_iterator(
        A2AClientTimeoutError,
        "Request timed out",
    )

    # Act
    event = await risk_check_tool.run_async(args=args, tool_context=tool_context)

    # Assert
    assert event.content is not None
    assert event.content.parts
    response_part = event.content.parts[0]
    assert response_part.function_response is not None
    response_data = response_part.function_response.response
    assert response_data is not None
    assert response_data["approved"] is False
    assert "Timeout Error: Request timed out" in response_data["reason"]


@pytest.mark.asyncio
async def test_run_async_a2a_http_error(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
    mock_alphabot_a2a,
    test_agent_card: AgentCard,
) -> None:
    """Tests that the tool handles an A2AClientHTTPError."""
    # Arrange
    args = {
        "trade_proposal": {
            "action": "BUY",
            "quantity": 10,
            "price": 100.0,
            "ticker": "TEST",
        },
        "portfolio_state": {"cash": 10000.0, "shares": 50, "total_value": 15000.0},
        "risk_params": {"risk_guard_url": "http://mock-riskguard.com"},
    }
    mock_resolver_instance = mock_alphabot_a2a["mock_resolver_instance"]
    mock_resolver_instance.get_agent_card.return_value = test_agent_card
    mock_a2a_client = mock_alphabot_a2a["mock_a2a_client"]

    # Replace the send_message method with our error iterator using the helper function
    mock_a2a_client.send_message = lambda *args, **kwargs: create_async_error_iterator(
        A2AClientError,
        message="Service Unavailable",
    )

    # Act
    event = await risk_check_tool.run_async(args=args, tool_context=tool_context)

    # Assert
    assert event.content is not None
    assert event.content.parts
    response_part = event.content.parts[0]
    assert response_part.function_response is not None
    response_data = response_part.function_response.response
    assert response_data is not None
    assert response_data["approved"] is False
    assert (
        "A2A SDK Error: Service Unavailable. Is RiskGuard running?"
        in response_data["reason"]
    )


@pytest.mark.asyncio
async def test_run_async_transport_resolution_error(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
    mock_alphabot_a2a,
) -> None:
    """Tests that the tool handles an A2ATransportResolutionError."""
    args = {
        "trade_proposal": {
            "action": "BUY",
            "quantity": 10,
            "price": 100.0,
            "ticker": "TEST",
        },
        "portfolio_state": {"cash": 10000.0, "shares": 50, "total_value": 15000.0},
        "risk_params": {"risk_guard_url": "http://mock-riskguard.com"},
    }
    mock_resolver_instance_risk_tool = mock_alphabot_a2a["mock_resolver_instance"]

    # Mock the agent card resolution to raise an AgentCardResolutionError
    error_message = "Could not resolve agent card"
    mock_resolver_instance_risk_tool.get_agent_card.side_effect = (
        AgentCardResolutionError(
            message=error_message,
            status_code=503,
        )
    )

    # Act
    event = await risk_check_tool.run_async(args=args, tool_context=tool_context)

    # Assert
    assert event.content is not None
    assert event.content.parts
    response_part = event.content.parts[0]
    assert response_part.function_response is not None
    response_data = response_part.function_response.response
    assert response_data is not None
    assert response_data["approved"] is False

    # Verify that the error message matches the expected format
    expected_reason = f"A2A SDK Error: {error_message}. Is RiskGuard running?"
    assert response_data["reason"] == expected_reason


def test_risk_tool_declaration_schema_match(risk_check_tool: A2ARiskCheckTool) -> None:
    """Test that the ADK FunctionDeclaration schema matches the RiskCheckPayload properties."""
    decl = risk_check_tool._get_declaration()
    assert decl.name == "a2a_risk_check"
    params = decl.parameters
    assert params is not None
    assert params.properties is not None
    assert params.required is not None
    assert "trade_proposal" in params.properties
    assert "portfolio_state" in params.properties
    assert "trade_proposal" in params.required
    assert "portfolio_state" in params.required


@pytest.mark.asyncio
async def test_risk_tool_close_client() -> None:
    """Test that closing the tool closes the internal httpx client."""
    mock_client = MagicMock()
    mock_client.aclose = AsyncMock()
    tool = A2ARiskCheckTool(httpx_client=mock_client)
    await tool.close()
    mock_client.aclose.assert_called_once()


@pytest.mark.asyncio
async def test_risk_tool_sends_wrapped_request_and_context_id(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
    mock_alphabot_a2a,
    test_agent_card: AgentCard,
    mock_a2a_send_message_generator,
) -> None:
    """Test that outgoing messages are wrapped in SendMessageRequest and propagate context_id."""
    args = {
        "trade_proposal": {
            "action": "BUY",
            "quantity": 10,
            "price": 100.0,
            "ticker": "TEST",
        },
        "portfolio_state": {"cash": 10000.0, "shares": 50, "total_value": 15000.0},
        "risk_params": {"risk_guard_url": "http://mock-riskguard.com"},
    }
    expected_result = {"approved": True, "reason": "Within limits"}
    mock_a2a_client = mock_alphabot_a2a["mock_a2a_client"]
    mock_resolver_instance = mock_alphabot_a2a["mock_resolver_instance"]
    mock_resolver_instance.get_agent_card.return_value = test_agent_card

    mock_send_message = mock_a2a_send_message_generator(
        create_success_response_message(expected_result),
    )
    spy_send_message = MagicMock(side_effect=mock_send_message)
    mock_a2a_client.send_message = spy_send_message

    await risk_check_tool.run_async(args=args, tool_context=tool_context)

    spy_send_message.assert_called_once()
    call_arg = spy_send_message.call_args[0][0]
    assert isinstance(call_arg, SendMessageRequest)
    assert call_arg.message.context_id == tool_context.session.id


@pytest.mark.asyncio
async def test_risk_tool_missing_arguments(
    risk_check_tool: A2ARiskCheckTool,
    tool_context: ToolContext,
) -> None:
    """Test that running with missing parameters returns tool validation error."""
    args = {
        "trade_proposal": {
            "action": "BUY",
            "quantity": 10,
            "price": 100.0,
            "ticker": "TEST",
        },
        # portfolio_state is missing!
    }
    event = await risk_check_tool.run_async(args=args, tool_context=tool_context)
    assert event.content is not None
    assert event.content.parts
    fn_response = event.content.parts[0].function_response
    assert fn_response is not None
    resp = fn_response.response
    assert resp is not None
    assert resp["approved"] is False
    assert "Tool Error: Missing input arguments." in resp["reason"]


@pytest.mark.asyncio
async def test_risk_tool_close_client_default() -> None:
    """Test that closing the tool closes the default internal httpx client when not injected."""
    with patch("httpx.AsyncClient.aclose", new_callable=AsyncMock) as mock_aclose:
        tool = A2ARiskCheckTool()
        assert tool.httpx_client is not None
        await tool.close()
        mock_aclose.assert_called_once()
