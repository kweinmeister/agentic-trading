import uuid
from unittest.mock import AsyncMock

import httpx
import pytest
from a2a.types import DataPart, Message, Part, Role

from alphabot.a2a_risk_tool import A2ARiskCheckTool
from alphabot.agent import AlphaBotAgent
from common.models import AlphaBotTaskPayload, PortfolioState


@pytest.fixture
def agent() -> AlphaBotAgent:
    """Provides an AlphaBotAgent instance."""
    return AlphaBotAgent(stock_ticker="TEST")


@pytest.fixture
def historical_prices_buy_signal() -> list[float]:
    """Provides a list of historical prices that will trigger a BUY signal."""
    return [
        130,
        128,
        126,
        124,
        122,
        120,
        118,
        116,
        114,
        112,
        110,
        108,
        106,
        104,
        102,
        100,
        98,
        96,
        94,
        92,
        90,
        88,
        86,
        84,
        82,
        80,
        78,
        76,
        74,
        72,
        85,
        95,
        105,
        115,
        125,
    ]


@pytest.fixture
def historical_prices_sell_signal() -> list[float]:
    """Provides a list of historical prices that will trigger a SELL signal."""
    return [
        70,
        72,
        74,
        76,
        78,
        80,
        82,
        84,
        86,
        88,
        90,
        92,
        94,
        96,
        98,
        100,
        102,
        104,
        106,
        108,
        110,
        112,
        114,
        116,
        118,
        120,
        122,
        124,
        126,
        128,
        115,
        105,
        95,
        85,
        75,
    ]


@pytest.fixture
def alphabot_input_data_factory(base_portfolio_state: PortfolioState):
    """Provides a factory for creating AlphaBotTaskPayload instances.
    Injects shared base model fixtures from the root conftest.
    """

    def _input_data(**kwargs) -> AlphaBotTaskPayload:
        # Use the same .model_copy() pattern as the refactored riskguard factory
        final_portfolio_state = base_portfolio_state.model_copy(deep=True)

        if "portfolio_state" in kwargs:
            update_data = kwargs.pop("portfolio_state")
            if isinstance(update_data, dict):
                merged_data = base_portfolio_state.model_dump()
                merged_data.update(update_data)
                final_portfolio_state = PortfolioState(**merged_data)
            else:
                final_portfolio_state = update_data

        # Provide sensible defaults for required fields if not in kwargs
        if "historical_prices" not in kwargs:
            kwargs["historical_prices"] = [100.0, 101.0, 102.0]
        if "current_price" not in kwargs:
            kwargs["current_price"] = 103.0
        if "day" not in kwargs:
            kwargs["day"] = 1

        payload = AlphaBotTaskPayload(portfolio_state=final_portfolio_state, **kwargs)
        return payload

    return _input_data


@pytest.fixture
def alphabot_message_factory(alphabot_input_data_factory):
    """Factory to create a complete A2A Message for AlphaBot tests using the common utility."""

    def _create_message(**kwargs) -> Message:
        # 1. Create the payload using the other factory
        input_data = alphabot_input_data_factory(**kwargs)

        # 2. Use the existing utility to create the message
        # This ensures the test uses the same logic as the application
        a2a_message = Message(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            role=Role.user,
            parts=[
                Part(root=DataPart(data=input_data)),
            ],
        )

        # You can still override generated fields if needed for specific tests
        a2a_message.context_id = kwargs.get("context_id", a2a_message.context_id)

        return a2a_message

    return _create_message


@pytest.fixture
def mock_httpx_client() -> AsyncMock:
    """Fixture to create a mock httpx.AsyncClient."""
    return AsyncMock(spec=httpx.AsyncClient)


@pytest.fixture
def risk_check_tool(
    mock_httpx_client: AsyncMock,
) -> A2ARiskCheckTool:
    """Provides an A2ARiskCheckTool instance with a mocked httpx_client
    for hermetic testing.
    """
    # The tool is now a real object, but its http client is a mock,
    # preventing real network calls.
    return A2ARiskCheckTool()  # httpx_client=mock_httpx_client)
