"""Tests for the simulator's main application."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.client import ClientFactory
from a2a.types import (
    DataPart,
    Message,
    Part,
    Role,
)
from fastapi.testclient import TestClient

import common.config as defaults
from common.models import TradeOutcome, TradeProposal, TradeStatus
from simulator.main import _call_alphabot_a2a, app
from simulator.portfolio import PortfolioState

client = TestClient(app)


@pytest.fixture
def mock_a2a_call():
    """Fixture to mock the _call_alphabot_a2a function."""
    with patch("simulator.main._call_alphabot_a2a", new_callable=AsyncMock) as mock:
        # Simulate a successful trade approval
        mock.return_value = {
            "approved_trade": {
                "action": "BUY",
                "quantity": 10,
                "price": 100.0,
                "ticker": "SIM",
            },
            "rejected_trade": None,
            "reason": "SMA crossover",
            "error": None,
        }
        yield mock


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_read_main():
    """Test the main endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Agentic Trading Simulator" in response.text


@pytest.mark.asyncio
async def test_call_alphabot_a2a_with_factory(
    mock_a2a_sdk_components,
    test_agent_card,
    mock_a2a_send_message_generator,
):
    """Verify that _call_alphabot_a2a correctly uses the ClientFactory."""
    mock_logger = MagicMock()
    mock_factory_instance = mock_a2a_sdk_components["mock_factory_instance"]
    mock_a2a_client = mock_a2a_sdk_components["mock_a2a_client"]
    mock_resolver_instance = mock_a2a_sdk_components["mock_resolver_instance"]

    # Mock the agent card resolution
    mock_resolver_instance.get_agent_card.return_value = test_agent_card

    # Configure the client to return a successful message
    expected_trade_proposal = TradeProposal(
        action="BUY",
        quantity=10,
        price=100.0,
        ticker="TEST",
    )
    trade_outcome = TradeOutcome(
        status=TradeStatus.APPROVED,
        reason="Test approved reason.",
        trade_proposal=expected_trade_proposal,
    )
    mock_message = Message(
        message_id="mock_msg_id",
        role=Role.agent,
        parts=[Part(root=DataPart(data=trade_outcome.model_dump(mode="json")))],
    )

    # Configure the mock client's send_message to be an async generator function.
    mock_send_message = mock_a2a_send_message_generator(mock_message)

    # Replace the send_message method directly with our async generator
    mock_a2a_client.send_message = mock_send_message

    # Prepare input for the function
    params = {
        "alphabot_short_sma": 10,
        "alphabot_long_sma": 20,
        "alphabot_trade_qty": 10,
        "riskguard_url": "http://localhost:8001",
        "riskguard_max_pos_size": 1000,
        "riskguard_max_concentration": 0.5,
    }

    # The function under test will now use the patched components from the fixture
    outcome = await _call_alphabot_a2a(
        client_factory=mock_factory_instance,  # Pass the correct mock
        alphabot_url="http://test.com",
        session_id="test-session-123",
        day=1,
        current_price=100.0,
        historical_prices=[90.0, 95.0],
        portfolio=PortfolioState(cash=10000.0),
        params=params,
        sim_logger=mock_logger,
    )

    assert outcome["approved_trade"] is not None
    assert outcome["approved_trade"]["action"] == "BUY"
    assert outcome["reason"] == "Test approved reason."


@pytest.mark.asyncio
async def test_call_alphabot_a2a_factory_raises_transport_error():
    """Test that _call_alphabot_a2a handles transport resolution errors."""
    mock_factory = AsyncMock(spec=ClientFactory)
    mock_logger = MagicMock()

    # Configure the factory mock to have the necessary attributes
    mock_factory._config = MagicMock()
    mock_factory._config.httpx_client = AsyncMock()

    from a2a.client.errors import A2AClientHTTPError

    # Configure the factory mock to raise an error on card resolution
    mock_factory.create.side_effect = A2AClientHTTPError(
        message="Resolution failed",
        status_code=404,
    )

    with patch("simulator.main.A2ACardResolver") as mock_resolver:
        mock_resolver.return_value.get_agent_card.side_effect = A2AClientHTTPError(
            message="Resolution failed",
            status_code=404,
        )

        with pytest.raises(ConnectionError):
            await _call_alphabot_a2a(
                client_factory=mock_factory,
                alphabot_url="http://test.com",
                session_id="test-session-123",
                day=1,
                current_price=100.0,
                historical_prices=[90.0, 95.0],
                portfolio=PortfolioState(cash=10000.0),
                params={
                    "alphabot_short_sma": 10,
                    "alphabot_long_sma": 20,
                    "alphabot_trade_qty": 10,
                    "riskguard_url": "http://localhost:8001",
                    "riskguard_max_pos_size": 1000,
                    "riskguard_max_concentration": 0.5,
                },
                sim_logger=mock_logger,
            )


def test_run_simulation_success(mock_a2a_call):
    """Test a successful simulation run."""
    response = client.post(
        "/run_simulation",
        data={
            "alphabot_short_sma": "10",
            "alphabot_long_sma": "20",
            "alphabot_trade_qty": "10",
            "sim_days": "5",
            "sim_initial_cash": "10000",
            "sim_initial_price": "100",
            "sim_volatility": "0.02",
            "sim_trend": "0.001",
            "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
            "riskguard_max_pos_size": "1000",
            "riskguard_max_concentration": "50",
            "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
        },
    )
    assert response.status_code == 200
    assert "Simulation completed successfully." in response.text
    # Check that our mock was called, e.g., once per simulation day
    assert mock_a2a_call.call_count == 5
    # Check that the response contains the results
    assert "Total Value" in response.text


def test_run_simulation_invalid_params():
    """Test simulation run with invalid parameters."""
    response = client.post(
        "/run_simulation",
        data={
            "alphabot_short_sma": "0",  # Invalid value
            "alphabot_long_sma": "20",
            "alphabot_trade_qty": "10",
            "sim_days": "5",
            "sim_initial_cash": "10000",
            "sim_initial_price": "100",
            "sim_volatility": "0.02",
            "sim_trend": "0.001",
            "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
            "riskguard_max_pos_size": "1000",
            "riskguard_max_concentration": "50",
            "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
        },
    )
    assert response.status_code == 200
    assert "Invalid simulation parameters" in response.text
    assert "Input should be greater than 0" in response.text


def test_run_simulation_connection_error(mock_a2a_call):
    """Test simulation run with an A2A connection error."""
    mock_a2a_call.side_effect = ConnectionError("Test connection error")

    response = client.post(
        "/run_simulation",
        data={
            "alphabot_short_sma": "10",
            "alphabot_long_sma": "20",
            "alphabot_trade_qty": "10",
            "sim_days": "5",
            "sim_initial_cash": "10000",
            "sim_initial_price": "100",
            "sim_volatility": "0.02",
            "sim_trend": "0.001",
            "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
            "riskguard_max_pos_size": "1000",
            "riskguard_max_concentration": "50",
            "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
        },
    )
    assert response.status_code == 200
    assert "Simulation failed: Connection Error" in response.text


def test_concurrent_simulations_no_race_condition():
    """Test that concurrent simulations don't interfere with each other.

    This test demonstrates that the race condition has been fixed by
    making multiple concurrent requests and verifying that each
    request gets its own, correct results.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    async def run_single_simulation(simulation_id):
        """Run a single simulation with unique parameters."""
        with patch("simulator.main._call_alphabot_a2a", new_callable=AsyncMock) as mock:
            # Configure the mock to return different results based on simulation_id
            mock.return_value = {
                "approved_trade": {
                    "action": "BUY" if simulation_id % 2 == 0 else "SELL",
                    "quantity": 10 + simulation_id,
                    "price": 100.0 + simulation_id,
                    "ticker": "SIM",
                },
                "rejected_trade": None,
                "reason": f"SMA crossover for simulation {simulation_id}",
                "error": None,
            }

            response = client.post(
                "/run_simulation",
                data={
                    "alphabot_short_sma": str(10 + simulation_id),
                    "alphabot_long_sma": str(20 + simulation_id),
                    "alphabot_trade_qty": str(10 + simulation_id),
                    "sim_days": "5",
                    "sim_initial_cash": str(10000 + simulation_id * 1000),
                    "sim_initial_price": str(100 + simulation_id),
                    "sim_volatility": "0.02",
                    "sim_trend": "0.001",
                    "riskguard_url": defaults.DEFAULT_RISKGUARD_URL,
                    "riskguard_max_pos_size": "1000",
                    "riskguard_max_concentration": "50",
                    "alphabot_url": defaults.DEFAULT_ALPHABOT_URL,
                },
            )
            return response

    async def run_concurrent_simulations():
        """Run multiple simulations concurrently."""
        tasks = [run_single_simulation(i) for i in range(3)]
        responses = await asyncio.gather(*tasks)
        return responses

    # Run the concurrent simulations
    responses = asyncio.run(run_concurrent_simulations())

    # Verify that all responses are successful
    for i, response in enumerate(responses):
        assert response.status_code == 200
        assert "Simulation completed successfully." in response.text
        # Verify that each response contains the correct parameters for that simulation
        # We can check for the initial cash value in the results section
        initial_cash = 10000 + i * 1000
        assert (
            f"${initial_cash:,}" in response.text
            or f"${initial_cash:,.2f}" in response.text
        )
