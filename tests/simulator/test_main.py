"""Tests for the simulator's main application."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from a2a.client import ClientFactory
from a2a.helpers import new_data_part
from a2a.types import (
    Message,
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


def test_health_check() -> None:
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_read_main() -> None:
    """Test the main endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Agentic Trading Simulator" in response.text


@pytest.mark.asyncio
async def test_call_alphabot_a2a_with_factory(
    mock_simulator_a2a,
    test_agent_card,
    mock_a2a_send_message_generator,
) -> None:
    """Verify that _call_alphabot_a2a correctly uses the ClientFactory."""
    mock_logger = MagicMock()
    mock_factory_instance = mock_simulator_a2a["mock_factory_instance"]
    mock_a2a_client = mock_simulator_a2a["mock_a2a_client"]
    mock_resolver_instance = mock_simulator_a2a["mock_resolver_instance"]

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
        role=Role.ROLE_AGENT,
        parts=[new_data_part(trade_outcome.model_dump(mode="json"))],
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
        httpx_client=mock_factory_instance._config.httpx_client,
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
async def test_call_alphabot_a2a_factory_raises_transport_error() -> None:
    """Test that _call_alphabot_a2a handles transport resolution errors."""
    mock_factory = AsyncMock(spec=ClientFactory)
    mock_logger = MagicMock()

    # Configure the factory mock to have the necessary attributes
    mock_factory._config = MagicMock()
    mock_factory._config.httpx_client = AsyncMock()

    from a2a.client.errors import A2AClientError

    # Configure the factory mock to raise an error on card resolution
    mock_factory.create.side_effect = A2AClientError(
        "Resolution failed",
    )

    with patch("simulator.main.A2ACardResolver") as mock_resolver:
        mock_resolver.return_value.get_agent_card.side_effect = A2AClientError(
            "Resolution failed",
        )

        with pytest.raises(ConnectionError):
            await _call_alphabot_a2a(
                client_factory=mock_factory,
                httpx_client=mock_factory._config.httpx_client,
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


def test_run_simulation_success(mock_a2a_call) -> None:
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


def test_run_simulation_success_sell() -> None:
    """Test a successful simulation run where a SELL trade is approved."""
    call_count = 0

    async def mock_sell_a2a(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
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
        return {
            "approved_trade": {
                "action": "SELL",
                "quantity": 5,
                "price": 100.0,
                "ticker": "SIM",
            },
            "rejected_trade": None,
            "reason": "SMA crossover",
            "error": None,
        }

    with patch("simulator.main._call_alphabot_a2a", new_callable=AsyncMock) as mock:
        mock.side_effect = mock_sell_a2a
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
        assert mock.call_count == 5
        assert "Approved Sell" in response.text


def test_run_simulation_invalid_params() -> None:
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


def test_run_simulation_connection_error(mock_a2a_call) -> None:
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


def test_concurrent_simulations_no_race_condition() -> None:
    """Test that concurrent simulations don't interfere with each other.

    This test demonstrates that the race condition has been fixed by
    making multiple concurrent requests and verifying that each
    request gets its own, correct results.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    async def run_single_simulation(simulation_id: int):
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

            return client.post(
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

    async def run_concurrent_simulations():
        """Run multiple simulations concurrently."""
        tasks = [run_single_simulation(i) for i in range(3)]
        return await asyncio.gather(*tasks)

    # Run the concurrent simulations
    responses = asyncio.run(run_concurrent_simulations())

    # Verify that all responses are successful
    for i, response in enumerate(responses):
        assert response.status_code == 200
        assert "Simulation completed successfully." in response.text
        initial_cash = 10000 + i * 1000
        assert (
            f"${initial_cash:,}" in response.text
            or f"${initial_cash:,.2f}" in response.text
        )


def test_format_currency() -> None:
    """Tests the format_currency utility function."""
    from simulator.main import format_currency

    assert format_currency(None) == "N/A"

    # ValueError fallback
    with patch("locale.currency", side_effect=ValueError("Borked")):
        assert format_currency(1234.56) == "$1,234.56"

    # Generic exception fallback
    with patch("locale.currency", side_effect=RuntimeError("Serious Error")):
        assert format_currency(1234.56) == "$1,234.56"

    # Locale success path
    with patch("locale.currency", return_value="€1.234,56"):
        assert format_currency(1234.56) == "€1.234,56"


def test_ui_log_handler() -> None:
    """Tests that UILogHandler appends records correctly."""
    import logging

    from simulator.main import UILogHandler

    log_list: list[str] = []
    handler = UILogHandler(log_list)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger = logging.getLogger("test_ui_log")
    logger.addHandler(handler)
    logger.warning("hello logging world")
    assert "hello logging world" in log_list


def test_simulation_run_params() -> None:
    """Tests that SimulationRunParams handles defaults, validations, and conversions."""
    from pydantic import ValidationError

    from simulator.main import SimulationRunParams

    # Valid parameters
    params = SimulationRunParams(
        alphabot_short_sma=5,
        alphabot_long_sma=30,
        alphabot_trade_qty=100,
        sim_days=100,
        sim_initial_cash=100000.0,
        sim_initial_price=100.0,
        sim_volatility=0.02,
        sim_trend=0.0005,
        riskguard_url="http://127.0.0.1:8080",
        riskguard_max_pos_size=10000.0,
        riskguard_max_concentration=50,
        alphabot_url="http://127.0.0.1:8081",
    )
    d = params.to_dict()
    assert d["alphabot_short_sma"] == 5
    assert d["riskguard_max_concentration"] == 50

    # Boundaries check
    with pytest.raises(ValidationError):
        # sim_days too high (must be <= 10000)
        SimulationRunParams.model_validate(
            {
                "alphabot_short_sma": 5,
                "sim_days": 10001,
                "sim_initial_cash": 100000.0,
                "sim_initial_price": 100.0,
                "sim_volatility": 1.0,
                "sim_trend": -0.1,
                "riskguard_url": "http://127.0.0.1:8080",
                "riskguard_max_pos_size": 10000.0,
                "riskguard_max_concentration": 50,
                "alphabot_url": "http://127.0.0.1:8081",
            },
        )


def test_create_results_figure() -> None:
    """Tests that _create_results_figure handles inputs and generates traces correctly."""
    import pandas as pd

    from simulator.main import _create_results_figure

    df = pd.DataFrame(
        {
            "Price": [100.0, 101.0],
            "SMA_Short": [99.0, 100.0],
            "SMA_Long": [98.0, 99.0],
            "Cash": [10000.0, 10000.0],
            "Shares": [0, 0],
            "TotalValue": [10000.0, 10000.0],
        },
        index=[1, 2],
    )
    df.index.name = "Day"
    params = {"alphabot_short_sma": 5, "alphabot_long_sma": 30}
    trade_markers: dict[str, list[Any]] = {
        "approved_buy_days": [1],
        "approved_buy_prices": [100.0],
        "rejected_buy_days": [],
        "rejected_buy_prices": [],
        "approved_sell_days": [],
        "approved_sell_prices": [],
        "rejected_sell_days": [],
        "rejected_sell_prices": [],
    }
    fig = _create_results_figure(df, params, trade_markers)
    assert fig is not None
    trace_names = [getattr(t, "name", "") for t in fig.data]
    assert "Price" in trace_names
    assert "Approved Buy" in trace_names

    # Empty DataFrame case
    empty_df = pd.DataFrame(
        columns=[
            "Price",
            "SMA_Short",
            "SMA_Long",
            "Cash",
            "Shares",
            "TotalValue",
        ],
    )
    empty_df.index.name = "Day"
    fig_empty = _create_results_figure(empty_df, params, trade_markers)
    assert fig_empty is not None


def test_render_error_page() -> None:
    """Tests that _render_error_page properly formats and renders the page."""
    from fastapi import Request

    from simulator.main import _render_error_page

    req = MagicMock(spec=Request)
    req.scope = {"type": "http"}
    response = _render_error_page(req, "Testing Error Page", {})
    assert response.status_code == 200
    assert "Testing Error Page" in bytes(response.body).decode()


@pytest.mark.asyncio
async def test_lifespan() -> None:
    """Tests lifespan startup and locale settings."""
    import locale

    from simulator.main import lifespan

    # Success setting locale
    with patch("locale.setlocale") as mock_set:
        async with lifespan(app):
            pass
        mock_set.assert_called_once_with(locale.LC_ALL, "en_US.UTF-8")

    # Error handling
    with patch("locale.setlocale", side_effect=locale.Error("Failed")):
        async with lifespan(app):
            pass  # Should handle gracefully and not crash


@pytest.mark.asyncio
async def test_call_alphabot_a2a_stream_handling_and_parsing(
    mock_simulator_a2a,
    test_agent_card,
) -> None:
    """Tests _call_alphabot_a2a response parsing and event stream paths."""
    from unittest.mock import AsyncMock, MagicMock

    from a2a.types import Role, TaskState
    from a2a.types.a2a_pb2 import (
        Message,
        StreamResponse,
        Task,
        TaskStatus,
        TaskStatusUpdateEvent,
    )

    from simulator.main import _call_alphabot_a2a

    mock_logger = MagicMock()
    mock_factory_instance = mock_simulator_a2a["mock_factory_instance"]
    mock_a2a_client = mock_simulator_a2a["mock_a2a_client"]
    mock_resolver_instance = mock_simulator_a2a["mock_resolver_instance"]
    mock_resolver_instance.get_agent_card.return_value = test_agent_card

    async def run_call(yield_values):
        async def mock_stream(*args, **kwargs):
            for val in yield_values:
                yield val

        mock_a2a_client.send_message = mock_stream
        return await _call_alphabot_a2a(
            client_factory=mock_factory_instance,
            httpx_client=AsyncMock(),
            alphabot_url="http://test.com",
            session_id="session-123",
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
                "riskguard_max_concentration": 50,
            },
            sim_logger=mock_logger,
        )

    # 1. NO_ACTION
    outcome_no_action = TradeOutcome(
        status=TradeStatus.NO_ACTION,
        reason="No SMA cross",
    )
    msg_no_action = Message(
        message_id="m1",
        role=Role.ROLE_AGENT,
        parts=[new_data_part(outcome_no_action.model_dump(mode="json"))],
    )
    res = await run_call([StreamResponse(message=msg_no_action)])
    assert res["reason"] == "No SMA cross"
    assert res["approved_trade"] is None

    # 2. ERROR
    outcome_error = TradeOutcome(status=TradeStatus.ERROR, reason="Calculation failed")
    msg_error = Message(
        message_id="m2",
        role=Role.ROLE_AGENT,
        parts=[new_data_part(outcome_error.model_dump(mode="json"))],
    )
    res_err = await run_call([StreamResponse(message=msg_error)])
    assert res_err["reason"] == "Calculation failed"

    # 3. Empty parts
    msg_empty = Message(message_id="m3", role=Role.ROLE_AGENT, parts=[])
    res_empty = await run_call([StreamResponse(message=msg_empty)])
    assert res_empty["error"] == "AlphaBot Response Format Issue or No Decision"

    # 4. Part fails validation
    msg_invalid = Message(
        message_id="m4",
        role=Role.ROLE_AGENT,
        parts=[new_data_part({"status": "INVALID_STATUS", "reason": "broken"})],
    )
    res_invalid = await run_call([StreamResponse(message=msg_invalid)])
    assert "A2A Processing Error" in res_invalid["error"]

    # 5. status_update event (TaskState.TASK_STATE_FAILED)
    status_update = TaskStatusUpdateEvent(
        status=TaskStatus(
            state=TaskState.TASK_STATE_FAILED,
        ),
    )
    res_failed = await run_call([StreamResponse(status_update=status_update)])
    assert res_failed["error"] == "AlphaBot task execution failed."

    # 6. task event (should be debug logged but not terminate)
    task_event = StreamResponse(
        task=Task(id="task-123", status=TaskStatus(state=TaskState.TASK_STATE_WORKING)),
    )
    res_task = await run_call([task_event, StreamResponse(message=msg_no_action)])
    assert res_task["reason"] == "No SMA cross"


@pytest.mark.asyncio
async def test_run_simulation_async_edge_cases() -> None:
    """Tests run_simulation_async orchestration edge cases."""
    from simulator.main import run_simulation_async

    # 0-day simulation
    params_0 = {
        "alphabot_short_sma": 5,
        "alphabot_long_sma": 30,
        "alphabot_trade_qty": 100,
        "sim_days": 0,
        "sim_initial_cash": 100000.0,
        "sim_initial_price": 100.0,
        "sim_volatility": 0.02,
        "sim_trend": 0.0005,
        "riskguard_url": "http://127.0.0.1:8080",
        "riskguard_max_pos_size": 10000.0,
        "riskguard_max_concentration": 50,
        "alphabot_url": "http://127.0.0.1:8081",
    }
    res_0 = await run_simulation_async(params_0)
    assert res_0["success"] is True
    assert "Final Portfolio" in res_0["signals_log"]

    # Mixed results: Day 2 returns ERROR
    day_counter = 0

    async def mock_call_error(*args, **kwargs):
        nonlocal day_counter
        day_counter += 1
        if day_counter == 2:
            return {
                "approved_trade": None,
                "rejected_trade": None,
                "reason": "Test error",
                "error": "Failed request mock",
            }
        return {
            "approved_trade": None,
            "rejected_trade": None,
            "reason": "No action",
            "error": None,
        }

    params_error = params_0.copy()
    params_error["sim_days"] = 3
    with patch("simulator.main._call_alphabot_a2a", side_effect=mock_call_error):
        res_error = await run_simulation_async(params_error)
        assert res_error["success"] is True
        assert "A2A ERROR: Failed request mock" in res_error["signals_log"]

    # Trade execution fails (insufficient funds)
    async def mock_call_buy(*args, **kwargs):
        return {
            "approved_trade": {
                "action": "BUY",
                "quantity": 10000,
                "price": 100.0,
                "ticker": "TECH",
            },
            "rejected_trade": None,
            "reason": "Large buy approved",
            "error": None,
        }

    params_fail = params_0.copy()
    params_fail["sim_days"] = 1
    params_fail["sim_initial_cash"] = 100.0
    with patch("simulator.main._call_alphabot_a2a", side_effect=mock_call_buy):
        res_fail = await run_simulation_async(params_fail)
        assert res_fail["success"] is True
        assert "Execution FAILED." in res_fail["signals_log"]
