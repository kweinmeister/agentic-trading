"""Tests for the AlphaBot agent."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

# ADK Imports
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types as genai_types

from alphabot.agent import A2ARiskCheckTool, AlphaBotAgent
from common.config import DEFAULT_TICKER
from common.models import PortfolioState


def _get_text(event: Event) -> str:
    """Helper to safely extract text content from an Event for assertions."""
    assert event.content is not None
    assert event.content.parts is not None
    assert len(event.content.parts) > 0
    text = event.content.parts[0].text
    assert text is not None
    return text


def test_alphabot_agent_instantiation() -> None:
    """Tests basic instantiation of the AlphaBotAgent."""
    try:
        agent = AlphaBotAgent(stock_ticker="TEST_TICKER")
        assert agent is not None
        assert agent.name == "AlphaBot"
        assert agent.ticker == "TEST_TICKER"
        assert agent.tools is not None
        assert len(agent.tools) == 1
        assert isinstance(agent.tools[0], A2ARiskCheckTool)
        default_agent = AlphaBotAgent()
        assert default_agent.ticker == DEFAULT_TICKER
    except Exception as e:
        pytest.fail(f"AlphaBotAgent instantiation failed: {e}")


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_no_signal(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
) -> None:
    """Tests _run_async_impl when no crossover signal is generated."""
    adk_ctx.session.state = {"should_be_long": False}

    # Mock the A2ARiskCheckTool's run_async method
    with patch.object(
        A2ARiskCheckTool,
        "run_async",
        new_callable=AsyncMock,
    ) as mock_run_async:
        mock_run_async.return_value = None

        input_data = alphabot_input_data_factory(
            historical_prices=[100, 101, 102, 103, 104, 105],
            current_price=105.5,
            short_sma_period=2,
            long_sma_period=4,
            day=1,
        )
        adk_ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )

        events = []
        async for event in agent._run_async_impl(adk_ctx):
            events.append(event)

        assert len(events) == 1
        final_event = events[0]
        assert final_event.author == agent.name
        assert "No signal (Conditions not met)" in _get_text(final_event)
        assert not final_event.actions.state_delta


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_buy_approved(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
) -> None:
    """Tests _run_async_impl for a BUY signal that is approved by RiskGuard."""
    adk_ctx.session.state = {"should_be_long": False}

    # Mock the A2ARiskCheckTool's run_async method to return an approved response
    with patch.object(A2ARiskCheckTool, "run_async") as mock_run_async:
        mock_run_async.return_value = Event(
            author="a2a_risk_check",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": True,
                                "reason": "Trade adheres to risk rules.",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

        input_data = alphabot_input_data_factory(
            historical_prices=historical_prices_buy_signal,
            current_price=129.0,
            day=35,
            portfolio_state={"cash": 10000, "shares": 0, "total_value": 10000},
        )
        adk_ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )

        events = []
        async for event in agent._run_async_impl(adk_ctx):
            events.append(event)

        # Expect two events: one for the proposal, one for the final result
        assert len(events) == 2

        # Check the proposal event
        proposal_event = events[0]
        assert proposal_event.author == agent.name
        assert "Proposing BUY" in _get_text(proposal_event)

        # Check the final event
        final_event = events[1]
        assert final_event.author == agent.name
        assert "Trade Approved (A2A)" in _get_text(final_event)
        assert final_event.actions.state_delta["should_be_long"] is True
        assert "approved_trade" in final_event.actions.state_delta


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_sell_approved(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_sell_signal,
) -> None:
    """Tests _run_async_impl for a SELL signal that is approved by RiskGuard."""
    adk_ctx.session.state = {"should_be_long": True}

    # Mock the A2ARiskCheckTool's run_async method to return an approved response
    with patch.object(A2ARiskCheckTool, "run_async") as mock_run_async:
        mock_run_async.return_value = Event(
            author="a2a_risk_check",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": True,
                                "reason": "Trade adheres to risk rules.",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

        input_data = alphabot_input_data_factory(
            historical_prices=historical_prices_sell_signal,
            current_price=66.0,
            portfolio_state={"cash": 10000, "shares": 100, "total_value": 17000},
            day=35,
        )
        adk_ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )

        events = []
        async for event in agent._run_async_impl(adk_ctx):
            events.append(event)

        # Expect two events: one for the proposal, one for the final result
        assert len(events) == 2

        # Check the proposal event
        proposal_event = events[0]
        assert proposal_event.author == agent.name
        assert "Proposing SELL" in _get_text(proposal_event)

        # Check the final event
        final_event = events[1]
        assert final_event.author == agent.name
        assert "Trade Approved (A2A)" in _get_text(final_event)
        assert final_event.actions.state_delta["should_be_long"] is False
        assert "approved_trade" in final_event.actions.state_delta


@pytest.mark.parametrize(
    ("sma_short", "sma_long", "prev_sma_short", "prev_sma_long", "expected_signal"),
    [
        (95.0, 100.0, 102.0, 101.0, "SELL"),  # Death cross
        (10.0, 10.0, 10.0, 10.0, None),  # Equal, no crossover
        (11.0, 10.0, 10.0, 10.0, "BUY"),  # Buy crossover (touch and separate)
        (9.0, 10.0, 10.0, 10.0, "SELL"),  # Sell crossover (touch and separate)
        (None, 100.0, 102.0, 101.0, None),  # Missing short SMA
        (95.0, None, 102.0, 101.0, None),  # Missing long SMA
        (95.0, 100.0, None, 101.0, None),  # Missing prev short SMA
        (95.0, 100.0, 102.0, None, None),  # Missing prev long SMA
    ],
)
def test_generate_signal_scenarios(
    agent: AlphaBotAgent,
    sma_short: float | None,
    sma_long: float | None,
    prev_sma_short: float | None,
    prev_sma_long: float | None,
    expected_signal: str | None,
) -> None:
    """Test _generate_signal crossover condition boundaries and missing values."""
    signal = agent._generate_signal(
        sma_short,
        sma_long,
        prev_sma_short,
        prev_sma_long,
        "test_invocation",
    )
    assert signal == expected_signal


def test_determine_trade_proposal_no_buy_when_long(agent: AlphaBotAgent) -> None:
    """Tests that _determine_trade_proposal returns None for a BUY signal when already long."""
    portfolio_state = PortfolioState(cash=10000, shares=10, total_value=11000)
    proposal = agent._determine_trade_proposal(
        signal="BUY",
        should_be_long=True,
        portfolio_state=portfolio_state,
        current_price=100.0,
        trade_quantity=10,
        last_rejected_trade=None,
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_invalid_input(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
) -> None:
    """Tests that the agent handles malformed input data gracefully."""
    adk_ctx.user_content = genai_types.Content(
        parts=[genai_types.Part(text="not a valid json")],
    )

    events = []
    async for event in agent._run_async_impl(adk_ctx):
        events.append(event)

    assert len(events) == 1
    final_event = events[0]
    assert final_event.author == agent.name
    assert "Error: Invalid input data structure or values." in _get_text(final_event)


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_buy_rejected(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
) -> None:
    """Tests _run_async_impl for a BUY signal that is rejected by RiskGuard."""
    adk_ctx.session.state = {"should_be_long": False}

    with patch.object(A2ARiskCheckTool, "run_async") as mock_run_async:
        mock_run_async.return_value = Event(
            author="a2a_risk_check",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": False,
                                "reason": "Exceeds max position size.",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

        input_data = alphabot_input_data_factory(
            historical_prices=historical_prices_buy_signal,
            current_price=129.0,
            day=35,
            portfolio_state={"cash": 10000, "shares": 0, "total_value": 10000},
        )
        adk_ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )

        events = []
        async for event in agent._run_async_impl(adk_ctx):
            events.append(event)

        assert len(events) == 2
        final_event = events[1]
        assert final_event.author == agent.name
        assert "Trade Rejected (A2A)" in _get_text(final_event)
        assert "should_be_long" not in final_event.actions.state_delta
        assert "rejected_trade_proposal" in final_event.actions.state_delta


def test_determine_trade_proposal_no_sell_when_not_long(agent: AlphaBotAgent) -> None:
    """Tests that _determine_trade_proposal returns None for a SELL signal when not long."""
    portfolio_state = PortfolioState(cash=10000, shares=0, total_value=10000)
    proposal = agent._determine_trade_proposal(
        signal="SELL",
        should_be_long=False,
        portfolio_state=portfolio_state,
        current_price=100.0,
        trade_quantity=10,
        last_rejected_trade=None,
    )
    assert proposal is None


def test_determine_trade_proposal_no_sell_when_long_no_shares(
    agent: AlphaBotAgent,
) -> None:
    """Tests that _determine_trade_proposal returns None for a SELL signal when long but with no shares."""
    portfolio_state = PortfolioState(cash=10000, shares=0, total_value=10000)
    proposal = agent._determine_trade_proposal(
        signal="SELL",
        should_be_long=True,
        portfolio_state=portfolio_state,
        current_price=100.0,
        trade_quantity=10,
        last_rejected_trade=None,
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_state_correction_sell_no_shares(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_sell_signal,
) -> None:
    """Tests _run_async_impl for a SELL signal that triggers state correction due to no shares held."""
    adk_ctx.session.state = {"should_be_long": True}

    # Mock the A2ARiskCheckTool's run_async method (not called in this path, but good practice)
    with patch.object(A2ARiskCheckTool, "run_async") as mock_run_async:
        mock_run_async.return_value = Event(
            author="a2a_risk_check",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": True,
                                "reason": "Trade adheres to risk rules.",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

        input_data = alphabot_input_data_factory(
            historical_prices=historical_prices_sell_signal,
            current_price=66.0,
            portfolio_state={
                "cash": 10000,
                "shares": 0,
                "total_value": 10000,
            },  # Shares are 0 here
            day=35,
        )
        adk_ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )

        events = []
        async for event in agent._run_async_impl(adk_ctx):
            events.append(event)

        # Expect one event for state correction
        assert len(events) == 1
        final_event = events[0]
        assert final_event.author == agent.name
        assert final_event.turn_complete is True
        assert "State correction" in _get_text(final_event)
        assert final_event.actions.state_delta["should_be_long"] is False


@pytest.mark.asyncio
async def test_alphabot_concurrency(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
) -> None:
    """Tests that the agent can handle concurrent requests without race conditions."""
    adk_ctx.session.state = {"should_be_long": False}

    # Mock the A2ARiskCheckTool's run_async method to return an approved response
    with patch.object(A2ARiskCheckTool, "run_async") as mock_run_async:
        mock_run_async.return_value = Event(
            author="a2a_risk_check",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": True,
                                "reason": "Trade adheres to risk rules.",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

        input_data = alphabot_input_data_factory(
            historical_prices=historical_prices_buy_signal,
            current_price=129.0,
            day=35,
            portfolio_state={"cash": 10000, "shares": 0, "total_value": 10000},
        )

        async def run_agent_and_collect_events(index: int):
            # Create isolated sessions and contexts for each request
            from google.adk.sessions import InMemorySessionService

            session_service = InMemorySessionService()
            session = await session_service.create_session(
                app_name="test_app",
                user_id=f"test_user_{index}",
            )
            session.state = {"should_be_long": False}
            ctx = InvocationContext(
                agent=agent,
                session_service=session_service,
                invocation_id=f"test_invocation_{index}",
                session=session,
            )
            ctx.user_content = genai_types.Content(
                parts=[genai_types.Part(text=input_data.model_dump_json())],
            )
            return [event async for event in agent._run_async_impl(ctx)]

        # Run the agent multiple times concurrently
        tasks = [run_agent_and_collect_events(i) for i in range(5)]
        results = await asyncio.gather(*tasks)

        # Each invocation should produce 2 events (proposal and final)
        for events in results:
            # The agent is not designed to be stateful across concurrent requests in this manner
            # so we just check that each request was processed independently and correctly.
            assert len(events) == 2
            final_event = events[1]
            assert final_event.author == agent.name
            assert "Trade Approved (A2A)" in _get_text(final_event)
            assert final_event.actions.state_delta["should_be_long"] is True


@pytest.mark.asyncio
async def test_alphabot_does_not_repropose_rejected_trade(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
) -> None:
    """Test that AlphaBot does not re-propose a recently rejected trade.

    This test simulates a scenario where a BUY signal is generated, the
    resulting trade is rejected by RiskGuard, and then the agent is run
    again with the same market conditions, ensuring it doesn't propose the
    same trade.
    """
    # 1. Initial State: Agent is not long
    adk_ctx.session.state = {"should_be_long": False}

    # 2. Mock RiskGuard to always REJECT trades
    with patch.object(A2ARiskCheckTool, "run_async") as mock_run_async:
        mock_run_async.return_value = Event(
            author="a2a_risk_check",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": False,
                                "reason": "Insufficient cash for BUY.",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

        # 3. Market data that generates a BUY signal
        input_data = alphabot_input_data_factory(
            historical_prices=historical_prices_buy_signal,
            current_price=129.0,
            day=35,
            portfolio_state={"cash": 100, "shares": 0, "total_value": 100},
        )
        adk_ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )

        # --- First Invocation: Propose and get rejected ---
        events_run1 = [event async for event in agent._run_async_impl(adk_ctx)]

        # Expect a rejection (proposal and rejection events)
        assert len(events_run1) == 2
        final_event_run1 = events_run1[1]
        assert "Trade Rejected (A2A)" in _get_text(final_event_run1)
        assert "should_be_long" not in final_event_run1.actions.state_delta

        # --- Second Invocation: Should NOT propose again ---
        # Update the session state with the delta from the first run, which includes the rejected trade
        adk_ctx.session.state.update(final_event_run1.actions.state_delta)

        # Rerun with the exact same input
        events_run2 = [event async for event in agent._run_async_impl(adk_ctx)]

        # Assert that NO new trade was proposed
        assert len(events_run2) == 1
        final_event_run2 = events_run2[0]
        assert (
            "Signal generated, but no trade action needed based on current state or recent rejections."
            in _get_text(final_event_run2)
        )


def test_determine_trade_proposal_rejects_sell_if_quantity_exceeds_shares(
    agent: AlphaBotAgent,
) -> None:
    """Test that `_determine_trade_proposal` returns None for a SELL signal.

    This occurs when the configured trade quantity exceeds the number of
    shares held.
    """
    portfolio_state = PortfolioState(cash=10000, shares=5, total_value=10500)
    trade_quantity = 10  # Attempting to sell more than owned
    proposal = agent._determine_trade_proposal(
        signal="SELL",
        should_be_long=True,
        portfolio_state=portfolio_state,
        current_price=100.0,
        trade_quantity=trade_quantity,
        last_rejected_trade=None,
    )
    assert proposal is None


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_insufficient_data(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
) -> None:
    """Tests that the agent handles insufficient historical data gracefully."""
    adk_ctx.session.state = {"should_be_long": False}

    input_data = alphabot_input_data_factory(
        historical_prices=[100, 101, 102],  # Not enough data for long_sma_period=10
        current_price=105.5,
        short_sma_period=5,
        long_sma_period=10,
        day=1,
    )
    adk_ctx.user_content = genai_types.Content(
        parts=[genai_types.Part(text=input_data.model_dump_json())],
    )

    events = []
    async for event in agent._run_async_impl(adk_ctx):
        events.append(event)

    assert len(events) == 1
    final_event = events[0]
    assert final_event.author == agent.name
    assert "No signal yet (calculating SMAs)." in _get_text(final_event)
    assert not final_event.actions.state_delta


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_buy_signal_corrects_state(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
) -> None:
    """Test that if a BUY signal is generated but the agent is already long, it produces a NO_ACTION outcome."""
    adk_ctx.session.state = {"should_be_long": True}  # Agent is already long

    input_data = alphabot_input_data_factory(
        historical_prices=historical_prices_buy_signal,
        current_price=129.0,
        day=35,
        portfolio_state={"cash": 10000, "shares": 10, "total_value": 11290},
    )
    adk_ctx.user_content = genai_types.Content(
        parts=[genai_types.Part(text=input_data.model_dump_json())],
    )

    events = []
    async for event in agent._run_async_impl(adk_ctx):
        events.append(event)

    assert len(events) == 1
    final_event = events[0]
    assert final_event.author == agent.name
    assert (
        "Signal generated, but no trade action needed based on current state or recent rejections."
        in _get_text(final_event)
    )
    assert not final_event.actions.state_delta


def test_calculate_indicators_edge_cases(agent: AlphaBotAgent) -> None:
    """Test _calculate_indicators boundary conditions."""
    # 1. Exactly long_sma_period points
    short_period = 3
    long_period = 5
    prices = [10.0, 11.0, 12.0, 13.0, 14.0]
    sma_short, sma_long, prev_sma_short, prev_sma_long = agent._calculate_indicators(
        prices,
        short_period,
        long_period,
        "test_inv",
    )
    assert sma_short is not None
    assert sma_long is not None
    assert prev_sma_short is not None
    assert (
        prev_sma_long is None
    )  # only 4 prices in previous, which is < 5 (long_period)

    # 2. Short and long periods are equal
    sma_short, sma_long, prev_sma_short, prev_sma_long = agent._calculate_indicators(
        prices,
        3,
        3,
        "test_inv",
    )
    assert sma_short == sma_long
    assert prev_sma_short == prev_sma_long

    # 3. Empty history
    sma_short, sma_long, prev_sma_short, prev_sma_long = agent._calculate_indicators(
        [],
        short_period,
        long_period,
        "test_inv",
    )
    assert sma_short is None
    assert sma_long is None
    assert prev_sma_short is None
    assert prev_sma_long is None


def test_determine_trade_proposal_rejected_history(agent: AlphaBotAgent) -> None:
    """Test _determine_trade_proposal respects rejected trade history."""
    portfolio_state = PortfolioState(cash=10000, shares=0, total_value=10000)
    last_rejected = {
        "action": "BUY",
        "ticker": "TECH",
        "quantity": 10,
        "price": 100.0,
    }

    agent.ticker = "TECH"

    # 1. Identical trade proposal should be skipped
    proposal = agent._determine_trade_proposal(
        signal="BUY",
        should_be_long=False,
        portfolio_state=portfolio_state,
        current_price=100.0,
        trade_quantity=10,
        last_rejected_trade=last_rejected,
    )
    assert proposal is None

    # 2. Proposal with different quantity is allowed
    proposal = agent._determine_trade_proposal(
        signal="BUY",
        should_be_long=False,
        portfolio_state=portfolio_state,
        current_price=100.0,
        trade_quantity=20,
        last_rejected_trade=last_rejected,
    )
    assert proposal is not None
    assert proposal["quantity"] == 20

    # 3. Proposal with different ticker is allowed
    agent.ticker = "OTHER"
    proposal = agent._determine_trade_proposal(
        signal="BUY",
        should_be_long=False,
        portfolio_state=portfolio_state,
        current_price=100.0,
        trade_quantity=10,
        last_rejected_trade=last_rejected,
    )
    assert proposal is not None
    assert proposal["ticker"] == "OTHER"
