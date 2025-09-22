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


def test_alphabot_agent_instantiation():
    """Tests basic instantiation of the AlphaBotAgent."""
    try:
        agent = AlphaBotAgent(stock_ticker="TEST_TICKER")
        assert agent is not None
        assert agent.name == "AlphaBot"
        assert agent.ticker == "TEST_TICKER"
        assert agent.tools is not None and len(agent.tools) == 1
        assert agent.tools is not None and isinstance(agent.tools[0], A2ARiskCheckTool)
        default_agent = AlphaBotAgent()
        assert default_agent.ticker == DEFAULT_TICKER
    except Exception as e:
        pytest.fail(f"AlphaBotAgent instantiation failed: {e}")


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_no_signal(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
):
    """Tests _run_async_impl when no crossover signal is generated."""
    adk_ctx.session.state = {"should_be_long": False}

    # Mock the A2ARiskCheckTool's run_async method
    with patch.object(
        A2ARiskCheckTool,
        "run_async",
        new_callable=AsyncMock,
    ) as mock_run_async:

        async def mock_tool_response(*args, **kwargs):
            # Simulate no response from the tool for this test case
            if False:  # pragma: no cover
                yield

        mock_run_async.return_value = mock_tool_response()

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
        assert "No signal (Conditions not met)" in final_event.content.parts[0].text
        assert not final_event.actions.state_delta


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_buy_approved(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
):
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
        assert "Proposing BUY" in proposal_event.content.parts[0].text

        # Check the final event
        final_event = events[1]
        assert final_event.author == agent.name
        assert "Trade Approved (A2A)" in final_event.content.parts[0].text
        assert final_event.actions.state_delta["should_be_long"] is True
        assert "approved_trade" in final_event.actions.state_delta


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_sell_approved(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_sell_signal,
):
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
        assert "Proposing SELL" in proposal_event.content.parts[0].text

        # Check the final event
        final_event = events[1]
        assert final_event.author == agent.name
        assert "Trade Approved (A2A)" in final_event.content.parts[0].text
        assert final_event.actions.state_delta["should_be_long"] is False
        assert "approved_trade" in final_event.actions.state_delta


@pytest.mark.asyncio
async def test_generate_signal_sell_death_cross(agent: AlphaBotAgent):
    """Tests that _generate_signal correctly identifies a 'SELL' signal on a death cross."""
    sma_short = 95.0
    sma_long = 100.0
    prev_sma_short = 102.0
    prev_sma_long = 101.0
    signal = agent._generate_signal(
        sma_short,
        sma_long,
        prev_sma_short,
        prev_sma_long,
        "test_invocation",
    )
    assert signal == "SELL"


@pytest.mark.asyncio
async def test_generate_signal_no_signal_sma_none(agent: AlphaBotAgent):
    """Tests that _generate_signal returns None when any SMA value is None."""
    assert agent._generate_signal(None, 100.0, 102.0, 101.0, "test_invocation") is None
    assert agent._generate_signal(95.0, None, 102.0, 101.0, "test_invocation") is None
    assert agent._generate_signal(95.0, 100.0, None, 101.0, "test_invocation") is None
    assert agent._generate_signal(95.0, 100.0, 102.0, None, "test_invocation") is None


def test_determine_trade_proposal_no_buy_when_long(agent: AlphaBotAgent):
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
async def test_alphabot_run_async_impl_sell_approved_e2e(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_sell_signal,
):
    """Tests the full end-to-end flow for a successful 'SELL' trade."""
    adk_ctx.session.state = {"should_be_long": True}

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

        assert len(events) == 2
        final_event = events[1]
        assert final_event.author == agent.name
        assert "Trade Approved (A2A)" in final_event.content.parts[0].text
        assert final_event.actions.state_delta["should_be_long"] is False
        assert "approved_trade" in final_event.actions.state_delta


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_invalid_input(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
):
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
    assert (
        "Error: Invalid input data structure or values."
        in final_event.content.parts[0].text
    )


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_buy_rejected(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
):
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
        assert "Trade Rejected (A2A)" in final_event.content.parts[0].text
        assert "should_be_long" not in final_event.actions.state_delta
        assert "rejected_trade_proposal" in final_event.actions.state_delta


def test_determine_trade_proposal_no_sell_when_not_long(agent: AlphaBotAgent):
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


def test_determine_trade_proposal_no_sell_when_long_no_shares(agent: AlphaBotAgent):
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
):
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
        assert "State correction" in final_event.content.parts[0].text
        assert final_event.actions.state_delta["should_be_long"] is False


@pytest.mark.asyncio
async def test_alphabot_concurrency(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
):
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
        adk_ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )

        async def run_agent_and_collect_events():
            return [event async for event in agent._run_async_impl(adk_ctx)]

        # Run the agent multiple times concurrently
        tasks = [run_agent_and_collect_events() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # Each invocation should produce 2 events (proposal and final)
        for events in results:
            # The agent is not designed to be stateful across concurrent requests in this manner
            # so we just check that each request was processed independently and correctly.
            assert len(events) == 2
            final_event = events[1]
            assert final_event.author == agent.name
            assert final_event.content is not None
            assert final_event.content.parts is not None
            assert final_event.content.parts[0].text is not None
            assert "Trade Approved (A2A)" in final_event.content.parts[0].text
            assert final_event.actions.state_delta["should_be_long"] is True


@pytest.mark.asyncio
async def test_alphabot_does_not_repropose_rejected_trade(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
):
    """Tests that AlphaBot does not propose the same trade again immediately after
    it has been rejected by RiskGuard. This test simulates the scenario where
    a BUY signal is generated, rejected, and then the agent is run again
    with the same market conditions.
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
        assert final_event_run1.content is not None
        assert final_event_run1.content.parts is not None
        assert final_event_run1.content.parts[0].text is not None
        assert "Trade Rejected (A2A)" in final_event_run1.content.parts[0].text
        assert "should_be_long" not in final_event_run1.actions.state_delta

        # --- Second Invocation: Should NOT propose again ---
        # Update the session state with the delta from the first run, which includes the rejected trade
        adk_ctx.session.state.update(final_event_run1.actions.state_delta)

        # Rerun with the exact same input
        events_run2 = [event async for event in agent._run_async_impl(adk_ctx)]

        # Assert that NO new trade was proposed
        assert len(events_run2) == 1
        final_event_run2 = events_run2[0]
        assert final_event_run2.content is not None
        assert final_event_run2.content.parts is not None
        assert final_event_run2.content.parts[0].text is not None
        assert (
            "Signal generated, but no trade action needed based on current state or recent rejections."
            in final_event_run2.content.parts[0].text
        )


def test_determine_trade_proposal_rejects_sell_if_quantity_exceeds_shares(
    agent: AlphaBotAgent,
):
    """Tests that _determine_trade_proposal for a SELL signal returns None if the
    configured trade_quantity exceeds the number of shares held.
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
):
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
    assert "No signal yet (calculating SMAs)." in final_event.content.parts[0].text
    assert not final_event.actions.state_delta


@pytest.mark.asyncio
async def test_alphabot_run_async_impl_buy_signal_corrects_state(
    agent: AlphaBotAgent,
    adk_ctx: InvocationContext,
    alphabot_input_data_factory,
    historical_prices_buy_signal,
):
    """Tests that if a BUY signal is generated but the agent is already long,
    it produces a NO_ACTION outcome.
    """
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
        in final_event.content.parts[0].text
    )
    assert not final_event.actions.state_delta
