"""Tests for the RiskGuard agent."""

import asyncio

import pytest
from google.adk.agents.invocation_context import InvocationContext
from google.genai import types as genai_types

from common.models import TradeProposal
from riskguard.agent import RiskGuardAgent


@pytest.fixture
def agent() -> RiskGuardAgent:
    """Provide a RiskGuardAgent instance."""
    return RiskGuardAgent()


def test_riskguard_agent_instantiation(agent: RiskGuardAgent):
    """Tests basic instantiation of the RiskGuardAgent."""
    assert agent is not None
    assert agent.name == "RiskGuard"
    assert (
        agent.description == "Evaluates proposed trades against predefined risk rules."
    )


@pytest.mark.asyncio
async def test_riskguard_run_async_impl_concurrent_requests(
    agent,
    adk_ctx: InvocationContext,
    riskguard_input_data_factory,
):
    """Tests _run_async_impl handles multiple concurrent requests."""

    async def run_agent_invocation(check_id):
        ctx = adk_ctx
        input_data = riskguard_input_data_factory(
            trade_proposal=TradeProposal(
                action="BUY",
                quantity=100,
                price=50.0,
                ticker=f"GENERIC_STOCK_{check_id}",
            ),
        )
        ctx.user_content = genai_types.Content(
            parts=[genai_types.Part(text=input_data.model_dump_json())],
        )
        events = []
        async for event in agent._run_async_impl(ctx):
            events.append(event)
        return (
            check_id,
            events[0].content.parts[0].function_response.response["approved"],
        )

    # Run multiple agent invocations concurrently
    tasks = [run_agent_invocation(i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    # Verify all completed successfully and were approved (assuming valid inputs)
    assert len(results) == 10
    for _, approved in results:
        assert approved is True
