"""Compliance and integration tests for Simulator, AlphaBot, and RiskGuard."""

import asyncio
from typing import Any

import httpx
import pytest
from a2a.client import A2ACardResolver, ClientConfig, ClientFactory
from a2a.helpers import get_data_parts, new_data_part
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
    Role,
    SendMessageRequest,
    TaskState,
)
from a2a.types import (
    Message as A2AMessage,
)
from fastapi import FastAPI
from google.adk.events import Event, EventActions
from google.genai import types as genai_types

from alphabot.__main__ import create_app as create_alphabot_app
from alphabot.a2a_risk_tool import A2ARiskCheckTool
from alphabot.agent_executor import AlphaBotAgentExecutor
from common.models import (
    AlphaBotTaskPayload,
    RiskCheckPayload,
    RiskCheckResult,
    TradeOutcome,
    TradeProposal,
    TradeStatus,
)
from common.models import (
    PortfolioState as CommonPortfolioState,
)
from riskguard.__main__ import create_app as create_riskguard_app
from riskguard.agent_executor import RiskGuardAgentExecutor


# Helper to parse A2A StreamResponse stream and return (statuses, response_artifact)
async def _parse_a2a_stream(stream) -> tuple[list[Any], Any]:
    statuses = []
    response_artifact = None
    async for event in stream:
        if event.HasField("status_update"):
            statuses.append(event.status_update.status.state)
        elif (
            event.HasField("artifact_update")
            and event.artifact_update.artifact.name == "response"
        ):
            response_artifact = event.artifact_update.artifact
        elif event.HasField("task"):
            statuses.append(event.task.status.state)
            for art in event.task.artifacts:
                if art.name == "response":
                    response_artifact = art
    return statuses, response_artifact


# Helper to create RiskGuard app
def make_riskguard_app(mock_run_async_generator) -> FastAPI:
    executor = RiskGuardAgentExecutor()
    executor._adk_runner.run_async = mock_run_async_generator
    card = AgentCard(
        name="RiskGuard",
        description="Evaluates proposed trades against predefined risk rules.",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.1.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[
            AgentSkill(
                id="check_trade_risk",
                name="Check Trade Risk",
                description="desc",
                examples=[],
                tags=[],
            )
        ],
        default_input_modes=["data"],
        default_output_modes=["data"],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url="http://localhost:8080/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url="http://localhost:8080/a2a/rest",
            ),
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return create_riskguard_app(card, handler)


# Helper to create AlphaBot app
def make_alphabot_app(
    mock_run_async_generator, risk_client: httpx.AsyncClient | None = None
) -> FastAPI:
    executor = AlphaBotAgentExecutor()
    executor._adk_runner.run_async = mock_run_async_generator
    if risk_client is not None:
        tools = executor._adk_agent.tools or []
        for tool in tools:
            if isinstance(tool, A2ARiskCheckTool):
                tool._httpx_client = risk_client
                tool.risk_guard_url = (
                    "http://localhost:8080"  # Base url to route through ASGITransport
                )

    card = AgentCard(
        name="AlphaBot Agent",
        description="Trading agent that analyzes market data and portfolio state to propose trades.",
        provider=AgentProvider(organization="A2A Samples", url="https://example.com"),
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[
            AgentSkill(
                id="provide_trade_signal",
                name="Provide Trade Signal",
                description="desc",
                examples=[],
                tags=[],
            )
        ],
        default_input_modes=["data"],
        default_output_modes=["data"],
        supported_interfaces=[
            AgentInterface(
                protocol_binding="JSONRPC",
                protocol_version="1.0",
                url="http://localhost:8081/a2a/jsonrpc",
            ),
            AgentInterface(
                protocol_binding="HTTP+JSON",
                protocol_version="1.0",
                url="http://localhost:8081/a2a/rest",
            ),
        ],
    )
    handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return create_alphabot_app(card, handler)


@pytest.mark.asyncio
async def test_riskguard_contract_and_lifecycle() -> None:
    """Test calling RiskGuard app and verify output contract, lifecycle events, and card endpoint."""

    async def mock_run_async(user_id, session_id, new_message):
        yield Event(
            author="RiskGuard",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": True,
                                "reason": "Passed integration check",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

    app = make_riskguard_app(mock_run_async)

    # Verify /.well-known/agent-card.json compliance
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8080"
    ) as client:
        card_response = await client.get("/.well-known/agent-card.json")
        assert card_response.status_code == 200
        card_data = card_response.json()
        assert card_data["name"] == "RiskGuard"
        assert card_data["version"] == "1.1.0"

        # Test Card Resolution and Client Construction
        resolver = A2ACardResolver(
            httpx_client=client, base_url="http://localhost:8080"
        )
        card = await resolver.get_agent_card()
        assert card.name == "RiskGuard"

        factory = ClientFactory(config=ClientConfig(httpx_client=client))
        sdk_client = factory.create(card)

        # Call RiskGuard
        payload = RiskCheckPayload(
            trade_proposal=TradeProposal(
                action="BUY", ticker="TEST", quantity=10, price=100.0
            ),
            portfolio_state=CommonPortfolioState(
                cash=10000.0, shares=0, total_value=10000.0
            ),
            max_pos_size=5000.0,
            max_concentration=0.5,
        )
        msg = A2AMessage(
            message_id="msg-1",
            role=Role.ROLE_USER,
            parts=[new_data_part(payload.model_dump(mode="json"))],
            context_id="ctx-1",
        )

        stream = sdk_client.send_message(SendMessageRequest(message=msg))
        statuses, response_artifact = await _parse_a2a_stream(stream)

        # Since the client runs in non-streaming mode due to the agent's card capability,
        # it will only receive the final completed task event.
        assert TaskState.TASK_STATE_COMPLETED in statuses
        assert response_artifact is not None

        # Verify Output Contract
        data = get_data_parts(response_artifact.parts)[0]
        result = RiskCheckResult.model_validate(data)
        assert result.approved is True
        assert result.reason == "Passed integration check"


@pytest.mark.asyncio
async def test_alphabot_contract_and_lifecycle() -> None:
    """Test calling AlphaBot app and verify output contract, lifecycle events, and card endpoint."""

    async def mock_run_async(user_id, session_id, new_message):
        yield Event(
            author="AlphaBot",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(text="Proposing BUY 10 TEST @ $100"),
                ],
            ),
        )
        yield Event(
            author="AlphaBot",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text="Trade Approved (A2A): Within risk parameters"
                    ),
                ],
            ),
            actions=EventActions(
                state_delta={
                    "should_be_long": True,
                    "approved_trade": {
                        "action": "BUY",
                        "ticker": "TEST",
                        "quantity": 10,
                        "price": 100.0,
                    },
                },
            ),
            turn_complete=True,
        )

    app = make_alphabot_app(mock_run_async)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8081"
    ) as client:
        card_response = await client.get("/.well-known/agent-card.json")
        assert card_response.status_code == 200
        card_data = card_response.json()
        assert card_data["name"] == "AlphaBot Agent"
        assert card_data["version"] == "1.0.0"

        resolver = A2ACardResolver(
            httpx_client=client, base_url="http://localhost:8081"
        )
        card = await resolver.get_agent_card()
        assert card.name == "AlphaBot Agent"

        factory = ClientFactory(config=ClientConfig(httpx_client=client))
        sdk_client = factory.create(card)

        payload = AlphaBotTaskPayload(
            historical_prices=[90.0, 95.0, 100.0],
            current_price=100.0,
            portfolio_state=CommonPortfolioState(
                cash=10000.0, shares=0, total_value=10000.0
            ),
            day=1,
            short_sma_period=2,
            long_sma_period=3,
            trade_quantity=10,
            riskguard_url="http://localhost:8080",
            max_pos_size=5000.0,
            max_concentration=0.5,
        )
        msg = A2AMessage(
            message_id="msg-1",
            role=Role.ROLE_USER,
            parts=[new_data_part(payload.model_dump(mode="json"))],
            context_id="ctx-1",
        )

        stream = sdk_client.send_message(SendMessageRequest(message=msg))
        statuses, response_artifact = await _parse_a2a_stream(stream)

        # Since the client runs in non-streaming mode due to the agent's card capability,
        # it will only receive the final completed task event.
        assert TaskState.TASK_STATE_COMPLETED in statuses
        assert response_artifact is not None

        # Verify Output Contract
        data = get_data_parts(response_artifact.parts)[0]
        outcome = TradeOutcome.model_validate(data)
        assert outcome.status == TradeStatus.APPROVED
        assert outcome.reason == "Trade Approved (A2A): Within risk parameters"
        assert outcome.trade_proposal is not None
        assert outcome.trade_proposal.quantity == 10


@pytest.mark.asyncio
async def test_simulator_alphabot_riskguard_e2e() -> None:
    """Full Simulator -> AlphaBot -> RiskGuard E2E test resolving in-memory A2A calls."""

    # 1. Setup RiskGuard app
    async def mock_rg_run_async(user_id, session_id, new_message):
        yield Event(
            author="RiskGuard",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={
                                "approved": True,
                                "reason": "Approved by E2E RiskGuard",
                            },
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

    riskguard_app = make_riskguard_app(mock_rg_run_async)
    risk_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=riskguard_app),
        base_url="http://localhost:8080",
    )

    # 2. Setup AlphaBot E2E proxy run_async
    async def mock_alphabot_e2e_run_async(user_id, session_id, new_message):
        text = new_message.parts[0].text
        input_payload = AlphaBotTaskPayload.model_validate_json(text)

        trade_proposal = {
            "action": "BUY",
            "ticker": "TEST",
            "quantity": 10,
            "price": 100.0,
        }

        yield Event(
            author="AlphaBot",
            content=genai_types.Content(
                parts=[genai_types.Part(text="Proposing BUY 10 TEST @ $100")],
            ),
        )

        # Manually invoke RiskGuard ASGI app using the A2A resolver and client
        resolver = A2ACardResolver(
            httpx_client=risk_client, base_url="http://localhost:8080"
        )
        rg_card = await resolver.get_agent_card()
        rg_sdk_client = ClientFactory(
            config=ClientConfig(httpx_client=risk_client)
        ).create(rg_card)

        rg_payload = RiskCheckPayload(
            trade_proposal=TradeProposal(
                action="BUY",
                ticker="TEST",
                quantity=10,
                price=100.0,
            ),
            portfolio_state=input_payload.portfolio_state,
            max_pos_size=input_payload.max_pos_size,
            max_concentration=input_payload.max_concentration,
        )
        rg_message = A2AMessage(
            message_id="msg-rg-1",
            role=Role.ROLE_USER,
            parts=[new_data_part(rg_payload.model_dump(mode="json"))],
            context_id=session_id,
        )

        rg_stream = rg_sdk_client.send_message(SendMessageRequest(message=rg_message))
        _, rg_artifact = await _parse_a2a_stream(rg_stream)
        rg_result = get_data_parts(rg_artifact.parts)[0] if rg_artifact else {}

        approved = rg_result.get("approved", False)
        reason = rg_result.get("reason", "")

        yield Event(
            author="AlphaBot",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        text=f"Trade Approved (A2A): {reason}"
                        if approved
                        else f"Trade Rejected (A2A): {reason}"
                    )
                ],
            ),
            actions=EventActions(
                state_delta={
                    "should_be_long": bool(approved),
                    "approved_trade"
                    if approved
                    else "rejected_trade_proposal": trade_proposal,
                },
            ),
            turn_complete=True,
        )

    alphabot_app = make_alphabot_app(
        mock_alphabot_e2e_run_async, risk_client=risk_client
    )

    # 3. Call AlphaBot from the Simulator perspective
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=alphabot_app),
        base_url="http://localhost:8081",
    ) as client:
        resolver = A2ACardResolver(
            httpx_client=client, base_url="http://localhost:8081"
        )
        card = await resolver.get_agent_card()
        sdk_client = ClientFactory(config=ClientConfig(httpx_client=client)).create(
            card
        )

        payload = AlphaBotTaskPayload(
            historical_prices=[90.0, 95.0, 100.0],
            current_price=100.0,
            portfolio_state=CommonPortfolioState(
                cash=10000.0, shares=0, total_value=10000.0
            ),
            day=1,
            short_sma_period=2,
            long_sma_period=3,
            trade_quantity=10,
            riskguard_url="http://localhost:8080",
            max_pos_size=5000.0,
            max_concentration=0.5,
        )
        msg = A2AMessage(
            message_id="msg-sim-1",
            role=Role.ROLE_USER,
            parts=[new_data_part(payload.model_dump(mode="json"))],
            context_id="ctx-sim-1",
        )

        stream = sdk_client.send_message(SendMessageRequest(message=msg))
        _, response_artifact = await _parse_a2a_stream(stream)

        assert response_artifact is not None
        data = get_data_parts(response_artifact.parts)[0]
        outcome = TradeOutcome.model_validate(data)
        assert outcome.status == TradeStatus.APPROVED
        assert "Approved by E2E RiskGuard" in outcome.reason
        assert outcome.trade_proposal is not None
        assert outcome.trade_proposal.ticker == "TEST"
        assert outcome.trade_proposal.quantity == 10


@pytest.mark.asyncio
async def test_integration_error_responses() -> None:
    """Test that invalid method calls on JSON-RPC return compliant errors."""

    async def mock_run_async(user_id, session_id, new_message):
        yield Event(
            author="Mock",
            content=genai_types.Content(parts=[genai_types.Part(text="")]),
            turn_complete=True,
        )

    app = make_riskguard_app(mock_run_async)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8080"
    ) as client:
        # Invalid JSON-RPC request format
        response = await client.post(
            "/a2a/jsonrpc",
            json={"method": "invalid_method_name", "params": {}, "id": 1},
        )
        assert response.status_code == 200
        resp_json = response.json()
        assert "error" in resp_json
        assert resp_json["error"]["code"] in (-32601, -32600)


@pytest.mark.asyncio
async def test_integration_concurrent_requests_isolation() -> None:
    """Test that concurrent requests are handled independently with task isolation."""

    async def mock_run_async(user_id, session_id, new_message):
        text = new_message.parts[0].text
        import json

        payload = json.loads(text)
        qty = int(payload["trade_proposal"]["quantity"])
        # Sleep for a small duration to test overlapping concurrent executions
        await asyncio.sleep(0.05)
        yield Event(
            author="RiskGuard",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",
                            response={"approved": True, "reason": f"Quantity: {qty}"},
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

    app = make_riskguard_app(mock_run_async)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://localhost:8080"
    ) as client:
        resolver = A2ACardResolver(
            httpx_client=client, base_url="http://localhost:8080"
        )
        card = await resolver.get_agent_card()
        sdk_client = ClientFactory(config=ClientConfig(httpx_client=client)).create(
            card
        )

        async def call_rg(qty: int) -> str:
            payload = RiskCheckPayload(
                trade_proposal=TradeProposal(
                    action="BUY", ticker="TEST", quantity=qty, price=100.0
                ),
                portfolio_state=CommonPortfolioState(
                    cash=10000.0, shares=0, total_value=10000.0
                ),
                max_pos_size=5000.0,
                max_concentration=0.5,
            )
            msg = A2AMessage(
                message_id=f"msg-{qty}",
                role=Role.ROLE_USER,
                parts=[new_data_part(payload.model_dump(mode="json"))],
                context_id=f"ctx-{qty}",
            )
            stream = sdk_client.send_message(SendMessageRequest(message=msg))
            _, art = await _parse_a2a_stream(stream)
            if art:
                return get_data_parts(art.parts)[0]["reason"]
            return ""

        # Make concurrent calls
        results = await asyncio.gather(call_rg(10), call_rg(20), call_rg(30))
        assert "Quantity: 10" in results
        assert "Quantity: 20" in results
        assert "Quantity: 30" in results
