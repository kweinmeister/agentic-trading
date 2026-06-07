"""A2A Risk Check Tool for AlphaBot."""

import logging
import os
import uuid
from typing import Any

import httpx
from a2a.client import (
    A2ACardResolver,
    A2AClientError,
    A2AClientTimeoutError,
    ClientConfig,
    ClientFactory,
)
from a2a.helpers import get_data_parts, new_data_part
from a2a.types import (
    Message,
    Role,
    SendMessageRequest,
    TaskState,
)
from google.adk.events import Event
from google.adk.tools import BaseTool, ToolContext
from google.genai import types as genai_types  # ADK uses google.genai.types
from pydantic import ValidationError

from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
)
from common.models import (
    PortfolioState,
    RiskCheckPayload,
    RiskCheckResult,
    TradeProposal,
)

logger = logging.getLogger(__name__)

RISKGUARD_A2A_TIMEOUT_SECONDS = 10


class A2ARiskCheckTool(BaseTool):
    """ADK Tool that makes an A2A call to the RiskGuard service."""

    name: str = "a2a_risk_check"
    description: str = "Sends a trade proposal to the RiskGuard service for validation and returns the approval status and reason."
    risk_guard_url: str
    _httpx_client: httpx.AsyncClient

    def __init__(self, **kwargs) -> None:
        """Initialize the A2ARiskCheckTool."""
        # Pop custom arguments for this class before calling super()
        # Configure httpx.AsyncClient with the timeout
        self._httpx_client = kwargs.pop(
            "httpx_client",
            httpx.AsyncClient(
                timeout=RISKGUARD_A2A_TIMEOUT_SECONDS,
            ),
        )
        risk_guard_service_url = os.environ.get(
            "RISKGUARD_SERVICE_URL",
            DEFAULT_RISKGUARD_URL,
        )
        self.risk_guard_url = kwargs.pop("risk_guard_url", risk_guard_service_url)
        self._a2a_sdk_clients: dict[str, Any] = {}

        # Now, kwargs only contains arguments meant for the parent class
        super().__init__(name=self.name, description=self.description, **kwargs)

    async def close(self) -> None:
        """Close the httpx.AsyncClient when the tool is no longer needed."""
        if self._httpx_client:
            await self._httpx_client.aclose()
            logger.info("A2ARiskCheckTool httpx.AsyncClient closed.")

    @property
    def httpx_client(self) -> httpx.AsyncClient:
        """Get the internal httpx.AsyncClient."""
        return self._httpx_client

    @httpx_client.setter
    def httpx_client(self, client: httpx.AsyncClient) -> None:
        """Set the internal httpx.AsyncClient."""
        self._httpx_client = client

    def _get_declaration(self) -> genai_types.FunctionDeclaration:
        """Return the ADK FunctionDeclaration for this tool.

        This describes the tool's interface to an LLM if it were to be called by one.
        """
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "trade_proposal": genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        description="Details of the trade being proposed, e.g., {'action': 'BUY', 'ticker': 'XYZ', 'quantity': 100, 'price': 50.0}",
                        properties={
                            "action": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="BUY or SELL",
                            ),
                            "ticker": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="Stock ticker symbol",
                            ),
                            "quantity": genai_types.Schema(
                                type=genai_types.Type.INTEGER,
                                description="Number of shares",
                            ),
                            "price": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Price per share",
                            ),
                        },
                        required=["action", "ticker", "quantity", "price"],
                    ),
                    "portfolio_state": genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        description="Current state of the portfolio, e.g., {'cash': 10000, 'shares': 50, 'total_value': 15000}",
                        properties={
                            "cash": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Current cash in portfolio",
                            ),
                            "shares": genai_types.Schema(
                                type=genai_types.Type.INTEGER,
                                description="Current shares held of the ticker",
                            ),
                            "total_value": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Total current value of the portfolio",
                            ),
                        },
                        required=["cash", "shares", "total_value"],
                    ),
                    "risk_params": genai_types.Schema(
                        type=genai_types.Type.OBJECT,
                        description="Risk parameters for the check, e.g., {'riskguard_url': 'http://...', 'max_pos_size': 10000, 'max_concentration': 0.5}",
                        properties={
                            "riskguard_url": genai_types.Schema(
                                type=genai_types.Type.STRING,
                                description="URL of the RiskGuard service (optional, uses default if not provided).",
                            ),
                            "max_pos_size": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Maximum position size for a single trade.",
                            ),
                            "max_concentration": genai_types.Schema(
                                type=genai_types.Type.NUMBER,
                                description="Maximum portfolio concentration (0.0 to 1.0).",
                            ),
                        },
                        required=[],
                    ),
                },
                required=[
                    "trade_proposal",
                    "portfolio_state",
                ],
            ),
        )

    async def run_async(self, **kwargs: Any) -> Event:
        """Make the actual A2A HTTP call."""
        tool_context: ToolContext = kwargs["tool_context"]
        args: dict[str, Any] = kwargs["args"]
        invocation_id_short = tool_context.invocation_id[:8]
        logger.debug(
            f"[{self.name} Tool ({invocation_id_short})] Received args: {args}",
        )

        trade_proposal = args.get("trade_proposal")
        portfolio_state = args.get("portfolio_state")
        risk_params_from_args = args.get("risk_params", {})

        # Use risk_params from args if provided, otherwise fallback to AlphaBot's defaults
        risk_guard_target_url = risk_params_from_args.get(
            "riskguard_url",
            self.risk_guard_url,
        )
        max_pos_size = risk_params_from_args.get(
            "max_pos_size",
            DEFAULT_RISKGUARD_MAX_POS_SIZE,
        )
        max_concentration = risk_params_from_args.get(
            "max_concentration",
            DEFAULT_RISKGUARD_MAX_CONCENTRATION,
        )

        logger.debug(
            f"[{self.name} Tool ({invocation_id_short})] Using Risk Params: url={risk_guard_target_url}, "
            f"max_pos_size={max_pos_size}, max_concentration={max_concentration}",
        )

        final_result_dict = {
            "approved": False,
            "reason": "A2A call failed or result not found.",
        }

        if not trade_proposal or not portfolio_state:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] Error - Missing trade_proposal or portfolio_state in args.",
            )
            final_result_dict["reason"] = "Tool Error: Missing input arguments."
            return Event(
                author=self.name,
                content=genai_types.Content(
                    parts=[
                        genai_types.Part(
                            function_response=genai_types.FunctionResponse(
                                name=self.name,
                                response=final_result_dict,
                            ),
                        ),
                    ],
                ),
                turn_complete=True,
            )

        logger.info(
            f"[{self.name} Tool ({invocation_id_short})] Preparing A2A call to {risk_guard_target_url}",
        )

        # --- REFACTORED SECTION ---
        # 1. Create Pydantic models from your data
        trade_proposal_model = TradeProposal(**trade_proposal)
        portfolio_state_model = PortfolioState(**portfolio_state)

        risk_payload = RiskCheckPayload(
            trade_proposal=trade_proposal_model,
            portfolio_state=portfolio_state_model,
            max_pos_size=max_pos_size,
            max_concentration=max_concentration,
        )

        try:
            a2a_sdk_client = self._a2a_sdk_clients.get(risk_guard_target_url)
            if not a2a_sdk_client:
                client_config = ClientConfig(httpx_client=self._httpx_client)
                client_factory = ClientFactory(config=client_config)
                card_resolver = A2ACardResolver(
                    httpx_client=self._httpx_client,
                    base_url=risk_guard_target_url,
                )
                agent_card = await card_resolver.get_agent_card()
                a2a_sdk_client = client_factory.create(agent_card)
                self._a2a_sdk_clients[risk_guard_target_url] = a2a_sdk_client

            # The new client uses a streaming response. We need to iterate.
            message_to_send = Message(
                message_id=f"msg-{uuid.uuid4().hex[:8]}",
                role=Role.ROLE_USER,
                parts=[
                    new_data_part(risk_payload.model_dump(mode="json")),
                ],
                context_id=tool_context.session.id,
            )
            request = SendMessageRequest(message=message_to_send)
            stream = a2a_sdk_client.send_message(request)

            def process_part_data(part_data: Any) -> None:
                nonlocal final_result_dict
                try:
                    risk_result_model = RiskCheckResult.model_validate(part_data)
                    final_result_dict = risk_result_model.model_dump()
                except ValidationError:
                    final_result_dict["reason"] = "Malformed response from RiskGuard"

            async for event in stream:
                if event.HasField("message"):
                    data_parts = get_data_parts(event.message.parts)
                    if data_parts:
                        process_part_data(data_parts[0])
                    else:
                        final_result_dict["reason"] = (
                            "Malformed response from RiskGuard"
                        )
                    break
                if event.HasField("artifact_update"):
                    if event.artifact_update.artifact.name == "response":
                        data_parts = get_data_parts(
                            event.artifact_update.artifact.parts,
                        )
                        if data_parts:
                            process_part_data(data_parts[0])
                        else:
                            final_result_dict["reason"] = (
                                "Malformed response from RiskGuard"
                            )
                        break
                    logger.debug(
                        f"Received unexpected artifact update '{event.artifact_update.artifact.name}' from RiskGuard, ignoring.",
                    )
                elif event.HasField("status_update"):
                    state = event.status_update.status.state
                    if state == TaskState.TASK_STATE_FAILED:
                        final_result_dict["reason"] = "RiskGuard execution failed."
                        break
                elif event.HasField("task"):
                    logger.debug(
                        f"Received task update for {event.task.id}: {event.task.status.state}",
                    )
                    # Extract response artifact if present in the final Task payload
                    response_artifact = next(
                        (art for art in event.task.artifacts if art.name == "response"),
                        None,
                    )
                    if response_artifact:
                        data_parts = get_data_parts(response_artifact.parts)
                        if data_parts:
                            process_part_data(data_parts[0])
                        else:
                            final_result_dict["reason"] = (
                                "Malformed response from RiskGuard"
                            )
                        break
                    if event.task.status.state == TaskState.TASK_STATE_FAILED:
                        error_msg = (
                            event.task.status.message.parts[0].text
                            if event.task.status.message
                            and len(event.task.status.message.parts) > 0
                            else "RiskGuard execution failed."
                        )
                        final_result_dict["reason"] = (
                            f"RiskGuard execution failed: {error_msg}"
                        )
                        break

        except A2AClientTimeoutError as e:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] A2A SDK Timeout error connecting to/from RiskGuard ({risk_guard_target_url}): {e}",
                exc_info=True,
            )
            final_result_dict["reason"] = f"Timeout Error: {e}"
        except A2AClientError as e:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] A2A SDK error connecting to/from RiskGuard ({risk_guard_target_url}): {e}",
                exc_info=True,
            )
            final_result_dict["reason"] = f"A2A SDK Error: {e}. Is RiskGuard running?"
        except ValidationError as e:
            logger.error(
                f"[{self.name} Tool ({invocation_id_short})] A2A Response Validation Error: {e}",
                exc_info=True,
            )
            final_result_dict["reason"] = (
                "A2A protocol error: Invalid response structure."
            )
        except Exception as e:
            logger.exception(
                f"[{self.name} Tool ({invocation_id_short})] Unexpected Error during A2A call",
            )
            final_result_dict["reason"] = f"A2A Client Error: {e!s}"

        logger.info(
            f"[{self.name} Tool ({invocation_id_short})] Yielding final result: {final_result_dict}",
        )
        return Event(
            author=self.name,
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=self.name,
                            response=final_result_dict,
                        ),
                    ),
                ],
            ),
            turn_complete=True,  # This tool completes its action in one go
        )
