"""RiskGuard ADK Agent."""

import logging
from collections.abc import AsyncGenerator

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai import types as genai_types

from common.models import RiskCheckPayload
from common.utils.agent_utils import parse_and_validate_input

from .rules import (
    RiskCheckResult,
    check_trade_risk_logic,
)

logger = logging.getLogger(__name__)


class RiskGuardAgent(BaseAgent):
    """ADK Agent implementing the RiskGuard logic.

    Designed to be hosted via an A2A server.
    """

    name: str = "RiskGuard"
    description: str = "Evaluates proposed trades against predefined risk rules."

    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """Process a risk check request received via A2A (simulated via message content)."""
        invocation_id_short = ctx.invocation_id[:8]
        logger.info(
            f"[{self.name} ({invocation_id_short})] Received risk check invocation",
        )

        validated_input = parse_and_validate_input(ctx, RiskCheckPayload, self.name)

        if not validated_input:
            result_dict = {
                "approved": False,
                "reason": "Internal Error: Invalid input data.",
            }
        else:
            try:
                logger.info(
                    f"[{self.name} ({invocation_id_short})] Successfully parsed and validated input: {validated_input.model_dump_json(indent=2)}",
                )

                result: RiskCheckResult = check_trade_risk_logic(
                    trade_proposal=validated_input.trade_proposal,
                    portfolio_state=validated_input.portfolio_state,
                    max_pos_size=validated_input.max_pos_size,
                    max_concentration=validated_input.max_concentration,
                )
                result_dict = {"approved": result.approved, "reason": result.reason}

            except Exception as e:
                logger.error(
                    f"[{self.name} ({invocation_id_short})] Unexpected error processing input: {e}",
                    exc_info=True,
                )
                result_dict = {
                    "approved": False,
                    "reason": f"Internal Error: {e}",
                }

        logger.info(
            f"[{self.name} ({invocation_id_short})] Yielding result: {result_dict}",
        )
        yield Event(
            author=self.name,
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name="risk_check_result",  # This name is used by AgentExecutor to extract the result
                            response=result_dict,
                        ),
                    ),
                ],
            ),
            turn_complete=True,  # RiskGuard is a single-turn agent for each request
        )
        logger.info(
            f"[{self.name} ({invocation_id_short})] Risk check invocation processed and result yielded.",
        )


root_agent = RiskGuardAgent()
logger.info("RiskGuardAgent root_agent instance created.")
