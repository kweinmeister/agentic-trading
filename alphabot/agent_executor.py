import logging

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import DataPart, Part
from a2a.utils.message import new_agent_parts_message
from google.adk import Runner
from google.adk.memory import InMemoryMemoryService
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types as genai_types
from pydantic import ValidationError

from alphabot.agent import root_agent as alphabot_adk_agent
from common.models import AlphaBotTaskPayload, TradeOutcome, TradeStatus

logger = logging.getLogger(__name__)


class AlphaBotAgentExecutor(AgentExecutor):
    """Executes the AlphaBot ADK agent logic in response to A2A requests."""

    def __init__(self):
        self._adk_agent = alphabot_adk_agent
        self._adk_runner = Runner(
            app_name="alphabot_adk_runner",
            agent=self._adk_agent,
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        logger.info("AlphaBotAgentExecutor initialized with ADK Runner.")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Receives a unified task payload, runs it through the ADK agent,
        and returns the structured result in a standard Artifact.
        """
        outcome = TradeOutcome(
            status=TradeStatus.ERROR,
            reason="Initialization failed.",
        )
        try:
            # 1. Simplified Payload Parsing
            if not context.context_id:
                raise ValueError("Context ID is missing, cannot execute.")

            if not context.message or not context.message.parts:
                raise ValueError("Received an empty or invalid message.")

            part = context.message.parts[0].root
            if not isinstance(part, DataPart):
                raise ValueError("Expected a DataPart with AlphaBotTaskPayload")

            validated_payload = AlphaBotTaskPayload.model_validate(part.data)
            agent_input_json = validated_payload.model_dump_json()
            adk_content = genai_types.Content(
                parts=[genai_types.Part(text=agent_input_json)],
            )

            # Ensure ADK Session Exists
            session_id_for_adk = context.context_id
            session: (
                Session | None
            ) = await self._adk_runner.session_service.get_session(
                app_name=self._adk_runner.app_name,
                user_id="a2a_user",
                session_id=session_id_for_adk,
            )
            if not session:
                session = await self._adk_runner.session_service.create_session(
                    app_name=self._adk_runner.app_name,
                    user_id="a2a_user",
                    session_id=session_id_for_adk,
                    state={},
                )
            if not session:
                raise RuntimeError("Failed to create or retrieve ADK session.")

            # 2. Process ADK Output and Wrap in a `TradeOutcome` and `Artifact`
            final_reason_text = "Reason not provided."
            captured_state_delta = {}
            async for event in self._adk_runner.run_async(
                user_id="a2a_user",
                session_id=session_id_for_adk,
                new_message=adk_content,
            ):
                if event.actions and event.actions.state_delta:
                    captured_state_delta.update(event.actions.state_delta)
                if event.is_final_response() and event.content and event.content.parts:
                    text_part = next(
                        (p for p in event.content.parts if hasattr(p, "text")),
                        None,
                    )
                    if text_part:
                        final_reason_text = text_part.text

            if "approved_trade" in captured_state_delta:
                trade_decision = {
                    "status": TradeStatus.APPROVED,
                    "reason": final_reason_text,
                    "trade_proposal": captured_state_delta.get("approved_trade"),
                }
            elif "rejected_trade_proposal" in captured_state_delta:
                trade_decision = {
                    "status": TradeStatus.REJECTED,
                    "reason": final_reason_text,
                    "trade_proposal": captured_state_delta.get(
                        "rejected_trade_proposal",
                    ),
                }
            else:
                trade_decision = {
                    "status": TradeStatus.NO_ACTION,
                    "reason": final_reason_text,
                }
            outcome = TradeOutcome.model_validate(trade_decision)
            final_message = new_agent_parts_message(
                parts=[Part(root=DataPart(data=outcome.model_dump()))],
                context_id=context.context_id,
                task_id=context.task_id,
            )
            await event_queue.enqueue_event(final_message)

        except (ValidationError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            outcome = TradeOutcome(status=TradeStatus.ERROR, reason=str(e))
            final_message = new_agent_parts_message(
                parts=[Part(root=DataPart(data=outcome.model_dump()))],
                context_id=context.context_id,
                task_id=context.task_id,
            )
            await event_queue.enqueue_event(final_message)
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            outcome = TradeOutcome(
                status=TradeStatus.ERROR,
                reason="An unexpected server error occurred.",
            )
            final_message = new_agent_parts_message(
                parts=[Part(root=DataPart(data=outcome.model_dump()))],
                context_id=context.context_id,
                task_id=context.task_id,
            )
            await event_queue.enqueue_event(final_message)
        finally:
            await event_queue.close()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        logger.warning(
            f"Cancellation not implemented for synchronous AlphaBot ADK agent task: {context.task_id}",
        )
        await event_queue.close()
