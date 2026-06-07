"""Agent Executor for the AlphaBot agent."""

import logging

from a2a.helpers import get_data_parts, new_data_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus
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

    def __init__(self) -> None:
        """Initialize the AlphaBotAgentExecutor."""
        self._adk_agent = alphabot_adk_agent
        self._adk_runner = Runner(
            app_name="alphabot_adk_runner",
            agent=self._adk_agent,
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )
        logger.info("AlphaBotAgentExecutor initialized with ADK Runner.")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Receive a unified task payload and run it through the ADK agent.

        The structured result is returned in a standard Artifact.
        """
        if not context.context_id:
            msg = "Context ID is missing, cannot execute."
            raise ValueError(msg)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id,
        )
        outcome = TradeOutcome(
            status=TradeStatus.ERROR,
            reason="Initialization failed.",
        )
        try:
            # Enqueue the initial Task object to start task mode
            await event_queue.enqueue_event(
                Task(
                    id=context.task_id or "",
                    context_id=context.context_id or "",
                    status=TaskStatus(state=TaskState.TASK_STATE_SUBMITTED),
                    history=[context.message] if context.message else [],
                ),
            )

            # Set task state to WORKING
            await updater.start_work(
                message=updater.new_agent_message(
                    parts=[Part(text="Analyzing market and portfolio state...")],
                ),
            )

            if not context.message or not context.message.parts:
                msg = "Received an empty or invalid message."
                raise ValueError(msg)

            data_parts = get_data_parts(context.message.parts)
            if not data_parts:
                msg = "Expected a DataPart with AlphaBotTaskPayload"
                raise ValueError(msg)

            validated_payload = AlphaBotTaskPayload.model_validate(data_parts[0])
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
                msg = "Failed to create or retrieve ADK session."
                raise RuntimeError(msg)

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

            # Save the result as artifact
            await updater.add_artifact(
                parts=[new_data_part(outcome.model_dump())],
                name="response",
                last_chunk=True,
            )

            # Mark task as COMPLETED
            await updater.complete()

        except (ValidationError, ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Error during agent execution: {e}", exc_info=True)
            try:
                error_msg = updater.new_agent_message(parts=[Part(text=str(e))])
                await updater.failed(message=error_msg)
            except Exception as e_inner:
                logger.exception(f"Failed to publish failure update: {e_inner}")
        except Exception as e:
            logger.exception(f"An unexpected error occurred: {e}")
            try:
                error_msg = updater.new_agent_message(
                    parts=[Part(text="An unexpected server error occurred.")],
                )
                await updater.failed(message=error_msg)
            except Exception as e_inner:
                logger.exception(f"Failed to publish failure update: {e_inner}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the agent execution."""
        logger.warning(
            f"Cancellation not implemented for synchronous AlphaBot ADK agent task: {context.task_id}",
        )
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()
