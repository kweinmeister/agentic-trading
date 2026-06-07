"""Agent Executor for the RiskGuard agent."""

import json
import logging
from typing import Any

from a2a.helpers import get_data_parts, new_data_part
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Part, Task, TaskState, TaskStatus
from google.adk import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types as genai_types

from .agent import root_agent as riskguard_adk_agent

logger = logging.getLogger(__name__)


class RiskGuardAgentExecutor(AgentExecutor):
    """Executes the RiskGuard ADK agent logic in response to A2A requests."""

    def __init__(self) -> None:
        """Initialize the RiskGuardAgentExecutor."""
        self._adk_agent = riskguard_adk_agent
        self._adk_runner = Runner(
            app_name="riskguard_adk_runner",
            agent=self._adk_agent,
            session_service=InMemorySessionService(),
            # Other services like memory and artifact can be added if needed by the ADK agent
        )
        logger.info("RiskGuardAgentExecutor initialized with ADK Runner.")

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Receive a trade proposal and run it through the ADK agent.

        The result is immediately returned in a single Message event.
        """
        if not context.context_id:
            msg = "Context ID is missing, cannot execute."
            raise ValueError(msg)

        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id,
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
                    parts=[Part(text="Checking trade risk...")],
                ),
            )

            agent_input_data = None
            if context.message and context.message.parts:
                data_parts = get_data_parts(context.message.parts)
                if data_parts:
                    agent_input_data = data_parts[0]

            if (
                not agent_input_data
                or "trade_proposal" not in agent_input_data
                or "portfolio_state" not in agent_input_data
            ):
                msg = "Missing 'trade_proposal' or 'portfolio_state' in data payload"
                raise ValueError(
                    msg,
                )

            agent_input_json = json.dumps(agent_input_data)
            adk_content = genai_types.Content(
                parts=[genai_types.Part(text=agent_input_json)],
            )

            # Ensure ADK Session Exists
            session_id_for_adk = context.context_id
            logger.info(
                f"Task {context.task_id}: Attempting to get/create ADK session for session_id: '{session_id_for_adk}'",
            )

            session: Session | None = None
            if session_id_for_adk:
                try:
                    session = await self._adk_runner.session_service.get_session(
                        app_name=self._adk_runner.app_name,
                        user_id="a2a_user",
                        session_id=session_id_for_adk,
                    )
                except Exception as e_get:
                    logger.exception(
                        f"Task {context.task_id}: Exception during ADK session get_session for session_id '{session_id_for_adk}': {e_get}",
                    )
                    session = None

                if not session:
                    logger.info(
                        f"Task {context.task_id}: ADK Session not found or failed to get for '{session_id_for_adk}'. Creating new session.",
                    )
                    try:
                        session = await self._adk_runner.session_service.create_session(
                            app_name=self._adk_runner.app_name,
                            user_id="a2a_user",
                            session_id=session_id_for_adk,
                            state={},
                        )
                        if session:
                            logger.info(
                                f"Task {context.task_id}: Successfully created ADK session '{session.id if hasattr(session, 'id') else 'ID_NOT_FOUND'}'.",
                            )
                        else:
                            logger.error(
                                f"Task {context.task_id}: ADK InMemorySessionService.create_session returned None for session_id '{session_id_for_adk}'.",
                            )
                    except Exception as e_create:
                        logger.exception(
                            f"Task {context.task_id}: Exception during ADK session create_session for session_id '{session_id_for_adk}': {e_create}",
                        )
                        session = None
                else:
                    logger.info(
                        f"Task {context.task_id}: Found existing ADK session '{session.id if hasattr(session, 'id') else 'ID_NOT_FOUND'}'.",
                    )
            else:
                logger.error(
                    f"Task {context.task_id}: ADK session_id (context.context_id) is None or empty. Cannot initialize ADK session.",
                )

            if not session:
                error_message_text = f"Failed to establish ADK session. session_id was '{session_id_for_adk}'."
                logger.error(
                    f"Task {context.task_id}: {error_message_text} Cannot proceed with ADK run.",
                )
                raise ConnectionError(error_message_text)

            # Core ADK logic execution
            risk_result_dict: dict[str, Any] = {
                "approved": False,
                "reason": "Agent did not produce a result.",
            }
            async for event in self._adk_runner.run_async(
                user_id="a2a_user",
                session_id=context.context_id,
                new_message=adk_content,
            ):
                if event.content and event.content.parts:
                    first_part = event.content.parts[0]
                    if (
                        hasattr(first_part, "function_response")
                        and first_part.function_response
                        and first_part.function_response.name == "risk_check_result"
                    ):
                        response_data = first_part.function_response.response
                        if isinstance(response_data, dict):
                            risk_result_dict = response_data
                            break

            # Save the result as artifact
            await updater.add_artifact(
                parts=[new_data_part(risk_result_dict)],
                name="response",
                last_chunk=True,
            )

            # Mark task as COMPLETED
            await updater.complete()

        except Exception as e:
            logger.exception("Error during RiskGuard execution")
            # Create an error message to send back
            try:
                error_msg = updater.new_agent_message(parts=[Part(text=str(e))])
                await updater.failed(message=error_msg)
            except Exception as e_inner:
                logger.exception(f"Failed to publish failure update: {e_inner}")

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the agent execution."""
        # This synchronous agent has nothing to cancel.
        logger.warning("Cancel called on synchronous RiskGuard agent; nothing to do.")
        updater = TaskUpdater(
            event_queue=event_queue,
            task_id=context.task_id or "",
            context_id=context.context_id or "",
        )
        await updater.cancel()
