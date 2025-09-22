import json
import logging
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.types import DataPart, Part
from a2a.utils.message import new_agent_parts_message
from google.adk import Runner
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types as genai_types

from .agent import root_agent as riskguard_adk_agent

logger = logging.getLogger(__name__)


class RiskGuardAgentExecutor(AgentExecutor):
    """Executes the RiskGuard ADK agent logic in response to A2A requests."""

    def __init__(self):
        self._adk_agent = riskguard_adk_agent
        self._adk_runner = Runner(
            app_name="riskguard_adk_runner",
            agent=self._adk_agent,
            session_service=InMemorySessionService(),
            # Other services like memory and artifact can be added if needed by the ADK agent
        )
        logger.info("RiskGuardAgentExecutor initialized with ADK Runner.")

    async def execute(self, context: RequestContext, event_queue: EventQueue):
        """Receives a trade proposal, runs it through the ADK agent, and immediately
        returns the result in a single Message event.
        """
        try:
            if not context.context_id:
                raise ValueError("Context ID is missing, cannot execute.")

            agent_input_data = None
            if context.message and context.message.parts:
                part = context.message.parts[0].root
                if isinstance(part, DataPart):
                    agent_input_data = part.data

            if (
                not agent_input_data
                or "trade_proposal" not in agent_input_data
                or "portfolio_state" not in agent_input_data
            ):
                raise ValueError(
                    "Missing 'trade_proposal' or 'portfolio_state' in data payload",
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

            # Instead of using TaskUpdater, create and enqueue a single message
            final_message = new_agent_parts_message(
                parts=[Part(root=DataPart(data=risk_result_dict))],
                context_id=context.context_id,
                task_id=context.task_id,  # Keep task_id for correlation
            )
            await event_queue.enqueue_event(final_message)

        except Exception as e:
            logger.exception("Error during RiskGuard execution")
            # Create an error message to send back
            error_message = new_agent_parts_message(
                parts=[
                    Part(
                        root=DataPart(
                            data={
                                "approved": False,
                                "reason": f"An internal error occurred: {e}",
                            },
                        ),
                    ),
                ],
                context_id=context.context_id,
                task_id=context.task_id,
            )
            await event_queue.enqueue_event(error_message)
        finally:
            # Always close the queue after the single event is sent.
            await event_queue.close()

    async def cancel(self, context: RequestContext, event_queue: EventQueue):
        # This synchronous agent has nothing to cancel.
        logger.warning("Cancel called on synchronous RiskGuard agent; nothing to do.")
        await event_queue.close()
