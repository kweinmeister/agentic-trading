import json
import logging
import uuid
from typing import Optional, Type, TypeVar

from a2a.types import (
    DataPart,
    Message,
    MessageSendParams,
    Part,
    Role,
    SendMessageRequest,
)
from google.adk.agents.invocation_context import InvocationContext
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
logger = logging.getLogger(__name__)


def parse_and_validate_input(
    ctx: InvocationContext,
    model: Type[T],
    agent_name: str,
) -> T | None:
    """Parses and validates input JSON into a Pydantic model."""
    invocation_id_short = ctx.invocation_id[:8]
    logger.debug(
        f"[{agent_name} ({invocation_id_short})] Attempting to parse input with Pydantic...",
    )
    if (
        not ctx.user_content
        or not ctx.user_content.parts
        or not hasattr(ctx.user_content.parts[0], "text")
    ):
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Input event text not found.",
        )
        return None

    input_text = ctx.user_content.parts[0].text
    if input_text is None:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Input event text is None.",
        )
        return None

    try:
        input_payload = json.loads(input_text)
        validated_input = model(**input_payload)
        logger.info(
            f"[{agent_name} ({invocation_id_short})] Successfully parsed and validated input.",
        )
        return validated_input
    except json.JSONDecodeError as e:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Failed to decode input JSON: '{input_text[:200]}...'. Error: {e}",
        )
        return None
    except ValidationError as e:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] ERROR - Input validation failed: {e}. Input was: '{input_text[:200]}...'",
        )
        return None


def create_a2a_message_from_payload(
    payload: BaseModel,
    role: Role,
    metadata: Optional[dict] = None,
    context_id: Optional[str] = None,
) -> Message:
    """Creates a standardized a2a.types.Message from a Pydantic model payload.
    The payload is serialized and placed into a single DataPart.
    """
    # model_dump() converts the Pydantic model to a dictionary
    data_part = DataPart(data=payload.model_dump())

    return Message(
        message_id=str(uuid.uuid4()),
        role=role,
        parts=[Part(root=data_part)],
        metadata=metadata or {},
        context_id=context_id,
    )


def create_a2a_request_from_payload(
    payload: BaseModel,
    role: Role = Role.user,
) -> SendMessageRequest:
    """Creates a complete SendMessageRequest from a Pydantic model payload."""
    message = create_a2a_message_from_payload(payload, role=role)
    send_params = MessageSendParams(message=message)
    return SendMessageRequest(id=str(uuid.uuid4()), params=send_params)
