import logging
import os
from typing import Any

from google.adk.agents import InvocationContext
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)


def get_service_url(env_var_name: str, host: str, port: int) -> str:
    """Determines the service URL by checking an environment variable first,
    then falling back to a host/port combination.

    Args:
        env_var_name: The name of the environment variable to check.
        host: The hostname to use as a fallback.
        port: The port to use as a fallback.

    Returns:
        The determined service URL.

    """
    if public_url := os.environ.get(env_var_name):
        logger.info(
            f"Using public URL from environment variable '{env_var_name}': {public_url}",
        )
        return public_url.rstrip("/")

    card_url = f"http://{host}:{port}"
    logger.info(
        f"No '{env_var_name}' env var found. Falling back to local URL: {card_url}",
    )
    return card_url


def parse_and_validate_input(
    ctx: InvocationContext,
    payload_model: type[BaseModel],
    agent_name: str,
) -> Any | None:
    """Parses and validates the input from the invocation context against a Pydantic model."""
    invocation_id_short = ctx.invocation_id[:8]

    # Correct way to access the input data from the InvocationContext
    if not (
        ctx.user_content and ctx.user_content.parts and ctx.user_content.parts[0].text
    ):
        logger.warning(
            f"[{agent_name} ({invocation_id_short})] Input data is missing or malformed.",
        )
        return None

    # Assuming the input is a JSON string in the text part
    input_data_str = ctx.user_content.parts[0].text

    try:
        # Pydantic's model_validate_json is ideal for parsing from a string
        validated_input = payload_model.model_validate_json(input_data_str)
        logger.debug(
            f"[{agent_name} ({invocation_id_short})] Validated input: {validated_input.model_dump_json(indent=2)}",
        )
        return validated_input
    except ValidationError as e:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] Input validation failed for data '{input_data_str}': {e}",
        )
        return None
    except Exception as e:
        logger.error(
            f"[{agent_name} ({invocation_id_short})] An unexpected error occurred during parsing: {e}",
        )
        return None
