import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

import common.config as defaults

# Import the specific AgentExecutor for AlphaBot
from .agent_executor import AlphaBotAgentExecutor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--host",
    default=defaults.DEFAULT_ALPHABOT_URL.split(":")[1].replace("//", ""),
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=int(defaults.DEFAULT_ALPHABOT_URL.split(":")[2]),
    help="Port to bind the server to.",
)
@click.option(
    "--proxy-headers",
    is_flag=True,
    default=False,
    help="Enable proxy headers.",
)
def main(host: str, port: int, proxy_headers: bool):
    """Runs the AlphaBot agent as an A2A server."""
    logger.info("Configuring AlphaBot A2A server...")

    # Define the Agent Card for AlphaBot
    try:
        # Get the public URL from an environment variable if it exists.
        public_url = os.environ.get("ALPHABOT_SERVICE_URL")

        # Use the public URL for the agent card, otherwise fall back to local host/port.
        if public_url:
            logger.info(f"Using public URL from environment: {public_url}")
            card_url = public_url
        else:
            card_url = f"http://{host}:{port}"
            logger.info(
                f"No ALPHABOT_SERVICE_URL env var found. Falling back to local URL: {card_url}",
            )
        agent_card = AgentCard(
            name="AlphaBot Agent",
            description="Trading agent that analyzes market data and portfolio state to propose trades.",
            url=card_url.rstrip("/"),
            version="1.0.0",
            capabilities=AgentCapabilities(
                streaming=False,  # AlphaBotTaskManager doesn't support streaming
                push_notifications=False,  # Not implemented
            ),
            skills=[
                AgentSkill(
                    id="provide_trade_signal",
                    name="Provide Trade Signal",
                    description="Analyzes market and portfolio data to decide whether to buy, sell, or hold.",
                    examples=[
                        "Given market data and portfolio, what trade should I make?",
                    ],
                    tags=[],
                ),
            ],
            default_input_modes=["data"],
            default_output_modes=["data"],
        )
    except Exception:
        logger.exception("Error creating AgentCard")
        raise

    # Instantiate the AlphaBot AgentExecutor
    try:
        agent_executor = AlphaBotAgentExecutor()
    except Exception:
        logger.exception("Error initializing AlphaBotAgentExecutor")
        raise

    # Instantiate the A2AStarletteApplication
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
    )
    try:
        app_builder = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler,
        )
    except Exception:
        logger.exception("Error initializing A2AStarletteApplication")
        raise

    # Start the Server
    import uvicorn

    logger.info(f"Starting AlphaBot A2A server on http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server.")
    server_config = uvicorn.Config(
        app_builder.build(),
        host=host,
        port=port,
        proxy_headers=proxy_headers,
    )
    server = uvicorn.Server(server_config)
    server.run()


if __name__ == "__main__":
    # Example: python -m alphabot --port 8081
    main()
