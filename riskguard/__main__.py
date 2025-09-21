import logging
import os

import click
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

import common.config as defaults

from .agent import root_agent as riskguard_adk_agent
from .agent_executor import RiskGuardAgentExecutor  # Renamed from RiskGuardTaskManager

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--host",
    default=defaults.DEFAULT_RISKGUARD_URL.split(":")[1].replace("//", ""),
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=int(defaults.DEFAULT_RISKGUARD_URL.split(":")[2]),
    help="Port to bind the server to.",
)
@click.option(
    "--proxy-headers",
    is_flag=True,
    default=False,
    help="Enable proxy headers.",
)
def main(host: str, port: int, proxy_headers: bool):
    """Runs the RiskGuard ADK agent as an A2A server."""
    logger.info("Configuring RiskGuard A2A server...")

    try:
        # Get public URL from environment variable for cloud deployment.
        public_url = os.environ.get("RISKGUARD_SERVICE_URL")

        # Use the public URL for the agent card, otherwise fall back to local host/port.
        if public_url:
            logger.info(f"Using public URL from environment: {public_url}")
            card_url = public_url
        else:
            card_url = f"http://{host}:{port}"
            logger.info(
                f"No RISKGUARD_SERVICE_URL env var found. Falling back to local URL: {card_url}",
            )
        agent_card = AgentCard(
            name=riskguard_adk_agent.name,
            description=riskguard_adk_agent.description,
            url=card_url.rstrip("/"),  # Use the determined URL
            version="1.1.0",
            capabilities=AgentCapabilities(
                streaming=False,
                push_notifications=False,
            ),
            skills=[
                AgentSkill(
                    id="check_trade_risk",
                    name="Check Trade Risk",
                    description="Validates if a proposed trade meets risk criteria.",
                    examples=["Check if buying 100 TECH_STOCK at $150 is allowed."],
                    tags=[],
                ),
            ],
            default_input_modes=["data"],
            default_output_modes=["data"],
        )
    except AttributeError as e:
        logger.error(
            f"Error accessing attributes from riskguard_adk_agent: {e}. Is riskguard/agent.py correct?",
        )
        raise

    try:
        agent_executor = RiskGuardAgentExecutor()
    except Exception as e:
        logger.error(f"Error initializing RiskGuardAgentExecutor: {e}")
        raise

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
    except Exception as e:
        logger.error(f"Error initializing A2AStarletteApplication: {e}")
        raise

    # Start the Server
    import uvicorn

    logger.info(f"Starting RiskGuard A2A server on http://{host}:{port}")
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
    main()
