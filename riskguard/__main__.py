"""Command-line interface for the RiskGuard agent."""

import logging
from urllib.parse import urlparse

import click
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import (
    create_agent_card_routes,
    create_jsonrpc_routes,
    create_rest_routes,
)
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentInterface,
    AgentProvider,
    AgentSkill,
)
from fastapi import FastAPI

import common.config as defaults
from common.utils.agent_utils import get_service_url

from .agent import root_agent as riskguard_adk_agent
from .agent_executor import RiskGuardAgentExecutor  # Renamed from RiskGuardTaskManager

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Parse host and port safely using urlparse
parsed_default_url = urlparse(defaults.DEFAULT_RISKGUARD_URL)
riskguard_default_host = parsed_default_url.hostname or "127.0.0.1"
riskguard_default_port = parsed_default_url.port or 8080


@click.command()
@click.option(
    "--host",
    default=riskguard_default_host,
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=riskguard_default_port,
    help="Port to bind the server to.",
)
@click.option(
    "--proxy-headers",
    is_flag=True,
    default=False,
    help="Enable proxy headers.",
)
def main(host: str, port: int, proxy_headers: bool) -> None:
    """Run the RiskGuard ADK agent as an A2A server."""
    logger.info("Configuring RiskGuard A2A server...")

    try:
        card_url = get_service_url("RISKGUARD_SERVICE_URL", host, port)
        agent_card = AgentCard(
            name=riskguard_adk_agent.name,
            description=riskguard_adk_agent.description,
            provider=AgentProvider(
                organization="A2A Samples",
                url="https://example.com",
            ),
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
            supported_interfaces=[
                AgentInterface(
                    protocol_binding="JSONRPC",
                    protocol_version="1.0",
                    url=f"{card_url.rstrip('/')}/a2a/jsonrpc",
                ),
                AgentInterface(
                    protocol_binding="HTTP+JSON",
                    protocol_version="1.0",
                    url=f"{card_url.rstrip('/')}/a2a/rest",
                ),
            ],
        )
    except AttributeError as e:
        logger.exception(
            f"Error accessing attributes from riskguard_adk_agent: {e}. Is riskguard/agent.py correct?",
        )
        raise

    try:
        agent_executor = RiskGuardAgentExecutor()
    except Exception as e:
        logger.exception(f"Error initializing RiskGuardAgentExecutor: {e}")
        raise

    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    try:
        agent_card_routes = create_agent_card_routes(agent_card)
        jsonrpc_routes = create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc")
        rest_routes = create_rest_routes(request_handler, path_prefix="/a2a/rest")

        app = FastAPI()
        app.routes.extend(jsonrpc_routes)
        app.routes.extend(agent_card_routes)
        app.routes.extend(rest_routes)
    except Exception as e:
        logger.exception(f"Error initializing routes and FastAPI application: {e}")
        raise

    # Start the Server
    import uvicorn

    logger.info(f"Starting RiskGuard A2A server on http://{host}:{port}")
    logger.info("Press Ctrl+C to stop the server.")
    server_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        proxy_headers=proxy_headers,
    )
    server = uvicorn.Server(server_config)
    server.run()


if __name__ == "__main__":
    main()
