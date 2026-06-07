"""Command-line interface for the AlphaBot agent."""

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

# Import the specific AgentExecutor for AlphaBot
from .agent_executor import AlphaBotAgentExecutor

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

# Parse host and port safely using urlparse
parsed_default_url = urlparse(defaults.DEFAULT_ALPHABOT_URL)
alphabot_default_host = parsed_default_url.hostname or "127.0.0.1"
alphabot_default_port = parsed_default_url.port or 8081


def create_app(agent_card, request_handler) -> FastAPI:
    """Create the FastAPI app with mounted A2A routes."""
    agent_card_routes = create_agent_card_routes(agent_card)
    jsonrpc_routes = create_jsonrpc_routes(request_handler, rpc_url="/a2a/jsonrpc")
    rest_routes = create_rest_routes(request_handler, path_prefix="/a2a/rest")

    app = FastAPI()
    app.routes.extend(jsonrpc_routes)
    app.routes.extend(agent_card_routes)
    app.routes.extend(rest_routes)
    return app


@click.command()
@click.option(
    "--host",
    default=alphabot_default_host,
    help="Host to bind the server to.",
)
@click.option(
    "--port",
    default=alphabot_default_port,
    help="Port to bind the server to.",
)
@click.option(
    "--proxy-headers",
    is_flag=True,
    default=False,
    help="Enable proxy headers.",
)
def main(host: str, port: int, proxy_headers: bool) -> None:
    """Run the AlphaBot agent as an A2A server."""
    logger.info("Configuring AlphaBot A2A server...")

    # Define the Agent Card for AlphaBot
    try:
        card_url = get_service_url("ALPHABOT_SERVICE_URL", host, port)
        agent_card = AgentCard(
            name="AlphaBot Agent",
            description="Trading agent that analyzes market data and portfolio state to propose trades.",
            provider=AgentProvider(
                organization="A2A Samples",
                url="https://example.com",
            ),
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
    except Exception:
        logger.exception("Error creating AgentCard")
        raise

    # Instantiate the AlphaBot AgentExecutor
    try:
        agent_executor = AlphaBotAgentExecutor()
    except Exception:
        logger.exception("Error initializing AlphaBotAgentExecutor")
        raise

    # Instantiate uvicorn / fastapi routes
    task_store = InMemoryTaskStore()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=task_store,
        agent_card=agent_card,
    )

    try:
        app = create_app(agent_card, request_handler)
    except Exception:
        logger.exception("Error initializing routes and FastAPI application")
        raise

    # Start the Server
    import uvicorn

    logger.info(f"Starting AlphaBot A2A server on http://{host}:{port}")
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
    # Example: python -m alphabot --port 8081
    main()
