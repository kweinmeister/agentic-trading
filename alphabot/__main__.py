"""Command-line interface for the AlphaBot agent."""

import logging

import click
from fastapi import FastAPI
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.routes.rest_routes import create_rest_routes
from a2a.server.routes.fastapi_routes import add_a2a_routes_to_fastapi

class A2AStarletteApplication:
    def __init__(self, agent_card, http_handler):
        self.agent_card = agent_card
        self.http_handler = http_handler

    def build(self) -> FastAPI:
        app = FastAPI()
        add_a2a_routes_to_fastapi(
            app,
            agent_card_routes=create_agent_card_routes(self.agent_card),
            jsonrpc_routes=create_jsonrpc_routes(self.http_handler, rpc_url='/'),
            rest_routes=create_rest_routes(self.http_handler),
        )
        return app
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

import common.config as defaults
from common.utils.agent_utils import get_service_url

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
    """Run the AlphaBot agent as an A2A server."""
    logger.info("Configuring AlphaBot A2A server...")

    # Define the Agent Card for AlphaBot
    try:
        from a2a.types import AgentCard
        global _agent_card_urls
        if "_agent_card_urls" not in globals():
            _agent_card_urls = {}
            _orig_setattr = AgentCard.__setattr__
            _orig_getattribute = AgentCard.__getattribute__
            def _new_setattr(self, name, value):
                if name == "url":
                    _agent_card_urls[id(self)] = value
                else:
                    _orig_setattr(self, name, value)
            def _new_getattribute(self, name):
                if name == "url":
                    return _agent_card_urls.get(id(self))
                return _orig_getattribute(self, name)
            AgentCard.__setattr__ = _new_setattr
            AgentCard.__getattribute__ = _new_getattribute

        card_url = get_service_url("ALPHABOT_SERVICE_URL", host, port)
        agent_card = AgentCard(
            name="AlphaBot Agent",
            description="Trading agent that analyzes market data and portfolio state to propose trades.",
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
        agent_card.url = card_url
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
        agent_card=agent_card,
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
