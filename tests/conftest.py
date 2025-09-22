"""Global fixtures for all tests."""

from typing import Optional  # Import Optional
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio
from a2a.server.events import EventQueue
from google.adk.agents import BaseAgent  # Import BaseAgent for type hinting
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.sessions import InMemorySessionService, Session
from google.genai import types as genai_types

from common.models import PortfolioState, TradeProposal


@pytest.fixture
def event_queue() -> EventQueue:
    """Provide a new EventQueue instance for each test."""
    return EventQueue()


@pytest.fixture
def test_agent_card():
    """Fixture to create a valid AgentCard for testing."""
    from a2a.types import AgentCapabilities, AgentCard, AgentSkill

    return AgentCard(
        name="Test Agent",
        description="A test agent.",
        url="http://test-agent.com",
        version="1.0.0",
        capabilities=AgentCapabilities(streaming=False, push_notifications=False),
        skills=[
            AgentSkill(
                id="test_skill",
                name="Test Skill",
                description="A test skill.",
                examples=["Test example"],
                tags=[],
            ),
        ],
        default_input_modes=["data"],
        default_output_modes=["data"],
    )


@pytest_asyncio.fixture
async def adk_session() -> Session:
    """Provide a reusable, awaited session instance for tests."""
    return await InMemorySessionService().create_session(
        app_name="test_app",
        user_id="test_user",
    )


@pytest.fixture
def adk_ctx(
    adk_session: Session,
    agent: Optional[BaseAgent] = None,
) -> InvocationContext:  # Use Optional
    """Provide a base InvocationContext."""
    # If no agent is provided, use a MagicMock to satisfy the BaseAgent type hint
    if agent is None:
        agent = MagicMock(spec=BaseAgent)
        agent.name = "MockAgent"  # Assign a name to the mock agent

    return InvocationContext(
        agent=agent,
        session_service=InMemorySessionService(),
        invocation_id="test_invocation",
        session=adk_session,
    )


@pytest.fixture
def base_portfolio_state() -> PortfolioState:
    """Provide a default portfolio state, shared across all tests."""
    return PortfolioState(cash=100000.0, shares=100, total_value=110000.0)


@pytest.fixture
def base_trade_proposal() -> TradeProposal:
    """Provide a default 'BUY' trade proposal, shared across all tests."""
    return TradeProposal(
        action="BUY",
        ticker="TECH",
        quantity=50,
        price=100.0,
    )


@pytest.fixture
def mock_runner_factory():
    """Create a mock Runner for a specific agent."""

    def _factory(agent_module_path: str):
        with patch(f"{agent_module_path}.Runner") as mock:
            mock_runner_instance = mock.return_value
            # Set the app_name on the mock_runner_instance
            mock_runner_instance.app_name = "test_app"
            mock_runner_instance.session_service = InMemorySessionService()
            return mock_runner_instance

    return _factory


@pytest.fixture
def adk_mock_alphabot_generator():
    """Mock the async generator for AlphaBot, yielding ADK Events instead of.

    raw genai types. This aligns with the behavior of the ADK Runner.
    """

    async def _generator(final_state_delta, final_reason):
        # Ensure the trade_proposal in the delta is a complete object
        if "approved_trade" in final_state_delta:
            trade = final_state_delta["approved_trade"]
            if "ticker" not in trade:
                trade["ticker"] = "TEST"
            if "price" not in trade:
                trade["price"] = 100.0
        if "rejected_trade_proposal" in final_state_delta:
            trade = final_state_delta["rejected_trade_proposal"]
            if "ticker" not in trade:
                trade["ticker"] = "TEST"
            if "price" not in trade:
                trade["price"] = 100.0

        # Yield an event with the state delta
        yield Event(
            author="test_author",
            actions=EventActions(state_delta=final_state_delta),
        )
        # Yield a final event with the reason text
        yield Event(
            author="test_author",
            content=genai_types.Content(parts=[genai_types.Part(text=final_reason)]),
            turn_complete=True,
        )

    return _generator


@pytest.fixture
def adk_mock_riskguard_generator():
    """Mock the async generator for RiskGuard, yielding a single ADK Event.

    with a function response. This aligns with the behavior of the ADK Runner.
    """

    async def _generator(result_name, result_data):
        yield Event(
            author="test_author",
            content=genai_types.Content(
                parts=[
                    genai_types.Part(
                        function_response=genai_types.FunctionResponse(
                            name=result_name,
                            response=result_data,
                        ),
                    ),
                ],
            ),
            turn_complete=True,
        )

    return _generator


@pytest.fixture
def mock_a2a_send_message_generator():
    """Create a mock async generator for a2a_client.send_message.

    This fixture provides a function that can be used to configure an AsyncMock's
    send_message method to work properly with 'async for' loops.
    """

    def _create_mock_send_message(*yield_values):
        """Create a mock send_message function that yields the provided values.

        Args:
            *yield_values: Values to yield from the generator

        Returns:
            An async function that can be used as a mock for send_message

        """

        async def mock_send_message_generator(*args, **kwargs):
            for value in yield_values:
                yield value

        return mock_send_message_generator

    return _create_mock_send_message


def create_async_error_iterator(exception_class, *args, **kwargs):
    """Create an async iterator that raises an exception on the first call to __anext__.

    Args:
        exception_class: The exception class to raise
        *args: Positional arguments to pass to the exception constructor
        **kwargs: Keyword arguments to pass to the exception constructor

    Returns:
        An async iterator that raises the specified exception on first __anext__ call

    """

    class MockSendMessageErrorIterator:
        def __init__(self, exception):
            self.exception = exception

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise self.exception

    return MockSendMessageErrorIterator(exception_class(*args, **kwargs))


@pytest.fixture
def mock_a2a_sdk_components():
    """Mock and patch the A2A SDK's ClientFactory and A2ACardResolver.

    This provides a hermetic environment for testing functions that make A2A calls,
    preventing real network requests. It returns a dictionary of the key mock
    instances for use in tests.
    """
    from unittest.mock import AsyncMock, MagicMock, patch

    from a2a.types import AgentCard

    with (
        patch(
            "alphabot.a2a_risk_tool.A2ACardResolver",
        ) as mock_resolver_class_risk_tool,
        patch("alphabot.a2a_risk_tool.ClientFactory") as mock_factory_class_risk_tool,
        patch("simulator.main.A2ACardResolver") as mock_resolver_class_main,
        patch("simulator.main.ClientFactory") as mock_factory_class_main,
    ):
        # --- Mock A2ACardResolver ---
        mock_resolver_instance_risk_tool = mock_resolver_class_risk_tool.return_value
        mock_resolver_instance_main = mock_resolver_class_main.return_value
        mock_agent_card = AgentCard.model_validate(
            {
                "url": "http://mock-riskguard.com",
                "name": "MockRiskGuard",
                "description": "A mock RiskGuard agent card",
                "version": "1.0",
                "capabilities": {},
                "defaultInputModes": [],
                "defaultOutputModes": [],
                "skills": [],
            },
        )
        mock_resolver_instance_risk_tool.get_agent_card = AsyncMock(
            return_value=mock_agent_card,
        )
        mock_resolver_instance_main.get_agent_card = AsyncMock(
            return_value=mock_agent_card,
        )

        # --- Mock ClientFactory and the Client it creates ---
        mock_factory_instance_risk_tool = mock_factory_class_risk_tool.return_value
        mock_factory_instance_main = mock_factory_class_main.return_value
        mock_a2a_client = AsyncMock()
        mock_factory_instance_risk_tool.create.return_value = mock_a2a_client
        mock_factory_instance_main.create.return_value = mock_a2a_client

        # Also mock the config attribute that might be accessed
        mock_factory_instance_risk_tool._config = MagicMock()
        mock_factory_instance_risk_tool._config.httpx_client = AsyncMock()
        mock_factory_instance_main._config = MagicMock()
        mock_factory_instance_main._config.httpx_client = AsyncMock()

        yield {
            "mock_resolver_class": mock_resolver_class_main,
            "mock_factory_class": mock_factory_class_main,
            "mock_resolver_instance": mock_resolver_instance_main,
            "mock_resolver_instance_risk_tool": mock_resolver_instance_risk_tool,
            "mock_factory_instance": mock_factory_instance_main,
            "mock_a2a_client": mock_a2a_client,
            "mock_agent_card": mock_agent_card,
        }
