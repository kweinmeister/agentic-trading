"""Tests for the RiskGuard CLI."""

import os
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from riskguard.__main__ import create_app
from riskguard.__main__ import main as riskguard_main


def test_riskguard_main_uses_env_var_for_url() -> None:
    """Tests that riskguard.__main__ uses the RISKGUARD_SERVICE_URL env var when set."""
    runner = CliRunner()
    public_url = "https://my-riskguard-service.com"

    with patch.dict(os.environ, {"RISKGUARD_SERVICE_URL": public_url}):
        with patch("riskguard.__main__.DefaultRequestHandler") as mock_handler:
            with patch("uvicorn.Server"):
                # Mock the root_agent which is used to build the card
                with patch("riskguard.__main__.riskguard_adk_agent") as mock_agent:
                    mock_agent.name = "Test RiskGuard"
                    mock_agent.description = "A test agent"
                    result = runner.invoke(riskguard_main, ["--port", "8080"])

                    assert result.exit_code == 0
                    mock_handler.assert_called_once()

                    # Check the agent_card argument passed to the constructor
                    _, kwargs = mock_handler.call_args
                    agent_card = kwargs.get("agent_card")
                    assert agent_card is not None
                    assert len(agent_card.supported_interfaces) > 0
                    assert (
                        agent_card.supported_interfaces[0].url
                        == f"{public_url}/a2a/jsonrpc"
                    )


def test_riskguard_main_falls_back_to_host_port() -> None:
    """Tests that riskguard.__main__ falls back to host/port when the env var is not set."""
    runner = CliRunner()
    host = "127.0.0.1"
    port = 8089

    # Ensure the environment variable is not set
    with patch.dict(os.environ, {}, clear=True):
        with patch("riskguard.__main__.DefaultRequestHandler") as mock_handler:
            with patch("uvicorn.Server"):
                with patch("riskguard.__main__.riskguard_adk_agent") as mock_agent:
                    mock_agent.name = "Test RiskGuard"
                    mock_agent.description = "A test agent"
                    result = runner.invoke(
                        riskguard_main,
                        ["--host", host, "--port", str(port)],
                    )

                    assert result.exit_code == 0
                    mock_handler.assert_called_once()

                    # Check the agent_card argument
                    _, kwargs = mock_handler.call_args
                    agent_card = kwargs.get("agent_card")
                    assert agent_card is not None
                    assert len(agent_card.supported_interfaces) > 0
                    assert (
                        agent_card.supported_interfaces[0].url
                        == f"http://{host}:{port}/a2a/jsonrpc"
                    )


def test_riskguard_agent_card_wiring() -> None:
    """Tests that RiskGuard's AgentCard is built with all expected metadata, capabilities, and skills."""
    runner = CliRunner()
    with patch("riskguard.__main__.DefaultRequestHandler") as mock_handler:
        with patch("uvicorn.Server"):
            with patch("riskguard.__main__.riskguard_adk_agent") as mock_agent:
                mock_agent.name = "RiskGuard Agent"
                mock_agent.description = "A description"
                result = runner.invoke(riskguard_main, ["--port", "8080"])
                assert result.exit_code == 0

                _, kwargs = mock_handler.call_args
                agent_card = kwargs.get("agent_card")
                assert agent_card is not None
                assert agent_card.name == "RiskGuard Agent"
                assert agent_card.version == "1.1.0"
                assert agent_card.capabilities.streaming is False
                assert agent_card.capabilities.push_notifications is False
                assert len(agent_card.skills) == 1
                skill = agent_card.skills[0]
                assert skill.id == "check_trade_risk"
                assert skill.name == "Check Trade Risk"
                assert agent_card.default_input_modes == ["data"]
                assert agent_card.default_output_modes == ["data"]

                bindings = [
                    interface.protocol_binding
                    for interface in agent_card.supported_interfaces
                ]
                assert "JSONRPC" in bindings
                assert "HTTP+JSON" in bindings


def test_riskguard_fastapi_app_routes() -> None:
    """Tests that create_app factory correctly registers RiskGuard's routes."""
    mock_card = MagicMock()
    mock_handler = MagicMock()
    mock_card.model_dump_json.return_value = "{}"

    with (
        patch("riskguard.__main__.create_agent_card_routes") as mock_card_routes,
        patch("riskguard.__main__.create_jsonrpc_routes") as mock_rpc_routes,
        patch("riskguard.__main__.create_rest_routes") as mock_rest_routes,
    ):
        from fastapi.routing import APIRoute

        mock_card_routes.return_value = [
            APIRoute(path="/.well-known/agent-card.json", endpoint=lambda: None),
        ]
        mock_rpc_routes.return_value = [
            APIRoute(path="/a2a/jsonrpc", endpoint=lambda: None),
        ]
        mock_rest_routes.return_value = [
            APIRoute(path="/a2a/rest/run", endpoint=lambda: None),
        ]

        app = create_app(mock_card, mock_handler)

        paths = {route.path for route in app.routes if hasattr(route, "path")}
        assert "/.well-known/agent-card.json" in paths
        assert "/a2a/jsonrpc" in paths
        assert "/a2a/rest/run" in paths
