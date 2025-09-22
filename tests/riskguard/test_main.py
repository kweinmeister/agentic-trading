import os
from unittest.mock import patch

from click.testing import CliRunner

from riskguard.__main__ import main as riskguard_main


def test_riskguard_main_uses_env_var_for_url():
    """Tests that riskguard.__main__ uses the RISKGUARD_SERVICE_URL env var when set."""
    runner = CliRunner()
    public_url = "https://my-riskguard-service.com"

    with patch.dict(os.environ, {"RISKGUARD_SERVICE_URL": public_url}):
        with patch("riskguard.__main__.A2AStarletteApplication") as mock_app:
            with patch("uvicorn.Server"):
                # Mock the root_agent which is used to build the card
                with patch("riskguard.__main__.riskguard_adk_agent") as mock_agent:
                    mock_agent.name = "Test RiskGuard"
                    mock_agent.description = "A test agent"
                    result = runner.invoke(riskguard_main, ["--port", "8080"])

                    assert result.exit_code == 0
                    mock_app.assert_called_once()

                    # Check the agent_card argument passed to the constructor
                    _, kwargs = mock_app.call_args
                    agent_card = kwargs.get("agent_card")
                    assert agent_card is not None
                    assert agent_card.url == public_url


def test_riskguard_main_falls_back_to_host_port():
    """Tests that riskguard.__main__ falls back to host/port when the env var is not set."""
    runner = CliRunner()
    host = "127.0.0.1"
    port = 8089

    # Ensure the environment variable is not set
    with patch.dict(os.environ, {}, clear=True):
        with patch("riskguard.__main__.A2AStarletteApplication") as mock_app:
            with patch("uvicorn.Server"):
                with patch("riskguard.__main__.riskguard_adk_agent") as mock_agent:
                    mock_agent.name = "Test RiskGuard"
                    mock_agent.description = "A test agent"
                    result = runner.invoke(
                        riskguard_main,
                        ["--host", host, "--port", str(port)],
                    )

                    assert result.exit_code == 0
                    mock_app.assert_called_once()

                    # Check the agent_card argument
                    _, kwargs = mock_app.call_args
                    agent_card = kwargs.get("agent_card")
                    assert agent_card is not None
                    assert agent_card.url == f"http://{host}:{port}"
