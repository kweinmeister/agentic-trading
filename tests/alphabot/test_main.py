"""Tests for the AlphaBot CLI."""

import os
from unittest.mock import patch

from click.testing import CliRunner

from alphabot.__main__ import main as alphabot_main


def test_alphabot_main_uses_env_var_for_url() -> None:
    """Tests that alphabot.__main__ uses the ALPHABOT_SERVICE_URL env var when set."""
    runner = CliRunner()
    public_url = "https://my-alphabot-service.com"

    with patch.dict(os.environ, {"ALPHABOT_SERVICE_URL": public_url}):
        with patch("alphabot.__main__.DefaultRequestHandler") as mock_handler:
            with patch("uvicorn.Server"):
                result = runner.invoke(alphabot_main, ["--port", "8080"])

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


def test_alphabot_main_falls_back_to_host_port() -> None:
    """Tests that alphabot.__main__ falls back to host/port when the env var is not set."""
    runner = CliRunner()
    host = "127.0.0.1"
    port = 8088

    # Ensure the environment variable is not set
    with patch.dict(os.environ, {}, clear=True):
        with patch("alphabot.__main__.DefaultRequestHandler") as mock_handler:
            with patch("uvicorn.Server"):
                result = runner.invoke(
                    alphabot_main,
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
