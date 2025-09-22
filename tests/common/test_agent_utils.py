"""Tests for the agent utilities."""

import json
import os
from unittest.mock import MagicMock, patch

import pytest
from google.adk.agents.invocation_context import InvocationContext
from pydantic import BaseModel, Field

from common.utils.agent_utils import get_service_url, parse_and_validate_input


class MockModel(BaseModel):
    """A mock Pydantic model for testing."""

    name: str
    value: int = Field(gt=0)


@pytest.fixture
def mock_ctx() -> InvocationContext:
    """Provide a base InvocationContext."""
    return MagicMock(spec=InvocationContext)


def test_parse_and_validate_input_success(mock_ctx):
    """Tests successful parsing and validation."""
    input_data = {"name": "test", "value": 10}
    mock_ctx.user_content = MagicMock()
    mock_ctx.user_content.parts = [MagicMock(text=json.dumps(input_data))]
    mock_ctx.invocation_id = "test_id"

    result = parse_and_validate_input(mock_ctx, MockModel, "TestAgent")

    assert result is not None
    assert isinstance(result, MockModel)
    assert result.name == "test"
    assert result.value == 10


def test_parse_and_validate_input_no_content(mock_ctx):
    """Tests handling of missing user content."""
    mock_ctx.user_content = None
    mock_ctx.invocation_id = "test_id"

    result = parse_and_validate_input(mock_ctx, MockModel, "TestAgent")

    assert result is None


def test_parse_and_validate_input_json_error(mock_ctx):
    """Tests handling of invalid JSON."""
    mock_ctx.user_content = MagicMock()
    mock_ctx.user_content.parts = [MagicMock(text="not a valid json")]
    mock_ctx.invocation_id = "test_id"

    result = parse_and_validate_input(mock_ctx, MockModel, "TestAgent")

    assert result is None


def test_parse_and_validate_input_validation_error(mock_ctx):
    """Tests handling of Pydantic validation errors."""
    input_data = {"name": "test", "value": -5}  # 'value' must be > 0
    mock_ctx.user_content = MagicMock()
    mock_ctx.user_content.parts = [MagicMock(text=json.dumps(input_data))]
    mock_ctx.invocation_id = "test_id"

    result = parse_and_validate_input(mock_ctx, MockModel, "TestAgent")

    assert result is None


def test_get_service_url_prefers_env_var():
    """Tests that get_service_url returns the URL from the environment variable when it is set."""
    env_var_name = "MY_TEST_SERVICE_URL"
    public_url = "https://my-test-service.com"

    with patch.dict(os.environ, {env_var_name: public_url}):
        result_url = get_service_url(env_var_name, "localhost", 8080)
        assert result_url == public_url


def test_get_service_url_falls_back_to_host_port():
    """Tests that get_service_url falls back to the host and port when the env var is not set."""
    env_var_name = "MY_NONEXISTENT_SERVICE_URL"
    host = "127.0.0.1"
    port = 9999

    # Ensure the environment variable is not set
    with patch.dict(os.environ, {}, clear=True):
        result_url = get_service_url(env_var_name, host, port)
        assert result_url == f"http://{host}:{port}"


def test_get_service_url_strips_trailing_slash_from_env_var():
    """Tests that get_service_url correctly strips a trailing slash from the environment variable URL."""
    env_var_name = "MY_SLASH_SERVICE_URL"
    public_url_with_slash = "https://my-service.com/"
    expected_url = "https://my-service.com"

    with patch.dict(os.environ, {env_var_name: public_url_with_slash}):
        result_url = get_service_url(env_var_name, "localhost", 8080)
        assert result_url == expected_url
