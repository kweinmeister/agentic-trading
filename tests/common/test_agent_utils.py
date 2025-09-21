import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

# ADK Imports
try:
    from google.adk.agents.invocation_context import InvocationContext
except ImportError:
    from tests.adk_mocks import InvocationContext

from common.utils.agent_utils import parse_and_validate_input


class MockModel(BaseModel):
    name: str
    value: int = Field(gt=0)


@pytest.fixture
def mock_ctx() -> InvocationContext:
    """Provides a base InvocationContext."""
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
