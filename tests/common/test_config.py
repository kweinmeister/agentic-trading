"""Tests for configuration parameters consistency."""

from urllib.parse import urlparse

from common.config import (
    DEFAULT_ALPHABOT_PORT,
    DEFAULT_ALPHABOT_URL,
    DEFAULT_RISKGUARD_PORT,
    DEFAULT_RISKGUARD_URL,
    DEFAULT_SIMULATOR_PORT,
    DEFAULT_SIMULATOR_URL,
)


def test_default_urls_port_consistency() -> None:
    """Tests that default URLs match default ports."""
    assert urlparse(DEFAULT_RISKGUARD_URL).port == DEFAULT_RISKGUARD_PORT
    assert urlparse(DEFAULT_ALPHABOT_URL).port == DEFAULT_ALPHABOT_PORT
    assert urlparse(DEFAULT_SIMULATOR_URL).port == DEFAULT_SIMULATOR_PORT
