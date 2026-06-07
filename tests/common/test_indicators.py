"""Tests for the indicator calculation utilities."""

import pytest

from common.utils.indicators import calculate_sma


@pytest.mark.parametrize(
    ("prices", "period", "expected_result"),
    [
        ([1.0, 2.0, 3.0, 4.0, 5.0], 3, 4.0),  # Valid data
        ([1.0, 2.0], 5, None),  # Insufficient data
        ([], 3, None),  # Empty list
        ([1.0, 2.0, 3.0], 0, None),  # Zero period division fallback
    ],
)
def test_calculate_sma_scenarios(
    prices: list[float],
    period: int,
    expected_result: float | None,
) -> None:
    """Test SMA calculation with various parameterized scenarios."""
    assert calculate_sma(prices, period) == expected_result
