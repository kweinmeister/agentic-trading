"""Tests for the indicator calculation utilities."""

from common.utils.indicators import calculate_sma


def test_calculate_sma_valid():
    """Test SMA calculation with valid data."""
    prices = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = calculate_sma(prices, 3)
    assert result == 4.0  # (3+4+5)/3


def test_calculate_sma_insufficient_data():
    """Test SMA calculation with insufficient data."""
    prices = [1.0, 2.0]
    result = calculate_sma(prices, 5)
    assert result is None


def test_calculate_sma_empty_list():
    """Test SMA calculation with empty list."""
    prices = []
    result = calculate_sma(prices, 3)
    assert result is None


def test_calculate_sma_zero_period():
    """Test SMA calculation with zero period to trigger division by zero."""
    prices = [1.0, 2.0, 3.0]
    # This should raise ZeroDivisionError with current implementation
    result = calculate_sma(prices, 0)
    # After fix, it should return None
    assert result is None
