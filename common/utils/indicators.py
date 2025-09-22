"""Indicator calculation utilities."""

from typing import List, Optional


def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Calculate the Simple Moving Average."""
    if period <= 0:
        return None
    price_slice = prices[-period:]
    if len(prices) < period:
        return None
    valid_prices = [float(p) for p in price_slice if isinstance(p, (int, float))]
    if len(valid_prices) < period:
        return None
    result = sum(valid_prices) / period
    return result
