"""Indicator calculation utilities."""


def calculate_sma(prices: list[float], period: int) -> float | None:
    """Calculate the Simple Moving Average."""
    if period <= 0:
        return None
    price_slice = prices[-period:]
    if len(prices) < period:
        return None
    valid_prices = [float(p) for p in price_slice if isinstance(p, (int, float))]
    if len(valid_prices) < period:
        return None
    return sum(valid_prices) / period
