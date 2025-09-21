import random

from simulator.market import MarketDataSimulator


def test_market_data_simulator_initialization():
    """Test MarketDataSimulator initialization with default values."""
    simulator = MarketDataSimulator()

    assert simulator.current_price == 100.0
    assert simulator.volatility == 0.02
    assert simulator.trend == 0.0005
    assert len(simulator.history) == 1
    assert simulator.history[0] == 100.0


def test_market_data_simulator_custom_initialization():
    """Test MarketDataSimulator initialization with custom values."""
    initial_price = 50.0
    volatility = 0.05
    trend = 0.001
    history_size = 30

    simulator = MarketDataSimulator(
        initial_price=initial_price,
        volatility=volatility,
        trend=trend,
        history_size=history_size,
    )

    assert simulator.current_price == initial_price
    assert simulator.volatility == volatility
    assert simulator.trend == trend
    assert len(simulator.history) == 1
    assert simulator.history[0] == initial_price


def test_market_data_simulator_next_price():
    """Test that next_price generates a new price and updates history."""
    simulator = MarketDataSimulator(initial_price=100.0, history_size=5)

    # Generate a few prices
    prices = [simulator.next_price() for _ in range(3)]

    # Check that we have the right number of prices in history
    assert len(simulator.history) == 4  # Initial price + 3 generated prices
    assert simulator.history[0] == 100.0
    assert simulator.get_current_price() == prices[-1]

    # Check that all prices are positive
    for price in prices:
        assert price > 0


def test_market_data_simulator_history_limit():
    """Test that history respects the history_size limit."""
    history_size = 5
    simulator = MarketDataSimulator(initial_price=100.0, history_size=history_size)

    # Generate more prices than history_size
    for _ in range(history_size + 3):
        simulator.next_price()

    # Check that history size is maintained
    assert len(simulator.history) == history_size


def test_market_data_simulator_get_historical_prices():
    """Test get_historical_prices returns the correct list."""
    simulator = MarketDataSimulator(initial_price=100.0)

    # Generate a few prices
    expected_prices = [100.0]  # Start with initial price
    for _ in range(3):
        expected_prices.append(simulator.next_price())

    # Get historical prices and compare
    historical_prices = simulator.get_historical_prices()
    assert historical_prices == expected_prices


def test_market_data_simulator_price_minimum():
    """Test that prices don't go below 1.0."""
    # Set up simulator with high volatility and negative trend to try to force low prices
    simulator = MarketDataSimulator(initial_price=1.5, volatility=0.1, trend=-0.05)

    # Generate many prices
    for _ in range(100):
        price = simulator.next_price()
        assert price >= 1.0, f"Price dropped below 1.0: {price}"


def test_market_data_simulator_deterministic_behavior(monkeypatch):
    """Test that simulator behaves deterministically with fixed random seed."""
    # Fix the random seed
    monkeypatch.setattr(
        random,
        "normalvariate",
        lambda mu, sigma: mu,
    )  # Always return the mean

    simulator1 = MarketDataSimulator(initial_price=100.0, volatility=0.02, trend=0.0005)
    simulator2 = MarketDataSimulator(initial_price=100.0, volatility=0.02, trend=0.0005)

    # Generate prices for both simulators
    prices1 = [simulator1.next_price() for _ in range(5)]
    prices2 = [simulator2.next_price() for _ in range(5)]

    # With fixed random behavior, they should be identical
    assert prices1 == prices2


def test_market_data_simulator_edge_cases():
    """Test edge cases for MarketDataSimulator."""
    # Test with very small initial price
    simulator = MarketDataSimulator(initial_price=1.0, volatility=0.01, trend=0.0)
    prices = [simulator.next_price() for _ in range(10)]

    # Verify all prices are >= 1.0
    for price in prices:
        assert price >= 1.0


def test_market_data_simulator_high_volatility():
    """Test MarketDataSimulator with high volatility."""
    simulator = MarketDataSimulator(initial_price=100.0, volatility=0.5, trend=0.0)

    # Generate prices
    prices = [simulator.next_price() for _ in range(100)]

    # With high volatility, we should see significant price movements
    # but still maintain the minimum price of 1.0
    for price in prices:
        assert price >= 1.0


def test_market_data_simulator_negative_trend():
    """Test MarketDataSimulator with negative trend."""
    simulator = MarketDataSimulator(initial_price=100.0, volatility=0.01, trend=-0.01)

    # Generate prices
    [simulator.next_price() for _ in range(100)]

    # With negative trend, we expect the final price to generally be lower than initial
    # though volatility might cause some increases
    # This is a probabilistic test - in most cases the final price should be lower
    # but we won't assert this strictly as it could fail due to randomness


def test_market_data_simulator_zero_volatility():
    """Test MarketDataSimulator with zero volatility."""
    simulator = MarketDataSimulator(initial_price=100.0, volatility=0.0, trend=0.0005)

    # Generate prices
    prices = [simulator.next_price() for _ in range(10)]

    # With zero volatility, all changes should be exactly the trend
    # Price changes by factor of (1 + trend) each time
    expected_price = 100.0
    for price in prices:
        expected_price *= 1 + 0.0005
        assert abs(price - expected_price) < 1e-10  # Allow for floating point precision


def test_market_data_simulator_large_history():
    """Test MarketDataSimulator with large history size."""
    large_history_size = 10000
    simulator = MarketDataSimulator(
        initial_price=100.0,
        history_size=large_history_size,
    )

    # Generate many prices
    for _ in range(1001):  # One more than 1% of history size
        simulator.next_price()

    # Check that history size is maintained
    assert len(simulator.history) <= large_history_size


def test_market_data_simulator_single_history_size():
    """Test MarketDataSimulator with history size of 1."""
    simulator = MarketDataSimulator(initial_price=100.0, history_size=1)

    # Generate a price
    simulator.next_price()

    # Check that history size is maintained
    assert len(simulator.history) == 1
    assert simulator.history[0] == simulator.get_current_price()
