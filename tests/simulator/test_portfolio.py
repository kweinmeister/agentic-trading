"""Tests for the portfolio state and trade action models."""

from simulator.portfolio import PortfolioState, TradeAction


def test_portfolio_state_initialization():
    """Test PortfolioState initialization with default values."""
    portfolio = PortfolioState()

    assert portfolio.cash == 100000.0
    assert portfolio.shares == 0
    assert portfolio.holdings_value == 0.0
    assert portfolio.total_value == 100000.0


def test_portfolio_state_custom_initialization():
    """Test PortfolioState initialization with custom values."""
    cash = 50000.0
    shares = 100
    holdings_value = 5000.0
    total_value = 55000.0

    portfolio = PortfolioState(
        cash=cash,
        shares=shares,
        holdings_value=holdings_value,
        total_value=total_value,
    )

    assert portfolio.cash == cash
    assert portfolio.shares == shares
    assert portfolio.holdings_value == holdings_value
    assert portfolio.total_value == total_value


def test_portfolio_update_valuation():
    """Test update_valuation method."""
    portfolio = PortfolioState(cash=10000.0, shares=100)
    current_price = 50.0

    portfolio.update_valuation(current_price)

    assert portfolio.holdings_value == 5000.0  # 100 shares * $50
    assert portfolio.total_value == 15000.0  # $10000 cash + $5000 holdings


def test_portfolio_execute_buy_trade_sufficient_funds():
    """Test executing a BUY trade with sufficient funds."""
    portfolio = PortfolioState(cash=10000.0, shares=0)
    action = TradeAction.BUY
    quantity = 10
    price = 100.0

    result = portfolio.execute_trade(action, quantity, price)

    assert result is True
    assert portfolio.cash == 9000.0  # $10000 - (10 * $100)
    assert portfolio.shares == 10  # 0 + 10
    assert portfolio.holdings_value == 1000.0
    assert portfolio.total_value == 10000.0


def test_portfolio_execute_buy_trade_insufficient_funds():
    """Test executing a BUY trade with insufficient funds."""
    portfolio = PortfolioState(cash=500.0, shares=0)
    action = TradeAction.BUY
    quantity = 10
    price = 100.0  # Total cost would be $1000

    result = portfolio.execute_trade(action, quantity, price)

    assert result is False
    assert portfolio.cash == 500.0  # No change
    assert portfolio.shares == 0  # No change


def test_portfolio_execute_sell_trade_sufficient_shares():
    """Test executing a SELL trade with sufficient shares."""
    portfolio = PortfolioState(cash=10000.0, shares=100)
    action = TradeAction.SELL
    quantity = 10
    price = 100.0

    result = portfolio.execute_trade(action, quantity, price)

    assert result is True
    assert portfolio.cash == 11000.0  # $10000 + (10 * $100)
    assert portfolio.shares == 90  # 100 - 10


def test_portfolio_execute_sell_trade_insufficient_shares():
    """Test executing a SELL trade with insufficient shares."""
    portfolio = PortfolioState(cash=10000.0, shares=5)
    action = TradeAction.SELL
    quantity = 10
    price = 100.0

    result = portfolio.execute_trade(action, quantity, price)

    assert result is False
    assert portfolio.cash == 10000.0  # No change
    assert portfolio.shares == 5  # No change


def test_portfolio_str_representation():
    """Test the string representation of PortfolioState."""
    portfolio = PortfolioState(cash=10000.0, shares=50)
    portfolio.update_valuation(100.0)  # Update to set holdings_value and total_value

    portfolio_str = str(portfolio)

    # Check that the string contains the expected information
    assert "Cash=" in portfolio_str
    assert "Shares=50" in portfolio_str
    assert "Holdings=" in portfolio_str
    assert "Total=" in portfolio_str


def test_trade_action_enum():
    """Test the TradeAction enum values."""
    assert TradeAction.BUY.name == "BUY"
    assert TradeAction.SELL.name == "SELL"
    assert TradeAction.BUY.value == 1
    assert TradeAction.SELL.value == 2

    # Test enum membership
    assert TradeAction.BUY in TradeAction
    assert TradeAction.SELL in TradeAction
    assert "BUY" in TradeAction.__members__
    assert "SELL" in TradeAction.__members__


def test_portfolio_edge_cases():
    """Test edge cases for PortfolioState."""
    # Test with zero cash and zero shares
    portfolio = PortfolioState(cash=0.0, shares=0)

    # Try to buy shares with no cash
    result = portfolio.execute_trade(TradeAction.BUY, 1, 10.0)
    assert result is False
    assert portfolio.cash == 0.0
    assert portfolio.shares == 0

    # Try to sell shares with no shares
    result = portfolio.execute_trade(TradeAction.SELL, 1, 10.0)
    assert result is False
    assert portfolio.cash == 0.0
    assert portfolio.shares == 0


def test_portfolio_large_numbers():
    """Test PortfolioState with large numbers."""
    large_cash = 1e9  # 1 billion
    large_shares = int(1e6)  # 1 million shares
    portfolio = PortfolioState(cash=large_cash, shares=large_shares)

    # Update valuation with a reasonable price
    portfolio.update_valuation(100.0)

    assert portfolio.holdings_value == 1e8  # 100 million
    assert portfolio.total_value == 1.1e9  # 1.1 billion


def test_portfolio_fractional_shares():
    """Test that PortfolioState correctly handles integer shares."""
    portfolio = PortfolioState(cash=1000.0, shares=10)

    # Try to execute a trade with fractional shares (should work with integers)
    result = portfolio.execute_trade(TradeAction.BUY, 5, 10.0)
    assert result is True
    assert portfolio.shares == 15

    # Try to execute a trade with fractional shares that would result in fractional holdings
    # (This should still work since we're using integers)
    result = portfolio.execute_trade(TradeAction.SELL, 3, 10.0)
    assert result is True
    assert portfolio.shares == 12


def test_portfolio_zero_price():
    """Test PortfolioState with zero price."""
    portfolio = PortfolioState(cash=1000.0, shares=100)

    # Update valuation with zero price
    portfolio.update_valuation(0.0)

    assert portfolio.holdings_value == 0.0
    assert portfolio.total_value == 1000.0


def test_portfolio_very_high_price():
    """Test PortfolioState with very high price."""
    portfolio = PortfolioState(cash=1000.0, shares=100)

    # Update valuation with very high price
    very_high_price = 1e9  # $1 billion per share
    portfolio.update_valuation(very_high_price)

    assert portfolio.holdings_value == 1e11  # $100 billion
    assert portfolio.total_value == 1e11 + 1000.0  # Holdings + cash
