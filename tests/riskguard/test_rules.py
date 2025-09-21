from common.models import PortfolioState, TradeProposal
from riskguard.rules import check_trade_risk_logic


def test_risk_check_with_very_small_limits(riskguard_input_data_factory):
    """Test risk check with very small limit values."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="SMALL",
            quantity=1,
            price=0.01,
        ),
        portfolio_state=PortfolioState(cash=0.01, shares=0, total_value=0.01),
        max_pos_size=0.001,
        max_concentration=0.001,
    )

    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_pos_size=input_data.max_pos_size,
        max_concentration=input_data.max_concentration,
    )

    assert result.approved is False


def test_risk_check_with_very_large_limits(riskguard_input_data_factory):
    """Test risk check with very large limit values."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="LARGE",
            quantity=1000000,
            price=1000.0,
        ),
        portfolio_state=PortfolioState(
            cash=1000000000.0,
            shares=0,
            total_value=1000000000.0,
        ),
        max_pos_size=10000000000.0,
        max_concentration=1.0,
    )

    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_pos_size=input_data.max_pos_size,
        max_concentration=input_data.max_concentration,
    )

    assert result.approved is True


def test_risk_check_approve_valid_trade(riskguard_input_data_factory):
    """Tests check_trade_risk_logic approves a valid trade."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=10,
            price=150.0,
        ),
        portfolio_state=PortfolioState(cash=10000.0, shares=0, total_value=10000.0),
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
    )
    assert result.approved is True
    assert result.reason == "Trade adheres to risk rules."


def test_risk_check_rejects_negative_portfolio_value(riskguard_input_data_factory):
    """Test that trades resulting in negative portfolio values are rejected.

    When a trade would result in a negative post-trade total value,
    the current logic uses the pre-trade total value for concentration calculation,
    which can mask the extreme risk of such trades.
    """
    # Create a scenario where a BUY trade would result in negative portfolio value
    # Portfolio: $100 cash, 0 shares, total value $100
    # Trade: Buy 50 shares at $10 each = $500 cost
    # Post-trade cash: $100 - $500 = -$400
    # Post-trade holdings value: 50 * $10 = $500
    # Post-trade total value (correct calculation): -$400 + $500 = $100

    # With the current implementation, if post_trade_total_value <= 0,
    # it uses the pre-trade total_value of $100 for concentration calculation
    # Concentration: $500 / $100 = 500% > 50% limit, so correctly rejected

    # However, the logic should be more strict about negative portfolio values
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=50,
            price=10.0,
        ),  # $500 trade
        portfolio_state=PortfolioState(cash=100.0, shares=0, total_value=100.0),
        max_pos_size=1000.0,  # Large enough to not block the trade
        max_concentration=0.5,  # 50% concentration limit
    )

    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_pos_size=input_data.max_pos_size,
        max_concentration=input_data.max_concentration,
    )

    # The trade should be rejected due to concentration limits
    # (This test passes with current implementation, but for the wrong reason)
    # After the fix, it should be rejected for the right reason
    assert result.approved is False


def test_risk_check_reject_pos_size(riskguard_input_data_factory):
    """Tests check_trade_risk_logic rejects a trade exceeding max position size."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=60,
            price=100.0,
        ),
        portfolio_state=PortfolioState(cash=10000.0, shares=0, total_value=10000.0),
        max_pos_size=5000,
        max_concentration=1.0,  # Ensure concentration doesn't block
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_pos_size=input_data.max_pos_size,
        max_concentration=input_data.max_concentration,
    )
    assert result.approved is False
    assert "Exceeds max position size per trade" in result.reason


def test_risk_check_reject_insufficient_cash(riskguard_input_data_factory):
    """Tests check_trade_risk_logic rejects a BUY trade with insufficient cash."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=100,
            price=150.0,
        ),
        portfolio_state=PortfolioState(cash=1000.0, shares=0, total_value=1000.0),
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
    )
    assert result.approved is False
    assert "Insufficient cash" in result.reason


def test_risk_check_reject_concentration_limit(riskguard_input_data_factory):
    """Tests check_trade_risk_logic rejects a trade exceeding concentration limit."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=50,
            price=100.0,
        ),
        portfolio_state=PortfolioState(cash=10000.0, shares=0, total_value=10000.0),
        max_concentration=0.3,
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_concentration=input_data.max_concentration,
    )
    assert result.approved is False
    assert "Exceeds max asset concentration" in result.reason


def test_risk_check_reject_insufficient_shares(riskguard_input_data_factory):
    """Tests check_trade_risk_logic rejects a SELL trade with insufficient shares."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="SELL",
            ticker="TECH",
            quantity=100,
            price=100.0,
        ),
        portfolio_state=PortfolioState(cash=10000.0, shares=50, total_value=15000.0),
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
    )
    assert result.approved is False
    assert "Insufficient shares to sell" in result.reason


def test_risk_check_reject_unknown_action(riskguard_input_data_factory):
    """Tests check_trade_risk_logic rejects a trade with unknown action."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=100,
            price=100.0,
        ),
    )
    input_data.trade_proposal.action = "HOLD"  # Manually set invalid action
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
    )
    assert result.approved is False
    assert "Unknown trade action" in result.reason


def test_risk_check_invalid_input(riskguard_input_data_factory):
    """Tests check_trade_risk_logic handles invalid input data."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=-10,
            price=100.0,
        ),
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
    )
    assert result.approved is False
    assert "Trade quantity and price must be positive" in result.reason


def test_risk_check_zero_values(riskguard_input_data_factory):
    """Tests check_trade_risk_logic handles zero values correctly."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=0,
            price=0.0,
        ),
        portfolio_state=PortfolioState(cash=10000.0, shares=0, total_value=10000.0),
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_pos_size=1000.0,  # Set to exact trade value
        max_concentration=1.0,  # Allow 100% concentration
    )
    assert result.approved is False
    assert "Trade quantity and price must be positive." in result.reason


def test_risk_check_approve_exact_cash(riskguard_input_data_factory):
    """Tests that a trade using the exact available cash is approved."""
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=10,
            price=100.0,
        ),
        portfolio_state=PortfolioState(cash=1000.0, shares=0, total_value=1000.0),
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_pos_size=1000.0,
        max_concentration=1.0,
    )
    assert result.approved is True
    assert result.reason == "Trade adheres to risk rules."


def test_risk_check_division_by_zero(riskguard_input_data_factory):
    """Tests that check_trade_risk_logic handles a portfolio with a total value of
    zero gracefully, preventing a ZeroDivisionError.
    """
    input_data = riskguard_input_data_factory(
        trade_proposal=TradeProposal(
            action="BUY",
            ticker="TECH",
            quantity=10,
            price=100.0,
        ),
        portfolio_state=PortfolioState(cash=0.0, shares=0, total_value=0.0),
        max_concentration=0.5,
    )
    result = check_trade_risk_logic(
        trade_proposal=input_data.trade_proposal,
        portfolio_state=input_data.portfolio_state,
        max_concentration=input_data.max_concentration,
    )
    # In this scenario, the trade should be rejected because there is no cash.
    assert result.approved is False
    assert "Invalid total portfolio value for risk check." in result.reason
