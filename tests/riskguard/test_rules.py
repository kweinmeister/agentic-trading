"""Tests for the RiskGuard rules."""


import pytest

from common.models import PortfolioState, TradeProposal
from riskguard.rules import check_trade_risk_logic


@pytest.mark.parametrize(
    (
        "action",
        "quantity",
        "price",
        "cash",
        "shares",
        "total_value",
        "max_pos_size",
        "max_concentration",
        "expected_approved",
        "expected_reason",
    ),
    [
        # Success Cases
        (
            "BUY",
            10,
            150.0,
            10000.0,
            0,
            10000.0,
            None,
            None,
            True,
            "Trade adheres to risk rules.",
        ),
        # Very small limits
        (
            "BUY",
            1,
            0.01,
            0.01,
            0,
            0.01,
            0.001,
            0.001,
            False,
            "Exceeds max asset concentration",
        ),
        # Very large limits
        (
            "BUY",
            1000000,
            1000.0,
            1000000000.0,
            0,
            1000000000.0,
            10000000000.0,
            1.0,
            True,
            "Trade adheres to risk rules.",
        ),
        # SELL with max concentration parameter check (does not affect SELL)
        (
            "SELL",
            10,
            100.0,
            10000.0,
            20,
            12000.0,
            None,
            0.5,
            True,
            "Trade adheres to risk rules.",
        ),
        (
            "BUY",
            10,
            100.0,
            1000.0,
            0,
            1000.0,
            1000.0,
            1.0,
            True,
            "Trade adheres to risk rules.",
        ),
        # Failures - Position Size / concentration
        (
            "BUY",
            60,
            100.0,
            10000.0,
            0,
            10000.0,
            5000,
            1.0,
            False,
            "Exceeds max position size per trade",
        ),
        (
            "BUY",
            50,
            100.0,
            10000.0,
            0,
            10000.0,
            None,
            0.3,
            False,
            "Exceeds max asset concentration",
        ),
        # Failures - Insufficient Funds / Holdings
        ("BUY", 100, 150.0, 1000.0, 0, 1000.0, None, None, False, "Insufficient cash"),
        (
            "SELL",
            100,
            100.0,
            10000.0,
            50,
            15000.0,
            None,
            None,
            False,
            "Insufficient shares to sell",
        ),
        # Failures - Invalid Inputs / Boundary values
        (
            "HOLD",
            100,
            100.0,
            10000.0,
            0,
            10000.0,
            None,
            None,
            False,
            "Unknown trade action",
        ),
        (
            "BUY",
            -10,
            100.0,
            10000.0,
            0,
            10000.0,
            None,
            None,
            False,
            "Trade quantity and price must be positive",
        ),
        (
            "BUY",
            0,
            0.0,
            10000.0,
            0,
            10000.0,
            1000.0,
            1.0,
            False,
            "Trade quantity and price must be positive",
        ),
        (
            "BUY",
            10,
            100.0,
            0.0,
            0,
            0.0,
            None,
            0.5,
            False,
            "Invalid total portfolio value for risk check.",
        ),
    ],
)
def test_check_trade_risk_logic_scenarios(
    action: str,
    quantity: int,
    price: float,
    cash: float,
    shares: int,
    total_value: float,
    max_pos_size: float | None,
    max_concentration: float | None,
    expected_approved: bool,
    expected_reason: str,
) -> None:
    """Test check_trade_risk_logic with various parameterized scenarios."""
    trade_proposal = TradeProposal(
        action=action if action in ("BUY", "SELL") else "BUY",
        ticker="TECH",
        quantity=quantity,
        price=price,
    )
    if action == "HOLD":
        setattr(trade_proposal, "action", "HOLD")

    portfolio_state = PortfolioState(cash=cash, shares=shares, total_value=total_value)

    kwargs = {}
    if max_pos_size is not None:
        kwargs["max_pos_size"] = max_pos_size
    if max_concentration is not None:
        kwargs["max_concentration"] = max_concentration

    result = check_trade_risk_logic(
        trade_proposal=trade_proposal,
        portfolio_state=portfolio_state,
        **kwargs,
    )

    assert result.approved == expected_approved
    assert expected_reason in result.reason
