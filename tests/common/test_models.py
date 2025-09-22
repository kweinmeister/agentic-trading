"""Tests for the common data models."""

import pytest
from pydantic import ValidationError

from common.config import DEFAULT_RISKGUARD_MAX_POS_SIZE
from common.models import (
    AlphaBotTaskPayload,
    PortfolioState,
    RiskCheckPayload,
    TradeOutcome,
    TradeProposal,
    TradeStatus,
)


def test_risk_check_payload_valid():
    """Tests that a valid RiskCheckPayload can be created."""
    trade_proposal = TradeProposal(
        action="BUY",
        ticker="TECH",
        quantity=100,
        price=150.0,
    )
    portfolio_state = PortfolioState(cash=50000.0, shares=200, total_value=80000.0)

    payload = RiskCheckPayload(
        trade_proposal=trade_proposal,
        portfolio_state=portfolio_state,
    )

    assert payload.max_pos_size == DEFAULT_RISKGUARD_MAX_POS_SIZE
    assert payload.trade_proposal.action == "BUY"


def test_trade_proposal_invalid_action():
    """Tests that TradeProposal rejects an invalid action."""
    with pytest.raises(ValidationError):
        TradeProposal(action="HOLD", ticker="TECH", quantity=100, price=150.0)  # type: ignore


def test_risk_check_payload_missing_required_field():
    """Tests that Pydantic raises an error if required nested models are missing."""
    # Missing 'portfolio_state'
    with pytest.raises(ValidationError) as exc_info:
        RiskCheckPayload(  # type: ignore
            trade_proposal=TradeProposal(
                action="SELL",
                ticker="TECH",
                quantity=50,
                price=155.0,
            ),
        )

    assert "portfolio_state" in str(exc_info.value)


def test_alphabot_task_payload_valid():
    """Tests that a valid AlphaBotTaskPayload can be created."""
    portfolio_state = PortfolioState(cash=10000, shares=100, total_value=20000)
    payload = AlphaBotTaskPayload(
        historical_prices=[100.0, 101.0],
        current_price=102.0,
        portfolio_state=portfolio_state,
        day=1,
    )
    assert payload.day == 1
    assert payload.current_price == 102.0


def test_trade_outcome_approved():
    """Tests a valid 'APPROVED' TradeOutcome."""
    proposal = TradeProposal(action="BUY", ticker="TEST", quantity=10, price=100)
    outcome = TradeOutcome(
        status=TradeStatus.APPROVED,
        reason="SMA Crossover",
        trade_proposal=proposal,
    )
    assert outcome.status == TradeStatus.APPROVED
    assert outcome.trade_proposal is not None
    assert outcome.trade_proposal.action == "BUY"


def test_trade_outcome_no_action():
    """Tests a valid 'NO_ACTION' TradeOutcome."""
    outcome = TradeOutcome(status=TradeStatus.NO_ACTION, reason="No signal detected")
    assert outcome.status == TradeStatus.NO_ACTION
    assert outcome.trade_proposal is None
