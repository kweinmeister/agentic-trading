"""Tests for the common data models."""

import pytest
from pydantic import ValidationError

from common.config import DEFAULT_RISKGUARD_MAX_POS_SIZE
from common.models import (
    AlphaBotTaskPayload,
    RiskCheckPayload,
    RiskCheckResult,
    TradeOutcome,
    TradeProposal,
    TradeStatus,
)


def test_risk_check_payload_valid(base_trade_proposal, base_portfolio_state) -> None:
    """Tests that a valid RiskCheckPayload can be created."""
    payload = RiskCheckPayload(
        trade_proposal=base_trade_proposal,
        portfolio_state=base_portfolio_state,
    )

    assert payload.max_pos_size == DEFAULT_RISKGUARD_MAX_POS_SIZE
    assert payload.trade_proposal.action == "BUY"


def test_trade_proposal_invalid_action() -> None:
    """Tests that TradeProposal rejects an invalid action."""
    from typing import Any

    invalid_data: dict[str, Any] = {
        "action": "HOLD",
        "ticker": "TECH",
        "quantity": 100,
        "price": 150.0,
    }
    with pytest.raises(ValidationError):
        TradeProposal(**invalid_data)


def test_risk_check_payload_missing_required_field(base_trade_proposal) -> None:
    """Tests that Pydantic raises an error if required nested models are missing."""
    # Missing 'portfolio_state' — use model_validate so the static type checker
    # sees a dict (Any) rather than a constructor call with a missing required arg.
    with pytest.raises(ValidationError) as exc_info:
        RiskCheckPayload.model_validate(
            {"trade_proposal": base_trade_proposal.model_dump()}
        )

    assert "portfolio_state" in str(exc_info.value)


def test_alphabot_task_payload_valid(base_portfolio_state) -> None:
    """Tests that a valid AlphaBotTaskPayload can be created."""
    payload = AlphaBotTaskPayload(
        historical_prices=[100.0, 101.0],
        current_price=102.0,
        portfolio_state=base_portfolio_state,
        day=1,
    )
    assert payload.day == 1
    assert payload.current_price == 102.0


def test_trade_outcome_approved(base_trade_proposal) -> None:
    """Tests a valid 'APPROVED' TradeOutcome."""
    outcome = TradeOutcome(
        status=TradeStatus.APPROVED,
        reason="SMA Crossover",
        trade_proposal=base_trade_proposal,
    )
    assert outcome.status == TradeStatus.APPROVED
    assert outcome.trade_proposal is not None
    assert outcome.trade_proposal.action == "BUY"


def test_trade_outcome_no_action() -> None:
    """Tests a valid 'NO_ACTION' TradeOutcome."""
    outcome = TradeOutcome(status=TradeStatus.NO_ACTION, reason="No signal detected")
    assert outcome.status == TradeStatus.NO_ACTION
    assert outcome.trade_proposal is None


def test_risk_check_result_validation() -> None:
    """Tests construction and validation of RiskCheckResult."""
    # Valid construction
    result = RiskCheckResult(approved=True, reason="Within limits")
    assert result.approved is True
    assert result.reason == "Within limits"

    # Default reason is empty string
    result_default = RiskCheckResult(approved=False)
    assert result_default.approved is False
    assert result_default.reason == ""

    # Validation failure: missing approved field
    with pytest.raises(ValidationError):
        RiskCheckResult.model_validate({"reason": "Missing approved"})


def test_trade_status_enum() -> None:
    """Tests that TradeStatus enum has expected members and values."""
    assert TradeStatus.APPROVED == "APPROVED"
    assert TradeStatus.REJECTED == "REJECTED"
    assert TradeStatus.NO_ACTION == "NO_ACTION"
    assert TradeStatus.ERROR == "ERROR"

    # Ensure all enum values are checked
    expected_members = {"APPROVED", "REJECTED", "NO_ACTION", "ERROR"}
    assert {status.name for status in TradeStatus} == expected_members


def test_trade_outcome_rejected_error_roundtrip(base_trade_proposal) -> None:
    """Tests REJECTED with proposal, ERROR status, and model round-trip."""
    # 1. REJECTED with proposal
    rejected_outcome = TradeOutcome(
        status=TradeStatus.REJECTED,
        reason="Exceeds concentration",
        trade_proposal=base_trade_proposal,
    )
    assert rejected_outcome.status == TradeStatus.REJECTED
    assert rejected_outcome.trade_proposal == base_trade_proposal

    # 2. ERROR status
    error_outcome = TradeOutcome(
        status=TradeStatus.ERROR,
        reason="Execution failed due to network error",
    )
    assert error_outcome.status == TradeStatus.ERROR
    assert error_outcome.trade_proposal is None

    # 3. Round-trip model_dump -> model_validate
    dumped_data = rejected_outcome.model_dump()
    validated_outcome = TradeOutcome.model_validate(dumped_data)
    assert validated_outcome == rejected_outcome
    assert validated_outcome.trade_proposal is not None
    assert validated_outcome.trade_proposal.ticker == base_trade_proposal.ticker


def test_alphabot_task_payload_defaults_and_boundaries(base_portfolio_state) -> None:
    """Tests default values for optional fields and boundaries of AlphaBotTaskPayload."""
    # Check default values are populated correctly
    payload = AlphaBotTaskPayload(
        historical_prices=[100.0, 101.0, 102.0],
        current_price=103.0,
        portfolio_state=base_portfolio_state,
        day=5,
    )
    from common.config import (
        DEFAULT_ALPHABOT_LONG_SMA,
        DEFAULT_ALPHABOT_SHORT_SMA,
        DEFAULT_ALPHABOT_TRADE_QTY,
        DEFAULT_RISKGUARD_MAX_CONCENTRATION,
        DEFAULT_RISKGUARD_MAX_POS_SIZE,
        DEFAULT_RISKGUARD_URL,
    )

    assert payload.short_sma_period == DEFAULT_ALPHABOT_SHORT_SMA
    assert payload.long_sma_period == DEFAULT_ALPHABOT_LONG_SMA
    assert payload.trade_quantity == DEFAULT_ALPHABOT_TRADE_QTY
    assert payload.riskguard_url == DEFAULT_RISKGUARD_URL
    assert payload.max_pos_size == DEFAULT_RISKGUARD_MAX_POS_SIZE
    assert payload.max_concentration == DEFAULT_RISKGUARD_MAX_CONCENTRATION

    # Boundary check: validation failure with empty historical_prices (if we ever restrict it)
    # Since historical_prices is List[float], let's check validation fails if we pass string
    with pytest.raises(ValidationError):
        AlphaBotTaskPayload.model_validate(
            {
                "historical_prices": "invalid_data",
                "current_price": 103.0,
                "portfolio_state": base_portfolio_state,
                "day": 5,
            }
        )
