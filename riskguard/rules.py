"""Risk management rules for the RiskGuard agent."""

import logging

# Import defaults from the common config
from common.config import (
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
)
from common.models import PortfolioState, RiskCheckResult, TradeProposal

logger = logging.getLogger(__name__)


def check_trade_risk_logic(
    trade_proposal: TradeProposal,
    portfolio_state: PortfolioState,
    max_pos_size: float = DEFAULT_RISKGUARD_MAX_POS_SIZE,  # Use imported default
    max_concentration: float = DEFAULT_RISKGUARD_MAX_CONCENTRATION,  # Use imported default
) -> RiskCheckResult:
    """Encapsulate the risk checking logic, using provided limits.

    Args:
        trade_proposal: TradeProposal model instance.
        portfolio_state: PortfolioState model instance.
        max_pos_size: The maximum position size for a single trade.
        max_concentration: The maximum portfolio concentration.

    Returns:
        RiskCheckResult indicating approval status and reason.

    """
    logger.info(f"Checking proposal: {trade_proposal}")

    # Basic validation
    ticker = trade_proposal.ticker
    action = trade_proposal.action
    quantity = trade_proposal.quantity
    price = trade_proposal.price

    # Ensure all required fields are present and have correct types
    if not (
        ticker
        and isinstance(action, str)
        and isinstance(
            quantity,
            (int, float),
        )  # Allow float for quantity if needed, though int is preferred
        and isinstance(price, (int, float))
    ):
        logger.warning(
            "Invalid trade proposal structure or values (missing fields or wrong types).",
        )
        return RiskCheckResult(
            approved=False,
            reason="Invalid trade proposal structure or values (missing fields or wrong types).",
        )

    # Ensure quantity and price are positive
    if quantity <= 0 or price <= 0:
        logger.warning("Trade quantity and price must be positive.")
        return RiskCheckResult(
            approved=False,
            reason="Trade quantity and price must be positive.",
        )

    # Get portfolio details safely
    cash = portfolio_state.cash
    current_shares = portfolio_state.shares
    total_value = portfolio_state.total_value
    if total_value <= 0:
        logger.warning("Invalid total portfolio value for risk check.")
        return RiskCheckResult(
            approved=False,
            reason="Invalid total portfolio value for risk check.",
        )

    proposed_trade_value = quantity * price
    action_upper = action.upper()

    if action_upper == "BUY":
        # Rule 1 (BUY): Sufficient Cash
        if proposed_trade_value > cash:
            reason = (
                f"Insufficient cash for BUY. "
                f"Cost (${proposed_trade_value:,.2f}) "
                f"exceeds available cash (${cash:,.2f})."
            )
            logger.warning(f"REJECTED - {reason}")
            return RiskCheckResult(approved=False, reason=reason)

        # Rule 2 (BUY): Max Asset Concentration (Post-Trade)
        post_trade_shares = current_shares + quantity
        post_trade_holdings_value = post_trade_shares * price
        post_trade_total_value = (
            cash - proposed_trade_value
        ) + post_trade_holdings_value  # More accurate
        if post_trade_total_value <= 0:
            post_trade_total_value = total_value  # Fallback if cash becomes negative

        concentration = (
            post_trade_holdings_value / post_trade_total_value
            if post_trade_total_value > 0
            else 1.0
        )
        if concentration > max_concentration:  # Use parameter
            reason = f"Exceeds max asset concentration ({concentration * 100:.1f}% > {max_concentration * 100:.1f}%)"
            logger.warning(f"REJECTED - {reason}")
            return RiskCheckResult(approved=False, reason=reason)

        # Rule 3 (BUY): Max Position Size per Trade (Check after cash/concentration)
        if proposed_trade_value > max_pos_size:  # Use parameter
            reason = f"Exceeds max position size per trade (${proposed_trade_value:.2f} > ${max_pos_size:.2f})"
            logger.warning(f"REJECTED - {reason}")
            return RiskCheckResult(approved=False, reason=reason)

    elif action_upper == "SELL":
        # Rule 1 (SELL): Sufficient Shares to Sell
        if quantity > current_shares:
            reason = (
                f"Insufficient shares to sell ({current_shares} < requested {quantity})"
            )
            logger.warning(f"REJECTED - {reason}")
            return RiskCheckResult(approved=False, reason=reason)

        # Rule 2 (SELL): Max Position Size per Trade (Check after sufficient shares)
        if proposed_trade_value > max_pos_size:  # Use parameter
            reason = f"Exceeds max position size per trade (${proposed_trade_value:.2f} > ${max_pos_size:.2f})"
            logger.warning(f"REJECTED - {reason}")
            return RiskCheckResult(approved=False, reason=reason)

    else:
        logger.warning(f"Unknown trade action '{action}'.")
        return RiskCheckResult(
            approved=False,
            reason=f"Unknown trade action '{action}'.",
        )

    # If all checks passed for the given action
    logger.info("APPROVED - Trade adheres to risk rules.")
    return RiskCheckResult(approved=True, reason="Trade adheres to risk rules.")
