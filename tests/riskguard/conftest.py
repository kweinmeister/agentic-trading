"""Fixtures for RiskGuard tests."""

import pytest

from common.models import (
    PortfolioState,
    RiskCheckPayload,
    TradeProposal,
)


@pytest.fixture
def riskguard_input_data_factory(
    base_trade_proposal: TradeProposal,
    base_portfolio_state: PortfolioState,
):
    """Provide a factory for creating RiskCheckPayload instances for RiskGuard tests.

    Injects shared base model fixtures from the root conftest.
    """

    def _input_data(**kwargs) -> RiskCheckPayload:
        # 1. Use the injected fixtures as the starting point
        final_trade_proposal = base_trade_proposal.model_copy(deep=True)
        final_portfolio_state = base_portfolio_state.model_copy(deep=True)

        # 2. Update models directly from kwargs if needed
        if "trade_proposal" in kwargs:
            update_data = kwargs.pop("trade_proposal")
            if isinstance(update_data, dict):
                # Create a new dictionary with the base model's data and update it
                merged_data = base_trade_proposal.model_dump()
                merged_data.update(update_data)
                final_trade_proposal = TradeProposal(**merged_data)
            else:
                final_trade_proposal = update_data
        if "portfolio_state" in kwargs:
            update_data = kwargs.pop("portfolio_state")
            if isinstance(update_data, dict):
                merged_data = base_portfolio_state.model_dump()
                merged_data.update(update_data)
                final_portfolio_state = PortfolioState(**merged_data)
            else:
                final_portfolio_state = update_data

        # 3. Create the final payload from the model instances
        payload = RiskCheckPayload(
            trade_proposal=final_trade_proposal,
            portfolio_state=final_portfolio_state,
            **kwargs,  # Apply any other top-level overrides
        )
        return payload

    return _input_data
