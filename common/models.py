"""Data models for the agentic trading simulator."""

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

# --- Import Defaults from config.py ---
from .config import (
    DEFAULT_ALPHABOT_LONG_SMA,
    DEFAULT_ALPHABOT_SHORT_SMA,
    DEFAULT_ALPHABOT_TRADE_QTY,
    DEFAULT_RISKGUARD_MAX_CONCENTRATION,
    DEFAULT_RISKGUARD_MAX_POS_SIZE,
    DEFAULT_RISKGUARD_URL,
)

# --- Shared Core Models ---


class TradeProposal(BaseModel):
    """Represents a trade proposal initiated by an agent."""

    action: Literal["BUY", "SELL"]
    ticker: str
    quantity: int
    price: float


class PortfolioState(BaseModel):
    """Represents the state of the portfolio at a given time."""

    cash: float
    shares: int
    total_value: float


# --- Models for RiskGuard Agent ---


class RiskCheckPayload(BaseModel):
    """Payload sent TO RiskGuard for a risk assessment."""

    trade_proposal: TradeProposal
    portfolio_state: PortfolioState
    max_pos_size: float = Field(default=DEFAULT_RISKGUARD_MAX_POS_SIZE)
    max_concentration: float = Field(default=DEFAULT_RISKGUARD_MAX_CONCENTRATION)


class RiskCheckResult(BaseModel):
    """Standardized result FROM RiskGuard for a risk assessment."""

    approved: bool
    reason: str = ""


# --- Models for AlphaBot Agent ---


class AlphaBotTaskPayload(BaseModel):
    """A unified payload representing a single task for the AlphaBot agent.

    This is sent FROM the Simulator TO AlphaBot.
    """

    historical_prices: List[float]
    current_price: float
    portfolio_state: PortfolioState
    day: int

    # Agent parameters can be grouped for clarity
    short_sma_period: int = Field(default=DEFAULT_ALPHABOT_SHORT_SMA)
    long_sma_period: int = Field(default=DEFAULT_ALPHABOT_LONG_SMA)
    trade_quantity: int = Field(default=DEFAULT_ALPHABOT_TRADE_QTY)

    # Parameters to be passed through to the RiskGuard tool
    riskguard_url: str = Field(default=DEFAULT_RISKGUARD_URL)
    max_pos_size: float = Field(default=DEFAULT_RISKGUARD_MAX_POS_SIZE)
    max_concentration: float = Field(default=DEFAULT_RISKGUARD_MAX_CONCENTRATION)


class TradeStatus(str, Enum):
    """Enum for the status of a trade decision from AlphaBot."""

    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    NO_ACTION = "NO_ACTION"  # No signal or trade proposed
    ERROR = "ERROR"


class TradeOutcome(BaseModel):
    """Standardized response FROM the AlphaBot agent back to the calling client (Simulator)."""

    status: TradeStatus
    reason: str
    trade_proposal: Optional[TradeProposal] = None
