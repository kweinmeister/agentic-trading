"""FastAPI application for the Trading Simulator UI.

Provides a web interface to configure and run trading simulations using
AlphaBot (via A2A) for trade signals and RiskGuard (via A2A within AlphaBot)
for risk checks. Displays results including portfolio value, trades,
and performance metrics using Plotly charts.
"""

import locale
import logging
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
import plotly.graph_objects as go

# A2A SDK Imports
from a2a.client import (
    A2ACardResolver,
    A2AClientHTTPError,
    A2AClientJSONError,
    ClientConfig,
    ClientFactory,
)
from a2a.client.errors import A2AClientInvalidStateError
from a2a.types import (
    DataPart,
    Part,
    Role,
)
from a2a.types import (
    Message as A2AMessage,
)
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from plotly.subplots import make_subplots
from pydantic import BaseModel, Field, ValidationError

# Common project imports
import common.config as defaults
from common.config import (
    DEFAULT_SIMULATOR_PORT,
)  # Keep this for the uvicorn runner at the bottom
from common.models import (
    AlphaBotTaskPayload,
    TradeOutcome,
    TradeStatus,
)
from common.models import (
    PortfolioState as CommonPortfolioState,
)
from common.utils.indicators import calculate_sma

from .market import MarketDataSimulator
from .portfolio import PortfolioState, TradeAction

SIMULATOR_UI_LOGGER = "SimulatorUI"
SIMULATOR_LOGIC_LOGGER = "SimulatorLogic"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(SIMULATOR_UI_LOGGER)

module_dir = Path(__file__).parent
templates_dir = module_dir / "templates"
static_dir = module_dir / "static"


def format_currency(value: Optional[float]) -> str:
    """Format an optional float value as currency, using locale settings if possible."""
    if value is None:
        return "N/A"
    try:
        return locale.currency(value, grouping=True)
    except ValueError:
        logger.debug("Locale formatting failed in format_currency, using fallback.")
        return f"${value:,.2f}"
    except Exception as e:
        logger.warning(
            f"Unexpected error during locale formatting: {e}. Using fallback.",
        )
        return f"${value:,.2f}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events, including locale setting."""
    logger.info("Simulator UI starting up...")
    locale_setting = "en_US.UTF-8"
    try:
        locale.setlocale(locale.LC_ALL, locale_setting)
        logger.info(f"Default locale set to: {locale.getlocale(locale.LC_ALL)}")
    except locale.Error as e:
        logger.warning(
            f"Could not set default locale ('{locale_setting}') at startup: {e}. "
            "Check system locale settings. Using fallback currency formatting.",
        )
    except Exception as e:
        logger.warning(
            f"Unexpected error setting locale ('{locale_setting}') at startup: {e}. Using fallback formatting.",
        )
    yield
    logger.info("Simulator UI shutting down...")


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory=str(templates_dir))
templates.env.filters["format_currency"] = format_currency
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


def _create_results_figure(
    results_df: pd.DataFrame,
    params: Dict[str, Any],
    trade_markers: Dict[str, List],
) -> go.Figure:
    """Create the Plotly figure for simulation results."""
    MARKER_SIZE = 10
    MARKER_LINE_WIDTH = 1
    APPROVED_COLOR = "lime"
    REJECTED_COLOR = "red"
    MARKER_LINE_COLOR = "black"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        specs=[[{"secondary_y": False}], [{"secondary_y": True}]],
    )

    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["Price"],
            name="Price",
            line=dict(color="skyblue"),
            legendgroup="price",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["SMA_Short"],
            name=f"SMA({params['alphabot_short_sma']})",
            line=dict(color="orange", dash="dot"),
            legendgroup="price",
            legendrank=2,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["SMA_Long"],
            name=f"SMA({params['alphabot_long_sma']})",
            line=dict(color="lightcoral", dash="dash"),
            legendgroup="price",
            legendrank=3,
        ),
        row=1,
        col=1,
    )

    if trade_markers["approved_buy_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["approved_buy_days"],
                y=trade_markers["approved_buy_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    color=APPROVED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Approved Buy",
                legendgroup="outcome",
                legendrank=4,
            ),
            row=1,
            col=1,
        )
    if trade_markers["rejected_buy_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["rejected_buy_days"],
                y=trade_markers["rejected_buy_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-up",
                    color=REJECTED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Rejected Buy",
                legendgroup="outcome",
                legendrank=5,
            ),
            row=1,
            col=1,
        )
    if trade_markers["approved_sell_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["approved_sell_days"],
                y=trade_markers["approved_sell_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    color=APPROVED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Approved Sell",
                legendgroup="outcome",
                legendrank=6,
            ),
            row=1,
            col=1,
        )
    if trade_markers["rejected_sell_days"]:
        fig.add_trace(
            go.Scatter(
                x=trade_markers["rejected_sell_days"],
                y=trade_markers["rejected_sell_prices"],
                mode="markers",
                marker=dict(
                    symbol="triangle-down",
                    color=REJECTED_COLOR,
                    size=MARKER_SIZE,
                    line=dict(color=MARKER_LINE_COLOR, width=MARKER_LINE_WIDTH),
                ),
                name="Rejected Sell",
                legendgroup="outcome",
                legendrank=7,
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["TotalValue"],
            name="Total Value",
            line=dict(color="green"),
            legendgroup="portfolio",
            legendrank=8,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["Cash"],
            name="Cash",
            line=dict(color="lightgreen", dash="dash"),
            legendgroup="portfolio",
            legendrank=9,
        ),
        row=2,
        col=1,
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=results_df.index,
            y=results_df["Shares"],
            name="Shares Held",
            line=dict(color="purple", dash="dot"),
            legendgroup="portfolio",
            legendrank=10,
        ),
        row=2,
        col=1,
        secondary_y=True,
    )

    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            traceorder="grouped+reversed",
        ),
        autosize=True,
    )
    fig.update_yaxes(title_text="Price ($)", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Value ($)", row=2, col=1, secondary_y=False)
    fig.update_yaxes(
        title_text="Shares",
        row=2,
        col=1,
        secondary_y=True,
        showgrid=False,
    )

    return fig


class UILogHandler(logging.Handler):
    """Custom log handler to capture logs into a provided list for UI display."""

    def __init__(self, log_list: List[str]):
        """Initialize the UILogHandler."""
        super().__init__()
        self._log_list = log_list

    def emit(self, record):
        """Format the record and append it to the log list."""
        log_entry = self.format(record)
        self._log_list.append(log_entry)


async def _call_alphabot_a2a(
    client_factory: ClientFactory,
    alphabot_url: str,
    session_id: str,
    day: int,
    current_price: float,
    historical_prices: List[float],
    portfolio: PortfolioState,
    params: Dict[str, Any],
    sim_logger: logging.Logger,
) -> Dict[str, Any]:
    """Prepare and send a message to the AlphaBot A2A server for a given simulation day.

    Args:
        client_factory: The A2A ClientFactory instance.
        alphabot_url: The base URL for the AlphaBot service.
        session_id: The simulation session ID (used as contextId).
        day: The current simulation day.
        current_price: The current market price.
        historical_prices: List of historical prices.
        portfolio: The current PortfolioState object.
        params: Dictionary containing simulation and agent parameters.
        sim_logger: Logger instance for simulation logic.

    Returns:
        A dictionary containing the outcome.

    """
    # 1. Create a single, unified payload object.
    payload = AlphaBotTaskPayload(
        historical_prices=historical_prices,
        current_price=current_price,
        portfolio_state=CommonPortfolioState(
            cash=portfolio.cash,
            shares=portfolio.shares,
            total_value=portfolio.total_value,
        ),
        day=day,
        short_sma_period=params["alphabot_short_sma"],
        long_sma_period=params["alphabot_long_sma"],
        trade_quantity=params["alphabot_trade_qty"],
        riskguard_url=params["riskguard_url"],
        max_pos_size=params["riskguard_max_pos_size"],
        max_concentration=params["riskguard_max_concentration"] / 100.0,
    )

    # 2. Use the new helper to create the A2A Request
    # This is now handled by the A2A SDK client

    sim_logger.info(f"--- Calling AlphaBot A2A Server (Session ID: {session_id}) ---")
    outcome: Dict[str, Any] = {
        "approved_trade": None,
        "rejected_trade": None,
        "reason": None,
        "error": None,
    }

    # Access the httpx_client from the factory's config to create the A2ACardResolver.
    # The ClientFactory doesn't provide a public method to access the httpx_client directly.
    httpx_client = client_factory._config.httpx_client
    if not httpx_client:
        raise ConnectionError("HTTPX client not configured in ClientFactory.")

    card_resolver = A2ACardResolver(httpx_client=httpx_client, base_url=alphabot_url)

    try:
        agent_card = await card_resolver.get_agent_card()
        a2a_sdk_client = client_factory.create(agent_card)

        # The new client returns a stream. We iterate and process the events.
        message_to_send = A2AMessage(
            message_id=f"msg-{uuid.uuid4().hex[:8]}",
            role=Role.user,
            parts=[
                Part(root=DataPart(data=payload.model_dump(mode="json"))),
            ],
        )
        async for event in a2a_sdk_client.send_message(message_to_send):
            if isinstance(event, A2AMessage):
                # This is the final response message
                if event.parts and isinstance(event.parts[0].root, DataPart):
                    part_data = event.parts[0].root.data
                    outcome_model = TradeOutcome.model_validate(part_data)

                    if outcome_model.status == TradeStatus.APPROVED:
                        if outcome_model.trade_proposal:
                            outcome["approved_trade"] = (
                                outcome_model.trade_proposal.model_dump()
                            )
                        sim_logger.info(
                            f"    >>> Approved Trade: {outcome.get('approved_trade')} (Reason: {outcome_model.reason})",
                        )
                    elif outcome_model.status == TradeStatus.REJECTED:
                        if outcome_model.trade_proposal:
                            outcome["rejected_trade"] = (
                                outcome_model.trade_proposal.model_dump()
                            )
                        sim_logger.info(
                            f"    >>> Rejected Trade: {outcome.get('rejected_trade')} (Reason: {outcome_model.reason})",
                        )
                    elif outcome_model.status == TradeStatus.NO_ACTION:
                        sim_logger.info(
                            f"  >> Received info message from AlphaBot: {outcome_model.reason}",
                        )
                    outcome["reason"] = outcome_model.reason
                else:
                    sim_logger.warning(
                        "AlphaBot response lacked expected DataPart with TradeOutcome.",
                    )
                    outcome["error"] = "AlphaBot Response Format Issue or No Decision"
                break  # We have the final message, so we can exit the loop.
            if isinstance(event, tuple):  # It's a ClientEvent (Task, Update)
                task, _ = event
                sim_logger.debug(
                    f"Received task update for {task.id}: {task.status.state}",
                )

    except A2AClientInvalidStateError as e:
        sim_logger.error(f"A2A SDK Invalid State Error: {e}", exc_info=True)
        outcome["error"] = f"A2A SDK Error: {e}"
    except A2AClientHTTPError as http_err:
        sim_logger.error(
            f"A2A HTTP Error to AlphaBot: {http_err.status_code} - {http_err.message}",
        )
        outcome["error"] = (
            f"AlphaBot Connection/HTTP Error: {http_err.status_code} - {http_err.message}"
        )
        raise ConnectionError(
            f"AlphaBot A2A HTTP Error: {http_err.message}",
        ) from http_err
    except A2AClientJSONError as json_err:
        sim_logger.error(f"A2A JSON Error from AlphaBot: {json_err.message}")
        outcome["error"] = f"AlphaBot JSON Response Error: {json_err.message}"
    except Exception as e:
        sim_logger.error(f"General A2A Client/Processing Error: {e}", exc_info=True)
        outcome["error"] = f"A2A Processing Error: {e}"
    return outcome


async def run_simulation_async(params: Dict[str, Any]) -> Dict[str, Any]:
    """Run the trading simulation and collects results. Returns dict with results or error."""
    logger.info(f"--- Starting Simulation with params: {params} ---")
    sim_log_list: List[str] = []
    ui_log_handler = UILogHandler(sim_log_list)
    formatter = logging.Formatter("%(levelname)s:%(name)s: %(message)s")
    ui_log_handler.setFormatter(formatter)

    sim_logger = logging.getLogger(SIMULATOR_LOGIC_LOGGER)
    if not any(isinstance(h, UILogHandler) for h in sim_logger.handlers):
        sim_logger.addHandler(ui_log_handler)
    sim_logger.setLevel(logging.INFO)
    sim_logger.propagate = False

    daily_results = []
    signals = []
    approved_buy_days, approved_buy_prices = [], []
    rejected_buy_days, rejected_buy_prices = [], []
    approved_sell_days, approved_sell_prices = [], []
    rejected_sell_days, rejected_sell_prices = [], []

    try:
        sim_logger.info("Initializing simulation components...")
        portfolio = PortfolioState(cash=params["sim_initial_cash"])
        market_sim = MarketDataSimulator(
            initial_price=params["sim_initial_price"],
            volatility=params["sim_volatility"],
            trend=params["sim_trend"],
            history_size=params["alphabot_long_sma"]
            + 20,  # Ensure enough history for longest SMA
        )

        alphabot_url = params.get(
            "alphabot_url",
            os.environ.get("ALPHABOT_SERVICE_URL", defaults.DEFAULT_ALPHABOT_URL),
        ).rstrip("/")  # Ensure no trailing slash for A2AClient
        sim_logger.info(f"Using AlphaBot Service URL: {alphabot_url}")

        # The A2AClient needs an httpx.AsyncClient. Manage its lifecycle.
        async with httpx.AsyncClient() as http_client:
            client_factory = ClientFactory(
                config=ClientConfig(httpx_client=http_client),
            )

            a2a_session_id = f"sim-session-{uuid.uuid4().hex[:8]}"
            sim_logger.info(f"Using A2A Session ID (contextId): {a2a_session_id}")

            initial_portfolio_str = f"Initial Portfolio: {portfolio}"
            sim_logger.info(initial_portfolio_str)
            signals.append({"day": 0, "log": initial_portfolio_str})

            total_days = params["sim_days"]
            sim_logger.info(f"Starting simulation loop for {total_days} days...")

            for day in range(1, total_days + 1):
                sim_logger.info(f"===== Day {day} =====")
                current_price = market_sim.next_price()
                historical_prices = market_sim.get_historical_prices()
                sim_logger.info(
                    f"Market Data: Price = {format_currency(current_price)}",
                )

                sma_short = calculate_sma(
                    historical_prices,
                    params["alphabot_short_sma"],
                )
                sma_long = calculate_sma(historical_prices, params["alphabot_long_sma"])

                portfolio.update_valuation(current_price)
                sim_logger.info(f"Portfolio (Start Day {day}): {portfolio}")

                daily_results.append(
                    {
                        "Day": day,
                        "Price": current_price,
                        "SMA_Short": sma_short,
                        "SMA_Long": sma_long,
                        "Cash": portfolio.cash,
                        "Shares": portfolio.shares,
                        "HoldingsValue": portfolio.holdings_value,
                        "TotalValue": portfolio.total_value,
                    },
                )

                a2a_outcome = await _call_alphabot_a2a(
                    client_factory=client_factory,
                    alphabot_url=alphabot_url,
                    session_id=a2a_session_id,
                    day=day,
                    current_price=current_price,
                    historical_prices=historical_prices,
                    portfolio=portfolio,
                    params=params,
                    sim_logger=sim_logger,
                )

                approved_trade = a2a_outcome["approved_trade"]
                rejected_trade = a2a_outcome["rejected_trade"]
                reason_text = a2a_outcome["reason"]
                a2a_error = a2a_outcome["error"]

                trade_details = approved_trade or rejected_trade
                is_approved = approved_trade is not None
                action = trade_details.get("action") if trade_details else None

                signal_log_entry = {
                    "day": day,
                    "log": f"Price={format_currency(current_price)}",
                }

                if a2a_error:
                    signal_log_entry["log"] += f" | A2A ERROR: {a2a_error}"
                    sim_logger.error(f"A2A Error on day {day}: {a2a_error}")
                elif trade_details:
                    qty = trade_details.get("quantity")
                    price = trade_details.get("price")
                    ticker = trade_details.get("ticker", "N/A")
                    status = "Approved" if is_approved else "Rejected"
                    reason_from_outcome = reason_text or (
                        "OK" if is_approved else "Reason not captured."
                    )
                    signal_log_entry["log"] += (
                        f" | {action} {qty} {ticker} @ {format_currency(price)} | {status}: {reason_from_outcome}"
                    )

                    if action == "BUY":
                        if is_approved:
                            approved_buy_days.append(day)
                            approved_buy_prices.append(current_price)
                        else:
                            rejected_buy_days.append(day)
                            rejected_buy_prices.append(current_price)
                    elif action == "SELL":
                        if is_approved:
                            approved_sell_days.append(day)
                            approved_sell_prices.append(current_price)
                        else:
                            rejected_sell_days.append(day)
                            rejected_sell_prices.append(current_price)

                    if is_approved:
                        sim_logger.info(
                            f"--- Executing Approved Trade: {action} {qty} @ {price} ---",
                        )
                        exec_action = trade_details.get("action")
                        exec_qty = trade_details.get("quantity")
                        exec_price = trade_details.get("price")

                        if (
                            exec_action
                            and exec_qty is not None
                            and exec_price is not None
                        ):
                            trade_action_enum: Optional[TradeAction] = None
                            if exec_action.upper() == "BUY":
                                trade_action_enum = TradeAction.BUY
                            elif exec_action.upper() == "SELL":
                                trade_action_enum = TradeAction.SELL

                            if trade_action_enum:
                                trade_executed = portfolio.execute_trade(
                                    action=trade_action_enum,
                                    quantity=exec_qty,
                                    price=exec_price,
                                )
                                if trade_executed:
                                    portfolio.update_valuation(current_price)
                                    sim_logger.info(
                                        f"Portfolio (Post-Trade Day {day}): {portfolio}",
                                    )
                                    signal_log_entry["log"] += " | Executed."
                                else:
                                    sim_logger.warning(
                                        "--- Trade Execution FAILED (Insufficient funds/shares?) ---",
                                    )
                                    signal_log_entry["log"] += " | Execution FAILED."
                            else:
                                sim_logger.error(
                                    f"--- Trade Execution SKIPPED - Unknown action '{exec_action}' ---",
                                )
                                signal_log_entry["log"] += (
                                    f" | Execution SKIPPED (Unknown Action: {exec_action})."
                                )
                        else:
                            sim_logger.error(
                                f"--- Trade Execution SKIPPED - Missing details: {trade_details} ---",
                            )
                            signal_log_entry["log"] += (
                                " | Execution SKIPPED (Missing Data)."
                            )
                        sim_logger.info(
                            f"Portfolio (After {action} attempt Day {day}): {portfolio}",
                        )
                    else:  # Trade was rejected
                        sim_logger.info(
                            f"--- Trade Rejected: {action} {qty} @ {price} (Reason: {reason_from_outcome}) ---",
                        )
                        sim_logger.info(
                            f"Portfolio (Rejected Trade Day {day}): {portfolio}",
                        )
                else:  # No trade_details and no A2A error means no trade was proposed
                    signal_log_entry["log"] += (
                        f" | No trade proposed by AlphaBot. Reason: {reason_text or 'N/A'}"
                    )
                    sim_logger.info(f"Portfolio (No Trade Day {day}): {portfolio}")

                signals.append(signal_log_entry)
        # End of httpx.AsyncClient context manager

        sim_logger.info("--- Simulation End ---")
        sim_logger.info(f"Final Portfolio: {portfolio}")
        signals.append({"day": total_days + 1, "log": f"Final Portfolio: {portfolio}"})

        results_df = (
            pd.DataFrame(daily_results).set_index("Day")
            if daily_results
            else pd.DataFrame(
                columns=[
                    "Day",
                    "Price",
                    "SMA_Short",
                    "SMA_Long",
                    "Cash",
                    "Shares",
                    "HoldingsValue",
                    "TotalValue",
                ],
            ).set_index("Day")
        )
        trade_markers = {
            "approved_buy_days": approved_buy_days,
            "approved_buy_prices": approved_buy_prices,
            "rejected_buy_days": rejected_buy_days,
            "rejected_buy_prices": rejected_buy_prices,
            "approved_sell_days": approved_sell_days,
            "approved_sell_prices": approved_sell_prices,
            "rejected_sell_days": rejected_sell_days,
            "rejected_sell_prices": rejected_sell_prices,
        }
        fig = _create_results_figure(results_df, params, trade_markers)
        charts = {"combined_chart_json": fig.to_json()}
        signals_log = "\n".join([f"Day {s['day']}: {s['log']}" for s in signals])
        detailed_log = "\n".join(sim_log_list)

        return {
            "success": True,
            "final_portfolio": portfolio.__dict__,
            "charts": charts,
            "signals_log": signals_log,
            "detailed_log": detailed_log,
        }

    except (ConnectionError, httpx.ConnectError) as ce:
        error_msg = f"Connection Error: {ce}. Ensure AlphaBot A2A server is running and accessible."
        logger.error(error_msg)
        sim_logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "detailed_log": "\n".join(sim_log_list),
        }
    except (A2AClientHTTPError, A2AClientJSONError) as a2a_err:
        error_msg = f"A2A Client Error: {a2a_err}. Check the AlphaBot server logs for more details."
        logger.error(error_msg, exc_info=True)
        sim_logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "detailed_log": "\n".join(sim_log_list),
        }
    except Exception as e:  # General fallback for other unexpected errors
        error_msg = f"Unexpected Simulation Error: {e}"
        logger.error(error_msg, exc_info=True)
        sim_logger.error(error_msg, exc_info=True)
        return {
            "success": False,
            "error": error_msg,
            "detailed_log": "\n".join(sim_log_list),
        }
    finally:
        if ui_log_handler in sim_logger.handlers:
            sim_logger.removeHandler(ui_log_handler)
        sim_logger.propagate = True


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request) -> HTMLResponse:
    """Serve the main HTML page, passing simulation status and defaults."""
    logger.info("Serving root page.")
    template_context = {
        "request": request,
        "status": {},
        "params": {},
        "DEFAULT_ALPHABOT_SHORT_SMA": defaults.DEFAULT_ALPHABOT_SHORT_SMA,
        "DEFAULT_ALPHABOT_LONG_SMA": defaults.DEFAULT_ALPHABOT_LONG_SMA,
        "DEFAULT_ALPHABOT_TRADE_QTY": defaults.DEFAULT_ALPHABOT_TRADE_QTY,
        "DEFAULT_ALPHABOT_URL": os.environ.get(
            "ALPHABOT_SERVICE_URL",
            defaults.DEFAULT_ALPHABOT_URL,
        ).rstrip("/"),  # Ensure no trailing slash for UI default
        "DEFAULT_RISKGUARD_URL": os.environ.get(
            "RISKGUARD_SERVICE_URL",
            defaults.DEFAULT_RISKGUARD_URL,
        ).rstrip("/"),  # Ensure no trailing slash
        "DEFAULT_RISKGUARD_MAX_POS_SIZE": defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE,
        "DEFAULT_RISKGUARD_MAX_CONCENTRATION": defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION,
        "DEFAULT_SIM_DAYS": defaults.DEFAULT_SIM_DAYS,
        "DEFAULT_SIM_INITIAL_CASH": defaults.DEFAULT_SIM_INITIAL_CASH,
        "DEFAULT_SIM_INITIAL_PRICE": defaults.DEFAULT_SIM_INITIAL_PRICE,
        "DEFAULT_SIM_VOLATILITY": defaults.DEFAULT_SIM_VOLATILITY,
        "DEFAULT_SIM_TREND": defaults.DEFAULT_SIM_TREND,
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=template_context,
    )


class SimulationRunParams(BaseModel):
    """Parameters for configuring and running a trading simulation, with validation."""

    alphabot_short_sma: int = Field(
        ...,
        gt=0,
        description="Short window for AlphaBot SMA (must be > 0).",
    )
    alphabot_long_sma: int = Field(
        defaults.DEFAULT_ALPHABOT_LONG_SMA,
        gt=0,
        description="Long window for AlphaBot SMA (must be > 0).",
    )
    alphabot_trade_qty: int = Field(
        defaults.DEFAULT_ALPHABOT_TRADE_QTY,
        gt=0,
        description="Quantity per trade for AlphaBot (must be > 0).",
    )
    sim_days: int = Field(
        defaults.DEFAULT_SIM_DAYS,
        gt=0,
        le=10000,
        description="Number of days to simulate (1-10000).",
    )
    sim_initial_cash: float = Field(
        defaults.DEFAULT_SIM_INITIAL_CASH,
        ge=0,
        description="Initial cash for simulation (must be >= 0).",
    )
    sim_initial_price: float = Field(
        defaults.DEFAULT_SIM_INITIAL_PRICE,
        gt=0,
        description="Initial price for market simulation (must be > 0).",
    )
    sim_volatility: float = Field(
        defaults.DEFAULT_SIM_VOLATILITY,
        ge=0,
        le=1.0,
        description="Volatility for market simulation (0.0-1.0).",
    )
    sim_trend: float = Field(
        defaults.DEFAULT_SIM_TREND,
        ge=-0.1,
        le=0.1,
        description="Trend for market simulation (-0.1 to 0.1).",
    )
    riskguard_url: str = Field(
        defaults.DEFAULT_RISKGUARD_URL,
        description="URL for RiskGuard service.",
    )
    riskguard_max_pos_size: float = Field(
        defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE,
        ge=0,
        description="Maximum position size allowed by RiskGuard (must be >= 0).",
    )
    riskguard_max_concentration: int = Field(  # Input as int 0-100 from form
        int(defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION * 100),
        ge=0,
        le=100,
        description="Maximum portfolio concentration (%) allowed by RiskGuard (0-100).",
    )
    alphabot_url: str = Field(
        defaults.DEFAULT_ALPHABOT_URL,
        description="URL for AlphaBot service.",
    )

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation of the model."""
        # Convert concentration back to float 0.0-1.0 for internal use if needed,
        # but the agent/tool might expect it directly as passed by metadata.
        # For now, keep as is, ensure AlphaBot's tool handles the percentage if necessary.
        return self.model_dump()


def _render_error_page(
    request: Request,
    error_message: str,
    form_values: Dict[str, Any],
) -> HTMLResponse:
    """Render the main page with an error message."""
    template_context = {
        "request": request,
        "status": {
            "message": error_message,
            "is_error": True,
        },
        "params": form_values,
        "DEFAULT_ALPHABOT_SHORT_SMA": defaults.DEFAULT_ALPHABOT_SHORT_SMA,
        "DEFAULT_ALPHABOT_LONG_SMA": defaults.DEFAULT_ALPHABOT_LONG_SMA,
        "DEFAULT_ALPHABOT_TRADE_QTY": defaults.DEFAULT_ALPHABOT_TRADE_QTY,
        "DEFAULT_ALPHABOT_URL": os.environ.get(
            "ALPHABOT_SERVICE_URL",
            defaults.DEFAULT_ALPHABOT_URL,
        ).rstrip("/"),
        "DEFAULT_RISKGUARD_URL": os.environ.get(
            "RISKGUARD_SERVICE_URL",
            defaults.DEFAULT_RISKGUARD_URL,
        ).rstrip("/"),
        "DEFAULT_RISKGUARD_MAX_POS_SIZE": defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE,
        "DEFAULT_RISKGUARD_MAX_CONCENTRATION": defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION,
        "DEFAULT_SIM_DAYS": defaults.DEFAULT_SIM_DAYS,
        "DEFAULT_SIM_INITIAL_CASH": defaults.DEFAULT_SIM_INITIAL_CASH,
        "DEFAULT_SIM_INITIAL_PRICE": defaults.DEFAULT_SIM_INITIAL_PRICE,
        "DEFAULT_SIM_VOLATILITY": defaults.DEFAULT_SIM_VOLATILITY,
        "DEFAULT_SIM_TREND": defaults.DEFAULT_SIM_TREND,
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=template_context,
    )


@app.post("/run_simulation", response_class=HTMLResponse)
async def handle_run_simulation(
    request: Request,
    alphabot_short_sma: int = Form(...),
    alphabot_long_sma: int = Form(defaults.DEFAULT_ALPHABOT_LONG_SMA),
    alphabot_trade_qty: int = Form(defaults.DEFAULT_ALPHABOT_TRADE_QTY),
    sim_days: int = Form(defaults.DEFAULT_SIM_DAYS),
    sim_initial_cash: float = Form(defaults.DEFAULT_SIM_INITIAL_CASH),
    sim_initial_price: float = Form(defaults.DEFAULT_SIM_INITIAL_PRICE),
    sim_volatility: float = Form(defaults.DEFAULT_SIM_VOLATILITY),
    sim_trend: float = Form(defaults.DEFAULT_SIM_TREND),
    riskguard_url: str = Form(
        os.environ.get("RISKGUARD_SERVICE_URL", defaults.DEFAULT_RISKGUARD_URL),
    ),  # Get from env or default
    riskguard_max_pos_size: float = Form(defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE),
    riskguard_max_concentration: int = Form(  # Input as int
        int(defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION * 100),
    ),
    alphabot_url: str = Form(
        os.environ.get("ALPHABOT_SERVICE_URL", defaults.DEFAULT_ALPHABOT_URL),
    ),  # Get from env or default
):
    """Handle the simulation run request, validating parameters via Pydantic."""
    # Note: We're removing the check for concurrent simulations as it was causing race conditions
    # with the global simulation_status dictionary. In a production environment, you might want
    # to implement a proper queuing mechanism or allow concurrent simulations with proper isolation.

    form_values = {
        "alphabot_short_sma": alphabot_short_sma,
        "alphabot_long_sma": alphabot_long_sma,
        "alphabot_trade_qty": alphabot_trade_qty,
        "sim_days": sim_days,
        "sim_initial_cash": sim_initial_cash,
        "sim_initial_price": sim_initial_price,
        "sim_volatility": sim_volatility,
        "sim_trend": sim_trend,
        "riskguard_url": riskguard_url,
        "riskguard_max_pos_size": riskguard_max_pos_size,
        "riskguard_max_concentration": riskguard_max_concentration,
        "alphabot_url": alphabot_url,
    }

    try:
        sim_params = SimulationRunParams(
            alphabot_short_sma=alphabot_short_sma,
            alphabot_long_sma=alphabot_long_sma,
            alphabot_trade_qty=alphabot_trade_qty,
            sim_days=sim_days,
            sim_initial_cash=sim_initial_cash,
            sim_initial_price=sim_initial_price,
            sim_volatility=sim_volatility,
            sim_trend=sim_trend,
            riskguard_url=riskguard_url.rstrip("/"),  # Ensure no trailing slash
            riskguard_max_pos_size=riskguard_max_pos_size,
            riskguard_max_concentration=riskguard_max_concentration,
            alphabot_url=alphabot_url.rstrip("/"),  # Ensure no trailing slash
        )
        params_dict = sim_params.to_dict()
    except ValidationError as e:
        logger.error(f"Simulation parameter validation failed: {e}")
        return _render_error_page(
            request,
            f"Invalid simulation parameters: {e}",
            form_values,
        )
    except Exception as e:  # Catch other unexpected errors during param processing
        logger.error(f"Unexpected error processing parameters: {e}", exc_info=True)
        return _render_error_page(
            request,
            f"Error processing parameters: {e}",
            form_values,
        )

    logger.info(f"Received simulation request with validated params: {params_dict}")

    results = await run_simulation_async(params_dict)

    # Render the results directly in the template
    template_context = {
        "request": request,
        "status": {
            "message": "Simulation completed successfully."
            if results.get("success")
            else f"Simulation failed: {results.get('error', 'Unknown error')}",
            "is_error": not results.get("success"),
            "results": results
            if results.get("success")
            else {
                "detailed_log": results.get(
                    "detailed_log",
                    "No detailed log available.",
                ),
            },
        },
        "params": params_dict,
        "DEFAULT_ALPHABOT_SHORT_SMA": defaults.DEFAULT_ALPHABOT_SHORT_SMA,
        "DEFAULT_ALPHABOT_LONG_SMA": defaults.DEFAULT_ALPHABOT_LONG_SMA,
        "DEFAULT_ALPHABOT_TRADE_QTY": defaults.DEFAULT_ALPHABOT_TRADE_QTY,
        "DEFAULT_ALPHABOT_URL": os.environ.get(
            "ALPHABOT_SERVICE_URL",
            defaults.DEFAULT_ALPHABOT_URL,
        ).rstrip("/"),
        "DEFAULT_RISKGUARD_URL": os.environ.get(
            "RISKGUARD_SERVICE_URL",
            defaults.DEFAULT_RISKGUARD_URL,
        ).rstrip("/"),
        "DEFAULT_RISKGUARD_MAX_POS_SIZE": defaults.DEFAULT_RISKGUARD_MAX_POS_SIZE,
        "DEFAULT_RISKGUARD_MAX_CONCENTRATION": defaults.DEFAULT_RISKGUARD_MAX_CONCENTRATION,
        "DEFAULT_SIM_DAYS": defaults.DEFAULT_SIM_DAYS,
        "DEFAULT_SIM_INITIAL_CASH": defaults.DEFAULT_SIM_INITIAL_CASH,
        "DEFAULT_SIM_INITIAL_PRICE": defaults.DEFAULT_SIM_INITIAL_PRICE,
        "DEFAULT_SIM_VOLATILITY": defaults.DEFAULT_SIM_VOLATILITY,
        "DEFAULT_SIM_TREND": defaults.DEFAULT_SIM_TREND,
    }
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=template_context,
    )


@app.get("/health")
async def health_check():
    """Return a simple health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn

    logger.info("--- Starting FastAPI server for Simulator UI ---")
    logger.info("Ensure dependent A2A services are running:")
    logger.info(
        f"  RiskGuard: python -m riskguard --port {defaults.DEFAULT_RISKGUARD_URL.split(':')[-1]}",
    )
    logger.info(
        f"  AlphaBot:  python -m alphabot --port {defaults.DEFAULT_ALPHABOT_URL.split(':')[-1]}",
    )
    logger.info("Required Environment Variables (if not using defaults):")
    logger.info(f"  RISKGUARD_SERVICE_URL (default: {defaults.DEFAULT_RISKGUARD_URL})")
    logger.info(f"  ALPHABOT_SERVICE_URL (default: {defaults.DEFAULT_ALPHABOT_URL})")
    logger.info(
        f"--- Access UI at http://0.0.0.0:{DEFAULT_SIMULATOR_PORT} ---",
    )  # Using imported DEFAULT_SIMULATOR_PORT

    uvicorn.run(
        "simulator.main:app",
        host="0.0.0.0",
        port=DEFAULT_SIMULATOR_PORT,
        reload=True,
    )
