"""
backtest/backtest.py
====================
PnL simulation with transaction costs, performance metrics, and
equity-curve / drawdown chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data.universe import pretty_ticker


# ──────────────────────────────────────────────────────────
# Configuration helpers
# ──────────────────────────────────────────────────────────

UNIVERSE_CONFIG = {
    "nifty": {
        "capital": 100_000,       # ₹1,00,000
        "currency": "₹",
        "tc_bps": 10,             # 10 bps round-trip
        "risk_free": 0.065,       # 6.5 % (India 10Y)
    },
    "forex": {
        "capital": 100_000,       # $100,000
        "currency": "$",
        "tc_bps": 2,              # 2 bps round-trip
        "risk_free": 0.05,        # 5 % (US 10Y)
    },
}


def _trading_days_per_year() -> int:
    return 252


# ──────────────────────────────────────────────────────────
# Backtest engine
# ──────────────────────────────────────────────────────────

def run_backtest(
    prices: pd.DataFrame,
    ticker_1: str,
    ticker_2: str,
    hedge_ratio: float,
    position: pd.Series,
    universe: str = "nifty",
) -> pd.DataFrame:
    """
    Simulate daily PnL for one pair.

    Strategy return on day t:
        r_t = position_{t-1} × (spread_return_t) − TC × |Δposition_t|

    where spread_return is the daily percentage return of a portfolio
    that is long ticker_1 and short hedge_ratio × ticker_2.

    Returns a DataFrame indexed by date with columns:
        spread_return, position, trade, tc, strategy_return, equity
    """
    cfg = UNIVERSE_CONFIG[universe]
    tc_frac = cfg["tc_bps"] / 10_000
    capital = cfg["capital"]

    # Daily returns of each leg
    r1 = prices[ticker_1].pct_change()
    r2 = prices[ticker_2].pct_change()

    # Spread return (dollar-neutral approximation):
    #   We go long 1 unit of t1 and short β units of t2 (normalised).
    #   Daily P&L ≈ r1 − β·r2   (for unit notional)
    spread_return = r1 - hedge_ratio * r2

    # Align position with returns
    common = spread_return.dropna().index.intersection(position.dropna().index)
    spread_return = spread_return.loc[common]
    pos = position.loc[common]

    # Shift position by 1 day to avoid lookahead
    pos_shifted = pos.shift(1).fillna(0)

    # Transaction costs on position changes
    trade = pos_shifted.diff().fillna(0)
    tc = tc_frac * trade.abs()

    # Strategy returns
    strat_return = pos_shifted * spread_return - tc

    # Equity curve
    equity = capital * (1 + strat_return).cumprod()

    result = pd.DataFrame({
        "spread_return": spread_return,
        "position": pos_shifted,
        "trade": trade,
        "tc": tc,
        "strategy_return": strat_return,
        "equity": equity,
    })

    return result


# ──────────────────────────────────────────────────────────
# Performance metrics
# ──────────────────────────────────────────────────────────

def compute_metrics(bt: pd.DataFrame, universe: str = "nifty") -> dict:
    """
    Compute annualised performance metrics.

    Returns dict with:
        CAGR, Sharpe, Max_Drawdown, Calmar, Win_Rate, Avg_Trade_Duration_Days
    """
    cfg = UNIVERSE_CONFIG[universe]
    rf = cfg["risk_free"]
    N = _trading_days_per_year()
    capital = cfg["capital"]
    currency = cfg["currency"]

    returns = bt["strategy_return"].dropna()
    equity = bt["equity"].dropna()

    if len(returns) < 2:
        return {"error": "Not enough data"}

    # ── CAGR ──
    total_days = (equity.index[-1] - equity.index[0]).days
    total_return = equity.iloc[-1] / capital
    years = total_days / 365.25
    cagr = total_return ** (1 / years) - 1 if years > 0 else 0.0

    # ── Annualised Sharpe Ratio ──
    excess = returns - rf / N
    sharpe = np.sqrt(N) * excess.mean() / excess.std() if excess.std() != 0 else 0.0

    # ── Max Drawdown ──
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = drawdown.min()

    # ── Calmar Ratio ──
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0.0

    # ── Win Rate ──
    trades = returns[returns != 0]
    win_rate = (trades > 0).sum() / len(trades) if len(trades) > 0 else 0.0

    # ── Avg Trade Duration ──
    position = bt["position"]
    in_trade = (position != 0).astype(int)
    # Group consecutive trade blocks
    trade_groups = (in_trade.diff().fillna(0) != 0).cumsum()
    trade_groups = trade_groups[in_trade == 1]
    if len(trade_groups) > 0:
        durations = trade_groups.groupby(trade_groups).count()
        avg_duration = durations.mean()
    else:
        avg_duration = 0.0

    metrics = {
        "Universe": universe.upper(),
        "Currency": currency,
        "Starting_Capital": f"{currency}{capital:,.0f}",
        "Final_Equity": f"{currency}{equity.iloc[-1]:,.0f}",
        "CAGR": f"{cagr:.2%}",
        "Sharpe_Ratio": f"{sharpe:.3f}",
        "Max_Drawdown": f"{max_dd:.2%}",
        "Calmar_Ratio": f"{calmar:.3f}",
        "Win_Rate": f"{win_rate:.2%}",
        "Avg_Trade_Duration_Days": f"{avg_duration:.1f}",
        "Total_Trading_Days": len(returns),
        "Risk_Free_Rate": f"{rf:.1%}",
    }
    return metrics


# ──────────────────────────────────────────────────────────
# Equity curve + drawdown chart
# ──────────────────────────────────────────────────────────

def plot_equity(
    bt: pd.DataFrame,
    ticker_1: str,
    ticker_2: str,
    universe: str = "nifty",
    output_dir: str = "output",
) -> str:
    """
    Plot equity curve (top) and drawdown (bottom).
    Saves to output/ and returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    cfg = UNIVERSE_CONFIG[universe]
    currency = cfg["currency"]

    t1 = pretty_ticker(ticker_1)
    t2 = pretty_ticker(ticker_2)

    equity = bt["equity"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1.2]})
    fig.suptitle(f"Pairs Trade: {t1} / {t2}  —  Equity & Drawdown",
                 fontsize=14, fontweight="bold", y=0.97)

    # ── Equity ──
    ax1.plot(equity.index, equity.values, linewidth=1.0, color="#1976D2")
    ax1.fill_between(equity.index, cfg["capital"], equity.values,
                     where=equity.values >= cfg["capital"],
                     alpha=0.15, color="#4CAF50")
    ax1.fill_between(equity.index, cfg["capital"], equity.values,
                     where=equity.values < cfg["capital"],
                     alpha=0.15, color="#F44336")
    ax1.axhline(cfg["capital"], color="grey", linestyle="--", linewidth=0.6)
    ax1.set_ylabel(f"Equity ({currency})")
    ax1.grid(alpha=0.3)

    # ── Drawdown ──
    ax2.fill_between(drawdown.index, drawdown.values, 0, alpha=0.45, color="#E53935")
    ax2.plot(drawdown.index, drawdown.values, linewidth=0.7, color="#B71C1C")
    ax2.set_ylabel("Drawdown")
    ax2.set_ylim(drawdown.min() * 1.15, 0.02)
    ax2.grid(alpha=0.3)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    fname = f"{t1}_{t2}_equity.png"
    filepath = os.path.join(output_dir, fname)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] Equity chart saved → {filepath}")
    return filepath
