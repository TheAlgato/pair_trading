"""
signals/signals.py
==================
Spread construction, rolling z-score computation, signal generation,
and 3-panel diagnostic chart.
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
# Spread & Z-score
# ──────────────────────────────────────────────────────────

def compute_spread(
    prices: pd.DataFrame,
    ticker_1: str,
    ticker_2: str,
    hedge_ratio: float,
) -> pd.Series:
    """Spread = price_t1  −  hedge_ratio × price_t2"""
    spread = prices[ticker_1] - hedge_ratio * prices[ticker_2]
    spread.name = "spread"
    return spread.dropna()


def compute_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    """
    Rolling z-score (no lookahead bias):
        z_t = (spread_t − μ_window) / σ_window
    Uses only the trailing `window` observations up to and including t.
    """
    rolling_mean = spread.rolling(window=window, min_periods=window).mean()
    rolling_std = spread.rolling(window=window, min_periods=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    zscore.name = "zscore"
    return zscore


# ──────────────────────────────────────────────────────────
# Signal generation
# ──────────────────────────────────────────────────────────

def generate_signals(
    zscore: pd.Series,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.Series:
    """
    Generate trading positions from the z-score series.

    Rules (executed sequentially, no future information):
      * If z < −entry_z  →  position = +1  (long spread: buy t1, sell t2)
      * If z >  entry_z  →  position = −1  (short spread: sell t1, buy t2)
      * If |z| < exit_z  →  position =  0  (exit / flatten)
      * Otherwise         →  hold previous position

    Returns a Series of positions: +1, −1, or 0.
    """
    position = pd.Series(index=zscore.index, data=0.0, name="position")

    current_pos = 0.0
    for i, z in enumerate(zscore.values):
        if np.isnan(z):
            position.iloc[i] = 0.0
            continue

        if z < -entry_z:
            current_pos = 1.0       # long spread
        elif z > entry_z:
            current_pos = -1.0      # short spread
        elif abs(z) < exit_z:
            current_pos = 0.0       # flatten

        position.iloc[i] = current_pos

    return position


# ──────────────────────────────────────────────────────────
# 3-panel chart
# ──────────────────────────────────────────────────────────

def plot_signals(
    spread: pd.Series,
    zscore: pd.Series,
    position: pd.Series,
    ticker_1: str,
    ticker_2: str,
    hedge_ratio: float,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    output_dir: str = "output",
) -> str:
    """
    Plot a 3-panel chart:
      1. Spread with mean
      2. Z-score with ±entry and ±exit threshold bands
      3. Position (+1 / 0 / −1)

    Saves to output/ and returns the file path.
    """
    os.makedirs(output_dir, exist_ok=True)

    t1_label = pretty_ticker(ticker_1)
    t2_label = pretty_ticker(ticker_2)
    title = f"{t1_label} / {t2_label}  (β = {hedge_ratio:.4f})"

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.5]})
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.97)

    # ── Panel 1: Spread ──
    ax1 = axes[0]
    ax1.plot(spread.index, spread.values, linewidth=0.9, color="#2196F3", label="Spread")
    ax1.axhline(spread.mean(), color="#FF9800", linestyle="--", linewidth=0.8, label="Mean")
    ax1.fill_between(spread.index,
                     spread.mean() - spread.std(),
                     spread.mean() + spread.std(),
                     alpha=0.12, color="#2196F3")
    ax1.set_ylabel("Spread")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Panel 2: Z-score ──
    ax2 = axes[1]
    ax2.plot(zscore.index, zscore.values, linewidth=0.9, color="#9C27B0", label="Z-score")
    ax2.axhline(0, color="grey", linewidth=0.5)
    for level, style, colour, lbl in [
        (entry_z,  "--", "#F44336", f"+{entry_z}σ"),
        (-entry_z, "--", "#F44336", f"−{entry_z}σ"),
        (exit_z,   ":",  "#4CAF50", f"+{exit_z}σ"),
        (-exit_z,  ":",  "#4CAF50", f"−{exit_z}σ"),
    ]:
        ax2.axhline(level, color=colour, linestyle=style, linewidth=0.8, label=lbl)
    ax2.fill_between(zscore.index, -entry_z, entry_z, alpha=0.06, color="#E1BEE7")
    ax2.set_ylabel("Z-score")
    ax2.legend(loc="upper left", fontsize=7, ncol=3)
    ax2.grid(alpha=0.3)

    # ── Panel 3: Position ──
    ax3 = axes[2]
    ax3.fill_between(position.index, position.values, step="post", alpha=0.5, color="#00BCD4")
    ax3.step(position.index, position.values, where="post", linewidth=0.8, color="#006064")
    ax3.set_ylabel("Position")
    ax3.set_yticks([-1, 0, 1])
    ax3.set_yticklabels(["Short", "Flat", "Long"])
    ax3.set_ylim(-1.4, 1.4)
    ax3.grid(alpha=0.3)

    # X-axis formatting
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate(rotation=30)

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    fname = f"{t1_label}_{t2_label}_signals.png"
    filepath = os.path.join(output_dir, fname)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[chart] Signal chart saved → {filepath}")
    return filepath
