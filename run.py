#!/usr/bin/env python3
"""
run.py  —  Pairs Trading Statistical Arbitrage Engine
=====================================================
Single entry-point.  Usage:

    python run.py                               # best Nifty pair
    python run.py --universe forex              # best Forex pair
    python run.py --pair TCS INFY               # specific Nifty pair
    python run.py --universe forex --pair EURUSD GBPUSD   # specific Forex pair
"""

import argparse
import sys
import os

# Ensure project root is on the path so that package imports work
# regardless of where the script is invoked from.
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Ensure Matplotlib and font caches use writable directories inside the project.
MPL_DIR = os.path.join(ROOT, ".mplconfig")
CACHE_DIR = os.path.join(ROOT, ".cache")
os.makedirs(MPL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", MPL_DIR)
os.environ.setdefault("XDG_CACHE_HOME", CACHE_DIR)

from data.universe import get_tickers, download_prices, scan_pairs, pretty_ticker
from signals.signals import compute_spread, compute_zscore, generate_signals, plot_signals
from backtest.backtest import run_backtest, compute_metrics, plot_equity


# ──────────────────────────────────────────────────────────
# CLI argument parser
# ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Pairs Trading Statistical Arbitrage Engine",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--universe",
        type=str,
        default="nifty",
        choices=["nifty", "forex"],
        help="Asset universe  (default: nifty)",
    )
    parser.add_argument(
        "--pair",
        nargs=2,
        metavar=("TICKER1", "TICKER2"),
        default=None,
        help="Specific pair to analyse (e.g. --pair TCS INFY)",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────
# Ticker resolution helpers
# ──────────────────────────────────────────────────────────

def resolve_ticker(raw: str, universe: str) -> str:
    """
    Append the Yahoo Finance suffix if the user supplied a bare ticker.
    Examples:  TCS → TCS.NS,   EURUSD → EURUSD=X
    """
    if universe == "nifty" and not raw.endswith(".NS"):
        return raw + ".NS"
    if universe == "forex" and not raw.endswith("=X"):
        return raw + "=X"
    return raw


# ──────────────────────────────────────────────────────────
# Separator util
# ──────────────────────────────────────────────────────────

LINE = "─" * 60


def banner(text: str):
    print(f"\n{LINE}")
    print(f"  {text}")
    print(LINE)


# ──────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────

def main():
    args = parse_args()
    universe = args.universe
    output_dir = os.path.join(ROOT, "output")

    banner(f"PAIRS TRADING ENGINE  │  Universe: {universe.upper()}")

    # ── 1.  Download prices ──
    tickers = get_tickers(universe)

    # If a specific pair is given, make sure the two tickers are included
    if args.pair:
        t1_raw, t2_raw = args.pair
        t1 = resolve_ticker(t1_raw, universe)
        t2 = resolve_ticker(t2_raw, universe)
        # Add them if missing (user might specify tickers outside the default list)
        for t in [t1, t2]:
            if t not in tickers:
                tickers.append(t)
    else:
        t1, t2 = None, None

    prices = download_prices(tickers, universe, output_dir=output_dir)

    # ── 2.  Cointegration scan or specific pair ──
    if t1 and t2:
        # Validate that both tickers are present in downloaded prices
        missing = [t for t in (t1, t2) if t not in prices.columns]
        if missing:
            print(f"\n✗  Ticker(s) not found in downloaded data: {missing}")
            print(f"   Available: {list(prices.columns)}")
            sys.exit(1)

        banner(f"PAIR: {pretty_ticker(t1)} / {pretty_ticker(t2)}")

        # Compute hedge ratio on the fly for user-specified pair
        import numpy as np
        s1, s2 = prices[t1].dropna(), prices[t2].dropna()
        common = s1.index.intersection(s2.index)
        s1, s2 = s1.loc[common], s2.loc[common]
        hedge_ratio = np.polyfit(s2.values, s1.values, 1)[0]
        print(f"  Hedge ratio (OLS): {hedge_ratio:.6f}")

    else:
        pairs_df = scan_pairs(prices, universe, output_dir=output_dir)

        if pairs_df.empty:
            print("\n✗  No cointegrated pairs found.  Try loosening filters.")
            sys.exit(1)

        # Best pair = lowest cointegration p-value
        best = pairs_df.iloc[0]
        t1, t2 = best["ticker_1"], best["ticker_2"]
        hedge_ratio = best["hedge_ratio"]

        banner(f"BEST PAIR: {pretty_ticker(t1)} / {pretty_ticker(t2)}")
        print(f"  Coint p-value : {best['coint_pvalue']:.6f}")
        print(f"  Hedge ratio   : {hedge_ratio:.6f}")
        print(f"  Half-life     : {best['half_life_days']:.1f} days")
        print(f"  Spread μ      : {best['spread_mean']:.4f}")
        print(f"  Spread σ      : {best['spread_std']:.4f}")

    # ── 3.  Signals ──
    banner("SIGNAL GENERATION")
    spread = compute_spread(prices, t1, t2, hedge_ratio)
    zscore = compute_zscore(spread, window=30)
    position = generate_signals(zscore, entry_z=2.0, exit_z=0.5)

    plot_signals(spread, zscore, position, t1, t2, hedge_ratio, output_dir=output_dir)

    # ── 4.  Backtest ──
    banner("BACKTEST")
    bt = run_backtest(prices, t1, t2, hedge_ratio, position, universe=universe)
    metrics = compute_metrics(bt, universe=universe)

    for k, v in metrics.items():
        print(f"  {k:<28s}: {v}")

    plot_equity(bt, t1, t2, universe=universe, output_dir=output_dir)

    banner("DONE  ✓")
    print(f"  Charts saved to:  {output_dir}/")
    print()


if __name__ == "__main__":
    main()
