"""
data/universe.py
================
Universe definitions, price downloading, cointegration scanning,
ADF testing, half-life filtering, and OLS hedge-ratio estimation.
"""

import os
import itertools
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import coint, adfuller
from scipy import stats

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# Universe definitions
# ──────────────────────────────────────────────────────────

NIFTY_TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "BAJFINANCE.NS", "ASIANPAINT.NS", "MARUTI.NS",
    "HCLTECH.NS", "SUNPHARMA.NS", "TITAN.NS", "WIPRO.NS", "ULTRACEMCO.NS",
    "NESTLEIND.NS", "TATAMOTORS.NS", "POWERGRID.NS", "NTPC.NS", "M&M.NS",
    "TECHM.NS", "INDUSINDBK.NS", "ADANIENT.NS", "BAJAJFINSV.NS", "ONGC.NS",
]

FOREX_TICKERS = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X",
    "NZDUSD=X", "USDCHF=X", "EURGBP=X", "EURJPY=X", "GBPJPY=X",
    "AUDJPY=X", "EURAUD=X", "EURCHF=X", "AUDNZD=X", "GBPAUD=X",
    "NZDJPY=X", "CADJPY=X", "GBPCHF=X", "AUDCAD=X", "EURCAD=X",
]


def get_tickers(universe: str) -> list:
    """Return the list of tickers for a given universe."""
    if universe == "nifty":
        return NIFTY_TICKERS
    elif universe == "forex":
        return FOREX_TICKERS
    else:
        raise ValueError(f"Unknown universe: {universe}. Choose 'nifty' or 'forex'.")


# ──────────────────────────────────────────────────────────
# Price download with CSV cache
# ──────────────────────────────────────────────────────────

def download_prices(
    tickers: list,
    universe: str,
    output_dir: str = "output",
    period: str = "2y",
) -> pd.DataFrame:
    """
    Download adjusted close prices via yfinance.
    Caches to output/prices_{universe}.csv so re-runs skip the download.
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, f"prices_{universe}.csv")

    if os.path.exists(cache_path):
        print(f"[cache] Loading prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices

    print(f"[download] Fetching {len(tickers)} tickers for '{universe}' universe …")
    data = yf.download(tickers, period=period, auto_adjust=True, progress=True)

    # yfinance returns multi-level columns when >1 ticker
    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Close"].copy()
    else:
        prices = data[["Close"]].copy()
        prices.columns = tickers

    # Drop columns with >20 % NaN, then forward-fill remaining gaps
    thresh = int(len(prices) * 0.80)
    prices = prices.dropna(axis=1, thresh=thresh)
    prices = prices.ffill().bfill()

    prices.to_csv(cache_path)
    print(f"[cache] Saved prices → {cache_path}  ({prices.shape[1]} tickers, {len(prices)} days)")
    return prices


# ──────────────────────────────────────────────────────────
# Half-life of mean reversion (AR(1) / Ornstein-Uhlenbeck)
# ──────────────────────────────────────────────────────────

def compute_half_life(spread: pd.Series) -> float:
    """
    Compute the half-life of mean reversion via AR(1) regression:
        Δs_t = φ · s_{t-1} + ε_t
    Half-life = -ln(2) / ln(1 + φ)  ≈  -ln(2) / φ  when φ is small.
    """
    spread_lag = spread.shift(1).dropna()
    spread_diff = spread.diff().dropna()
    # Align
    spread_lag = spread_lag.iloc[1:]
    spread_diff = spread_diff.iloc[1:]

    if len(spread_lag) < 30:
        return np.nan

    # OLS: Δs = φ·s_lag + const
    slope, _intercept, _r, _p, _se = stats.linregress(spread_lag, spread_diff)

    if slope >= 0:
        return np.inf  # no mean reversion

    half_life = -np.log(2) / slope
    return half_life


# ──────────────────────────────────────────────────────────
# Cointegration scanner
# ──────────────────────────────────────────────────────────

def scan_pairs(
    prices: pd.DataFrame,
    universe: str,
    output_dir: str = "output",
    coint_pvalue: float = 0.05,
    hl_min: float = 5.0,
    hl_max: float = 60.0,
) -> pd.DataFrame:
    """
    Run pairwise Engle-Granger cointegration tests, filter by ADF and
    half-life, compute OLS hedge ratios, and output a ranked CSV.

    Returns a DataFrame with columns:
        ticker_1, ticker_2, coint_pvalue, hedge_ratio,
        half_life_days, spread_mean, spread_std
    """
    os.makedirs(output_dir, exist_ok=True)
    cache_path = os.path.join(output_dir, f"pairs_{universe}.csv")

    if os.path.exists(cache_path):
        print(f"[cache] Loading pairs from {cache_path}")
        return pd.read_csv(cache_path)

    tickers = list(prices.columns)
    n = len(tickers)
    print(f"[scan] Testing {n * (n - 1) // 2} pairs for cointegration …")

    results = []

    for t1, t2 in itertools.combinations(tickers, 2):
        s1 = prices[t1].dropna()
        s2 = prices[t2].dropna()

        # Align on common index
        common = s1.index.intersection(s2.index)
        if len(common) < 100:
            continue
        s1, s2 = s1.loc[common], s2.loc[common]

        # ── Engle-Granger cointegration test ──
        try:
            score, pvalue, _ = coint(s1, s2)
        except Exception:
            continue

        if pvalue > coint_pvalue:
            continue

        # ── OLS hedge ratio via polyfit (y = β·x + α) ──
        hedge_ratio = np.polyfit(s2.values, s1.values, 1)[0]

        # ── Spread ──
        spread = s1 - hedge_ratio * s2

        # ── ADF test on spread ──
        try:
            adf_stat, adf_p, *_ = adfuller(spread, maxlag=1)
        except Exception:
            continue
        if adf_p > coint_pvalue:
            continue

        # ── Half-life filter ──
        hl = compute_half_life(spread)
        if not (hl_min <= hl <= hl_max):
            continue

        results.append(
            {
                "ticker_1": t1,
                "ticker_2": t2,
                "coint_pvalue": round(pvalue, 6),
                "hedge_ratio": round(hedge_ratio, 6),
                "half_life_days": round(hl, 2),
                "spread_mean": round(spread.mean(), 6),
                "spread_std": round(spread.std(), 6),
            }
        )

    if not results:
        print("[scan] ⚠  No cointegrated pairs found with current filters.")
        return pd.DataFrame(
            columns=[
                "ticker_1", "ticker_2", "coint_pvalue", "hedge_ratio",
                "half_life_days", "spread_mean", "spread_std",
            ]
        )

    pairs_df = (
        pd.DataFrame(results)
        .sort_values("coint_pvalue")
        .reset_index(drop=True)
    )

    pairs_df.to_csv(cache_path, index=False)
    print(f"[scan] Found {len(pairs_df)} cointegrated pairs → {cache_path}")
    return pairs_df


# ──────────────────────────────────────────────────────────
# Helper for run.py: format a ticker for display
# ──────────────────────────────────────────────────────────

def pretty_ticker(ticker: str) -> str:
    """Strip Yahoo suffixes for display."""
    return ticker.replace(".NS", "").replace("=X", "")
