# Pairs Trading — Statistical Arbitrage Engine

A quantitative Python engine that scans a universe of equities (Nifty 50) or
Forex currency pairs, discovers statistically cointegrated pairs using the
**Engle-Granger method**, generates **z-score based entry/exit signals**, and
backtests the strategy with realistic transaction costs.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run with defaults (Nifty universe, auto-select best pair)
python run.py

# 3. Forex universe
python run.py --universe forex

# 4. Specific Nifty pair
python run.py --pair TCS INFY

# 5. Specific Forex pair
python run.py --universe forex --pair EURUSD GBPUSD
```

All charts and CSV outputs are saved to the `output/` directory.  
Re-runs use cached price / pair data — delete the CSVs to force a refresh.

---

## Project Structure

```
pairs_trading/
├── run.py                  ← CLI entry-point
├── requirements.txt
├── README.md
├── data/
│   ├── __init__.py
│   └── universe.py         ← price download, cointegration scanner, ADF, half-life
├── signals/
│   ├── __init__.py
│   └── signals.py          ← spread, z-score, signal generation, 3-panel chart
├── backtest/
│   ├── __init__.py
│   └── backtest.py         ← PnL sim, transaction costs, metrics, equity chart
└── output/                 ← cached CSVs + generated charts
```

---

## The Math

### 1. Cointegration (Engle-Granger Two-Step)

Two price series $P_1$ and $P_2$ are **cointegrated** if there exists a
constant $\beta$ (the hedge ratio) such that the linear combination

$$S_t = P_{1,t} - \beta \cdot P_{2,t}$$

is **stationary** — it mean-reverts rather than drifting.

**Step 1 — Estimate $\beta$** via OLS regression:

$$P_{1,t} = \beta \cdot P_{2,t} + \alpha + \varepsilon_t$$

We use `numpy.polyfit(P_2, P_1, 1)` to get $\beta$.

**Step 2 — Test the residual $S_t$ for stationarity** using:
- The **Engle-Granger cointegration test** (`statsmodels.tsa.stattools.coint`)
  which returns a p-value.  We require **p < 0.05**.
- The **Augmented Dickey-Fuller (ADF) test** on $S_t$ as a secondary check.

### 2. Half-Life of Mean Reversion

Model the spread as an **Ornstein-Uhlenbeck process**:

$$\Delta S_t = \varphi \cdot S_{t-1} + \varepsilon_t$$

Estimate $\varphi$ via OLS.  The half-life is:

$$\tau_{1/2} = -\frac{\ln 2}{\varphi}$$

We keep only pairs with $5 \leq \tau_{1/2} \leq 60$ days — fast enough to
trade but not so fast that transaction costs dominate.

### 3. Z-Score Trading Signal

Compute a **rolling 30-day z-score** (no lookahead bias):

$$z_t = \frac{S_t - \bar{S}_{[t-29:t]}}{\sigma_{[t-29:t]}}$$

| Condition | Action |
|--|--|
| $z_t < -2.0$ | **Long** spread (buy leg 1, sell β × leg 2) |
| $z_t > +2.0$ | **Short** spread (sell leg 1, buy β × leg 2) |
| $\|z_t\| < 0.5$ | **Exit** (flatten) |
| Otherwise | **Hold** previous position |

### 4. Backtest Mechanics

- **Position-shifted returns**: position at end of day $t-1$ earns the spread
  return on day $t$.  This prevents lookahead bias.
- **Transaction costs**: charged proportionally on every position change.
  - Nifty: 10 bps (covers STT + brokerage + exchange fees)
  - Forex: 2 bps (minimal spread costs for major pairs)

---

## Performance Metrics

| Metric | Formula |
|--|--|
| **CAGR** | $(V_T / V_0)^{1/Y} - 1$ |
| **Sharpe Ratio** | $\sqrt{252}\;\frac{\bar{r}_e}{\sigma_e}$ where $r_e = r - r_f/252$ |
| **Max Drawdown** | $\max_{t}\left(\frac{\text{Peak}_t - V_t}{\text{Peak}_t}\right)$ |
| **Calmar Ratio** | CAGR / \|Max Drawdown\| |
| **Win Rate** | % of non-zero daily returns that are positive |
| **Avg Trade Duration** | Mean length (days) of contiguous non-zero position blocks |

Risk-free rates: **6.5 %** (India 10Y) for Nifty, **5 %** (US 10Y) for Forex.

---

## Known Limitations

1. **Survivorship bias** — uses current index constituents only; delisted or
   removed stocks are not included.
2. **Slippage not modelled** — real execution would face slippage, especially
   on less liquid NSE mid-caps.
3. **NSE short selling** — requires F&O access; not available on all stocks.
   Forex has no such restriction.
4. **Static OLS hedge ratio** — the hedge ratio is estimated once over the
   full sample.  In production, a **Kalman filter** should dynamically update
   $\beta_t$ (planned for Week 4).
5. **Multiple testing problem** — scanning many pairs inflates false discovery
   rate.  Bonferroni or FDR corrections are recommended for production use.
6. **Single regime** — the strategy assumes a constant cointegration
   relationship.  Regime-switching models would improve robustness.

---

## 6-Week Roadmap

| Week | Milestone | Details |
|--|--|--|
| **1** | Core engine ✅ | Cointegration scanner, z-score signals, backtest with metrics |
| **2** | Walk-forward validation | Rolling-window OOS test, expanding-window hedge ratio re-estimation |
| **3** | Portfolio of pairs | Multi-pair portfolio construction, capital allocation, correlation-aware sizing |
| **4** | Kalman filter hedge ratio | Dynamic β estimation via `pykalman`, adaptive entry/exit thresholds |
| **5** | Risk management layer | Volatility targeting, max position limits, stop-loss / time-stop rules |
| **6** | Live paper trading | WebSocket price feed (Alpaca / IBKR), real-time signal engine, tearsheet dashboard |

---

## License

This project is released for **educational and research purposes only**.
It is not financial advice.  Always validate strategies with out-of-sample data
before risking real capital.
