"""Historical data ingestion.

Two sources for the MVP:

- Yahoo Finance (``.NS`` suffix) — free daily OHLCV for NSE equities,
  auto-adjusted for splits/dividends so the lake holds a continuous
  back-adjusted series (the TDD's corporate-actions requirement at MVP
  fidelity). Swappable for a licensed vendor feed later.
- Synthetic generator — regime-switching GBM so the full pipeline runs
  offline and deterministically in CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .universe import symbols as universe_symbols


def fetch_yahoo(symbols: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    import yfinance as yf

    tickers = [f"{s}.NS" for s in symbols]
    raw = yf.download(
        tickers, start=start, end=end, auto_adjust=True,
        group_by="ticker", progress=False, threads=True,
    )
    frames = []
    for sym, ticker in zip(symbols, tickers):
        try:
            df = raw[ticker].dropna(how="all")
        except KeyError:
            continue
        if df.empty:
            continue
        df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]
        df = df.reset_index().rename(columns={"Date": "ts", "index": "ts"})
        df["symbol"] = sym
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize(None)
        frames.append(df)
    if not frames:
        raise RuntimeError("Yahoo Finance returned no data for any symbol")
    return pd.concat(frames, ignore_index=True)


def generate_synthetic(
    symbols: list[str] | None = None,
    start: str = "2018-01-01",
    end: str = "2024-12-31",
    seed: int = 42,
) -> pd.DataFrame:
    """Regime-switching GBM with cross-sectional dispersion and volume."""
    symbols = symbols or universe_symbols()
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end)
    n_days, n_syms = len(dates), len(symbols)

    # Market regimes: calm bull / choppy / stressed bear, persistent via Markov chain
    regimes = np.zeros(n_days, dtype=int)
    trans = np.array([[0.985, 0.010, 0.005],
                      [0.020, 0.970, 0.010],
                      [0.010, 0.030, 0.960]])
    for t in range(1, n_days):
        regimes[t] = rng.choice(3, p=trans[regimes[t - 1]])
    mkt_mu = np.array([0.0008, 0.0000, -0.0015])[regimes]
    mkt_vol = np.array([0.007, 0.012, 0.025])[regimes]
    mkt_ret = rng.normal(mkt_mu, mkt_vol)

    betas = rng.uniform(0.6, 1.4, n_syms)
    alphas = rng.normal(0.0, 0.0002, n_syms)
    idio_vol = rng.uniform(0.008, 0.020, n_syms)
    # Slow-moving idiosyncratic trend gives momentum strategies signal to find
    trend = np.cumsum(rng.normal(0, 0.0004, (n_days, n_syms)), axis=0)

    rets = (alphas + np.outer(mkt_ret, betas)
            + rng.normal(0, 1, (n_days, n_syms)) * idio_vol
            + np.diff(np.vstack([np.zeros(n_syms), trend]), axis=0))
    close = 100 * np.exp(np.cumsum(rets, axis=0)) * rng.uniform(0.5, 30, n_syms)

    frames = []
    for j, sym in enumerate(symbols):
        c = close[:, j]
        o = c * (1 + rng.normal(0, 0.003, n_days))
        h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.004, n_days)))
        l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.004, n_days)))
        base_vol = rng.uniform(2e5, 8e6)
        v = base_vol * np.exp(rng.normal(0, 0.4, n_days)) * (1 + 2 * mkt_vol / 0.02)
        frames.append(pd.DataFrame({
            "symbol": sym, "ts": dates, "open": o, "high": h,
            "low": l, "close": c, "volume": v.astype(np.int64),
        }))
    return pd.concat(frames, ignore_index=True)
