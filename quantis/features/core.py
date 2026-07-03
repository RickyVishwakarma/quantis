"""Point-in-time feature computation (MVP feature set).

Contract (the TDD's look-ahead-bias prevention rule): every feature value
at row ``t`` uses ONLY data up to and including the close of ``t``.
Strategies decide at close ``t``; the engines execute at open ``t+1``.
``tests/test_no_lookahead.py`` asserts this contract mechanically.

All features are wide frames (index=date, columns=symbol).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class FeaturePanel:
    close: pd.DataFrame
    open: pd.DataFrame
    features: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __getitem__(self, name: str) -> pd.DataFrame:
        return self.features[name]

    def names(self) -> list[str]:
        return sorted(self.features)


def _rsi(close: pd.DataFrame, window: int) -> pd.DataFrame:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _atr(high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, window: int) -> pd.DataFrame:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        keys=["hl", "hc", "lc"],
    ).groupby(level=1).max()
    return tr.ewm(alpha=1 / window, adjust=False).mean()


def compute_features(wide: dict[str, pd.DataFrame]) -> FeaturePanel:
    """Compute the MVP feature set from wide OHLCV frames."""
    close, high, low = wide["close"], wide["high"], wide["low"]
    volume = wide["volume"]

    f: dict[str, pd.DataFrame] = {}

    # Returns & momentum
    f["ret_1d"] = close.pct_change(1)
    f["ret_5d"] = close.pct_change(5)
    f["ret_21d"] = close.pct_change(21)
    f["mom_63"] = close.pct_change(63)
    # 12-1 momentum: 126d lookback skipping most recent 21d (reversal zone)
    f["mom_126_21"] = close.shift(21).pct_change(105)

    # Trend
    f["sma_20"] = close.rolling(20).mean()
    f["sma_50"] = close.rolling(50).mean()
    f["sma_200"] = close.rolling(200).mean()
    f["px_vs_sma_200"] = close / f["sma_200"] - 1

    # Oscillators / mean reversion
    f["rsi_14"] = _rsi(close, 14)
    roll_mean = close.rolling(20).mean()
    roll_std = close.rolling(20).std()
    f["zscore_20"] = (close - roll_mean) / roll_std

    # Volatility
    f["vol_21"] = f["ret_1d"].rolling(21).std() * np.sqrt(252)
    f["atr_14"] = _atr(high, low, close, 14)
    f["atr_pct"] = f["atr_14"] / close

    # Liquidity — rupee average daily volume, the risk engine's ADV input
    f["adv_20"] = (close * volume).rolling(20).mean()

    return FeaturePanel(close=close, open=wide["open"], features=f)
