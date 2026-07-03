"""Mean reversion: enter oversold names (RSI < entry) in uptrends, exit when RSI normalizes."""

from __future__ import annotations

import pandas as pd

from . import register
from ..features import FeaturePanel
from .base import Strategy


@register
class RSIMeanReversion(Strategy):
    name = "mean_reversion"
    default_params = {"entry_rsi": 30.0, "exit_rsi": 55.0, "max_positions": 8}

    def target_weights(self, panel: FeaturePanel) -> pd.DataFrame:
        rsi = panel["rsi_14"]
        trend_ok = panel["px_vs_sma_200"] > 0  # only fade dips in uptrends
        cap = self.params["max_positions"]
        w_each = 1.0 / cap

        weights = pd.DataFrame(0.0, index=rsi.index, columns=rsi.columns)
        holding = pd.Series(False, index=rsi.columns)
        for dt in rsi.index:
            r = rsi.loc[dt]
            up = trend_ok.loc[dt].fillna(False)
            holding &= ~(r > self.params["exit_rsi"]).fillna(False)
            candidates = r[(r < self.params["entry_rsi"]).fillna(False) & up & ~holding]
            room = cap - int(holding.sum())
            if room > 0 and len(candidates) > 0:
                holding[candidates.nsmallest(room).index] = True
            weights.loc[dt, holding[holding].index] = w_each
        return weights
