"""Cross-sectional momentum: rank by 12-1 momentum, hold top N, monthly rebalance."""

from __future__ import annotations

import pandas as pd

from . import register
from ..features import FeaturePanel
from .base import Strategy


@register
class CrossSectionalMomentum(Strategy):
    name = "momentum"
    default_params = {"top_n": 10, "rebalance_days": 21, "vol_filter": 0.60}

    def target_weights(self, panel: FeaturePanel) -> pd.DataFrame:
        mom = panel["mom_126_21"]
        vol = panel["vol_21"]
        top_n = self.params["top_n"]
        reb = self.params["rebalance_days"]

        # Exclude names in extreme-volatility blowups from the ranking
        eligible = mom.where(vol < self.params["vol_filter"])

        weights = pd.DataFrame(0.0, index=mom.index, columns=mom.columns)
        current = pd.Series(0.0, index=mom.columns)
        for i, dt in enumerate(mom.index):
            if i % reb == 0:
                ranked = eligible.loc[dt].dropna().nlargest(top_n)
                current = pd.Series(0.0, index=mom.columns)
                if len(ranked) > 0:
                    current[ranked.index] = 1.0 / top_n
            weights.loc[dt] = current
        return weights
