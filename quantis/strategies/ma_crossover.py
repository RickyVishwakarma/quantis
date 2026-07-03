"""Trend following: long names whose fast SMA is above slow SMA, equal weight."""

from __future__ import annotations

import pandas as pd

from . import register
from ..features import FeaturePanel
from .base import Strategy


@register
class MACrossover(Strategy):
    name = "ma_crossover"
    default_params = {"fast": 20, "slow": 50, "max_positions": 15}

    def target_weights(self, panel: FeaturePanel) -> pd.DataFrame:
        fast = panel.close.rolling(self.params["fast"]).mean()
        slow = panel.close.rolling(self.params["slow"]).mean()
        long_mask = (fast > slow) & slow.notna()

        n_long = long_mask.sum(axis=1).clip(lower=1)
        cap = self.params["max_positions"]
        # Equal weight across signals, never concentrating past 1/cap each
        per_name = (1.0 / n_long.clip(lower=cap)).where(n_long > 0, 0.0)
        weights = long_mask.astype(float).mul(per_name, axis=0)
        return weights.fillna(0.0)
