"""Single strategy interface shared by the vectorized and event engines.

A strategy maps a FeaturePanel to a frame of target weights, one row per
decision date. Row ``t`` may only use information through the close of
``t``; both engines execute row ``t`` at the open of ``t+1``. Because the
same weights frame feeds both engines, a strategy cannot behave
differently in research and simulation — the TDD's "one code path"
principle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ..features import FeaturePanel


class Strategy(ABC):
    name: str = "base"
    default_params: dict = {}

    def __init__(self, **params):
        self.params = {**self.default_params, **params}

    @abstractmethod
    def target_weights(self, panel: FeaturePanel) -> pd.DataFrame:
        """Return wide frame of target portfolio weights in [0, 1] per symbol.

        Row t = desired weights decided at close of t. NaNs are treated
        as 0. Long-only in the MVP; gross exposure <= 1 is the strategy's
        job, but the risk engine enforces it regardless.
        """

    def describe(self) -> str:
        kv = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({kv})"
