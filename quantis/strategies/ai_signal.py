"""AI-signal strategy: model predictions -> target weights.

This is deliberately just another Strategy plug-in. AI-sourced weights
flow through the identical engines and the identical risk gate as the
template strategies — the TDD's rule that limits apply regardless of
signal source, with two extra AI-specific safeguards implemented here:

  sanity bound     predictions outside the model's training-time signal
                   bounds (0.5th–99.5th percentile) are treated as
                   hallucinations: zeroed and counted, never traded
  per-model cap    ``gross_cap`` limits how much of the book one model
                   can direct (TDD: per-model capital caps)

Explainability: ``explain(date)`` returns the top feature contributions
behind each held name on that date — every AI trade idea ships with
attribution, not just a score.
"""

from __future__ import annotations

import pandas as pd

from . import register
from ..features import FeaturePanel
from .base import Strategy


@register
class AISignalStrategy(Strategy):
    name = "ai_signal"
    default_params = {
        "model_id": "production",     # or explicit id / "production:<name>"
        "top_n": 8,
        "rebalance_days": 5,
        "gross_cap": 0.5,
        "registry_root": "models",
    }

    def __init__(self, **params):
        super().__init__(**params)
        from ..ai.registry import ModelRegistry

        registry = ModelRegistry(self.params["registry_root"])
        self.entry, self.model = registry.load_model(self.params["model_id"])
        self.sanity_rejections = 0
        self.last_predictions: pd.DataFrame | None = None
        self._panel: FeaturePanel | None = None

    # ------------------------------------------------------------------
    def _predict_panel(self, panel: FeaturePanel) -> pd.DataFrame:
        """Predict for every (date, symbol); returns wide prediction frame."""
        feats = self.entry["feature_names"]
        stacked = pd.concat(
            [panel[f].stack().rename(f) for f in feats], axis=1
        ).dropna()
        if stacked.empty:
            return pd.DataFrame(index=panel.close.index,
                                columns=panel.close.columns, dtype=float)
        preds = pd.Series(
            self.model.predict(stacked.to_numpy()), index=stacked.index
        )

        # AI-hallucination sanity bound: zero out out-of-distribution signals
        bounds = self.entry.get("signal_bounds")
        if bounds:
            lo, hi = bounds
            insane = (preds < lo) | (preds > hi)
            self.sanity_rejections += int(insane.sum())
            preds = preds.where(~insane)

        return preds.unstack().reindex(index=panel.close.index,
                                       columns=panel.close.columns)

    def target_weights(self, panel: FeaturePanel) -> pd.DataFrame:
        self._panel = panel
        preds = self._predict_panel(panel)
        self.last_predictions = preds

        top_n = self.params["top_n"]
        reb = self.params["rebalance_days"]
        cap = self.params["gross_cap"]

        weights = pd.DataFrame(0.0, index=preds.index, columns=preds.columns)
        current = pd.Series(0.0, index=preds.columns)
        for i, dt in enumerate(preds.index):
            if i % reb == 0:
                ranked = preds.loc[dt].dropna()
                ranked = ranked[ranked > 0].nlargest(top_n)   # long-only: positive signal
                current = pd.Series(0.0, index=preds.columns)
                if len(ranked) > 0:
                    current[ranked.index] = cap / top_n
            weights.loc[dt] = current
        return weights

    # ------------------------------------------------------------------
    def explain(self, date, symbols: list[str] | None = None,
                top_k: int = 3) -> dict[str, dict]:
        """Feature attribution for held/candidate names on a date."""
        if self._panel is None or self.last_predictions is None:
            raise RuntimeError("call target_weights first")
        date = pd.Timestamp(date)
        feats = self.entry["feature_names"]
        row_preds = self.last_predictions.loc[date].dropna()
        symbols = symbols or list(row_preds.nlargest(self.params["top_n"]).index)
        out = {}
        for sym in symbols:
            x = [self._panel[f].loc[date, sym] for f in feats]
            if any(pd.isna(v) for v in x):
                continue
            attrib = self.model.attribution(pd.Series(x, dtype=float).to_numpy())
            out[sym] = {
                "prediction": round(float(row_preds.get(sym, float("nan"))), 6),
                "top_features": dict(list(attrib.items())[:top_k]),
            }
        return out
