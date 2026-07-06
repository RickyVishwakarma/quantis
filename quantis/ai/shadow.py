"""Shadow mode: infer, don't trade (TDD Part 4 promotion pipeline).

A CANDIDATE model runs against the most recent data window producing
hypothetical target weights, which are evaluated with the vectorized
engine — no orders, no broker, no capital. The resulting report (realized
IC, hypothetical Sharpe, benchmark comparison, sanity-bound violations)
is attached to the registry entry and the model moves to SHADOW.

Promotion to PRODUCTION then requires this report PLUS a human
``approved_by`` — enforced by the registry, not by convention.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..backtest.metrics import compute_metrics
from ..backtest.vectorized import vectorized_returns
from ..strategies import get as get_strategy
from .registry import ModelRegistry, Stage
from .train import _per_date_ic


def run_shadow(
    wide: dict[str, pd.DataFrame],
    model_ref: str,
    shadow_days: int = 126,
    top_n: int = 8,
    gross_cap: float = 0.5,
    registry_root: str = "models",
) -> dict:
    registry = ModelRegistry(registry_root)
    entry = registry.resolve(model_ref)

    strat = get_strategy("ai_signal")(
        model_id=entry["model_id"], top_n=top_n, gross_cap=gross_cap,
        registry_root=registry_root,
    )
    from ..features import compute_features
    panel = compute_features(wide)
    weights = strat.target_weights(panel)

    dates = weights.index
    window = dates[-shadow_days:] if len(dates) > shadow_days else dates
    w_window = weights.loc[window]

    daily = vectorized_returns(wide, weights).loc[window]
    equity = (1 + daily).cumprod() * 1_000_000
    metrics = compute_metrics(pd.Series(equity, index=window, name="equity"))

    # Benchmark: equal-weight universe over the same window
    eq_w = weights.copy() * 0 + 1.0 / len(weights.columns)
    bench_daily = vectorized_returns(wide, eq_w).loc[window]

    # Realized IC over the shadow window
    horizon = entry.get("label_horizon", 5)
    fwd = wide["close"].pct_change(horizon).shift(-horizon)
    preds = strat.last_predictions
    ic = float("nan")
    if preds is not None:
        merged = pd.concat(
            [preds.loc[window].stack().rename("pred"),
             fwd.loc[window].stack().rename("label")], axis=1
        ).dropna().reset_index(names=["ts", "symbol"])
        if len(merged):
            ic = _per_date_ic(merged, "pred", "label")

    report = {
        "shadow_days": len(window),
        "window_start": str(window[0].date()),
        "window_end": str(window[-1].date()),
        "hypothetical_sharpe": metrics.get("sharpe"),
        "hypothetical_return": metrics.get("total_return"),
        "max_drawdown": metrics.get("max_drawdown"),
        "benchmark_return": float((1 + bench_daily).prod() - 1),
        "realized_ic": None if np.isnan(ic) else round(ic, 4),
        "signals_rejected_by_sanity_bound": strat.sanity_rejections,
        "avg_gross_exposure": float(w_window.sum(axis=1).mean()),
    }
    report = {k: (round(v, 4) if isinstance(v, float) else v)
              for k, v in report.items()}

    registry.attach_shadow_report(entry["model_id"], report)
    if entry["stage"] == Stage.CANDIDATE.value:
        registry.promote(entry["model_id"], Stage.SHADOW)
    return report
