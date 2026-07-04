"""Vectorized engine for fast parameter sweeps.

Whole-history matrix math: weights (decided at close t) are shifted one
bar and applied to open-to-open returns, with a per-unit-turnover cost
drawn from the SAME NSECostModel as the event engine, so a parameter
combo cannot look good here and fail there (TDD Part 9 requirement).
Use for grid searches; validate winners in the event engine, which adds
the risk gate, integer cash, and per-order slippage.
"""

from __future__ import annotations

from itertools import product

import pandas as pd

from ..features import compute_features
from ..strategies import get as get_strategy
from .costs import NSECostModel
from .metrics import compute_metrics


def vectorized_returns(
    wide: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    cost_model: NSECostModel | None = None,
) -> pd.Series:
    """Daily portfolio returns net of turnover costs (whole history)."""
    costs = cost_model or NSECostModel()
    open_px = wide["open"]
    rets = open_px.pct_change().reindex(weights.index)

    w = weights.fillna(0.0).clip(lower=0.0)
    gross = w.sum(axis=1).clip(lower=1.0)
    w = w.div(gross, axis=0)                      # enforce gross <= 1

    held = w.shift(2)                              # decide t, trade t+1 open, earn t+1→t+2
    port_ret = (held * rets).sum(axis=1)

    turnover = (w.shift(1) - w.shift(2)).abs().sum(axis=1) / 2
    cost_per_turnover = costs.round_trip_bps() / 10_000
    return (port_ret - turnover * cost_per_turnover).fillna(0.0)


def vectorized_run(
    wide: dict[str, pd.DataFrame],
    weights: pd.DataFrame,
    cost_model: NSECostModel | None = None,
    initial_capital: float = 1_000_000.0,
) -> pd.Series:
    port_ret = vectorized_returns(wide, weights, cost_model)
    equity = initial_capital * (1 + port_ret).cumprod()
    equity.name = "equity"
    return equity


def sweep(
    wide: dict[str, pd.DataFrame],
    strategy_name: str,
    grid: dict[str, list],
    cost_model: NSECostModel | None = None,
) -> pd.DataFrame:
    """Run every parameter combination; return a metrics leaderboard."""
    panel = compute_features(wide)
    cls = get_strategy(strategy_name)
    rows = []
    keys = list(grid)
    for combo in product(*(grid[k] for k in keys)):
        params = dict(zip(keys, combo))
        strat = cls(**params)
        weights = strat.target_weights(panel)
        equity = vectorized_run(wide, weights, cost_model)
        m = compute_metrics(equity)
        rows.append({**params,
                     "sharpe": m.get("sharpe"), "cagr": m.get("cagr"),
                     "max_dd": m.get("max_drawdown"), "calmar": m.get("calmar")})
    return (pd.DataFrame(rows)
            .sort_values("sharpe", ascending=False)
            .reset_index(drop=True))
