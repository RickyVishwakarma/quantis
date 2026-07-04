"""Walk-forward validation (TDD Part 9 validation methodology).

For each rolling (or expanding) train/test split:

  1. Sweep the parameter grid on the TRAIN window with the vectorized
     engine and pick the best combo by Sharpe.
  2. Run the chosen combo through the EVENT engine (risk gate, cash,
     per-order slippage) on the unseen TEST window.
  3. Stitch the test segments into one continuous out-of-sample equity
     curve — the only number treated as evidence. A single in-sample
     backtest is a research artifact, never evidence (TDD rule).

The aggregate is then stress-checked two ways:
  - Block bootstrap of OOS daily returns → distribution of Sharpe and
    max-drawdown outcomes (Monte Carlo per Part 9).
  - Deflated Sharpe ratio (Bailey & López de Prado) penalizing the
    number of grid combinations tried — the TDD's overfitting gate for
    parameter sweeps.

Point-in-time note: features at row t only use data through t, so
computing the feature panel once over the full history leaks nothing;
the only fitted quantity is the parameter choice, which is always taken
from the train window alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from statistics import NormalDist

import numpy as np
import pandas as pd

from ..backtest.costs import NSECostModel
from ..backtest.engine import EventBacktester
from ..backtest.metrics import TRADING_DAYS, compute_metrics
from ..backtest.vectorized import vectorized_returns
from ..features import compute_features
from ..risk import RiskLimits
from ..strategies import get as get_strategy

_EULER_GAMMA = 0.5772156649


@dataclass
class WalkForwardConfig:
    train_days: int = 504          # ~2y
    test_days: int = 126           # ~6m
    expanding: bool = False        # False = rolling window
    min_train_days: int = 252


@dataclass
class WalkForwardResult:
    windows: pd.DataFrame          # one row per split: ranges, params, IS/OOS metrics
    oos_equity: pd.Series          # stitched, continuous
    oos_metrics: dict
    monte_carlo: dict              # bootstrap percentiles
    deflated_sharpe: float         # P[true Sharpe > 0 | n_trials]
    n_trials: int
    strategy: str = ""
    grid: dict = field(default_factory=dict)


def make_windows(dates: pd.DatetimeIndex, cfg: WalkForwardConfig) -> list[tuple[slice, slice]]:
    """Index-position slices; train always strictly precedes test."""
    windows = []
    start = 0
    train_end = cfg.train_days
    while train_end + 1 < len(dates):
        test_end = min(train_end + cfg.test_days, len(dates))
        train_start = 0 if cfg.expanding else max(0, train_end - cfg.train_days)
        if train_end - train_start >= cfg.min_train_days:
            windows.append((slice(train_start, train_end), slice(train_end, test_end)))
        train_end += cfg.test_days
    return windows


def _sharpe(daily: pd.Series) -> float:
    sd = daily.std()
    return float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def _max_dd(daily: np.ndarray) -> float:
    equity = np.cumprod(1 + daily)
    peak = np.maximum.accumulate(equity)
    return float((equity / peak - 1).min())


def block_bootstrap(daily: pd.Series, n_sims: int = 500, block: int = 21,
                    seed: int = 7) -> dict:
    """Resample OOS returns in blocks (preserves short-range autocorrelation)."""
    rng = np.random.default_rng(seed)
    arr = daily.to_numpy()
    n = len(arr)
    if n < block * 2:
        return {"error": "insufficient OOS data for bootstrap"}
    n_blocks = int(np.ceil(n / block))
    sharpes, maxdds = [], []
    for _ in range(n_sims):
        starts = rng.integers(0, n - block, size=n_blocks)
        sim = np.concatenate([arr[s:s + block] for s in starts])[:n]
        sd = sim.std()
        sharpes.append(sim.mean() / sd * np.sqrt(TRADING_DAYS) if sd > 0 else np.nan)
        maxdds.append(_max_dd(sim))
    sh, dd = np.array(sharpes), np.array(maxdds)
    pct = lambda a, q: float(np.nanpercentile(a, q))
    return {
        "n_sims": n_sims, "block": block,
        "sharpe_p05": pct(sh, 5), "sharpe_p50": pct(sh, 50), "sharpe_p95": pct(sh, 95),
        "prob_sharpe_negative": float(np.nanmean(sh < 0)),
        "maxdd_p05": pct(dd, 5), "maxdd_p50": pct(dd, 50), "maxdd_p95": pct(dd, 95),
    }


def deflated_sharpe(daily: pd.Series, n_trials: int,
                    trial_sharpe_std: float | None = None) -> float:
    """P[true Sharpe > 0] after deflating for multiple testing.

    Bailey & López de Prado (2014): benchmark the observed Sharpe against
    the expected maximum Sharpe of ``n_trials`` unskilled strategies,
    adjusting for the return distribution's skew and kurtosis.
    """
    nd = NormalDist()
    T = len(daily)
    if T < 30:
        return float("nan")
    sr = daily.mean() / daily.std() if daily.std() > 0 else 0.0   # per-period SR
    if n_trials <= 1:
        sr0 = 0.0
    else:
        sd_trials = trial_sharpe_std if trial_sharpe_std and trial_sharpe_std > 0 \
            else max(abs(sr) / 2, 1e-4)
        e_max = ((1 - _EULER_GAMMA) * nd.inv_cdf(1 - 1 / n_trials)
                 + _EULER_GAMMA * nd.inv_cdf(1 - 1 / (n_trials * np.e)))
        sr0 = sd_trials * e_max
    x = daily.to_numpy()
    x = (x - x.mean()) / (x.std() or 1.0)
    skew = float((x ** 3).mean())
    kurt = float((x ** 4).mean())
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr ** 2, 1e-12))
    z = (sr - sr0) * np.sqrt(T - 1) / denom
    return float(nd.cdf(z))


def run_walkforward(
    wide: dict[str, pd.DataFrame],
    strategy_name: str,
    grid: dict[str, list],
    cfg: WalkForwardConfig | None = None,
    initial_capital: float = 1_000_000.0,
    cost_model: NSECostModel | None = None,
    risk_limits: RiskLimits | None = None,
) -> WalkForwardResult:
    cfg = cfg or WalkForwardConfig()
    cost_model = cost_model or NSECostModel()
    panel = compute_features(wide)
    dates = panel.close.index
    windows = make_windows(dates, cfg)
    if not windows:
        raise ValueError(
            f"History too short: {len(dates)} bars < train {cfg.train_days} "
            f"+ test {cfg.test_days}"
        )

    # Weights + vectorized daily returns computed once per combo (PIT-safe)
    cls = get_strategy(strategy_name)
    keys = list(grid)
    combos = [dict(zip(keys, c)) for c in product(*(grid[k] for k in keys))]
    combo_rets: list[pd.Series] = []
    combo_weights: list[pd.DataFrame] = []
    for params in combos:
        w = cls(**params).target_weights(panel)
        combo_weights.append(w)
        combo_rets.append(vectorized_returns(wide, w, cost_model))

    engine = EventBacktester(initial_capital=initial_capital,
                             cost_model=cost_model, risk_limits=risk_limits)

    rows, oos_daily = [], []
    for train_sl, test_sl in windows:
        train_dates, test_dates = dates[train_sl], dates[test_sl]

        is_sharpes = [_sharpe(r.loc[train_dates[0]:train_dates[-1]]) for r in combo_rets]
        best = int(np.nanargmax(is_sharpes))
        best_params = combos[best]

        # Event engine on the unseen test window (+1 lookback row so the
        # first test day executes the weights decided the prior close)
        w_slice = combo_weights[best].loc[dates[max(test_sl.start - 1, 0)]:test_dates[-1]]
        result = engine.run_weights(panel, w_slice,
                                    strategy_name=f"{strategy_name}_wf",
                                    params=best_params)
        seg = result.equity.pct_change().fillna(
            result.equity.iloc[0] / initial_capital - 1
        )
        oos_daily.append(seg)

        rows.append({
            "train_start": train_dates[0].date(), "train_end": train_dates[-1].date(),
            "test_start": test_dates[0].date(), "test_end": test_dates[-1].date(),
            **{f"param_{k}": v for k, v in best_params.items()},
            "is_sharpe": round(is_sharpes[best], 3),
            "oos_sharpe": round(_sharpe(seg), 3),
            "oos_return": round(float((1 + seg).prod() - 1), 4),
            "risk_rejections": result.n_rejected,
        })

    oos = pd.concat(oos_daily).sort_index()
    oos = oos[~oos.index.duplicated(keep="first")]
    oos_equity = initial_capital * (1 + oos).cumprod()
    oos_equity.name = "equity"

    trial_std = float(pd.Series(
        [_sharpe(r) / np.sqrt(TRADING_DAYS) for r in combo_rets]
    ).std()) if len(combos) > 1 else None

    return WalkForwardResult(
        windows=pd.DataFrame(rows),
        oos_equity=oos_equity,
        oos_metrics=compute_metrics(oos_equity),
        monte_carlo=block_bootstrap(oos),
        deflated_sharpe=deflated_sharpe(oos, n_trials=len(combos),
                                        trial_sharpe_std=trial_std),
        n_trials=len(combos),
        strategy=strategy_name,
        grid=grid,
    )
