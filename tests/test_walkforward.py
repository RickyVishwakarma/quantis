import numpy as np
import pandas as pd
import pytest

from quantis.data.ingest import generate_synthetic
from quantis.data.store import to_wide
from quantis.research.walkforward import (
    WalkForwardConfig,
    block_bootstrap,
    deflated_sharpe,
    make_windows,
    run_walkforward,
)

SYMS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "SBIN", "LT"]


@pytest.fixture(scope="module")
def wide():
    bars = generate_synthetic(SYMS, start="2019-01-01", end="2023-12-31", seed=3)
    return to_wide(bars)


def test_windows_train_strictly_precedes_test():
    dates = pd.date_range("2020-01-01", periods=800, freq="B")
    cfg = WalkForwardConfig(train_days=252, test_days=63)
    for train_sl, test_sl in make_windows(dates, cfg):
        assert train_sl.stop == test_sl.start          # contiguous, no gap
        assert dates[train_sl.stop - 1] < dates[test_sl.start]
        assert test_sl.stop <= len(dates)


def test_rolling_test_windows_never_overlap():
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    cfg = WalkForwardConfig(train_days=252, test_days=63)
    windows = make_windows(dates, cfg)
    for (_, a), (_, b) in zip(windows, windows[1:]):
        assert a.stop == b.start                        # OOS segments tile exactly


def test_walkforward_end_to_end(wide):
    cfg = WalkForwardConfig(train_days=252, test_days=126, min_train_days=200)
    wf = run_walkforward(wide, "ma_crossover",
                         {"fast": [10, 20], "slow": [50, 100]}, cfg)
    assert len(wf.windows) >= 3
    assert wf.n_trials == 4
    # Stitched OOS equity covers exactly the union of test windows
    assert len(wf.oos_equity) == len(wf.oos_equity.index.unique())
    assert wf.oos_equity.index.is_monotonic_increasing
    assert "sharpe" in wf.oos_metrics
    assert 0.0 <= wf.deflated_sharpe <= 1.0
    # chosen params must come from the grid
    assert set(wf.windows["param_fast"]).issubset({10, 20})


def test_history_too_short_raises(wide):
    small = {k: v.iloc[:100] for k, v in wide.items()}
    with pytest.raises(ValueError, match="too short"):
        run_walkforward(small, "ma_crossover", {"fast": [10], "slow": [50]},
                        WalkForwardConfig(train_days=504, test_days=126))


def test_deflated_sharpe_penalizes_trials():
    rng = np.random.default_rng(0)
    daily = pd.Series(rng.normal(0.0006, 0.01, 1000))
    one = deflated_sharpe(daily, n_trials=1)
    many = deflated_sharpe(daily, n_trials=200, trial_sharpe_std=0.03)
    assert many < one                    # more trials → less confidence


def test_block_bootstrap_shape():
    rng = np.random.default_rng(1)
    daily = pd.Series(rng.normal(0.0005, 0.012, 500))
    mc = block_bootstrap(daily, n_sims=100)
    assert mc["sharpe_p05"] <= mc["sharpe_p50"] <= mc["sharpe_p95"]
    assert -1 <= mc["maxdd_p50"] <= 0
    assert 0 <= mc["prob_sharpe_negative"] <= 1
