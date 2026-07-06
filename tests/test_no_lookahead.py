"""Mechanical look-ahead bias checks (TDD Part 9: 'preventing the four
classic biases' — data leakage is unit-tested, not just documented).

Method: perturb all bars strictly AFTER a cutoff date; features and
strategy weights at/before the cutoff must be bit-identical. If any
feature or strategy peeked past its timestamp, this fails.
"""

import numpy as np
import pandas as pd
import pytest

from quantis.data.ingest import generate_synthetic
from quantis.data.store import to_wide
from quantis.features import compute_features
from quantis import strategies

SYMS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "ITC", "SBIN", "LT", "MARUTI", "SUNPHARMA"]


@pytest.fixture(scope="module")
def bars():
    return generate_synthetic(SYMS, start="2021-01-01", end="2023-12-31", seed=7)


def _perturbed(bars, cutoff):
    future = bars["ts"] > cutoff
    out = bars.copy()
    rng = np.random.default_rng(99)
    for col in ["open", "high", "low", "close"]:
        out.loc[future, col] *= rng.uniform(0.5, 1.5, future.sum())
    return out


def test_features_are_point_in_time(bars):
    cutoff = pd.Timestamp("2022-12-30")
    panel_a = compute_features(to_wide(bars))
    panel_b = compute_features(to_wide(_perturbed(bars, cutoff)))
    for name in panel_a.names():
        a = panel_a[name].loc[:cutoff]
        b = panel_b[name].loc[:cutoff]
        pd.testing.assert_frame_equal(a, b, check_exact=True), name


@pytest.mark.parametrize("strategy_name", strategies.available())
def test_strategy_weights_are_point_in_time(bars, strategy_name, tmp_path):
    cutoff = pd.Timestamp("2022-12-30")
    panel_a = compute_features(to_wide(bars))
    panel_b = compute_features(to_wide(_perturbed(bars, cutoff)))

    params = {}
    if strategy_name == "ai_signal":
        # AI predictions must be as point-in-time as any feature: train a
        # throwaway model and run the same perturbation check through it
        from quantis.ai.train import train_and_register
        entry = train_and_register(
            to_wide(bars), model_type="ridge",
            fstore_root=str(tmp_path / "fstore"),
            registry_root=str(tmp_path / "models"),
        )
        params = {"model_id": entry["model_id"],
                  "registry_root": str(tmp_path / "models")}

    strat = strategies.get(strategy_name)(**params)
    w_a = strat.target_weights(panel_a).loc[:cutoff]
    w_b = strat.target_weights(panel_b).loc[:cutoff]
    pd.testing.assert_frame_equal(w_a, w_b, check_exact=True)
