"""End-to-end AI pipeline: train -> registry -> shadow -> promote -> trade
through the risk-gated backtester (the full TDD Part 4 loop at MVP scale)."""

import numpy as np
import pytest

from quantis.ai.registry import ModelRegistry, Stage
from quantis.ai.shadow import run_shadow
from quantis.ai.train import train_and_register
from quantis.backtest import EventBacktester
from quantis.data.ingest import generate_synthetic
from quantis.data.store import to_wide
from quantis.features import compute_features
from quantis.risk import RiskLimits
from quantis.strategies import get as get_strategy

SYMS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "ITC", "SBIN", "LT", "MARUTI", "SUNPHARMA"]


@pytest.fixture(scope="module")
def wide():
    bars = generate_synthetic(SYMS, start="2020-01-01", end="2023-12-31", seed=13)
    return to_wide(bars)


@pytest.fixture(scope="module")
def trained(wide, tmp_path_factory):
    root = tmp_path_factory.mktemp("ai")
    entry = train_and_register(
        wide, model_type="ridge",
        fstore_root=str(root / "fstore"),
        registry_root=str(root / "models"),
    )
    return root, entry


def test_training_registers_with_metrics(trained):
    _, entry = trained
    m = entry["metrics"]
    assert entry["stage"] in ("EXPERIMENTAL", "CANDIDATE")
    assert "ic" in m and "hit_rate" in m and m["n_train_rows"] > 300
    assert entry["signal_bounds"][0] < entry["signal_bounds"][1]
    assert entry["feature_schema_version"]


def test_ai_strategy_weights_respect_gross_cap(trained, wide):
    root, entry = trained
    strat = get_strategy("ai_signal")(
        model_id=entry["model_id"], gross_cap=0.5, top_n=5,
        registry_root=str(root / "models"),
    )
    panel = compute_features(wide)
    weights = strat.target_weights(panel)
    gross = weights.sum(axis=1)
    assert gross.max() <= 0.5 + 1e-9
    assert (weights >= 0).all().all()


def test_sanity_bound_zeroes_out_of_distribution_signals(trained, wide):
    root, entry = trained
    # shrink the bounds so most predictions look "insane"
    reg = ModelRegistry(str(root / "models"))
    entries = reg._load_index()
    for e in entries:
        if e["model_id"] == entry["model_id"]:
            e["signal_bounds"] = [-1e-9, 1e-9]
    reg._save_index(entries)

    strat = get_strategy("ai_signal")(
        model_id=entry["model_id"], registry_root=str(root / "models"),
    )
    panel = compute_features(wide)
    weights = strat.target_weights(panel)
    assert strat.sanity_rejections > 0
    assert weights.sum().sum() == pytest.approx(0.0)  # nothing traded

    # restore real bounds for later tests
    for e in entries:
        if e["model_id"] == entry["model_id"]:
            e["signal_bounds"] = entry["signal_bounds"]
    reg._save_index(entries)


def test_ai_orders_transit_the_risk_gate(trained, wide):
    root, entry = trained
    strat = get_strategy("ai_signal")(
        model_id=entry["model_id"], registry_root=str(root / "models"),
        top_n=3, gross_cap=0.9,
    )
    result = EventBacktester(risk_limits=RiskLimits()).run(wide, strat)
    # every AI order was evaluated; identical limit set as any other source
    assert len(result.risk_decisions) > 0
    assert set(result.risk_decisions["outcome"]) <= {"APPROVE", "REJECT"}


def test_explainability_ships_with_signals(trained, wide):
    root, entry = trained
    strat = get_strategy("ai_signal")(
        model_id=entry["model_id"], registry_root=str(root / "models"),
    )
    panel = compute_features(wide)
    strat.target_weights(panel)
    date = panel.close.index[-10]
    explained = strat.explain(date)
    assert len(explained) > 0
    for sym, info in explained.items():
        assert "prediction" in info
        assert 1 <= len(info["top_features"]) <= 3


def test_shadow_then_human_promotion(trained, wide):
    root, entry = trained
    reg = ModelRegistry(str(root / "models"))
    if reg.get(entry["model_id"])["stage"] == "EXPERIMENTAL":
        reg.promote(entry["model_id"], Stage.CANDIDATE)

    report = run_shadow(wide, entry["model_id"], shadow_days=60,
                        registry_root=str(root / "models"))
    assert report["shadow_days"] > 0
    assert "hypothetical_sharpe" in report and "realized_ic" in report

    e = reg.get(entry["model_id"])
    assert e["stage"] == "SHADOW"
    assert e["shadow_report"]["shadow_days"] == report["shadow_days"]

    promoted = reg.promote(entry["model_id"], Stage.PRODUCTION, approved_by="ricky")
    assert promoted["stage"] == "PRODUCTION"
    # resolve by stage now works
    assert reg.resolve("production")["model_id"] == entry["model_id"]
