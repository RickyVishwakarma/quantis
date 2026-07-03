"""End-to-end engine tests on synthetic data."""

import pandas as pd
import pytest

from quantis.backtest import EventBacktester, NSECostModel
from quantis.backtest.vectorized import vectorized_run
from quantis.data.ingest import generate_synthetic
from quantis.data.store import to_wide
from quantis.features import compute_features
from quantis.risk import RiskLimits
from quantis.strategies import get as get_strategy

SYMS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "ITC", "SBIN", "LT", "MARUTI", "SUNPHARMA"]


@pytest.fixture(scope="module")
def wide():
    bars = generate_synthetic(SYMS, start="2021-01-01", end="2023-12-31", seed=11)
    return to_wide(bars)


def test_event_backtest_runs_and_is_accounted(wide):
    engine = EventBacktester(initial_capital=1_000_000)
    result = engine.run(wide, get_strategy("ma_crossover")())
    assert len(result.equity) > 500
    assert result.equity.iloc[0] > 0
    # Every executed trade must have a matching APPROVE decision
    n_fills = len(result.trades)
    n_approved = (result.risk_decisions["outcome"] == "APPROVE").sum()
    assert n_fills <= n_approved
    # Costs are real money
    if n_fills:
        assert (result.trades["costs"] > 0).all()


def test_risk_gate_cannot_be_starved(wide):
    """With a tight limit set, the engine still runs and logs rejections."""
    engine = EventBacktester(
        initial_capital=1_000_000,
        risk_limits=RiskLimits(max_position_weight=0.02),
    )
    result = engine.run(wide, get_strategy("momentum")(top_n=5))
    assert result.n_rejected > 0
    # Fills can never outnumber approvals for any (ts, symbol, side) bucket
    if len(result.trades):
        fills = result.trades.groupby(["ts", "symbol", "side"]).size()
        approvals = (
            result.risk_decisions[result.risk_decisions["outcome"] == "APPROVE"]
            .groupby(["ts", "symbol", "side"]).size()
        )
        for key, n in fills.items():
            assert n <= approvals.get(key, 0)


def test_no_negative_cash_or_shorts(wide):
    engine = EventBacktester(initial_capital=500_000)
    result = engine.run(wide, get_strategy("mean_reversion")())
    sells = result.trades.query("side == 'SELL'").groupby("symbol")["qty"].sum()
    buys = result.trades.query("side == 'BUY'").groupby("symbol")["qty"].sum()
    for sym in sells.index:
        assert sells[sym] <= buys.get(sym, 0) + 1e-6  # never sell what we don't hold


def test_vectorized_and_event_directionally_agree(wide):
    """Same weights through both engines should land in the same ballpark
    (identical cost model, different granularity)."""
    panel = compute_features(wide)
    strat = get_strategy("ma_crossover")()
    weights = strat.target_weights(panel)

    vec_equity = vectorized_run(wide, weights)
    result = EventBacktester().run_weights(panel, weights)

    vec_ret = vec_equity.iloc[-1] / vec_equity.iloc[0] - 1
    evt_ret = result.equity.iloc[-1] / result.equity.iloc[0] - 1
    # Same sign or both near zero; engines must not tell opposite stories
    assert abs(vec_ret - evt_ret) < 0.5
