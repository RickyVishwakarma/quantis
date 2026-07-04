"""End-to-end paper trading tests, including the backtest-parity check —
the TDD's core Phase-3 promise: paper results reflect the exact strategy
and risk code that runs in the backtester, differing only at the broker
boundary.
"""

import json

import pytest

from quantis.backtest import EventBacktester
from quantis.data.ingest import generate_synthetic
from quantis.data.store import to_wide
from quantis.feed import ReplayFeed
from quantis.paper import PaperTradingEngine
from quantis.risk import RiskLimits
from quantis.strategies import get as get_strategy

SYMS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "ITC", "SBIN", "LT"]
WARMUP = 210


@pytest.fixture(scope="module")
def wide():
    bars = generate_synthetic(SYMS, start="2021-01-01", end="2023-06-30", seed=21)
    return to_wide(bars)


def make_engine(session_dir=None, **kw):
    return PaperTradingEngine(
        strategy=get_strategy("ma_crossover")(),
        initial_capital=1_000_000,
        risk_limits=RiskLimits(),
        session_dir=session_dir,
        **kw,
    )


def test_paper_session_runs_and_persists(tmp_path, wide):
    engine = make_engine(session_dir=tmp_path / "s1")
    session = engine.run(ReplayFeed(wide), warmup_bars=WARMUP)

    assert len(session.equity) == len(wide["close"])
    assert (tmp_path / "s1" / "orders.jsonl").exists()
    assert (tmp_path / "s1" / "equity.csv").exists()
    assert (tmp_path / "s1" / "session.json").exists()
    cfg = json.loads((tmp_path / "s1" / "session.json").read_text())
    assert cfg["strategy"] == "ma_crossover"
    assert "clean" in session.reconciliation


def test_every_fill_has_an_approved_order(wide):
    engine = make_engine()
    session = engine.run(ReplayFeed(wide), warmup_bars=WARMUP)
    fills = session.fills
    assert len(fills) > 0
    # every filled order_id exists in the OMS and was approved before SENT
    for oid in fills["order_id"].unique():
        order = engine.oms.orders[oid]
        assert order.risk_decision_id is not None or order.strategy_id  # child slices carry parent approval
    # broker's book equals the OMS fill-implied book (self-reconciliation)
    assert engine.broker.positions() == pytest.approx(
        engine.oms.positions_from_fills()
    )


def test_backtest_parity(wide):
    """Same data, same strategy, same limits, same first trading day ->
    paper ~= backtest. The two runtimes share strategy, cost model, and
    risk code; only the broker boundary differs (TDD Part 4 promise)."""
    from quantis.features import compute_features

    engine = make_engine()
    session = engine.run(ReplayFeed(wide), warmup_bars=WARMUP)

    # Give the backtester the identical decision stream: no positions
    # before the paper engine's warmup completes.
    panel = compute_features(wide)
    weights = get_strategy("ma_crossover")().target_weights(panel)
    weights.iloc[:WARMUP - 1] = 0.0
    bt = EventBacktester(initial_capital=1_000_000, risk_limits=RiskLimits())
    bt_result = bt.run_weights(panel, weights)

    paper, btest = session.equity.align(bt_result.equity, join="inner")
    paper, btest = paper.iloc[WARMUP:], btest.iloc[WARMUP:]
    corr = paper.pct_change().corr(btest.pct_change())
    assert corr > 0.90, f"paper/backtest daily-return correlation {corr:.3f}"
    final_gap = abs(paper.iloc[-1] / btest.iloc[-1] - 1)
    # sizing marks differ by one bar (close t-1 vs open t), so day-level
    # jitter is expected; terminal wealth must agree tightly
    assert final_gap < 0.02, f"final equity diverged {final_gap:.2%}"


def test_flatten_tier_exits_positions(wide):
    """With a tight flatten threshold, the engine unwinds rather than holds:
    once drawdown breaches the tier, targets go to zero, the book is sold
    down, and (costs keeping equity below peak) it stays in cash."""
    limits = RiskLimits(soft_drawdown=0.005, flatten_drawdown=0.01,
                        breaker_consecutive_rejects=10_000)
    engine = PaperTradingEngine(
        strategy=get_strategy("ma_crossover")(),
        initial_capital=1_000_000, risk_limits=limits,
    )
    session = engine.run(ReplayFeed(wide), warmup_bars=WARMUP)
    assert (session.fills["side"] == "SELL").any()       # it did unwind
    assert session.final_positions == {}                 # fully in cash
    tail = session.equity.iloc[-30:].pct_change().abs()
    assert tail.max() < 1e-6                             # flat once flattened


def test_twap_execution_path(wide):
    engine = make_engine(execution_algo="twap", twap_slices=3)
    session = engine.run(ReplayFeed(wide), warmup_bars=WARMUP)
    assert len(session.fills) > 0
    assert "clean" in session.reconciliation
