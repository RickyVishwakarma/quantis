"""Live engine: arming interlock, audit trail, reconciliation cadence,
breaker response. All against sim/fake brokers — no real venue, ever."""

import json

import pytest

from quantis.audit import AuditLog
from quantis.broker import DryRunBroker, SimulatedBroker, ZerodhaKiteBroker
from quantis.data.ingest import generate_synthetic
from quantis.data.store import to_wide
from quantis.feed import ReplayFeed
from quantis.live import LiveTradingEngine
from quantis.risk import RiskLimits
from quantis.strategies import get as get_strategy

from test_zerodha_adapter import FakeKite

SYMS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK",
        "ITC", "SBIN", "LT"]
WARMUP = 210


@pytest.fixture(scope="module")
def wide():
    bars = generate_synthetic(SYMS, start="2021-06-01", end="2023-06-30", seed=31)
    return to_wide(bars)


def make_engine(tmp_path, wide=None, **kw):
    defaults = dict(
        strategy=get_strategy("ma_crossover")(),
        initial_capital=1_000_000,
        risk_limits=RiskLimits(),
        session_dir=tmp_path / "live",
        algo_id="SEBI-TEST-001",
    )
    defaults.update(kw)
    return LiveTradingEngine(**defaults)


def test_real_broker_without_arming_is_wrapped_in_dryrun(tmp_path):
    real = ZerodhaKiteBroker(kite=FakeKite(), retry_wait=0.0)
    engine = make_engine(tmp_path, broker=real, armed=False)
    assert isinstance(engine.broker, DryRunBroker)
    assert engine.broker.inner is real
    assert not engine.armed


def test_armed_without_algo_id_refused(tmp_path):
    real = ZerodhaKiteBroker(kite=FakeKite(), retry_wait=0.0)
    with pytest.raises(ValueError, match="algo_id"):
        make_engine(tmp_path, broker=real, armed=True, algo_id="")


def test_dry_run_session_places_nothing(tmp_path, wide):
    real = ZerodhaKiteBroker(kite=FakeKite(), retry_wait=0.0)
    engine = make_engine(tmp_path, broker=real, armed=False)
    session = engine.run(ReplayFeed(wide), warmup_bars=WARMUP)
    assert real.kite.place_calls == 0              # nothing reached the venue
    assert len(engine.broker.would_be_orders) > 0  # but intent was journaled
    assert all(o["algo_id"] == "SEBI-TEST-001"
               for o in engine.broker.would_be_orders)
    assert len(session.fills) == 0                 # dry run never fills


def test_sim_session_audits_everything(tmp_path, wide):
    engine = make_engine(tmp_path, broker=SimulatedBroker(starting_cash=1_000_000))
    session = engine.run(ReplayFeed(wide), warmup_bars=WARMUP)

    recs = engine.audit.records()
    types = [r["event_type"] for r in recs]
    assert types[0] == "session_start"
    assert types[-1] == "session_end"
    assert types.count("reconciliation") >= 3      # start + periodic + eod
    n_decisions = sum(t == "risk_decision" for t in types)
    n_fills = sum(t == "fill" for t in types)
    assert n_decisions > 0 and n_fills > 0
    assert n_fills == len(session.fills)           # every fill audited

    # chain is intact and every decision carries the SEBI tag
    ok, bad = engine.audit.verify()
    assert ok, f"audit chain broken at {bad}"
    for r in recs:
        if r["event_type"] == "risk_decision":
            assert r["payload"]["algo_id"] == "SEBI-TEST-001"


def test_reconciliation_mismatch_trips_breaker_when_armed(tmp_path, wide):
    engine = make_engine(tmp_path, broker=SimulatedBroker(starting_cash=1_000_000))
    engine.armed = True                            # simulate an armed session
    # inject drift mid-flight: broker suddenly claims shares the OMS never saw
    engine.broker.holdings["RELIANCE"] = 999.0
    engine._reconcile("test_injection")
    assert engine.risk.breaker.tripped
    assert "reconciliation mismatch" in engine.risk.breaker.reason
    last = engine.audit.records()[-1]
    assert last["event_type"] == "reconciliation"
    assert not last["payload"]["clean"]


def test_breaker_trip_cancels_open_orders_and_audits(tmp_path, wide):
    limits = RiskLimits(breaker_consecutive_rejects=3, max_position_weight=0.001)
    engine = make_engine(tmp_path, broker=SimulatedBroker(starting_cash=1_000_000),
                         risk_limits=limits)
    engine.run(ReplayFeed(wide), warmup_bars=WARMUP)
    assert engine.risk.breaker.tripped             # every buy breaches -> trip
    events = [r for r in engine.audit.records()
              if r["event_type"] == "circuit_breaker"]
    assert len(events) == 1                        # audited once, at the trip
    assert "consecutive risk rejections" in events[0]["payload"]["reason"]
    assert engine.broker.open_order_ids() == set() # nothing left resting


def test_audit_file_is_the_tamper_evident_artifact(tmp_path, wide):
    engine = make_engine(tmp_path, broker=SimulatedBroker(starting_cash=1_000_000))
    engine.run(ReplayFeed(wide), warmup_bars=WARMUP)
    path = tmp_path / "live" / "audit.jsonl"
    assert path.exists()

    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[5])
    rec["payload"]["outcome"] = "FORGED"           # rewrite history
    lines[5] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    ok, bad = AuditLog(path).verify()
    assert not ok and bad == 6
