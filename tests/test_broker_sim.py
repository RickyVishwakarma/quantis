import pandas as pd
import pytest

from quantis.broker import SimulatedBroker, reconcile
from quantis.oms import OMS, ManagedOrder, OrderStatus

TS = pd.Timestamp("2025-01-10")


def order(**kw):
    defaults = dict(symbol="INFY", side="BUY", qty=10, ref_price=1500.0, ts=TS)
    defaults.update(kw)
    return ManagedOrder(**defaults)


def test_market_order_fills_at_next_open_with_slippage_and_charges():
    b = SimulatedBroker(starting_cash=100_000)
    b.set_adv({"INFY": 50_000_000})
    b.place(order(qty=10))
    assert b.on_bar(TS, {"INFY": 1500.0}) == 1
    (f,) = b.poll_fills()
    assert f.price > 1500.0                      # buy slips up
    assert f.costs > 0
    assert b.positions() == {"INFY": pytest.approx(10.0)}
    assert b.cash == pytest.approx(100_000 - 10 * f.price - f.costs)
    assert b.poll_fills() == []                  # each fill returned once


def test_place_is_idempotent_on_client_order_id():
    b = SimulatedBroker()
    o = order()
    id1 = b.place(o)
    id2 = b.place(o)                             # network retry
    assert id1 == id2
    assert len(b.open_order_ids()) == 1


def test_limit_order_waits_for_cross():
    b = SimulatedBroker(starting_cash=100_000)
    bid = b.place(order(order_type="LIMIT", limit_price=1450.0))
    b.on_bar(TS, {"INFY": 1500.0})
    assert b.poll_fills() == []                  # above limit, still pending
    b.on_bar(TS, {"INFY": 1440.0})
    (f,) = b.poll_fills()
    assert f.qty == 10
    assert bid not in b.open_order_ids()


def test_cannot_sell_what_you_dont_hold():
    b = SimulatedBroker(starting_cash=100_000)
    b.place(order(side="SELL", qty=5))
    b.on_bar(TS, {"INFY": 1500.0})
    assert b.poll_fills() == []                  # rejected fill, stays pending


def test_insufficient_cash_leaves_order_pending():
    b = SimulatedBroker(starting_cash=1_000)
    b.place(order(qty=100))                      # ~150k needed
    b.on_bar(TS, {"INFY": 1500.0})
    assert b.poll_fills() == []
    assert len(b.open_order_ids()) == 1


def test_reconciliation_detects_position_drift():
    oms = OMS()
    b = SimulatedBroker(starting_cash=1_000_000)
    b.set_adv({"INFY": 50_000_000})

    o = oms.submit(order(qty=10))
    oms.transition(o.order_id, OrderStatus.APPROVED)
    o.broker_order_id = b.place(o)
    oms.transition(o.order_id, OrderStatus.SENT)
    b.on_bar(TS, {"INFY": 1500.0})
    for f in b.poll_fills():
        oms.apply_fill(f)

    assert reconcile(oms, b, b.open_order_ids()).clean

    # Inject drift: broker says 12 shares, OMS fills say 10
    b.holdings["INFY"] = 12.0
    report = reconcile(oms, b, b.open_order_ids())
    assert not report.clean
    assert report.position_mismatches["INFY"] == {"local": 10.0, "broker": 12.0}
    assert "INFY" in report.summary()
