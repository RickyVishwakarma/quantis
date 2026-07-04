import json

import pandas as pd
import pytest

from quantis.oms import OMS, Fill, InvalidTransition, ManagedOrder, OrderStatus

TS = pd.Timestamp("2025-01-10")


def order(**kw):
    defaults = dict(symbol="TCS", side="BUY", qty=10, ref_price=4000.0, ts=TS)
    defaults.update(kw)
    return ManagedOrder(**defaults)


def fill(o, qty, price=4000.0):
    return Fill(order_id=o.order_id, symbol=o.symbol, side=o.side,
                qty=qty, price=price, costs=5.0, ts=TS)


def test_happy_path_lifecycle():
    oms = OMS()
    o = oms.submit(order())
    oms.transition(o.order_id, OrderStatus.APPROVED)
    oms.transition(o.order_id, OrderStatus.SENT)
    oms.apply_fill(fill(o, 4))
    assert o.status == OrderStatus.PARTIALLY_FILLED
    oms.apply_fill(fill(o, 6, price=4010.0))
    assert o.status == OrderStatus.FILLED
    assert o.filled_qty == 10
    assert o.avg_fill_price == pytest.approx((4 * 4000 + 6 * 4010) / 10)


def test_rejected_order_cannot_fill():
    oms = OMS()
    o = oms.submit(order())
    oms.transition(o.order_id, OrderStatus.RISK_REJECTED)
    with pytest.raises(InvalidTransition):
        oms.apply_fill(fill(o, 10))


def test_illegal_transitions_raise():
    oms = OMS()
    o = oms.submit(order())
    with pytest.raises(InvalidTransition):
        oms.transition(o.order_id, OrderStatus.FILLED)     # PENDING_RISK -> FILLED
    oms.transition(o.order_id, OrderStatus.APPROVED)
    with pytest.raises(InvalidTransition):
        oms.transition(o.order_id, OrderStatus.APPROVED)   # double approval


def test_overfill_rejected():
    oms = OMS()
    o = oms.submit(order(qty=5))
    oms.transition(o.order_id, OrderStatus.APPROVED)
    oms.transition(o.order_id, OrderStatus.SENT)
    with pytest.raises(InvalidTransition, match="overfill"):
        oms.apply_fill(fill(o, 6))


def test_positions_from_fills_nets_out():
    oms = OMS()
    buy = oms.submit(order(qty=10))
    oms.transition(buy.order_id, OrderStatus.APPROVED)
    oms.transition(buy.order_id, OrderStatus.SENT)
    oms.apply_fill(fill(buy, 10))
    sell = oms.submit(order(side="SELL", qty=4))
    oms.transition(sell.order_id, OrderStatus.APPROVED)
    oms.transition(sell.order_id, OrderStatus.SENT)
    oms.apply_fill(fill(sell, 4))
    assert oms.positions_from_fills() == {"TCS": pytest.approx(6.0)}


def test_journal_is_append_only_audit_trail(tmp_path):
    oms = OMS(journal_dir=tmp_path)
    o = oms.submit(order())
    oms.transition(o.order_id, OrderStatus.APPROVED)
    oms.transition(o.order_id, OrderStatus.SENT)
    oms.apply_fill(fill(o, 10))
    lines = [json.loads(x) for x in
             (tmp_path / "orders.jsonl").read_text().splitlines()]
    statuses = [x["status"] for x in lines]
    assert statuses == ["PENDING_RISK", "APPROVED", "SENT", "FILLED"]
