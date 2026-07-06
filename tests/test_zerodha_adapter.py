"""Zerodha Kite adapter contract tests against a fake Kite client —
no credentials, no network, no kiteconnect dependency required."""

import pandas as pd
import pytest

from quantis.broker import BrokerError, ZerodhaKiteBroker
from quantis.oms import ManagedOrder

TS = pd.Timestamp("2026-07-06")


class FakeKite:
    """Minimal stand-in for kiteconnect.KiteConnect."""

    def __init__(self):
        self._orders: list[dict] = []
        self._seq = 0
        self.place_calls = 0
        self.fail_next_place = 0          # raise N times before succeeding
        self.landed_despite_error = False # simulate timeout-after-land

    def place_order(self, **params):
        self.place_calls += 1
        if self.fail_next_place > 0:
            self.fail_next_place -= 1
            if self.landed_despite_error:
                self._seq += 1
                self._orders.append({
                    "order_id": str(1000 + self._seq), "status": "OPEN",
                    "tag": params.get("tag"),
                    "tradingsymbol": params["tradingsymbol"],
                    "transaction_type": params["transaction_type"],
                    "quantity": params["quantity"],
                    "filled_quantity": 0, "average_price": 0,
                })
            raise TimeoutError("gateway timeout")
        self._seq += 1
        oid = str(1000 + self._seq)
        self._orders.append({
            "order_id": oid, "status": "OPEN", "tag": params.get("tag"),
            "tradingsymbol": params["tradingsymbol"],
            "transaction_type": params["transaction_type"],
            "quantity": params["quantity"],
            "filled_quantity": 0, "average_price": 0,
        })
        return oid

    def cancel_order(self, variety, order_id):
        for o in self._orders:
            if o["order_id"] == order_id:
                o["status"] = "CANCELLED"
                return order_id
        raise ValueError("unknown order")

    def orders(self):
        return list(self._orders)

    def fill(self, order_id, qty, price):
        for o in self._orders:
            if o["order_id"] == order_id:
                o["filled_quantity"] = o.get("filled_quantity", 0) + qty
                o["average_price"] = price
                o["status"] = "COMPLETE"

    def positions(self):
        return {"net": [{"tradingsymbol": "TCS", "quantity": 10},
                        {"tradingsymbol": "INFY", "quantity": 0}]}

    def margins(self):
        return {"equity": {"available": {"cash": 250_000.0}}}


def order(**kw):
    d = dict(symbol="TCS", side="BUY", qty=10, ref_price=3500.0, ts=TS)
    d.update(kw)
    return ManagedOrder(**d)


@pytest.fixture()
def broker():
    return ZerodhaKiteBroker(kite=FakeKite(), retry_wait=0.0)


def test_place_maps_fields_and_tags(broker):
    o = order()
    bid = broker.place(o)
    kite_order = broker.kite.orders()[0]
    assert kite_order["tradingsymbol"] == "TCS"
    assert kite_order["transaction_type"] == "BUY"
    assert kite_order["quantity"] == 10
    assert kite_order["tag"] == o.client_order_id[-20:]
    assert bid == kite_order["order_id"]


def test_place_is_idempotent(broker):
    o = order()
    assert broker.place(o) == broker.place(o)
    assert broker.kite.place_calls == 1


def test_retry_then_success(broker):
    broker.kite.fail_next_place = 2       # two transient failures
    bid = broker.place(order())
    assert bid
    assert broker.error_count == 2        # feeds the broker-error breaker


def test_timeout_after_landing_does_not_double_execute(broker):
    """The classic double-execution bug: request times out but the order
    actually landed. The adapter must find it by tag, not re-place it."""
    broker.kite.fail_next_place = 3       # exhaust all retries
    broker.kite.landed_despite_error = True
    o = order()
    bid = broker.place(o)
    landed = [k for k in broker.kite.orders() if k["tag"] == o.client_order_id[-20:]]
    assert len(landed) >= 1
    assert bid == landed[0]["order_id"]
    # a later retry of the same order returns the same id, no new kite order
    n_before = len(broker.kite.orders())
    assert broker.place(o) == bid
    assert len(broker.kite.orders()) == n_before


def test_poll_fills_emits_increments_once(broker):
    o = order(qty=10)
    bid = broker.place(o)
    broker.kite.fill(bid, 6, 3501.0)
    fills = broker.poll_fills()
    assert len(fills) == 1
    assert fills[0].qty == 6 and fills[0].price == 3501.0
    assert fills[0].order_id == o.order_id      # OMS id, not broker id
    assert broker.poll_fills() == []            # no re-emission
    broker.kite.fill(bid, 4, 3502.0)
    (f2,) = broker.poll_fills()
    assert f2.qty == 4                          # only the increment


def test_cancel_and_open_orders(broker):
    bid = broker.place(order())
    assert bid in broker.open_order_ids()
    assert broker.cancel(bid)
    assert bid not in broker.open_order_ids()


def test_positions_and_margins_mapping(broker):
    assert broker.positions() == {"TCS": 10.0}   # zero-qty rows dropped
    assert broker.margins() == {"cash": 250_000.0}


def test_hard_failure_raises_broker_error(broker):
    broker.kite.fail_next_place = 99
    with pytest.raises(BrokerError, match="failed after"):
        broker.place(order())
