import pandas as pd
import pytest

from quantis.broker import SimulatedBroker
from quantis.broker.base import BrokerError
from quantis.ems import EMS
from quantis.oms import OMS, ManagedOrder, OrderStatus

TS = pd.Timestamp("2025-01-10")


def approved_order(oms, qty=1000):
    o = oms.submit(ManagedOrder(symbol="INFY", side="BUY", qty=qty,
                                ref_price=1500.0, ts=TS))
    oms.transition(o.order_id, OrderStatus.APPROVED)
    return o


def test_ems_refuses_unapproved_orders():
    oms = OMS()
    b = SimulatedBroker()
    ems = EMS(oms=oms, broker=b)
    o = oms.submit(ManagedOrder(symbol="INFY", side="BUY", qty=10,
                                ref_price=1500.0, ts=TS))     # still PENDING_RISK
    with pytest.raises(BrokerError, match="APPROVED"):
        ems.execute(o)


def test_twap_slices_sum_to_parent_and_release_across_bars():
    oms = OMS()
    b = SimulatedBroker(starting_cash=10_000_000)
    b.set_adv({"INFY": 20_000_000})
    ems = EMS(oms=oms, broker=b)
    parent = approved_order(oms, qty=1000)
    children = ems.execute(parent, algo="twap", slices=4)
    assert len(children) == 4
    assert sum(c.qty for c in children) == pytest.approx(1000)

    fills = []
    for _ in range(5):                       # 4 slices need 4 bars
        ems.on_bar()
        b.on_bar(TS, {"INFY": 1500.0})
        fills.extend(b.poll_fills())
    assert sum(f.qty for f in fills) == pytest.approx(1000)
    # slices land on different bars, not all at once
    assert len(fills) == 4


def test_twap_impact_cheaper_than_block():
    """Each TWAP child participates 1/N of ADV -> sqrt impact is smaller."""
    adv = {"INFY": 2_000_000}

    def avg_fill(algo, slices=1):
        oms = OMS()
        b = SimulatedBroker(starting_cash=10_000_000)
        b.set_adv(adv)
        ems = EMS(oms=oms, broker=b)
        parent = approved_order(oms, qty=600)      # 900k notional vs 2M ADV
        ems.execute(parent, algo=algo, slices=slices)
        fills = []
        for _ in range(slices + 1):
            ems.on_bar()
            b.on_bar(TS, {"INFY": 1500.0})
            fills.extend(b.poll_fills())
        assert sum(f.qty for f in fills) == pytest.approx(600)
        return sum(f.price * f.qty for f in fills) / 600

    block_px = avg_fill("immediate")
    twap_px = avg_fill("twap", slices=4)
    assert twap_px < block_px                      # buys: lower avg price is better
