"""Execution Management System — owns order execution STRATEGY (TDD Part 10).

The OMS tracks what state an order is in; the EMS decides HOW a
risk-approved parent order reaches the market:

  immediate   one market child order, filled at the next bar
  twap        parent split into N equal child slices released on
              consecutive bars — each slice's ADV participation is
              1/N of the block's, so the square-root impact model
              prices each child tighter than the single block

Child orders inherit the parent's ``strategy_id``; parent state in the
OMS aggregates from child fills. Daily-bar MVP: a "slice interval" is
one bar; intraday slicing arrives with intraday data in later phases.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..broker.base import BrokerAdapter, BrokerError
from ..oms import OMS, ManagedOrder, OrderStatus


@dataclass
class _ScheduledSlice:
    order: ManagedOrder
    bars_until_release: int


@dataclass
class EMS:
    oms: OMS
    broker: BrokerAdapter
    error_count: int = 0
    _queue: list[_ScheduledSlice] = field(default_factory=list)

    def execute(self, parent: ManagedOrder, algo: str = "immediate",
                slices: int = 4) -> list[ManagedOrder]:
        """Slice a risk-APPROVED parent and route children to the broker."""
        if parent.status != OrderStatus.APPROVED:
            raise BrokerError(
                f"EMS refuses order in {parent.status.value}; only APPROVED "
                "orders may be executed (risk gate is upstream, not optional)"
            )
        if algo == "immediate" or parent.qty <= 0:
            children = [parent]                      # parent is its own child
            self._route(parent, delay=0)
        elif algo == "twap":
            self.oms.transition(parent.order_id, OrderStatus.SENT, note="twap parent")
            qty_each = parent.qty / slices
            children = []
            for i in range(slices):
                child = ManagedOrder(
                    symbol=parent.symbol, side=parent.side, qty=qty_each,
                    order_type="MARKET", strategy_id=parent.strategy_id,
                    ref_price=parent.ref_price, ts=parent.ts,
                )
                child.status = OrderStatus.APPROVED   # covered by parent's approval
                self.oms.submit(child)
                self._route(child, delay=i)
                children.append(child)
        else:
            raise BrokerError(f"unknown execution algo {algo!r}")
        return children

    def _route(self, order: ManagedOrder, delay: int) -> None:
        if delay <= 0:
            self._send(order)
        else:
            self._queue.append(_ScheduledSlice(order, delay))

    def _send(self, order: ManagedOrder) -> None:
        try:
            order.broker_order_id = self.broker.place(order)
            self.oms.transition(order.order_id, OrderStatus.SENT)
        except BrokerError:
            self.error_count += 1
            self.oms.transition(order.order_id, OrderStatus.ERROR, note="broker place failed")

    def on_bar(self) -> None:
        """Advance the slice schedule by one bar; release due children."""
        due, rest = [], []
        for s in self._queue:
            s.bars_until_release -= 1
            (due if s.bars_until_release <= 0 else rest).append(s)
        self._queue = rest
        for s in due:
            self._send(s.order)
