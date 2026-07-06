"""Dry-run broker: the safety interlock between paper and live.

Wraps a real adapter but NEVER forwards an order. ``place`` journals the
would-be order and returns a synthetic id; ``poll_fills`` is always empty
(nothing was executed, so nothing fills). Read-only calls (positions,
margins) pass through to the real broker when one is wrapped — so a
dry-run session exercises real connectivity, real reconciliation, and
real market data without a single rupee at risk.

The live engine wraps any real broker in this automatically unless the
session is explicitly armed (``--arm-live``).
"""

from __future__ import annotations

import itertools

from ..oms import Fill, ManagedOrder
from .base import BrokerAdapter

_dry_seq = itertools.count(1)


class DryRunBroker(BrokerAdapter):
    name = "dryrun"

    def __init__(self, inner: BrokerAdapter | None = None):
        self.inner = inner
        self.would_be_orders: list[dict] = []
        self._by_client_id: dict[str, str] = {}

    def place(self, order: ManagedOrder) -> str:
        if order.client_order_id in self._by_client_id:
            return self._by_client_id[order.client_order_id]
        dry_id = f"DRY-{next(_dry_seq):08d}"
        self.would_be_orders.append({
            "dry_id": dry_id, "symbol": order.symbol, "side": order.side,
            "qty": order.qty, "order_type": order.order_type,
            "limit_price": order.limit_price, "ref_price": order.ref_price,
            "algo_id": order.algo_id, "ts": str(order.ts),
        })
        self._by_client_id[order.client_order_id] = dry_id
        return dry_id

    def cancel(self, broker_order_id: str) -> bool:
        return True

    def poll_fills(self) -> list[Fill]:
        return []                                   # nothing executed, ever

    def positions(self) -> dict[str, float]:
        return self.inner.positions() if self.inner else {}

    def margins(self) -> dict:
        return self.inner.margins() if self.inner else {"cash": 0.0}

    def open_order_ids(self) -> set[str]:
        return set()
