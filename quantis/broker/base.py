"""Broker abstraction layer (TDD Part 10).

One interface — ``place / cancel / positions / margins`` — implemented
per venue. Paper trading uses ``SimulatedBroker``; Phase 5 adds Zerodha
Kite / Upstox / IBKR adapters behind the SAME interface, so the paper →
live promotion is a constructor swap, never a strategy or risk change.

``place`` is idempotent on ``client_order_id``: a retry after a network
timeout returns the original broker order id instead of double-executing.

``reconcile`` is the TDD's network-partition safeguard: diff the OMS's
local view of positions/open orders against the broker's source of
truth; run after every reconnect and every session start.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..oms import OMS, Fill, ManagedOrder


class BrokerError(RuntimeError):
    pass


class BrokerAdapter(ABC):
    name: str = "abstract"

    @abstractmethod
    def place(self, order: ManagedOrder) -> str:
        """Submit; returns broker_order_id. Idempotent on client_order_id."""

    @abstractmethod
    def cancel(self, broker_order_id: str) -> bool: ...

    @abstractmethod
    def poll_fills(self) -> list[Fill]:
        """Fills since the last poll (each returned exactly once)."""

    @abstractmethod
    def positions(self) -> dict[str, float]:
        """Broker's view: symbol -> net shares."""

    @abstractmethod
    def margins(self) -> dict:
        """Broker's view of available funds."""


@dataclass
class ReconciliationReport:
    position_mismatches: dict[str, dict] = field(default_factory=dict)
    unknown_broker_orders: list[str] = field(default_factory=list)
    stale_local_orders: list[str] = field(default_factory=list)

    @property
    def clean(self) -> bool:
        return (not self.position_mismatches and not self.unknown_broker_orders
                and not self.stale_local_orders)

    def summary(self) -> str:
        if self.clean:
            return "reconciliation clean: local and broker state agree"
        lines = ["RECONCILIATION MISMATCH:"]
        for sym, d in self.position_mismatches.items():
            lines.append(f"  position {sym}: local={d['local']} broker={d['broker']}")
        for oid in self.unknown_broker_orders:
            lines.append(f"  broker order unknown to OMS: {oid}")
        for oid in self.stale_local_orders:
            lines.append(f"  OMS open order unknown to broker: {oid}")
        return "\n".join(lines)


def reconcile(oms: OMS, broker: BrokerAdapter,
              open_broker_order_ids: set[str] | None = None) -> ReconciliationReport:
    report = ReconciliationReport()

    local = oms.positions_from_fills()
    remote = broker.positions()
    for sym in sorted(set(local) | set(remote)):
        lq, rq = local.get(sym, 0.0), remote.get(sym, 0.0)
        if abs(lq - rq) > 1e-6:
            report.position_mismatches[sym] = {"local": lq, "broker": rq}

    if open_broker_order_ids is not None:
        local_open = {o.broker_order_id for o in oms.open_orders() if o.broker_order_id}
        report.unknown_broker_orders = sorted(open_broker_order_ids - local_open)
        report.stale_local_orders = sorted(local_open - open_broker_order_ids)

    return report
