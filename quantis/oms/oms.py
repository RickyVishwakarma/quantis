"""Order Management System — owns order STATE (TDD Part 10).

State machine (exactly the ``orders.status`` enum from the TDD schema):

    PENDING_RISK ──> RISK_REJECTED                (terminal)
    PENDING_RISK ──> APPROVED ──> SENT ──> PARTIALLY_FILLED ──> FILLED
                                   │                │
                                   ├──> CANCELLED   ├──> CANCELLED
                                   └──> ERROR       └──> ERROR

Illegal transitions raise — a fill can never arrive on a rejected order,
an approval can never be granted twice. Every transition is appended to
``orders.jsonl`` (the audit trail); the in-memory book can be rebuilt
from that log after a crash, which is what reconciliation diffs against
the broker's state.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pandas as pd


class OrderStatus(str, Enum):
    PENDING_RISK = "PENDING_RISK"
    RISK_REJECTED = "RISK_REJECTED"
    APPROVED = "APPROVED"
    SENT = "SENT"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    ERROR = "ERROR"


TERMINAL = {OrderStatus.RISK_REJECTED, OrderStatus.FILLED,
            OrderStatus.CANCELLED, OrderStatus.ERROR}

VALID_TRANSITIONS: dict[OrderStatus, set[OrderStatus]] = {
    OrderStatus.PENDING_RISK: {OrderStatus.RISK_REJECTED, OrderStatus.APPROVED},
    OrderStatus.APPROVED: {OrderStatus.SENT, OrderStatus.CANCELLED, OrderStatus.ERROR},
    OrderStatus.SENT: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
                       OrderStatus.CANCELLED, OrderStatus.ERROR},
    OrderStatus.PARTIALLY_FILLED: {OrderStatus.PARTIALLY_FILLED, OrderStatus.FILLED,
                                   OrderStatus.CANCELLED, OrderStatus.ERROR},
}


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    costs: float
    ts: pd.Timestamp


@dataclass
class ManagedOrder:
    symbol: str
    side: str                       # BUY | SELL
    qty: float
    order_type: str = "MARKET"      # MARKET | LIMIT | TWAP
    limit_price: float | None = None
    strategy_id: str = ""
    ref_price: float = 0.0          # decision-time price (risk sizing)
    ts: pd.Timestamp | None = None
    order_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # idempotency key sent to the broker — a retry after a network timeout
    # re-uses it, so the broker can never double-execute (TDD Part 10)
    client_order_id: str = field(default_factory=lambda: f"q-{uuid.uuid4().hex[:16]}")
    status: OrderStatus = OrderStatus.PENDING_RISK
    filled_qty: float = 0.0
    avg_fill_price: float = 0.0
    risk_decision_id: int | None = None
    broker_order_id: str | None = None

    @property
    def remaining_qty(self) -> float:
        return max(self.qty - self.filled_qty, 0.0)

    @property
    def notional(self) -> float:
        return self.qty * self.ref_price


class InvalidTransition(RuntimeError):
    pass


class OMS:
    def __init__(self, journal_dir: str | Path | None = None):
        self.orders: dict[str, ManagedOrder] = {}
        self.fills: list[Fill] = []
        self._journal = None
        if journal_dir is not None:
            journal_dir = Path(journal_dir)
            journal_dir.mkdir(parents=True, exist_ok=True)
            self._journal = journal_dir / "orders.jsonl"

    # ------------------------------------------------------------------
    def submit(self, order: ManagedOrder) -> ManagedOrder:
        if order.order_id in self.orders:
            return self.orders[order.order_id]        # idempotent
        self.orders[order.order_id] = order
        self._log(order, "submitted")
        return order

    def transition(self, order_id: str, new_status: OrderStatus, note: str = "") -> ManagedOrder:
        order = self.orders[order_id]
        allowed = VALID_TRANSITIONS.get(order.status, set())
        if new_status not in allowed:
            raise InvalidTransition(
                f"{order.status.value} -> {new_status.value} is illegal "
                f"(order {order_id[:8]}, {order.symbol})"
            )
        order.status = new_status
        self._log(order, note or new_status.value.lower())
        return order

    def apply_fill(self, fill: Fill) -> ManagedOrder:
        order = self.orders[fill.order_id]
        if order.status not in (OrderStatus.SENT, OrderStatus.PARTIALLY_FILLED):
            raise InvalidTransition(
                f"fill arrived on order in {order.status.value} (order {fill.order_id[:8]})"
            )
        if fill.qty > order.remaining_qty + 1e-9:
            raise InvalidTransition(
                f"overfill: {fill.qty} > remaining {order.remaining_qty}"
            )
        total_val = order.avg_fill_price * order.filled_qty + fill.price * fill.qty
        order.filled_qty += fill.qty
        order.avg_fill_price = total_val / order.filled_qty
        self.fills.append(fill)
        done = order.remaining_qty <= 1e-9
        self.transition(fill.order_id,
                        OrderStatus.FILLED if done else OrderStatus.PARTIALLY_FILLED,
                        note=f"fill {fill.qty}@{fill.price:.2f}")
        return order

    # ------------------------------------------------------------------
    def open_orders(self) -> list[ManagedOrder]:
        return [o for o in self.orders.values() if o.status not in TERMINAL]

    def positions_from_fills(self) -> dict[str, float]:
        """Net share position per symbol implied by the fill history."""
        pos: dict[str, float] = {}
        for f in self.fills:
            delta = f.qty if f.side == "BUY" else -f.qty
            pos[f.symbol] = pos.get(f.symbol, 0.0) + delta
        return {s: q for s, q in pos.items() if abs(q) > 1e-9}

    def fills_frame(self) -> pd.DataFrame:
        if not self.fills:
            return pd.DataFrame(columns=["ts", "symbol", "side", "qty", "price", "costs"])
        return pd.DataFrame([{
            "ts": f.ts, "symbol": f.symbol, "side": f.side,
            "qty": round(f.qty, 4), "price": round(f.price, 2),
            "costs": round(f.costs, 2), "order_id": f.order_id,
        } for f in self.fills])

    # ------------------------------------------------------------------
    def _log(self, order: ManagedOrder, note: str) -> None:
        if self._journal is None:
            return
        rec = {
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "order_id": order.order_id,
            "client_order_id": order.client_order_id,
            "symbol": order.symbol, "side": order.side,
            "qty": order.qty, "order_type": order.order_type,
            "status": order.status.value,
            "filled_qty": order.filled_qty,
            "avg_fill_price": order.avg_fill_price,
            "strategy_id": order.strategy_id,
            "note": note,
        }
        with self._journal.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
