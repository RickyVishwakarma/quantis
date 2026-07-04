"""Simulated Broker Adapter (TDD Part 4: the paper-trading swap point).

Fills pending orders against the next market-data bar using the SAME
``NSECostModel`` (slippage + full charge schedule) as the backtester:

  MARKET      fills at next bar's open, slippage-adjusted
  LIMIT       BUY fills iff next open <= limit; SELL iff next open >= limit
              (unfilled limit orders stay pending until cancelled)

Cash and holdings are tracked broker-side so ``positions()`` / ``margins()``
give the independent source of truth reconciliation diffs against.
"""

from __future__ import annotations

import itertools

import pandas as pd

from ..backtest.costs import NSECostModel
from ..oms import Fill, ManagedOrder
from .base import BrokerAdapter, BrokerError

_broker_seq = itertools.count(1)


class SimulatedBroker(BrokerAdapter):
    name = "sim"

    def __init__(self, cost_model: NSECostModel | None = None,
                 starting_cash: float = 1_000_000.0,
                 allow_short: bool = False):
        self.costs = cost_model or NSECostModel()
        self.cash = starting_cash
        self.holdings: dict[str, float] = {}
        self.allow_short = allow_short
        self._pending: dict[str, ManagedOrder] = {}       # broker_order_id -> order
        self._by_client_id: dict[str, str] = {}           # idempotency map
        self._fill_queue: list[Fill] = []
        self._adv: dict[str, float] = {}

    # ------------------------------------------------------------------
    # BrokerAdapter interface
    # ------------------------------------------------------------------
    def place(self, order: ManagedOrder) -> str:
        if order.client_order_id in self._by_client_id:      # retry-safe
            return self._by_client_id[order.client_order_id]
        if order.qty <= 0:
            raise BrokerError("qty must be positive")
        broker_id = f"SIM-{next(_broker_seq):08d}"
        self._pending[broker_id] = order
        self._by_client_id[order.client_order_id] = broker_id
        return broker_id

    def cancel(self, broker_order_id: str) -> bool:
        return self._pending.pop(broker_order_id, None) is not None

    def poll_fills(self) -> list[Fill]:
        out, self._fill_queue = self._fill_queue, []
        return out

    def positions(self) -> dict[str, float]:
        return {s: q for s, q in self.holdings.items() if abs(q) > 1e-9}

    def margins(self) -> dict:
        return {"cash": self.cash}

    def open_order_ids(self) -> set[str]:
        return set(self._pending)

    def pending_orders(self) -> dict[str, "ManagedOrder"]:
        return dict(self._pending)

    # ------------------------------------------------------------------
    # Market simulation
    # ------------------------------------------------------------------
    def set_adv(self, adv: dict[str, float]) -> None:
        self._adv = dict(adv)

    def on_bar(self, ts: pd.Timestamp, open_prices: dict[str, float]) -> int:
        """Attempt to fill every pending order at this bar's open."""
        filled = 0
        for broker_id in list(self._pending):
            order = self._pending[broker_id]
            px = open_prices.get(order.symbol)
            if px is None or pd.isna(px) or px <= 0:
                continue
            if order.order_type == "LIMIT" and order.limit_price is not None:
                crosses = (px <= order.limit_price if order.side == "BUY"
                           else px >= order.limit_price)
                if not crosses:
                    continue

            qty = order.remaining_qty
            adv = self._adv.get(order.symbol, 0.0)
            fill_px = self.costs.fill_price(order.side, px, qty * px, adv)
            notional = qty * fill_px
            charges = self.costs.charges(order.side, notional)

            if order.side == "BUY":
                if self.cash < notional + charges:
                    continue                           # insufficient funds; stays pending
                self.cash -= notional + charges
                self.holdings[order.symbol] = self.holdings.get(order.symbol, 0.0) + qty
            else:
                held = self.holdings.get(order.symbol, 0.0)
                if not self.allow_short and qty > held + 1e-9:
                    continue                           # can't short in the MVP
                self.cash += notional - charges
                self.holdings[order.symbol] = held - qty
                if abs(self.holdings[order.symbol]) < 1e-9:
                    self.holdings.pop(order.symbol, None)

            self._fill_queue.append(Fill(
                order_id=order.order_id, symbol=order.symbol, side=order.side,
                qty=qty, price=fill_px, costs=charges, ts=ts,
            ))
            del self._pending[broker_id]
            filled += 1
        return filled
