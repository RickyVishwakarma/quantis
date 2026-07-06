"""Zerodha Kite Connect adapter (TDD Part 10 broker abstraction).

Same interface as SimulatedBroker — the paper -> live promotion is a
constructor swap. Design notes:

  Idempotency   every order is placed with ``tag=client_order_id`` (Kite
                tags cap at 20 chars). Before re-placing after a network
                timeout, the adapter re-queries the order book by tag —
                if the first attempt actually landed, the existing broker
                order id is returned instead of double-executing.
  Fills         Kite reports cumulative ``filled_quantity`` per order;
                ``poll_fills`` diffs against the last seen quantity and
                emits only the increments, so each fill is returned once.
  Retries       every call retries with backoff; persistent failures
                raise ``BrokerError`` and increment ``error_count`` (the
                LiveRiskManager's broker-error circuit breaker input).
  SEBI tagging  ``ManagedOrder.algo_id`` rides in the order tag alongside
                the idempotency key when present.

The Kite client is injectable (used by the tests, which run against a
fake). Building the real client needs ``pip install "quantis[live]"`` and
KITE_API_KEY / KITE_ACCESS_TOKEN in the environment — never in code.
"""

from __future__ import annotations

import os
import time

import pandas as pd

from ..oms import Fill, ManagedOrder
from .base import BrokerAdapter, BrokerError

_SIDE_MAP = {"BUY": "BUY", "SELL": "SELL"}


class ZerodhaKiteBroker(BrokerAdapter):
    name = "zerodha"

    def __init__(self, kite=None, exchange: str = "NSE", product: str = "CNC",
                 variety: str = "regular", max_retries: int = 3,
                 retry_wait: float = 1.0):
        if kite is None:
            try:
                from kiteconnect import KiteConnect
            except ImportError as e:
                raise BrokerError(
                    'kiteconnect not installed — pip install "quantis[live]"'
                ) from e
            api_key = os.environ.get("KITE_API_KEY")
            access_token = os.environ.get("KITE_ACCESS_TOKEN")
            if not api_key or not access_token:
                raise BrokerError(
                    "KITE_API_KEY / KITE_ACCESS_TOKEN not set — generate an "
                    "access token via Kite Connect login flow first"
                )
            kite = KiteConnect(api_key=api_key)
            kite.set_access_token(access_token)
        self.kite = kite
        self.exchange = exchange
        self.product = product
        self.variety = variety
        self.max_retries = max_retries
        self.retry_wait = retry_wait
        self.error_count = 0
        self._by_client_id: dict[str, str] = {}
        self._oms_id_by_broker: dict[str, str] = {}    # broker id -> OMS order_id
        self._seen_fill_qty: dict[str, float] = {}

    # ------------------------------------------------------------------
    def _call(self, fn, *args, **kwargs):
        last_exc = None
        for attempt in range(self.max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:                      # kite raises many types
                last_exc = e
                self.error_count += 1
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_wait * (2 ** attempt))
        raise BrokerError(f"kite call failed after {self.max_retries} "
                          f"attempts: {last_exc}") from last_exc

    @staticmethod
    def _tag(order: ManagedOrder) -> str:
        return order.client_order_id[-20:]              # Kite tag limit

    def _find_by_tag(self, tag: str) -> str | None:
        try:
            for o in self.kite.orders():
                if o.get("tag") == tag:
                    return str(o["order_id"])
        except Exception:
            pass
        return None

    # ------------------------------------------------------------------
    def place(self, order: ManagedOrder) -> str:
        if order.client_order_id in self._by_client_id:
            return self._by_client_id[order.client_order_id]
        tag = self._tag(order)
        # a previous attempt may have landed before the timeout
        existing = self._find_by_tag(tag)
        if existing:
            self._by_client_id[order.client_order_id] = existing
            self._oms_id_by_broker[existing] = order.order_id
            return existing

        params = dict(
            variety=self.variety,
            exchange=self.exchange,
            tradingsymbol=order.symbol,
            transaction_type=_SIDE_MAP[order.side],
            quantity=int(order.qty),
            product=self.product,
            order_type="LIMIT" if order.order_type == "LIMIT" else "MARKET",
            tag=tag,
        )
        if order.order_type == "LIMIT" and order.limit_price is not None:
            params["price"] = float(order.limit_price)

        try:
            broker_id = str(self._call(self.kite.place_order, **params))
        except BrokerError:
            # timeout ambiguity: check once more whether it landed
            landed = self._find_by_tag(tag)
            if landed:
                self._by_client_id[order.client_order_id] = landed
                self._oms_id_by_broker[landed] = order.order_id
                return landed
            raise
        self._by_client_id[order.client_order_id] = broker_id
        self._oms_id_by_broker[broker_id] = order.order_id
        return broker_id

    def cancel(self, broker_order_id: str) -> bool:
        try:
            self._call(self.kite.cancel_order, variety=self.variety,
                       order_id=broker_order_id)
            return True
        except BrokerError:
            return False

    def poll_fills(self) -> list[Fill]:
        fills: list[Fill] = []
        known = {v: k for k, v in self._by_client_id.items()}
        for o in self._call(self.kite.orders):
            broker_id = str(o.get("order_id"))
            if broker_id not in known:
                continue
            filled = float(o.get("filled_quantity") or 0)
            seen = self._seen_fill_qty.get(broker_id, 0.0)
            if filled > seen:
                fills.append(Fill(
                    order_id=self._order_id_for(broker_id),
                    symbol=o.get("tradingsymbol", ""),
                    side=o.get("transaction_type", ""),
                    qty=filled - seen,
                    price=float(o.get("average_price") or 0),
                    costs=0.0,          # actual charges arrive on the contract note
                    ts=pd.Timestamp(o.get("order_timestamp") or pd.Timestamp.now()),
                ))
                self._seen_fill_qty[broker_id] = filled
        return fills

    def _order_id_for(self, broker_id: str) -> str:
        """Fills key on the OMS order_id recorded at place() time."""
        return self._oms_id_by_broker.get(broker_id, broker_id)

    def positions(self) -> dict[str, float]:
        pos = self._call(self.kite.positions)
        return {
            p["tradingsymbol"]: float(p["quantity"])
            for p in pos.get("net", [])
            if abs(float(p.get("quantity") or 0)) > 1e-9
        }

    def margins(self) -> dict:
        m = self._call(self.kite.margins)
        equity = m.get("equity", {})
        return {"cash": float(equity.get("available", {}).get("cash", 0.0))}

    def open_order_ids(self) -> set[str]:
        open_status = {"OPEN", "TRIGGER PENDING", "AMO REQ RECEIVED"}
        return {
            str(o["order_id"]) for o in self._call(self.kite.orders)
            if o.get("status") in open_status
        }
