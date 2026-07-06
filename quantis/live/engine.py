"""Live trading engine (TDD Phase 5).

Extends the paper engine — same strategy code, same risk gate, same
OMS/EMS — and adds what only matters when the broker is real:

  Arming interlock    a real broker is auto-wrapped in DryRunBroker
                      unless ``armed=True``. An armed session refuses to
                      start without a SEBI ``algo_id`` (order tagging is
                      regulatory, not optional).
  Audit trail         every risk decision, order transition, fill,
                      breaker event, and reconciliation result lands in
                      the hash-chained AuditLog (Part 13).
  Reconciliation      at session start, every ``reconcile_every`` bars,
                      and at EOD (Part 10: market-open / reconnect / EOD).
                      A mismatch trips the circuit breaker — trading
                      halts until a human investigates and resets.
  Breaker response    when any breaker trips, all resting orders are
                      cancelled immediately (Part 8: open orders
                      auto-cancelled) and the event is audited.
"""

from __future__ import annotations

from pathlib import Path

from ..broker import DryRunBroker, SimulatedBroker, reconcile
from ..audit import AuditLog
from ..feed import Bar, MarketDataFeed
from ..oms import ManagedOrder
from ..paper.engine import PaperSession, PaperTradingEngine
from ..risk import PortfolioState


class LiveTradingEngine(PaperTradingEngine):
    def __init__(self, *, broker=None, armed: bool = False,
                 algo_id: str = "", session_dir: str | Path | None = None,
                 reconcile_every: int = 20, **kw):
        broker = broker or SimulatedBroker(
            starting_cash=kw.get("initial_capital", 1_000_000.0))
        self.is_simulated = isinstance(broker, (SimulatedBroker, DryRunBroker))

        if not self.is_simulated and not armed:
            broker = DryRunBroker(inner=broker)      # safety interlock
            self.is_simulated = True
        if not self.is_simulated and armed and not algo_id:
            raise ValueError(
                "an ARMED live session requires an algo_id — SEBI order "
                "tagging is mandatory for algorithmic orders"
            )

        super().__init__(broker=broker, algo_id=algo_id,
                         session_dir=session_dir, **kw)
        self.armed = armed and not isinstance(broker, DryRunBroker)
        self.reconcile_every = reconcile_every
        audit_path = (Path(session_dir) / "audit.jsonl") if session_dir \
            else Path("live_sessions") / "adhoc_audit.jsonl"
        self.audit = AuditLog(audit_path)
        self._breaker_was_tripped = False

    # ------------------------------------------------------------------
    def run(self, feed: MarketDataFeed, warmup_bars: int = 210) -> PaperSession:
        self.audit.append("session_start", {
            "strategy": self.strategy.describe(),
            "broker": self.broker.name,
            "armed": self.armed,
            "algo_id": self.algo_id,
            "initial_capital": self.initial_capital,
            "limits": self.risk.limits.snapshot(),
        })
        self._reconcile("session_start")
        session = super().run(feed, warmup_bars=warmup_bars)
        self._reconcile("eod")
        ok, bad_seq = self.audit.verify()
        self.audit.append("session_end", {
            "final_equity": float(session.equity.iloc[-1]) if len(session.equity) else None,
            "risk_status": session.risk_status,
            "chain_verified_before_this_record": ok,
            "first_bad_seq": bad_seq,
        })
        return session

    def on_bar(self, bar: Bar, warmup_bars: int = 210) -> None:
        super().on_bar(bar, warmup_bars=warmup_bars)
        if self._bars_seen % self.reconcile_every == 0 and self._bars_seen > warmup_bars:
            self._reconcile(f"periodic@{bar.ts.date()}")
        self._check_breaker(bar)

    # ------------------------------------------------------------------
    def _gate_and_route(self, mo: ManagedOrder, state: PortfolioState):
        decision = super()._gate_and_route(mo, state)
        self.audit.append("risk_decision", {
            "order_id": mo.order_id,
            "client_order_id": mo.client_order_id,
            "symbol": mo.symbol, "side": mo.side, "qty": mo.qty,
            "ref_price": mo.ref_price,
            "algo_id": mo.algo_id,
            "outcome": decision.outcome,
            "breached_rule": decision.breached_rule,
            "order_status": mo.status.value,
            "broker_order_id": mo.broker_order_id,
        })
        return decision

    def _on_fill(self, fill) -> None:
        self.audit.append("fill", {
            "order_id": fill.order_id, "symbol": fill.symbol,
            "side": fill.side, "qty": fill.qty, "price": fill.price,
            "costs": fill.costs, "ts": str(fill.ts),
        })

    # ------------------------------------------------------------------
    def _reconcile(self, when: str) -> None:
        report = reconcile(self.oms, self.broker,
                           open_broker_order_ids=self.broker.open_order_ids())
        self.audit.append("reconciliation", {
            "when": when, "clean": report.clean,
            "position_mismatches": report.position_mismatches,
            "unknown_broker_orders": report.unknown_broker_orders,
            "stale_local_orders": report.stale_local_orders,
        })
        # DryRunBroker never fills, so OMS-vs-broker divergence is expected
        # there; only a REAL book disagreeing with the OMS is an incident.
        if not report.clean and self.armed:
            self.risk.trip(f"reconciliation mismatch at {when}")

    def _check_breaker(self, bar: Bar) -> None:
        tripped = self.risk.breaker.tripped
        if tripped and not self._breaker_was_tripped:
            cancelled = []
            for broker_id in list(self.broker.open_order_ids()):
                if self.broker.cancel(broker_id):
                    cancelled.append(broker_id)
            self.audit.append("circuit_breaker", {
                "reason": self.risk.breaker.reason,
                "ts": str(bar.ts),
                "open_orders_cancelled": cancelled,
                "action_required": "manual reset (quantis risk engine .reset())",
            })
        self._breaker_was_tripped = tripped
