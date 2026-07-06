"""Paper trading engine (TDD Phase 3).

The live loop, one bar at a time:

    bar arrives ─> broker fills pending orders at bar open ─> OMS updated
    ─> portfolio marked ─> strategy decides target weights from history
    through this close (same Strategy code path as the backtester)
    ─> vol-target + tier sizing scale the targets ─> rebalance orders
    ─> EVERY order transits LiveRiskManager (veto + breakers) ─> OMS
    ─> EMS routes (immediate / TWAP) ─> broker holds until next bar.

Decision at close t, fill at open t+1 — the identical timing convention
as ``EventBacktester``, which is what makes the backtest-parity check
meaningful: same weights, same costs, same risk gate, different runtime.

Every session persists its journal (orders.jsonl), fills, equity curve,
risk decisions, and reconciliation report to ``paper_sessions/<name>/``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..backtest.costs import NSECostModel
from ..backtest.metrics import compute_metrics
from ..broker import SimulatedBroker, reconcile
from ..ems import EMS
from ..features import compute_features
from ..feed import Bar, MarketDataFeed
from ..oms import OMS, ManagedOrder, OrderStatus
from ..risk import LiveRiskManager, Order, PortfolioState, RiskLimits
from ..strategies.base import Strategy


@dataclass
class PaperSession:
    name: str
    strategy: str
    params: dict
    equity: pd.Series
    fills: pd.DataFrame
    risk_decisions: pd.DataFrame
    final_positions: dict[str, float]
    risk_status: dict
    reconciliation: str
    session_dir: Path | None = None
    metrics: dict = field(default_factory=dict)


class PaperTradingEngine:
    def __init__(
        self,
        strategy: Strategy,
        initial_capital: float = 1_000_000.0,
        cost_model: NSECostModel | None = None,
        risk_limits: RiskLimits | None = None,
        execution_algo: str = "immediate",
        twap_slices: int = 4,
        min_trade_notional: float = 5_000.0,
        vol_targeting: bool = False,
        session_dir: str | Path | None = None,
        history_window: int | None = None,   # cap bars kept; None = full history
        broker=None,                          # injected adapter; default sim
        algo_id: str = "",                    # SEBI algo tag on every order
    ):                                        # (full history = exact backtest parity)
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.costs = cost_model or NSECostModel()
        self.risk = LiveRiskManager(risk_limits or RiskLimits())
        self.execution_algo = execution_algo
        self.twap_slices = twap_slices
        self.min_trade_notional = min_trade_notional
        self.vol_targeting = vol_targeting
        self.session_dir = Path(session_dir) if session_dir else None
        self.algo_id = algo_id

        self.oms = OMS(journal_dir=self.session_dir)
        self.broker = broker or SimulatedBroker(cost_model=self.costs,
                                                starting_cash=initial_capital)
        self.ems = EMS(oms=self.oms, broker=self.broker)

        self.history_window = history_window
        self._history: list[Bar] = []
        self._bars_seen = 0
        self._equity_hist: list[tuple[pd.Timestamp, float]] = []
        self._peak = initial_capital
        self._prev_equity = initial_capital
        self._prev_day_ret = 0.0

    # ------------------------------------------------------------------
    def run(self, feed: MarketDataFeed, warmup_bars: int = 210) -> PaperSession:
        """Consume the feed to exhaustion (replay) or interruption (live)."""
        for bar in feed:
            self.on_bar(bar, warmup_bars=warmup_bars)
            self.risk.on_feed_staleness(feed.staleness_seconds())
            self.risk.on_broker_errors(self.ems.error_count)
        return self.finalize()

    def on_bar(self, bar: Bar, warmup_bars: int = 210) -> None:
        # 1. Fills first: pending orders execute at THIS bar's open
        self._bars_seen += 1
        self.broker.on_bar(bar.ts, bar.open.to_dict())
        for fill in self.broker.poll_fills():
            self.oms.apply_fill(fill)
            self._on_fill(fill)
        # Day-order semantics: whatever didn't fill at its bar is cancelled
        # (matches the backtester, which never carries an order overnight);
        # LIMIT orders are the exception — they rest until crossed.
        for broker_id, pending in self.broker.pending_orders().items():
            if pending.order_type != "LIMIT":
                self.broker.cancel(broker_id)
                if pending.status not in (OrderStatus.CANCELLED,):
                    self.oms.transition(pending.order_id, OrderStatus.CANCELLED,
                                        note="day order expired unfilled")
        # Release due TWAP slices AFTER the fill pass: a slice released on
        # bar t rests with the broker and executes at bar t+1's open.
        self.ems.on_bar()

        self._history.append(bar)
        if self.history_window and len(self._history) > self.history_window:
            self._history = self._history[-self.history_window:]
        if self._bars_seen < warmup_bars:
            self._mark_equity(bar)
            return

        # 2. Decide from history through this close (identical code path)
        wide = self._wide()
        panel = compute_features(wide)
        weights = self.strategy.target_weights(panel).iloc[-1].fillna(0.0).clip(lower=0.0)

        equity = self._mark_equity(bar)
        adv_row = panel["adv_20"].iloc[-1].fillna(0.0).to_dict()
        state = self._state(bar, equity, adv_row)

        # 3. Defensive overlays can only shrink the book
        scale = self.risk.size_factor(state)
        if self.vol_targeting:
            scale *= self.risk.vol_scale(
                pd.Series([e for _, e in self._equity_hist]).pct_change().to_numpy()
            )
        targets = weights * scale
        if self.risk.tier(state).value == "FLATTEN":
            targets = weights * 0.0                # exits only

        # 4. Rebalance orders: sells first so freed cash funds buys
        holdings = self.broker.positions()
        orders: list[ManagedOrder] = []
        for sym in weights.index:
            px = bar.close.get(sym)
            if px is None or pd.isna(px) or px <= 0:
                continue
            target_val = float(targets.get(sym, 0.0)) * equity
            cur_val = holdings.get(sym, 0.0) * px
            delta = target_val - cur_val
            if abs(delta) < self.min_trade_notional:
                continue
            qty = abs(delta) / px
            if delta < 0:
                qty = min(qty, holdings.get(sym, 0.0))
                if qty <= 0:
                    continue
            orders.append(ManagedOrder(
                symbol=sym, side="BUY" if delta > 0 else "SELL", qty=qty,
                ref_price=float(px), ts=bar.ts,
                strategy_id=self.strategy.describe(),
                algo_id=self.algo_id,
            ))
        orders.sort(key=lambda o: 0 if o.side == "SELL" else 1)

        # 5. Risk gate -> OMS -> EMS -> broker (fills at next bar's open)
        if hasattr(self.broker, "set_adv"):
            self.broker.set_adv(adv_row)
        for mo in orders:
            self._gate_and_route(mo, state)

    def _gate_and_route(self, mo: ManagedOrder, state: PortfolioState):
        """Risk gate one order, then hand approved orders to the EMS.

        Overridable hook: the live engine wraps this with audit logging."""
        self.oms.submit(mo)
        decision = self.risk.evaluate(
            Order(symbol=mo.symbol, side=mo.side, qty=mo.qty,
                  ref_price=mo.ref_price, ts=mo.ts),
            state,
        )
        mo.risk_decision_id = decision.order_id
        if not decision.approved:
            self.oms.transition(mo.order_id, OrderStatus.RISK_REJECTED,
                                note=decision.breached_rule or "rejected")
            return decision
        self.oms.transition(mo.order_id, OrderStatus.APPROVED)
        self.ems.execute(mo, algo=self.execution_algo, slices=self.twap_slices)
        return decision

    def _on_fill(self, fill) -> None:
        """Overridable hook: the live engine audits every fill."""

    # ------------------------------------------------------------------
    def finalize(self) -> PaperSession:
        equity = pd.Series(dict(self._equity_hist), name="equity", dtype=float).sort_index()
        recon = reconcile(self.oms, self.broker,
                          open_broker_order_ids=self.broker.open_order_ids())
        metrics = compute_metrics(equity) if len(equity) > 2 else {}
        session = PaperSession(
            name=self.session_dir.name if self.session_dir else "adhoc",
            strategy=self.strategy.name,
            params=self.strategy.params,
            equity=equity,
            fills=self.oms.fills_frame(),
            risk_decisions=self.risk.decisions_frame(),
            final_positions=self.broker.positions(),
            risk_status=self.risk.status(),
            reconciliation=recon.summary(),
            session_dir=self.session_dir,
            metrics=metrics,
        )
        if self.session_dir:
            self._persist(session)
        return session

    # ------------------------------------------------------------------
    def _wide(self) -> dict[str, pd.DataFrame]:
        idx = [b.ts for b in self._history]
        return {
            fldname: pd.DataFrame([getattr(b, fldname) for b in self._history], index=idx)
            for fldname in ["open", "high", "low", "close", "volume"]
        }

    def _broker_cash(self) -> float:
        cash = getattr(self.broker, "cash", None)     # sim tracks it directly
        if cash is None:
            cash = self.broker.margins().get("cash", 0.0)
        return float(cash)

    def _mark_equity(self, bar: Bar) -> float:
        holdings = self.broker.positions()
        pos_val = sum(q * bar.close.get(s, float("nan"))
                      for s, q in holdings.items()
                      if not pd.isna(bar.close.get(s, float("nan"))))
        equity = self._broker_cash() + pos_val
        self._equity_hist.append((bar.ts, equity))
        self._prev_day_ret = (equity / self._prev_equity - 1
                              if self._prev_equity > 0 else 0.0)
        self._prev_equity = equity
        self._peak = max(self._peak, equity)
        return equity

    def _state(self, bar: Bar, equity: float, adv_row: dict) -> PortfolioState:
        holdings = self.broker.positions()
        return PortfolioState(
            equity=equity,
            positions={s: q * float(bar.close.get(s, 0.0)) for s, q in holdings.items()},
            peak_equity=self._peak,
            prev_day_return=self._prev_day_ret,
            adv=adv_row,
        )

    def _persist(self, session: PaperSession) -> None:
        d = self.session_dir
        d.mkdir(parents=True, exist_ok=True)
        session.equity.to_csv(d / "equity.csv")
        session.fills.to_csv(d / "fills.csv", index=False)
        session.risk_decisions.to_csv(d / "risk_decisions.csv", index=False)
        (d / "session.json").write_text(json.dumps({
            "strategy": session.strategy,
            "params": session.params,
            "execution_algo": self.execution_algo,
            "final_positions": session.final_positions,
            "cash": self._broker_cash(),
            "risk_status": session.risk_status,
            "metrics": {k: v for k, v in session.metrics.items()
                        if isinstance(v, (int, float, str))},
        }, indent=2, default=str))
        (d / "reconciliation.txt").write_text(session.reconciliation, encoding="utf-8")
