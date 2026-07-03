"""Risk engine with unconditional veto authority.

Every order — regardless of source (template strategy today, AI signal
later) — transits ``RiskEngine.evaluate`` before it can reach execution.
There is no bypass path. Each evaluation emits an immutable
``RiskDecision`` carrying the breached rule and a snapshot of the limits
in force, which the backtester persists as the audit trail (mirrors the
``risk_decisions`` table in the TDD's Postgres schema).

Rule ordering: cheap sanity bounds first, then halts (drawdown / daily
loss), then concentration, exposure, and liquidity. Risk-REDUCING orders
(sells in the long-only MVP) are exempt from halts and concentration
rules — a risk engine must never block you from getting smaller.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import pandas as pd

from ..data.universe import sector
from .limits import RiskLimits

_order_seq = itertools.count(1)


@dataclass
class Order:
    symbol: str
    side: str            # BUY | SELL
    qty: float
    ref_price: float     # decision-time reference price
    ts: pd.Timestamp
    order_id: int = field(default_factory=lambda: next(_order_seq))

    @property
    def notional(self) -> float:
        return self.qty * self.ref_price


@dataclass
class PortfolioState:
    """Marked-to-market snapshot the risk engine judges an order against."""
    equity: float
    positions: dict[str, float]        # symbol -> market value
    peak_equity: float
    prev_day_return: float             # yesterday's close-to-close equity return
    adv: dict[str, float]              # symbol -> 20d rupee ADV

    @property
    def gross_exposure(self) -> float:
        return sum(abs(v) for v in self.positions.values())

    def sector_value(self, sec: str) -> float:
        return sum(abs(v) for s, v in self.positions.items() if sector(s) == sec)


@dataclass(frozen=True)
class RiskDecision:
    order_id: int
    ts: pd.Timestamp
    symbol: str
    side: str
    notional: float
    outcome: str                 # APPROVE | REJECT
    breached_rule: str | None
    limit_snapshot: dict

    @property
    def approved(self) -> bool:
        return self.outcome == "APPROVE"


class RiskEngine:
    def __init__(self, limits: RiskLimits | None = None):
        self.limits = limits or RiskLimits()
        self.decisions: list[RiskDecision] = []

    def evaluate(self, order: Order, state: PortfolioState) -> RiskDecision:
        breached = self._check(order, state)
        decision = RiskDecision(
            order_id=order.order_id,
            ts=order.ts,
            symbol=order.symbol,
            side=order.side,
            notional=order.notional,
            outcome="REJECT" if breached else "APPROVE",
            breached_rule=breached,
            limit_snapshot=self.limits.snapshot(),
        )
        self.decisions.append(decision)
        return decision

    def _check(self, order: Order, state: PortfolioState) -> str | None:
        lim = self.limits
        equity = state.equity
        if equity <= 0:
            return "equity_non_positive"
        if order.qty <= 0 or order.ref_price <= 0:
            return "invalid_order"

        reduces_risk = order.side == "SELL"  # long-only MVP

        # Sanity bound applies to everything, including exits
        if order.notional > lim.max_order_notional_pct * equity:
            return "max_order_notional_pct"

        # Liquidity applies to everything — an illiquid exit is still a fill problem
        sym_adv = state.adv.get(order.symbol, 0.0)
        if sym_adv > 0 and order.notional > lim.max_adv_participation * sym_adv:
            return "max_adv_participation"
        if sym_adv <= 0 and not reduces_risk:
            return "no_liquidity_data"

        if reduces_risk:
            return None

        # Halts: block new risk after a loss event
        drawdown = 1.0 - equity / state.peak_equity if state.peak_equity > 0 else 0.0
        if drawdown > lim.max_drawdown:
            return "max_drawdown_halt"
        if state.prev_day_return < -lim.max_daily_loss:
            return "max_daily_loss_halt"

        # Concentration
        pos_after = abs(state.positions.get(order.symbol, 0.0)) + order.notional
        if pos_after > lim.max_position_weight * equity:
            return "max_position_weight"
        sec_after = state.sector_value(sector(order.symbol)) + order.notional
        if sec_after > lim.max_sector_weight * equity:
            return "max_sector_weight"

        # Exposure
        if state.gross_exposure + order.notional > lim.max_gross_exposure * equity:
            return "max_gross_exposure"

        return None

    def decisions_frame(self) -> pd.DataFrame:
        if not self.decisions:
            return pd.DataFrame(
                columns=["order_id", "ts", "symbol", "side", "notional",
                         "outcome", "breached_rule"]
            )
        return pd.DataFrame([
            {"order_id": d.order_id, "ts": d.ts, "symbol": d.symbol,
             "side": d.side, "notional": round(d.notional, 2),
             "outcome": d.outcome, "breached_rule": d.breached_rule}
            for d in self.decisions
        ])
