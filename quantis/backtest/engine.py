"""Event-driven daily backtester.

Timeline per bar t:
  1. Strategy decided target weights at close of t-1 (point-in-time).
  2. At open of t, portfolio is marked and rebalance orders are generated.
  3. EVERY order transits the RiskEngine; rejected orders do not execute.
  4. Approved orders fill at open of t adjusted for slippage, and pay the
     full NSE charge schedule.
  5. Equity is marked at close of t.

The engine consumes the same weights frame as the vectorized engine, so a
strategy cannot diverge between research sweeps and simulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..features import FeaturePanel, compute_features
from ..risk import Order, PortfolioState, RiskEngine, RiskLimits
from ..strategies.base import Strategy
from .costs import NSECostModel


@dataclass
class BacktestResult:
    equity: pd.Series
    trades: pd.DataFrame
    risk_decisions: pd.DataFrame
    weights: pd.DataFrame
    strategy: str
    params: dict = field(default_factory=dict)

    @property
    def n_rejected(self) -> int:
        if len(self.risk_decisions) == 0:
            return 0
        return int((self.risk_decisions["outcome"] == "REJECT").sum())


class EventBacktester:
    def __init__(
        self,
        initial_capital: float = 1_000_000.0,
        cost_model: NSECostModel | None = None,
        risk_limits: RiskLimits | None = None,
        min_trade_notional: float = 5_000.0,   # ignore dust rebalances
    ):
        self.initial_capital = initial_capital
        self.costs = cost_model or NSECostModel()
        self.risk_limits = risk_limits or RiskLimits()
        self.min_trade_notional = min_trade_notional

    def run(self, wide: dict[str, pd.DataFrame], strategy: Strategy) -> BacktestResult:
        panel = compute_features(wide)
        weights = strategy.target_weights(panel).fillna(0.0).clip(lower=0.0)
        return self.run_weights(panel, weights, strategy.describe(), strategy.params)

    def run_weights(
        self,
        panel: FeaturePanel,
        weights: pd.DataFrame,
        strategy_name: str = "external",
        params: dict | None = None,
    ) -> BacktestResult:
        open_px, close_px = panel.open, panel.close
        adv = panel["adv_20"]
        dates = weights.index

        risk = RiskEngine(self.risk_limits)
        cash = self.initial_capital
        shares: dict[str, float] = {}
        cost_basis: dict[str, float] = {}   # avg entry price per open position
        equity_hist: list[tuple[pd.Timestamp, float]] = []
        trades: list[dict] = []
        peak = self.initial_capital
        prev_equity = self.initial_capital
        prev_day_ret = 0.0

        for i in range(1, len(dates)):
            t, t_prev = dates[i], dates[i - 1]
            opens = open_px.loc[t]
            target_w = weights.loc[t_prev]

            # Mark portfolio at today's open
            pos_val = {s: q * opens[s] for s, q in shares.items()
                       if q > 0 and pd.notna(opens.get(s))}
            equity = cash + sum(pos_val.values())
            adv_row = adv.loc[t_prev].fillna(0.0).to_dict()

            state = PortfolioState(
                equity=equity,
                positions=pos_val,
                peak_equity=peak,
                prev_day_return=prev_day_ret,
                adv=adv_row,
            )

            # Build rebalance orders: sells first so freed cash funds buys
            orders: list[Order] = []
            for sym in weights.columns:
                px = opens.get(sym)
                if pd.isna(px) or px <= 0:
                    continue
                target_val = float(target_w.get(sym, 0.0)) * equity
                cur_val = pos_val.get(sym, 0.0)
                delta = target_val - cur_val
                if abs(delta) < self.min_trade_notional:
                    continue
                qty = abs(delta) / px
                if delta < 0:
                    qty = min(qty, shares.get(sym, 0.0))
                    if qty <= 0:
                        continue
                side = "BUY" if delta > 0 else "SELL"
                orders.append(Order(symbol=sym, side=side, qty=qty, ref_price=px, ts=t))
            orders.sort(key=lambda o: 0 if o.side == "SELL" else 1)

            for order in orders:
                decision = risk.evaluate(order, state)   # no bypass path
                if not decision.approved:
                    continue
                sym_adv = adv_row.get(order.symbol, 0.0)
                fill_px = self.costs.fill_price(order.side, order.ref_price,
                                                order.notional, sym_adv)
                notional = order.qty * fill_px
                charges = self.costs.charges(order.side, notional)
                realized = None
                if order.side == "BUY":
                    if cash < notional + charges:   # never lever up on rounding
                        continue
                    cash -= notional + charges
                    old_q = shares.get(order.symbol, 0.0)
                    old_b = cost_basis.get(order.symbol, 0.0)
                    shares[order.symbol] = old_q + order.qty
                    cost_basis[order.symbol] = (
                        (old_q * old_b + order.qty * fill_px) / (old_q + order.qty)
                    )
                else:
                    cash += notional - charges
                    realized = (fill_px - cost_basis.get(order.symbol, fill_px)) * order.qty - charges
                    shares[order.symbol] = shares.get(order.symbol, 0.0) - order.qty
                    if shares[order.symbol] <= 1e-9:
                        shares.pop(order.symbol, None)
                        cost_basis.pop(order.symbol, None)
                trades.append({
                    "ts": t, "symbol": order.symbol, "side": order.side,
                    "qty": round(order.qty, 4), "fill_price": round(fill_px, 2),
                    "notional": round(notional, 2), "costs": round(charges, 2),
                    "realized_pnl": None if realized is None else round(realized, 2),
                })
                # Keep state current so later orders in the same bar see
                # the updated book (sequential risk, not stale snapshot)
                pos_val = {s: q * opens[s] for s, q in shares.items()
                           if pd.notna(opens.get(s))}
                state = PortfolioState(
                    equity=cash + sum(pos_val.values()),
                    positions=pos_val,
                    peak_equity=peak,
                    prev_day_return=prev_day_ret,
                    adv=adv_row,
                )

            closes = close_px.loc[t]
            eod_equity = cash + sum(
                q * closes[s] for s, q in shares.items() if pd.notna(closes.get(s))
            )
            equity_hist.append((t, eod_equity))
            prev_day_ret = eod_equity / prev_equity - 1 if prev_equity > 0 else 0.0
            prev_equity = eod_equity
            peak = max(peak, eod_equity)

        equity_series = pd.Series(
            dict(equity_hist), name="equity", dtype=float
        ).sort_index()
        return BacktestResult(
            equity=equity_series,
            trades=pd.DataFrame(trades),
            risk_decisions=risk.decisions_frame(),
            weights=weights,
            strategy=strategy_name,
            params=params or {},
        )
