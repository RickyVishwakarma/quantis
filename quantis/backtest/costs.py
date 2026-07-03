"""NSE transaction cost model (equity segment, discount-broker schedule).

Rates modeled per the NSE/SEBI charge schedule for equity delivery and
intraday (as of FY25; all rates are config, not constants baked into the
engine, so a schedule change is a one-line edit):

  brokerage       delivery: 0 · intraday: min(0.03%, ₹20) per executed order
  STT             delivery: 0.1% both sides · intraday: 0.025% sell side
  exchange txn    NSE 0.00297% of turnover
  SEBI charges    0.0001% (₹10 / crore)
  stamp duty      buy side only — delivery 0.015%, intraday 0.003%
  GST             18% on (brokerage + exchange + SEBI)

Slippage: volume-participation model — a base half-spread plus impact
that grows with order size relative to ADV (square-root impact).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NSECostModel:
    segment: str = "delivery"          # delivery | intraday
    brokerage_pct: float = 0.0003      # intraday only
    brokerage_cap: float = 20.0
    stt_delivery: float = 0.001
    stt_intraday_sell: float = 0.00025
    exchange_txn: float = 0.0000297
    sebi: float = 0.000001
    stamp_delivery_buy: float = 0.00015
    stamp_intraday_buy: float = 0.00003
    gst: float = 0.18
    base_slippage_bps: float = 3.0     # effective half-spread, large caps
    impact_coeff_bps: float = 25.0     # bps at 100% ADV participation

    def charges(self, side: str, notional: float) -> float:
        """Statutory + brokerage charges for one executed order (₹)."""
        if notional <= 0:
            return 0.0
        side = side.upper()
        if self.segment == "delivery":
            brokerage = 0.0
            stt = notional * self.stt_delivery
            stamp = notional * self.stamp_delivery_buy if side == "BUY" else 0.0
        else:
            brokerage = min(notional * self.brokerage_pct, self.brokerage_cap)
            stt = notional * self.stt_intraday_sell if side == "SELL" else 0.0
            stamp = notional * self.stamp_intraday_buy if side == "BUY" else 0.0
        exchange = notional * self.exchange_txn
        sebi = notional * self.sebi
        gst = (brokerage + exchange + sebi) * self.gst
        return brokerage + stt + stamp + exchange + sebi + gst

    def slippage_bps(self, notional: float, adv: float) -> float:
        if adv <= 0:
            return self.base_slippage_bps + self.impact_coeff_bps  # worst case
        participation = min(notional / adv, 1.0)
        return self.base_slippage_bps + self.impact_coeff_bps * np.sqrt(participation)

    def fill_price(self, side: str, ref_price: float, notional: float, adv: float) -> float:
        slip = self.slippage_bps(notional, adv) / 10_000
        return ref_price * (1 + slip) if side.upper() == "BUY" else ref_price * (1 - slip)

    def round_trip_bps(self, adv_participation: float = 0.01) -> float:
        """Approximate round-trip cost in bps for the vectorized engine."""
        notional, adv = adv_participation * 1e9, 1e9
        buy = self.charges("BUY", notional) / notional * 10_000
        sell = self.charges("SELL", notional) / notional * 10_000
        slip = 2 * self.slippage_bps(notional, adv)
        return buy + sell + slip
