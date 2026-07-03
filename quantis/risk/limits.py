"""Risk limit configuration (the Phase-1 limit set from TDD Part 8)."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskLimits:
    # Concentration
    max_position_weight: float = 0.15      # single name, fraction of equity
    max_sector_weight: float = 0.35        # per-sector gross, fraction of equity
    # Exposure
    max_gross_exposure: float = 1.0        # long-only MVP: no leverage
    # Liquidity
    max_adv_participation: float = 0.05    # order notional vs 20d rupee ADV
    # Loss limits — breaches halt NEW risk-increasing orders, never exits
    max_daily_loss: float = 0.03           # yesterday's equity dd triggers halt today
    max_drawdown: float = 0.15             # trailing peak-to-trough kill switch
    # Sanity bound (TDD's AI-hallucination safeguard, applied to ALL sources)
    max_order_notional_pct: float = 0.25   # single order vs equity

    def snapshot(self) -> dict:
        return asdict(self)
