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

    # --- Phase 3: tiered drawdown response (TDD Part 8: halve -> flatten -> halt)
    soft_drawdown: float = 0.08            # tier 1: new orders sized at half
    flatten_drawdown: float = 0.15         # tier 2: exit everything, no new risk
    # (tier 3 halt = circuit breaker, manual reset required)

    # --- Phase 3: volatility targeting
    target_vol: float = 0.18               # annualized; book scales down above band
    vol_scale_floor: float = 0.25          # never scale below 25% of target weights

    # --- Phase 3: circuit breakers
    breaker_consecutive_rejects: int = 10  # trip after N rejects in a row
    breaker_feed_stale_secs: float = 300.0 # trip when feed silent this long

    def snapshot(self) -> dict:
        return asdict(self)
