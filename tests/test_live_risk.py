import pandas as pd
import pytest

from quantis.risk import LiveRiskManager, Order, PortfolioState, RiskLimits, RiskTier

TS = pd.Timestamp("2025-01-10")
ADV = {"RELIANCE": 50_000_000.0}


def state(equity=1_000_000, peak=None, positions=None):
    return PortfolioState(equity=equity, positions=positions or {},
                          peak_equity=peak if peak is not None else equity,
                          prev_day_return=0.0, adv=ADV)


def order(side="BUY", qty=50, price=1000.0):
    return Order(symbol="RELIANCE", side=side, qty=qty, ref_price=price, ts=TS)


def test_tiers_escalate_with_drawdown():
    rm = LiveRiskManager(RiskLimits(soft_drawdown=0.08, flatten_drawdown=0.15))
    assert rm.tier(state()) == RiskTier.NORMAL
    assert rm.tier(state(equity=900_000, peak=1_000_000)) == RiskTier.HALVED
    assert rm.tier(state(equity=800_000, peak=1_000_000)) == RiskTier.FLATTEN
    assert rm.size_factor(state()) == 1.0
    assert rm.size_factor(state(equity=900_000, peak=1_000_000)) == 0.5
    assert rm.size_factor(state(equity=800_000, peak=1_000_000)) == 0.0


def test_flatten_tier_blocks_buys_allows_sells():
    rm = LiveRiskManager(RiskLimits(flatten_drawdown=0.10))
    s = state(equity=850_000, peak=1_000_000,
              positions={"RELIANCE": 100_000.0})
    buy = rm.evaluate(order(side="BUY"), s)
    sell = rm.evaluate(order(side="SELL"), s)
    assert not buy.approved and buy.breached_rule == "flatten_tier_active"
    assert sell.approved


def test_breaker_trips_after_consecutive_rejects():
    rm = LiveRiskManager(RiskLimits(breaker_consecutive_rejects=3,
                                    max_position_weight=0.01))
    s = state()
    for _ in range(3):
        rm.evaluate(order(qty=500), s)          # each breaches position weight
    assert rm.breaker.tripped
    # once tripped, even a tiny order is rejected with the breaker rule
    d = rm.evaluate(order(qty=1), s)
    assert not d.approved and d.breached_rule.startswith("circuit_breaker")


def test_approval_resets_consecutive_counter():
    rm = LiveRiskManager(RiskLimits(breaker_consecutive_rejects=3,
                                    max_position_weight=0.10))
    s = state()
    rm.evaluate(order(qty=500), s)              # reject (50% weight)
    rm.evaluate(order(qty=500), s)              # reject
    rm.evaluate(order(qty=10), s)               # approve resets
    rm.evaluate(order(qty=500), s)              # reject
    assert not rm.breaker.tripped


def test_panic_and_manual_reset():
    rm = LiveRiskManager()
    rm.panic()
    assert rm.breaker.tripped
    d = rm.evaluate(order(qty=1), state())
    assert not d.approved
    rm.reset()
    assert not rm.breaker.tripped
    assert rm.evaluate(order(qty=10), state()).approved


def test_feed_staleness_trips_breaker():
    rm = LiveRiskManager(RiskLimits(breaker_feed_stale_secs=300))
    rm.on_feed_staleness(200)
    assert not rm.breaker.tripped
    rm.on_feed_staleness(400)
    assert rm.breaker.tripped and "stale" in rm.breaker.reason


def test_vol_targeting_scales_down_hot_books():
    import numpy as np
    rm = LiveRiskManager(RiskLimits(target_vol=0.15, vol_scale_floor=0.25))
    rng = np.random.default_rng(0)
    calm = rng.normal(0, 0.15 / np.sqrt(252), 100)
    hot = rng.normal(0, 0.60 / np.sqrt(252), 100)
    assert rm.vol_scale(calm) == pytest.approx(1.0, abs=0.15)
    hot_scale = rm.vol_scale(hot)
    assert 0.25 <= hot_scale < 0.5
