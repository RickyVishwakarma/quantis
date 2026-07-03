import pandas as pd
import pytest

from quantis.risk import Order, PortfolioState, RiskEngine, RiskLimits

TS = pd.Timestamp("2024-01-15")
ADV = {"RELIANCE": 50_000_000.0, "TCS": 50_000_000.0, "INFY": 50_000_000.0,
       "HDFCBANK": 50_000_000.0, "ICICIBANK": 50_000_000.0}


def make_state(equity=1_000_000, positions=None, peak=None, prev_ret=0.0, adv=None):
    return PortfolioState(
        equity=equity,
        positions=positions or {},
        peak_equity=peak if peak is not None else equity,
        prev_day_return=prev_ret,
        adv=adv if adv is not None else ADV,
    )


def order(symbol="RELIANCE", side="BUY", qty=10, price=1000.0):
    return Order(symbol=symbol, side=side, qty=qty, ref_price=price, ts=TS)


def test_normal_order_approved():
    eng = RiskEngine(RiskLimits())
    d = eng.evaluate(order(qty=50), make_state())  # 50k on 1M equity
    assert d.approved and d.breached_rule is None


def test_position_weight_veto():
    eng = RiskEngine(RiskLimits(max_position_weight=0.10))
    d = eng.evaluate(order(qty=200), make_state())  # 200k = 20% > 10%
    assert not d.approved and d.breached_rule == "max_position_weight"


def test_sector_limit_veto():
    limits = RiskLimits(max_sector_weight=0.20, max_position_weight=0.15)
    eng = RiskEngine(limits)
    # 150k already in Financials, adding 100k HDFC breaches 20% sector cap
    state = make_state(positions={"ICICIBANK": 150_000.0})
    d = eng.evaluate(order(symbol="HDFCBANK", qty=100), state)
    assert not d.approved and d.breached_rule == "max_sector_weight"


def test_gross_exposure_veto():
    eng = RiskEngine(RiskLimits(max_gross_exposure=1.0))
    state = make_state(positions={"TCS": 950_000.0})
    d = eng.evaluate(order(qty=100), state)  # +100k would exceed 1.0x
    assert not d.approved and d.breached_rule == "max_gross_exposure"


def test_daily_loss_halts_new_risk_but_not_exits():
    eng = RiskEngine(RiskLimits(max_daily_loss=0.03))
    state = make_state(prev_ret=-0.05, positions={"RELIANCE": 100_000.0})
    buy = eng.evaluate(order(side="BUY", qty=50), state)
    sell = eng.evaluate(order(side="SELL", qty=50), state)
    assert not buy.approved and buy.breached_rule == "max_daily_loss_halt"
    assert sell.approved


def test_drawdown_kill_switch():
    eng = RiskEngine(RiskLimits(max_drawdown=0.15))
    state = make_state(equity=800_000, peak=1_000_000)  # 20% dd
    d = eng.evaluate(order(qty=50), state)
    assert not d.approved and d.breached_rule == "max_drawdown_halt"


def test_liquidity_veto():
    eng = RiskEngine(RiskLimits(max_adv_participation=0.05))
    # 24% of equity (under the 25% notional bound) but 12% of a thin ADV
    state = make_state(adv={"RELIANCE": 2_000_000.0} | {k: v for k, v in ADV.items() if k != "RELIANCE"},
                       equity=1_000_000)
    d = eng.evaluate(order(qty=240), state)  # 240k > 5% of 2M ADV
    assert not d.approved and d.breached_rule == "max_adv_participation"


def test_sanity_bound_applies_to_all_orders():
    eng = RiskEngine(RiskLimits(max_order_notional_pct=0.25))
    d = eng.evaluate(order(qty=400), make_state())  # 400k = 40% of equity
    assert not d.approved and d.breached_rule == "max_order_notional_pct"


def test_every_evaluation_is_logged():
    eng = RiskEngine(RiskLimits())
    eng.evaluate(order(qty=50), make_state())
    eng.evaluate(order(qty=9999), make_state())
    frame = eng.decisions_frame()
    assert len(frame) == 2
    assert set(frame["outcome"]) == {"APPROVE", "REJECT"}


def test_limit_snapshot_recorded_on_decision():
    eng = RiskEngine(RiskLimits(max_position_weight=0.07))
    d = eng.evaluate(order(qty=50), make_state())
    assert d.limit_snapshot["max_position_weight"] == 0.07
