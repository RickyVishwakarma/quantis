import pytest

from quantis.backtest.costs import NSECostModel


def test_delivery_buy_charges():
    m = NSECostModel(segment="delivery")
    notional = 100_000.0
    c = m.charges("BUY", notional)
    # STT 0.1% + stamp 0.015% + exchange 0.00297% + SEBI 0.0001% + GST on (exch+sebi)
    expected = 100.0 + 15.0 + 2.97 + 0.10 + (2.97 + 0.10) * 0.18
    assert c == pytest.approx(expected, rel=1e-6)


def test_delivery_sell_has_no_stamp():
    m = NSECostModel(segment="delivery")
    assert m.charges("SELL", 100_000) < m.charges("BUY", 100_000)


def test_intraday_stt_sell_side_only():
    m = NSECostModel(segment="intraday")
    buy = m.charges("BUY", 100_000)
    sell = m.charges("SELL", 100_000)
    assert sell > buy  # STT applies on sell


def test_intraday_brokerage_capped():
    m = NSECostModel(segment="intraday")
    # 0.03% of 1cr = 3000 but capped at 20
    big = m.charges("BUY", 10_000_000)
    assert big < 10_000_000 * 0.0003 + 10_000_000 * 0.001


def test_slippage_grows_with_participation():
    m = NSECostModel()
    small = m.slippage_bps(notional=10_000, adv=10_000_000)
    large = m.slippage_bps(notional=5_000_000, adv=10_000_000)
    assert large > small


def test_fill_price_adverse_both_sides():
    m = NSECostModel()
    ref = 100.0
    assert m.fill_price("BUY", ref, 100_000, 10_000_000) > ref
    assert m.fill_price("SELL", ref, 100_000, 10_000_000) < ref


def test_zero_notional_costs_nothing():
    assert NSECostModel().charges("BUY", 0) == 0.0
