import numpy as np
import pandas as pd
import pytest

from quantis.backtest.metrics import compute_metrics


def make_equity(daily_ret=0.001, days=504, start=1_000_000):
    idx = pd.bdate_range("2022-01-03", periods=days)
    return pd.Series(start * (1 + daily_ret) ** np.arange(days), index=idx)


def test_cagr_positive_for_rising_equity():
    m = compute_metrics(make_equity(0.001))
    assert m["cagr"] > 0.20  # 0.1%/day compounds to >25%/yr
    assert m["total_return"] > 0


def test_max_drawdown_flat_series_is_zero():
    m = compute_metrics(make_equity(0.0))
    assert m["max_drawdown"] == 0.0


def test_max_drawdown_detects_crash():
    eq = make_equity(0.001)
    eq.iloc[300:] *= 0.7  # 30% gap down
    m = compute_metrics(eq)
    assert m["max_drawdown"] <= -0.29


def test_sharpe_sign():
    up = compute_metrics(make_equity(0.002))
    assert up["sharpe"] > 0


def test_trade_stats():
    trades = pd.DataFrame({
        "notional": [100_000] * 4,
        "costs": [120.0] * 4,
        "realized_pnl": [5_000.0, -2_000.0, 3_000.0, None],
    })
    m = compute_metrics(make_equity(), trades)
    assert m["n_trades"] == 4
    assert m["win_rate"] == pytest.approx(2 / 3)
    assert m["profit_factor"] == pytest.approx(8_000 / 2_000)


def test_insufficient_data():
    eq = pd.Series([100.0], index=[pd.Timestamp("2024-01-01")])
    assert "error" in compute_metrics(eq)
