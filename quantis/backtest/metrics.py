"""Performance metrics reported on every run (TDD Part 9 list)."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_metrics(
    equity: pd.Series,
    trades: pd.DataFrame | None = None,
    risk_free_rate: float = 0.065,   # RBI repo-ish annual
) -> dict:
    equity = equity.dropna()
    if len(equity) < 2:
        return {"error": "insufficient data"}
    rets = equity.pct_change().dropna()
    n_years = len(rets) / TRADING_DAYS

    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    cagr = (1 + total_return) ** (1 / n_years) - 1 if n_years > 0 else np.nan

    rf_daily = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    excess = rets - rf_daily
    vol = rets.std() * np.sqrt(TRADING_DAYS)
    sharpe = excess.mean() / rets.std() * np.sqrt(TRADING_DAYS) if rets.std() > 0 else np.nan

    downside = rets[rets < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = (cagr - risk_free_rate) / downside if downside and downside > 0 else np.nan

    peak = equity.cummax()
    dd = equity / peak - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd < 0 else np.nan

    out = {
        "start": str(equity.index[0].date()),
        "end": str(equity.index[-1].date()),
        "total_return": total_return,
        "cagr": cagr,
        "volatility": vol,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_drawdown": max_dd,
        "var_95_daily": rets.quantile(0.05),
        "cvar_95_daily": rets[rets <= rets.quantile(0.05)].mean(),
        "best_day": rets.max(),
        "worst_day": rets.min(),
    }

    if trades is not None and len(trades) > 0:
        out["n_trades"] = len(trades)
        out["turnover_annual"] = (
            trades["notional"].sum() / equity.mean() / n_years if n_years > 0 else np.nan
        )
        out["total_costs"] = trades["costs"].sum()
        if "realized_pnl" in trades.columns:
            closed = trades[trades["realized_pnl"].notna()]
            if len(closed) > 0:
                wins = closed[closed["realized_pnl"] > 0]["realized_pnl"]
                losses = closed[closed["realized_pnl"] < 0]["realized_pnl"]
                out["win_rate"] = len(wins) / len(closed)
                gross_win, gross_loss = wins.sum(), abs(losses.sum())
                out["profit_factor"] = gross_win / gross_loss if gross_loss > 0 else np.inf
                out["expectancy"] = closed["realized_pnl"].mean()
    return out


def format_metrics(m: dict) -> str:
    pct = lambda v: f"{v:+.2%}" if isinstance(v, (int, float)) and not np.isnan(v) else "n/a"
    num = lambda v: f"{v:.2f}" if isinstance(v, (int, float)) and not np.isnan(v) else "n/a"
    lines = [
        f"Period            {m.get('start')} -> {m.get('end')}",
        f"Total return      {pct(m.get('total_return'))}",
        f"CAGR              {pct(m.get('cagr'))}",
        f"Volatility        {pct(m.get('volatility'))}",
        f"Sharpe            {num(m.get('sharpe'))}",
        f"Sortino           {num(m.get('sortino'))}",
        f"Calmar            {num(m.get('calmar'))}",
        f"Max drawdown      {pct(m.get('max_drawdown'))}",
        f"Daily VaR 95%     {pct(m.get('var_95_daily'))}",
        f"Daily CVaR 95%    {pct(m.get('cvar_95_daily'))}",
    ]
    if "n_trades" in m:
        lines += [
            f"Trades            {m['n_trades']}",
            f"Annual turnover   {num(m.get('turnover_annual'))}x",
            f"Total costs       Rs {m.get('total_costs', 0):,.0f}",
        ]
    if "win_rate" in m:
        lines += [
            f"Win rate          {pct(m.get('win_rate'))}",
            f"Profit factor     {num(m.get('profit_factor'))}",
            f"Expectancy/trade  Rs {m.get('expectancy', 0):,.0f}",
        ]
    return "\n".join(lines)
