"""Run artifacts: every backtest writes a self-contained run directory.

runs/<timestamp>_<strategy>/
    report.txt          human-readable summary
    metrics.json        machine-readable metrics
    equity.csv          daily equity curve
    trades.csv          every executed fill with costs
    risk_decisions.csv  every risk evaluation incl. rejections (audit trail)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .backtest.engine import BacktestResult
from .backtest.metrics import compute_metrics, format_metrics


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, float) and np.isnan(obj):
        return None
    return obj


def write_run(result: BacktestResult, out_root: str | Path = "runs") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = result.strategy.split("(")[0]
    run_dir = Path(out_root) / f"{stamp}_{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(result.equity, result.trades)
    (run_dir / "metrics.json").write_text(
        json.dumps(_json_safe({**metrics, "strategy": result.strategy,
                               "params": result.params}), indent=2)
    )
    result.equity.to_csv(run_dir / "equity.csv")
    result.trades.to_csv(run_dir / "trades.csv", index=False)
    result.risk_decisions.to_csv(run_dir / "risk_decisions.csv", index=False)

    n_evals = len(result.risk_decisions)
    report = "\n".join([
        "QUANTIS BACKTEST REPORT",
        "=" * 50,
        f"Strategy          {result.strategy}",
        "",
        format_metrics(metrics),
        "",
        f"Risk evaluations  {n_evals} orders gated",
        f"Risk rejections   {result.n_rejected}"
        + (f" ({result.n_rejected / n_evals:.1%})" if n_evals else ""),
    ])
    (run_dir / "report.txt").write_text(report, encoding="utf-8")
    return run_dir
