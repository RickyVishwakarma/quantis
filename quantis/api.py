"""Quantis research API (FastAPI) + static research workspace.

Endpoints follow the TDD Appendix A API spec shape (`/v1/...`). Backtests
run synchronously in-process at MVP scale; Phase 3 moves them behind the
scheduler/event bus.

Run: `quantis ui` → http://127.0.0.1:8000
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import strategies
from .backtest import EventBacktester, NSECostModel
from .data import store
from .report import write_run
from .research import WalkForwardConfig, get_tracker, run_walkforward
from .risk import RiskLimits

WEB_DIR = Path(__file__).resolve().parent.parent / "web" / "research-workspace"


class BacktestRequest(BaseModel):
    strategy: str
    params: dict = Field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    capital: float = 1_000_000.0
    segment: str = "delivery"


class WalkForwardRequest(BaseModel):
    strategy: str
    grid: dict[str, list] = Field(default_factory=dict)
    train_days: int = 504
    test_days: int = 126
    expanding: bool = False
    start: str | None = None
    end: str | None = None
    capital: float = 1_000_000.0


def create_app(lake_root: str = "data/lake", runs_root: str = "runs") -> FastAPI:
    app = FastAPI(title="Quantis Research API", version="0.2.0")

    def load_wide(start=None, end=None):
        lake = store.BarLake(lake_root)
        syms = lake.available_symbols()
        if not syms:
            raise HTTPException(409, "Data lake is empty — run `quantis ingest` first")
        return store.to_wide(lake.load_bars(syms, start=start, end=end))

    @app.get("/v1/strategies")
    def list_strategies():
        return [
            {"name": name, "default_params": strategies.get(name).default_params}
            for name in strategies.available()
        ]

    @app.get("/v1/lake")
    def lake_status():
        lake = store.BarLake(lake_root)
        syms = lake.available_symbols()
        out = {"symbols": syms, "n_symbols": len(syms), "start": None, "end": None}
        if syms:
            bars = lake.load_bars(syms[:1])
            out["start"], out["end"] = str(bars["ts"].min().date()), str(bars["ts"].max().date())
        return out

    @app.get("/v1/runs")
    def list_runs():
        root = Path(runs_root)
        out = []
        if root.exists():
            for d in sorted(root.iterdir(), reverse=True):
                mfile = d / "metrics.json"
                if mfile.exists():
                    m = json.loads(mfile.read_text())
                    out.append({
                        "run_id": d.name,
                        "strategy": m.get("strategy"),
                        "sharpe": m.get("sharpe"),
                        "cagr": m.get("cagr"),
                        "max_drawdown": m.get("max_drawdown"),
                        "n_trades": m.get("n_trades"),
                    })
        return out

    @app.get("/v1/runs/{run_id}")
    def run_detail(run_id: str):
        d = Path(runs_root) / run_id
        if not (d / "metrics.json").exists():
            raise HTTPException(404, f"run {run_id} not found")
        metrics = json.loads((d / "metrics.json").read_text())
        equity = pd.read_csv(d / "equity.csv", index_col=0)
        eq = equity.iloc[:, 0]
        decisions_path = d / "risk_decisions.csv"
        risk_summary = {}
        if decisions_path.exists():
            rd = pd.read_csv(decisions_path)
            if len(rd):
                risk_summary = {
                    "evaluated": int(len(rd)),
                    "rejected": int((rd["outcome"] == "REJECT").sum()),
                    "by_rule": rd[rd["outcome"] == "REJECT"]["breached_rule"]
                    .value_counts().to_dict(),
                }
        return {
            "run_id": run_id,
            "metrics": metrics,
            "equity": {"ts": list(equity.index.astype(str)),
                       "value": [float(v) for v in eq]},
            "risk": risk_summary,
        }

    @app.post("/v1/backtests")
    def create_backtest(req: BacktestRequest):
        if req.strategy not in strategies.available():
            raise HTTPException(422, f"unknown strategy {req.strategy!r}")
        wide = load_wide(req.start, req.end)
        strat = strategies.get(req.strategy)(**req.params)
        engine = EventBacktester(
            initial_capital=req.capital,
            cost_model=NSECostModel(segment=req.segment),
            risk_limits=RiskLimits(),
        )
        result = engine.run(wide, strat)
        run_dir = write_run(result, runs_root)
        metrics = json.loads((run_dir / "metrics.json").read_text())
        get_tracker(runs_root).log_run(
            name=strat.describe(), params=req.params, metrics=metrics,
            artifacts_dir=run_dir, tags={"engine": "event", "source": "api"},
        )
        return {"run_id": run_dir.name, "metrics": metrics}

    @app.post("/v1/walkforward")
    def create_walkforward(req: WalkForwardRequest):
        if req.strategy not in strategies.available():
            raise HTTPException(422, f"unknown strategy {req.strategy!r}")
        wide = load_wide(req.start, req.end)
        cfg = WalkForwardConfig(train_days=req.train_days, test_days=req.test_days,
                                expanding=req.expanding)
        grid = req.grid or {k: [v] for k, v in
                            strategies.get(req.strategy).default_params.items()
                            if isinstance(v, (int, float))}
        try:
            wf = run_walkforward(wide, req.strategy, grid, cfg,
                                 initial_capital=req.capital)
        except ValueError as e:
            raise HTTPException(422, str(e))
        get_tracker(runs_root).log_run(
            name=f"{req.strategy}_walkforward",
            params={"grid": json.dumps(grid), "train_days": cfg.train_days,
                    "test_days": cfg.test_days},
            metrics={**{k: v for k, v in wf.oos_metrics.items()
                        if isinstance(v, (int, float))},
                     "deflated_sharpe": wf.deflated_sharpe},
            tags={"engine": "walkforward", "source": "api"},
        )
        return {
            "strategy": req.strategy,
            "n_windows": len(wf.windows),
            "n_trials": wf.n_trials,
            "windows": wf.windows.astype(str).to_dict(orient="records"),
            "oos_metrics": {k: v for k, v in wf.oos_metrics.items()},
            "monte_carlo": wf.monte_carlo,
            "deflated_sharpe": wf.deflated_sharpe,
            "oos_equity": {"ts": [str(t.date()) for t in wf.oos_equity.index],
                           "value": [float(v) for v in wf.oos_equity]},
        }

    @app.get("/")
    def index():
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "research workspace not built")
        return FileResponse(page)

    return app


app = create_app()
