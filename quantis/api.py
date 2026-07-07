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


def _build_strategy(name: str, params: dict):
    """Construct a strategy; surface bad configs as 422s, not 500s
    (e.g. ai_signal with no PRODUCTION model in the registry)."""
    if name not in strategies.available():
        raise HTTPException(422, f"unknown strategy {name!r}")
    try:
        return strategies.get(name)(**params)
    except (KeyError, FileNotFoundError, TypeError, ValueError) as e:
        raise HTTPException(422, f"cannot construct {name!r}: {e}")


class BacktestRequest(BaseModel):
    strategy: str
    params: dict = Field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    capital: float = 1_000_000.0
    segment: str = "delivery"


class PaperReplayRequest(BaseModel):
    strategy: str
    params: dict = Field(default_factory=dict)
    start: str | None = None
    end: str | None = None
    capital: float = 1_000_000.0
    algo: str = "immediate"
    warmup: int = 210


class WalkForwardRequest(BaseModel):
    strategy: str
    grid: dict[str, list] = Field(default_factory=dict)
    train_days: int = 504
    test_days: int = 126
    expanding: bool = False
    start: str | None = None
    end: str | None = None
    capital: float = 1_000_000.0


class PromoteRequest(BaseModel):
    to: str
    approved_by: str | None = None


class CopilotRequest(BaseModel):
    prompt: str
    use_llm: bool = True


def create_app(lake_root: str = "data/lake", runs_root: str = "runs",
               paper_root: str = "paper_sessions",
               registry_root: str = "models") -> FastAPI:
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
        strat = _build_strategy(req.strategy, req.params)
        wide = load_wide(req.start, req.end)
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
        except (ValueError, KeyError, FileNotFoundError) as e:
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

    @app.get("/v1/paper/sessions")
    def list_paper_sessions():
        root = Path(paper_root)
        out = []
        if root.exists():
            for d in sorted(root.iterdir(), reverse=True):
                sfile = d / "session.json"
                if sfile.exists():
                    s = json.loads(sfile.read_text())
                    out.append({
                        "session_id": d.name,
                        "strategy": s.get("strategy"),
                        "algo": s.get("execution_algo"),
                        "sharpe": s.get("metrics", {}).get("sharpe"),
                        "breaker_tripped": s.get("risk_status", {}).get("breaker_tripped"),
                    })
        return out

    @app.get("/v1/paper/sessions/{session_id}")
    def paper_session_detail(session_id: str):
        d = Path(paper_root) / session_id
        if not (d / "session.json").exists():
            raise HTTPException(404, f"paper session {session_id} not found")
        s = json.loads((d / "session.json").read_text())
        equity = pd.read_csv(d / "equity.csv", index_col=0)
        recon = (d / "reconciliation.txt")
        return {
            "session_id": session_id,
            **s,
            "equity": {"ts": list(equity.index.astype(str)),
                       "value": [float(v) for v in equity.iloc[:, 0]]},
            "reconciliation": recon.read_text(encoding="utf-8") if recon.exists() else "",
        }

    @app.post("/v1/paper/replay")
    def run_paper_replay(req: PaperReplayRequest):
        from datetime import datetime

        from .feed import ReplayFeed
        from .paper import PaperTradingEngine

        strat = _build_strategy(req.strategy, req.params)
        wide = load_wide(req.start, req.end)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = Path(paper_root) / f"{stamp}_{req.strategy}"
        engine = PaperTradingEngine(
            strategy=strat,
            initial_capital=req.capital,
            risk_limits=RiskLimits(),
            execution_algo=req.algo,
            session_dir=session_dir,
        )
        session = engine.run(ReplayFeed(wide), warmup_bars=req.warmup)
        get_tracker(runs_root).log_run(
            name=f"{req.strategy}_paper", params=req.params,
            metrics={k: v for k, v in session.metrics.items()
                     if isinstance(v, (int, float))},
            artifacts_dir=session_dir,
            tags={"engine": "paper", "algo": req.algo, "source": "api"},
        )
        return {
            "session_id": session_dir.name,
            "metrics": {k: v for k, v in session.metrics.items()
                        if isinstance(v, (int, float, str))},
            "risk_status": session.risk_status,
            "final_positions": session.final_positions,
            "reconciliation": session.reconciliation,
        }

    @app.get("/v1/models")
    def list_models():
        from .ai.registry import ModelRegistry
        return [
            {k: e.get(k) for k in ("model_id", "name", "version", "stage",
                                   "metrics", "shadow_report", "approved_by",
                                   "trained_at")}
            for e in ModelRegistry(registry_root).list_models()
        ]

    @app.post("/v1/models/{model_id}/promote")
    def promote_model(model_id: str, req: PromoteRequest):
        from .ai.registry import ModelRegistry, PromotionError
        try:
            entry = ModelRegistry(registry_root).promote(
                model_id, req.to, approved_by=req.approved_by
            )
        except KeyError as e:
            raise HTTPException(404, str(e))
        except (PromotionError, ValueError) as e:
            raise HTTPException(422, str(e))
        return {"model_id": entry["model_id"], "stage": entry["stage"],
                "approved_by": entry["approved_by"]}

    @app.post("/v1/copilot/query")
    def copilot_query(req: CopilotRequest):
        from .ai.copilot import ask, build_context
        ctx = build_context(lake_root=lake_root, runs_root=runs_root,
                            registry_root=registry_root, paper_root=paper_root)
        return ask(req.prompt, context=ctx, use_llm=req.use_llm)

    @app.get("/")
    def index():
        page = WEB_DIR / "index.html"
        if not page.exists():
            raise HTTPException(404, "research workspace not built")
        return FileResponse(page)

    return app


app = create_app()
