"""AI Copilot (TDD Part 15: the research workspace's Q&A surface).

Retrieval layer: assembles the platform's current state — lake, strategy
catalog, recent experiment runs, model registry, latest paper-session
risk status — into a context document. That context plus the user's
question goes to Claude (Anthropic SDK, adaptive thinking). Without an
API key / SDK the copilot degrades to a deterministic local summary of
the same context, so the endpoint never breaks.

The copilot is strictly read-only: it can describe state and suggest
next steps but has no execution path — an LLM can never place an order
here (the risk engine wouldn't let it anyway, but the copilot doesn't
even have the plumbing).
"""

from __future__ import annotations

import json
from pathlib import Path

SYSTEM_PROMPT = """You are the Quantis AI Copilot — the assistant inside an \
institutional-grade quantitative trading research platform for Indian markets \
(NSE equities and derivatives).

You will receive a JSON context document describing the platform's current \
state: data lake, available strategies, recent experiment runs, the model \
registry with stages (EXPERIMENTAL/CANDIDATE/SHADOW/PRODUCTION), and paper \
trading risk status.

Ground every answer in that context; when the context doesn't contain the \
answer, say so. Be precise with numbers. When discussing strategy performance \
be honest about overfitting risk, and always distinguish in-sample results \
from walk-forward/out-of-sample evidence. You are read-only: you can suggest \
CLI commands (quantis backtest / walkforward / paper / ai ...) but never \
claim to have executed anything."""


def build_context(
    lake_root: str = "data/lake",
    runs_root: str = "runs",
    registry_root: str = "models",
    paper_root: str = "paper_sessions",
) -> dict:
    ctx: dict = {}

    try:
        from ..data.store import BarLake
        lake = BarLake(lake_root)
        syms = lake.available_symbols()
        ctx["lake"] = {"n_symbols": len(syms), "symbols": syms[:30]}
    except Exception:
        ctx["lake"] = {"n_symbols": 0}

    try:
        from ..strategies import available
        ctx["strategies"] = available()
    except Exception:
        pass

    exp = Path(runs_root) / "experiments.jsonl"
    if exp.exists():
        lines = exp.read_text(encoding="utf-8").splitlines()
        ctx["recent_experiments"] = [json.loads(x) for x in lines[-8:]]

    try:
        from .registry import ModelRegistry
        ctx["model_registry"] = [
            {k: e.get(k) for k in ("model_id", "name", "version", "stage",
                                   "metrics", "shadow_report", "approved_by")}
            for e in ModelRegistry(registry_root).list_models()
        ]
    except Exception:
        ctx["model_registry"] = []

    sessions = sorted(Path(paper_root).glob("*/session.json"), reverse=True)
    if sessions:
        ctx["latest_paper_session"] = json.loads(sessions[0].read_text())

    return ctx


def _local_answer(question: str, ctx: dict) -> str:
    """Deterministic fallback: summarize the relevant slice of context."""
    q = question.lower()
    parts = []
    if any(w in q for w in ("model", "registry", "shadow", "production", "candidate")):
        models = ctx.get("model_registry", [])
        if models:
            parts.append("Model registry:")
            for m in models:
                ic = (m.get("metrics") or {}).get("ic")
                parts.append(f"  - {m['name']} v{m['version']} [{m['stage']}]"
                             + (f" IC={ic}" if ic is not None else ""))
        else:
            parts.append("No models in the registry yet — run `quantis ai train`.")
    if any(w in q for w in ("run", "backtest", "sharpe", "experiment", "performance")):
        for r in ctx.get("recent_experiments", [])[-3:]:
            m = r.get("metrics", {})
            parts.append(f"Run {r.get('name')}: sharpe={m.get('sharpe')}, "
                         f"cagr={m.get('cagr')}, max_dd={m.get('max_drawdown')}")
    if any(w in q for w in ("risk", "breaker", "paper", "position")):
        s = ctx.get("latest_paper_session")
        if s:
            parts.append(f"Latest paper session ({s.get('strategy')}): "
                         f"risk={s.get('risk_status')}, "
                         f"positions={len(s.get('final_positions', {}))}")
    if any(w in q for w in ("data", "lake", "symbol", "universe")):
        parts.append(f"Lake holds {ctx.get('lake', {}).get('n_symbols', 0)} symbols.")
    if not parts:
        parts = [
            "Platform state summary:",
            f"  lake: {ctx.get('lake', {}).get('n_symbols', 0)} symbols",
            f"  strategies: {', '.join(ctx.get('strategies', []))}",
            f"  models: {len(ctx.get('model_registry', []))} registered",
            f"  recent experiments: {len(ctx.get('recent_experiments', []))}",
        ]
    parts.append("(local fallback answer — set ANTHROPIC_API_KEY for the full copilot)")
    return "\n".join(parts)


def ask(question: str, context: dict | None = None, use_llm: bool = True) -> dict:
    ctx = context if context is not None else build_context()
    if use_llm:
        try:
            import anthropic

            client = anthropic.Anthropic()
            response = client.messages.create(
                model="claude-opus-4-8",
                max_tokens=2048,
                thinking={"type": "adaptive"},
                system=SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": (
                        "PLATFORM CONTEXT (JSON):\n"
                        + json.dumps(ctx, indent=1, default=str)[:60_000]
                        + f"\n\nQUESTION: {question}"
                    ),
                }],
            )
            text = "".join(b.text for b in response.content if b.type == "text")
            if response.stop_reason == "refusal" or not text:
                return {"answer": _local_answer(question, ctx), "backend": "local"}
            return {"answer": text, "backend": "claude"}
        except Exception:
            pass
    return {"answer": _local_answer(question, ctx), "backend": "local"}
