"""Quantis CLI.

    quantis ingest       --source yahoo|synthetic --start 2018-01-01
    quantis backtest     --strategy momentum --start 2019-01-01
    quantis sweep        --strategy ma_crossover --grid fast=10,20 --grid slow=50,100,200
    quantis walkforward  --strategy momentum --grid top_n=5,10 --train-days 504 --test-days 126
    quantis materialize  (compute + version the feature set into the feature store)
    quantis ui           (research workspace at http://127.0.0.1:8000)
    quantis list         (available strategies + lake contents)
"""

from __future__ import annotations

import argparse
import sys

from . import strategies
from .backtest import EventBacktester, NSECostModel
from .backtest.vectorized import sweep as run_sweep
from .data import ingest, store, universe
from .report import write_run
from .risk import RiskLimits


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lake", default="data/lake", help="data lake root")
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--symbols", default=None,
                   help="comma-separated symbols (default: whole lake)")


def cmd_ingest(args) -> int:
    syms = args.symbols.split(",") if args.symbols else universe.symbols(limit=args.limit)
    if args.source == "yahoo":
        print(f"Fetching {len(syms)} symbols from Yahoo Finance…")
        bars = ingest.fetch_yahoo(syms, start=args.start or "2018-01-01", end=args.end)
    else:
        print(f"Generating synthetic bars for {len(syms)} symbols…")
        bars = ingest.generate_synthetic(syms, start=args.start or "2018-01-01",
                                         end=args.end or "2024-12-31", seed=args.seed)
    lake = store.BarLake(args.lake)
    n = lake.save_bars(bars)
    print(f"Lake now holds {n} bars across {len(lake.available_symbols())} symbols "
          f"at {lake.bars_dir}")
    return 0


def _load_wide(args):
    lake = store.BarLake(args.lake)
    syms = args.symbols.split(",") if args.symbols else lake.available_symbols()
    bars = lake.load_bars(syms, start=args.start, end=args.end)
    return store.to_wide(bars)


def _parse_params(pairs: list[str] | None) -> dict:
    out = {}
    for pair in pairs or []:
        k, v = pair.split("=", 1)
        try:
            out[k] = int(v)
        except ValueError:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


def cmd_backtest(args) -> int:
    wide = _load_wide(args)
    cls = strategies.get(args.strategy)
    strat = cls(**_parse_params(args.param))
    engine = EventBacktester(
        initial_capital=args.capital,
        cost_model=NSECostModel(segment=args.segment),
        risk_limits=RiskLimits(),
    )
    print(f"Running event-driven backtest: {strat.describe()} "
          f"on {len(wide['close'].columns)} symbols, "
          f"{len(wide['close'])} bars…")
    result = engine.run(wide, strat)
    run_dir = write_run(result)

    from .research import get_tracker
    import json
    metrics = json.loads((run_dir / "metrics.json").read_text())
    tracker = get_tracker()
    tracker.log_run(name=strat.describe(), params=strat.params, metrics=metrics,
                    artifacts_dir=run_dir, tags={"engine": "event", "source": "cli"})

    print()
    print((run_dir / "report.txt").read_text(encoding="utf-8"))
    print(f"\nArtifacts: {run_dir}  (experiment logged via {tracker.backend})")
    return 0


def cmd_sweep(args) -> int:
    wide = _load_wide(args)
    grid = {}
    for spec in args.grid:
        k, vals = spec.split("=", 1)
        parsed = []
        for v in vals.split(","):
            try:
                parsed.append(int(v))
            except ValueError:
                parsed.append(float(v))
        grid[k] = parsed
    n_combos = 1
    for v in grid.values():
        n_combos *= len(v)
    print(f"Vectorized sweep: {args.strategy} × {n_combos} combos…")
    board = run_sweep(wide, args.strategy, grid)
    print(board.to_string(index=False,
                          float_format=lambda x: f"{x:.3f}"))
    print("\nNote: validate the winner in the event engine (`quantis backtest`) —")
    print("it adds the risk gate, cash constraints, and per-order slippage.")
    return 0


def _parse_grid(specs: list[str]) -> dict:
    grid = {}
    for spec in specs:
        k, vals = spec.split("=", 1)
        parsed = []
        for v in vals.split(","):
            try:
                parsed.append(int(v))
            except ValueError:
                parsed.append(float(v))
        grid[k] = parsed
    return grid


def cmd_walkforward(args) -> int:
    from .backtest.metrics import format_metrics
    from .research import WalkForwardConfig, get_tracker, run_walkforward

    wide = _load_wide(args)
    grid = _parse_grid(args.grid)
    cfg = WalkForwardConfig(train_days=args.train_days, test_days=args.test_days,
                            expanding=args.expanding)
    mode = "expanding" if cfg.expanding else "rolling"
    print(f"Walk-forward: {args.strategy}, {mode} train={cfg.train_days}d "
          f"test={cfg.test_days}d, grid={grid}")
    wf = run_walkforward(wide, args.strategy, grid, cfg,
                         initial_capital=args.capital)

    print("\nPER-WINDOW (params selected on train, evaluated OOS):")
    print(wf.windows.to_string(index=False))
    print("\nSTITCHED OUT-OF-SAMPLE PERFORMANCE (the only evidence):")
    print(format_metrics(wf.oos_metrics))
    mc = wf.monte_carlo
    if "error" not in mc:
        print(f"\nMonte Carlo ({mc['n_sims']} block-bootstrap resamples):")
        print(f"  Sharpe p5/p50/p95   {mc['sharpe_p05']:.2f} / "
              f"{mc['sharpe_p50']:.2f} / {mc['sharpe_p95']:.2f}")
        print(f"  MaxDD  p5/p50/p95   {mc['maxdd_p05']:.1%} / "
              f"{mc['maxdd_p50']:.1%} / {mc['maxdd_p95']:.1%}")
        print(f"  P(Sharpe < 0)       {mc['prob_sharpe_negative']:.1%}")
    print(f"\nDeflated Sharpe (P[true SR > 0], {wf.n_trials} trials): "
          f"{wf.deflated_sharpe:.1%}")

    tracker = get_tracker()
    tracker.log_run(
        name=f"{args.strategy}_walkforward",
        params={"grid": str(grid), "train_days": cfg.train_days,
                "test_days": cfg.test_days, "mode": mode},
        metrics={**{k: v for k, v in wf.oos_metrics.items()
                    if isinstance(v, (int, float))},
                 "deflated_sharpe": wf.deflated_sharpe},
        tags={"engine": "walkforward", "source": "cli"},
    )
    print(f"Experiment logged via {tracker.backend}")
    return 0


def cmd_paper(args) -> int:
    from datetime import datetime

    from .backtest.metrics import format_metrics
    from .feed import ReplayFeed
    from .paper import PaperTradingEngine
    from .research import get_tracker

    wide = _load_wide(args)
    cls = strategies.get(args.strategy)
    strat = cls(**_parse_params(args.param))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = f"paper_sessions/{stamp}_{args.strategy}"
    engine = PaperTradingEngine(
        strategy=strat,
        initial_capital=args.capital,
        cost_model=NSECostModel(segment=args.segment),
        risk_limits=RiskLimits(),
        execution_algo=args.algo,
        twap_slices=args.slices,
        vol_targeting=args.vol_target,
        session_dir=session_dir,
    )
    feed = ReplayFeed(wide, start=args.start, end=args.end)
    n_bars = len(feed.index)
    print(f"Paper session (replay): {strat.describe()}, {n_bars} bars, "
          f"algo={args.algo}, warmup={args.warmup}")
    session = engine.run(feed, warmup_bars=args.warmup)

    print("\nPAPER SESSION REPORT")
    print("=" * 50)
    if session.metrics:
        print(format_metrics(session.metrics))
    n_evals = len(session.risk_decisions)
    n_rej = int((session.risk_decisions["outcome"] == "REJECT").sum()) if n_evals else 0
    print(f"\nRisk evaluations  {n_evals} orders gated, {n_rej} vetoed")
    print(f"Risk status       {session.risk_status}")
    print(f"Open positions    {len(session.final_positions)}")
    print(f"Reconciliation    {session.reconciliation}")

    if args.parity:
        from .backtest import EventBacktester
        from .features import compute_features

        panel = compute_features(wide)
        weights = cls(**_parse_params(args.param)).target_weights(panel)
        weights.iloc[:max(args.warmup - 1, 0)] = 0.0    # same first trading day
        bt = EventBacktester(initial_capital=args.capital,
                             cost_model=NSECostModel(segment=args.segment),
                             risk_limits=RiskLimits())
        bt_result = bt.run_weights(panel, weights)
        paper_eq, bt_eq = session.equity.align(bt_result.equity, join="inner")
        if len(paper_eq) > 10:
            corr = paper_eq.pct_change().corr(bt_eq.pct_change())
            gap = abs(paper_eq.iloc[-1] / bt_eq.iloc[-1] - 1)
            print("\nBACKTEST PARITY (same window, same first trading day):")
            print(f"  daily-return correlation  {corr:.4f}")
            print(f"  paper final equity        Rs {paper_eq.iloc[-1]:,.0f}")
            print(f"  backtest final equity     Rs {bt_eq.iloc[-1]:,.0f}")
            print(f"  terminal wealth gap       {gap:.2%}")

    get_tracker().log_run(
        name=f"{args.strategy}_paper", params=strat.params,
        metrics={k: v for k, v in session.metrics.items()
                 if isinstance(v, (int, float))},
        artifacts_dir=session.session_dir,
        tags={"engine": "paper", "algo": args.algo, "source": "cli"},
    )
    print(f"\nArtifacts: {session.session_dir}")
    return 0


def cmd_materialize(args) -> int:
    from .fstore import FEATURE_SCHEMA_VERSION, FeatureStore

    wide = _load_wide(args)
    fs = FeatureStore(args.feature_store)
    counts = fs.materialize(wide)
    print(f"Materialized {len(counts)} features (schema v{FEATURE_SCHEMA_VERSION}) "
          f"into {fs.root}:")
    for name, n in sorted(counts.items()):
        print(f"  {name:<16} {n:>8} rows")
    return 0


def cmd_ai_train(args) -> int:
    from .ai.train import train_and_register
    from .research import get_tracker

    wide = _load_wide(args)
    print(f"Training {args.model} model, label = fwd {args.label_horizon}d return, "
          f"{len(wide['close'].columns)} symbols, {len(wide['close'])} bars…")
    entry = train_and_register(
        wide, model_type=args.model, label_horizon=args.label_horizon,
        train_frac=args.train_frac,
    )
    m = entry["metrics"]
    print(f"\nModel {entry['name']} v{entry['version']}  ({entry['model_id']})")
    print(f"  stage              {entry['stage']}")
    print(f"  validation IC      {m['ic']}   (baseline momentum IC: {m['baseline_ic']})")
    print(f"  hit rate           {m['hit_rate']:.1%}")
    print(f"  top-bottom spread  {m['top_bottom_spread']:+.4%} per {args.label_horizon}d")
    print(f"  rows train/val     {m['n_train_rows']} / {m['n_val_rows']}")
    if entry["stage"] == "CANDIDATE":
        print("\nBeat the baseline -> promoted to CANDIDATE. "
              "Next: `quantis ai shadow --model " + entry["model_id"] + "`")
    else:
        print("\nDid NOT beat the baseline -> stays EXPERIMENTAL.")
    get_tracker().log_run(
        name=f"train_{entry['name']}", params={"model_type": args.model,
                                               "label_horizon": args.label_horizon},
        metrics={k: v for k, v in m.items() if isinstance(v, (int, float))},
        tags={"engine": "ai_train", "model_id": entry["model_id"]},
    )
    return 0


def cmd_ai_models(args) -> int:
    from .ai.registry import ModelRegistry

    models = ModelRegistry(args.registry).list_models()
    if not models:
        print("Registry empty — run `quantis ai train`.")
        return 0
    for e in models:
        m = e.get("metrics", {})
        shadow = " +shadow" if e.get("shadow_report") else ""
        print(f"{e['model_id']}  {e['name']:<18} v{e['version']}  "
              f"{e['stage']:<12} IC={m.get('ic')}{shadow}"
              + (f"  approved_by={e['approved_by']}" if e.get("approved_by") else ""))
    return 0


def cmd_ai_shadow(args) -> int:
    from .ai.shadow import run_shadow

    wide = _load_wide(args)
    print(f"Shadow mode (infer, don't trade): model {args.model}, "
          f"{args.days} day window…")
    report = run_shadow(wide, args.model, shadow_days=args.days)
    for k, v in report.items():
        print(f"  {k:<34} {v}")
    print("\nModel moved to SHADOW. Promote with:")
    print(f"  quantis ai promote --model {args.model} --to PRODUCTION --approved-by <you>")
    return 0


def cmd_ai_promote(args) -> int:
    from .ai.registry import ModelRegistry, PromotionError

    try:
        entry = ModelRegistry(args.registry).promote(
            args.model, args.to, approved_by=args.approved_by
        )
    except (PromotionError, KeyError) as e:
        print(f"REFUSED: {e}")
        return 1
    print(f"{entry['name']} v{entry['version']} -> {entry['stage']}"
          + (f" (approved by {entry['approved_by']})" if entry["approved_by"] else ""))
    return 0


def cmd_ai_ask(args) -> int:
    from .ai.copilot import ask

    result = ask(args.question, use_llm=not args.local)
    print(f"[{result['backend']}]")
    print(result["answer"])
    return 0


def cmd_ui(args) -> int:
    import uvicorn

    print(f"Quantis research workspace: http://127.0.0.1:{args.port}")
    uvicorn.run("quantis.api:app", host=args.host, port=args.port, log_level="warning")
    return 0


def cmd_list(args) -> int:
    print("Strategies:", ", ".join(strategies.available()))
    try:
        lake = store.BarLake(args.lake)
        syms = lake.available_symbols()
        print(f"Lake ({lake.bars_dir}): {len(syms)} symbols")
    except Exception:
        print("Lake: empty (run `quantis ingest`)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="quantis",
                                     description="Quantis quant platform (MVP)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="ingest historical bars into the lake")
    p.add_argument("--source", choices=["yahoo", "synthetic"], default="yahoo")
    p.add_argument("--limit", type=int, default=None, help="cap symbol count")
    p.add_argument("--seed", type=int, default=42)
    _add_common(p)
    p.set_defaults(fn=cmd_ingest)

    p = sub.add_parser("backtest", help="event-driven backtest with risk gate")
    p.add_argument("--strategy", required=True)
    p.add_argument("--param", action="append", metavar="KEY=VAL")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--segment", choices=["delivery", "intraday"], default="delivery")
    _add_common(p)
    p.set_defaults(fn=cmd_backtest)

    p = sub.add_parser("sweep", help="vectorized parameter sweep")
    p.add_argument("--strategy", required=True)
    p.add_argument("--grid", action="append", required=True,
                   metavar="KEY=V1,V2,...")
    _add_common(p)
    p.set_defaults(fn=cmd_sweep)

    p = sub.add_parser("walkforward", help="walk-forward validation with OOS stitching")
    p.add_argument("--strategy", required=True)
    p.add_argument("--grid", action="append", required=True, metavar="KEY=V1,V2,...")
    p.add_argument("--train-days", type=int, default=504)
    p.add_argument("--test-days", type=int, default=126)
    p.add_argument("--expanding", action="store_true")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    _add_common(p)
    p.set_defaults(fn=cmd_walkforward)

    p = sub.add_parser("paper", help="paper trading session (replay feed, OMS/EMS, full risk)")
    p.add_argument("--strategy", required=True)
    p.add_argument("--param", action="append", metavar="KEY=VAL")
    p.add_argument("--capital", type=float, default=1_000_000.0)
    p.add_argument("--segment", choices=["delivery", "intraday"], default="delivery")
    p.add_argument("--algo", choices=["immediate", "twap"], default="immediate")
    p.add_argument("--slices", type=int, default=4)
    p.add_argument("--warmup", type=int, default=210,
                   help="bars of history before trading begins")
    p.add_argument("--vol-target", action="store_true",
                   help="enable volatility-targeting overlay")
    p.add_argument("--parity", action="store_true",
                   help="also run the event backtest and report divergence")
    _add_common(p)
    p.set_defaults(fn=cmd_paper)

    p = sub.add_parser("materialize", help="materialize features into the feature store")
    p.add_argument("--feature-store", default="data/feature_store")
    _add_common(p)
    p.set_defaults(fn=cmd_materialize)

    p_ai = sub.add_parser("ai", help="model training, registry, shadow mode, copilot")
    ai_sub = p_ai.add_subparsers(dest="ai_command", required=True)

    p = ai_sub.add_parser("train", help="train a signal model from the feature store")
    p.add_argument("--model", choices=["ridge", "gbt"], default="ridge")
    p.add_argument("--label-horizon", type=int, default=5)
    p.add_argument("--train-frac", type=float, default=0.75)
    _add_common(p)
    p.set_defaults(fn=cmd_ai_train)

    p = ai_sub.add_parser("models", help="list the model registry")
    p.add_argument("--registry", default="models")
    p.set_defaults(fn=cmd_ai_models)

    p = ai_sub.add_parser("shadow", help="shadow-mode evaluation (infer, don't trade)")
    p.add_argument("--model", required=True, help="model_id or production:<name>")
    p.add_argument("--days", type=int, default=126)
    _add_common(p)
    p.set_defaults(fn=cmd_ai_shadow)

    p = ai_sub.add_parser("promote", help="stage promotion (PRODUCTION needs sign-off)")
    p.add_argument("--model", required=True)
    p.add_argument("--to", required=True,
                   choices=["CANDIDATE", "SHADOW", "PRODUCTION", "RETIRED"])
    p.add_argument("--approved-by", default=None)
    p.add_argument("--registry", default="models")
    p.set_defaults(fn=cmd_ai_promote)

    p = ai_sub.add_parser("ask", help="ask the AI copilot about platform state")
    p.add_argument("question")
    p.add_argument("--local", action="store_true", help="skip the LLM, local summary only")
    p.set_defaults(fn=cmd_ai_ask)

    p = sub.add_parser("ui", help="serve the research workspace + API")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(fn=cmd_ui)

    p = sub.add_parser("list", help="list strategies and lake contents")
    p.add_argument("--lake", default="data/lake")
    p.set_defaults(fn=cmd_list)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
