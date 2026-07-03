"""Quantis CLI.

    quantis ingest    --source yahoo|synthetic --start 2018-01-01
    quantis backtest  --strategy momentum --start 2019-01-01
    quantis sweep     --strategy ma_crossover --grid fast=10,20 --grid slow=50,100,200
    quantis list      (available strategies + lake contents)
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
    print()
    print((run_dir / "report.txt").read_text(encoding="utf-8"))
    print(f"\nArtifacts: {run_dir}")
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

    p = sub.add_parser("list", help="list strategies and lake contents")
    p.add_argument("--lake", default="data/lake")
    p.set_defaults(fn=cmd_list)

    args = parser.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
