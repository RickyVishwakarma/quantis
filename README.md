# Quantis

AI quantitative research and execution platform for Indian equities and derivatives.
This repo implements **Phases 1–2** of the Quantis TDD: the full research loop — data →
feature store → strategy → walk-forward validation → risk-gated backtest → tracked
experiment — built so that every later phase (paper trading, AI signals, live execution)
extends it without a rewrite.

## Phase 2 — research platform

- **Feature store** (`quantis/fstore/`) — materialized, versioned features using the TDD's
  offline-table schema `(instrument_id, feature_name, as_of_ts, value, schema_version)`.
  `get_asof()` is a point-in-time join that structurally cannot return future values;
  `training_frame()` builds supervised datasets with strictly-future labels (Phase 4 prep).
  Feast is deferred until there's an online serving path (Phase 3) — same schema, so it's
  a backend swap, not an API change.
- **Walk-forward validation** (`quantis/research/walkforward.py`) — rolling/expanding
  train→test splits: grid selected on train (vectorized), evaluated OOS in the event
  engine (risk gate on), OOS segments stitched into the only equity curve treated as
  evidence. Plus block-bootstrap Monte Carlo (Sharpe/maxDD distributions) and the
  deflated Sharpe ratio penalizing the number of trials.
- **Experiment tracking** (`quantis/research/tracking.py`) — every backtest/walk-forward
  logs params, metrics, and artifacts to MLflow when installed, else to a local JSONL
  registry (`runs/experiments.jsonl`). The loop never depends on a tracking server.
- **Research workspace** (`quantis ui`) — FastAPI (`/v1/...` per the TDD API spec) +
  a terminal-style web UI: run risk-gated backtests, walk-forwards, browse the run
  registry, and see vetoes broken down by risk rule.

## What's in the MVP (Phase 1)

- **Data lake** — daily NSE OHLCV as per-symbol Parquet (`quantis/data/`). Sources:
  Yahoo Finance (`.NS`, split/dividend-adjusted) or a deterministic synthetic generator
  for offline work and CI.
- **Point-in-time features** (`quantis/features/`) — returns, momentum (12-1), SMAs,
  RSI, ATR, realized vol, z-score, rupee ADV. Contract: row *t* uses only data through
  the close of *t*, enforced by a mechanical perturbation test.
- **Strategy plug-ins** (`quantis/strategies/`) — one interface, three templates:
  cross-sectional momentum, MA-crossover trend following, RSI mean reversion.
  One weights frame feeds both engines (no research/production divergence).
- **Risk engine with veto authority** (`quantis/risk/`) — every order transits
  `RiskEngine.evaluate()`; there is no bypass path. Phase-1 limit set: position weight,
  sector weight, gross exposure, ADV participation, daily-loss halt, drawdown
  kill-switch, and an order-notional sanity bound. Every decision (including
  rejections) is persisted as the audit trail.
- **Two backtest engines** (`quantis/backtest/`) sharing one NSE cost model
  (brokerage/STT/stamp/exchange/SEBI/GST + square-root impact slippage):
  - *Event-driven* — bar-by-bar, integer cash, per-order risk gate. Final validation.
  - *Vectorized* — whole-history matrix math for parameter sweeps.
- **Run artifacts** — every backtest writes `runs/<ts>_<strategy>/` with report,
  metrics JSON, equity curve, fills, and the full risk-decision log.

## Quickstart

```bash
pip install -e ".[data,research,dev]"

# 1. Ingest data (real NSE via Yahoo, or --source synthetic for offline)
quantis ingest --source yahoo --start 2018-01-01
quantis ingest --source synthetic            # offline alternative

# 2. Run a risk-gated backtest
quantis backtest --strategy momentum --start 2019-01-01
quantis backtest --strategy ma_crossover --param fast=10 --param slow=100

# 3. Parameter sweep (vectorized), then validate the winner in the event engine
quantis sweep --strategy ma_crossover --grid fast=10,20,50 --grid slow=50,100,200

# 4. Walk-forward validation — out-of-sample evidence, Monte Carlo, deflated Sharpe
quantis walkforward --strategy momentum --grid top_n=5,10 --grid rebalance_days=21,42

# 5. Feature store + research workspace
quantis materialize                          # versioned point-in-time feature tables
quantis ui                                   # http://127.0.0.1:8000

quantis list
pytest                                        # cost model, risk rules, look-ahead, PIT store, walk-forward
```

## Design commitments carried from the TDD

| TDD principle | Where it lives here |
|---|---|
| Risk engine has unconditional veto; no order bypasses it | `risk/engine.py`, enforced in `backtest/engine.py` |
| One code path for research and simulation | `strategies/base.py` weights contract feeds both engines |
| Same cost model in fast and slow engines | `backtest/costs.py` shared by `engine.py` and `vectorized.py` |
| Look-ahead bias prevented structurally and tested | `tests/test_no_lookahead.py` perturbation test |
| Every risk decision logged with limit snapshot (audit trail) | `RiskDecision.limit_snapshot`, `risk_decisions.csv` per run |
| Lake format portable to S3/Timescale later | Parquet long-format bars, `data/store.py` |

## Roadmap (from the TDD, Part 16)

1. **MVP** — data, features, backtester, strategy templates, risk limits ✅
2. **Research platform** — feature store, walk-forward validation, experiment tracking, research UI ✅
3. Paper trading — real-time feed, simulated broker adapter, OMS/EMS skeleton, full limit set
4. AI integration — GBT + sequence models, model registry, shadow-mode promotion, LLM copilot
5. Live trading — broker connectors (Zerodha/Upstox), reconciliation, circuit breakers, SEBI audit tagging
6. Institutional — multi-asset, multi-tenant RBAC, strategy marketplace, white-label API

Full design: `docs/` (Quantis TDD PDF).
