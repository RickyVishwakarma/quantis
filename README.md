# Quantis

AI quantitative research and execution platform for Indian equities and derivatives.
This repo implements **Phases 1–4** of the Quantis TDD: the research loop, paper trading
through a real OMS/EMS, and AI signal generation with a governed model lifecycle —
every AI order transits the same risk gate as every template strategy.

## Phase 4 — AI integration

- **Signal models** (`quantis/ai/models.py`) — one contract (`fit / predict /
  attribution`): a dependency-free closed-form ridge baseline (prediction == sum of
  contributions, exactly attributable) and LightGBM (`pip install "quantis[ai]"`,
  SHAP-style attribution). Many small understood signals, per the TDD, not one opaque net.
- **Training pipeline** (`quantis/ai/train.py`) — training frames from the feature store
  (PIT features, strictly-future labels), date-based splits (never shuffled), and
  promotion to CANDIDATE only if validation IC beats a naive momentum baseline.
- **Model registry** (`quantis/ai/registry.py`) — the TDD stage lifecycle
  `EXPERIMENTAL → CANDIDATE → SHADOW → PRODUCTION → RETIRED`, enforced: no stage
  skipping, PRODUCTION requires a shadow report **and** human `--approved-by`,
  one PRODUCTION model per name, `feature_schema_version` pinned on every entry.
- **Shadow mode** (`quantis/ai/shadow.py`) — infer, don't trade: recent-window
  hypothetical performance, realized IC, benchmark comparison, sanity-bound rejections.
- **AI strategy plug-in** (`quantis/strategies/ai_signal.py`) — model scores → weights
  through the standard Strategy interface, with the TDD's AI safeguards: out-of-
  distribution predictions are zeroed (hallucination bound), `gross_cap` limits any one
  model's share of the book, and `explain()` ships feature attribution with every pick.
- **AI copilot** (`quantis/ai/copilot.py`, `POST /v1/copilot/query`) — Claude answering
  questions grounded in live platform state (lake, runs, registry, risk status);
  degrades to a deterministic local summary without an API key. Strictly read-only.
- **UI** — MODELS view (registry, stages, shadow reports, promote-with-sign-off) and
  AI COPILOT view in the research workspace.

```bash
quantis ai train --model gbt --label-horizon 5
quantis ai shadow --model <id> --days 126
quantis ai promote --model <id> --to PRODUCTION --approved-by ricky
quantis backtest --strategy ai_signal --param model_id=<id>
quantis ai ask "which model should go to production?"
```

## Phase 3 — paper trading

- **Broker abstraction** (`quantis/broker/`) — one interface (`place / cancel /
  poll_fills / positions / margins`), idempotent on `client_order_id` so a network retry
  can never double-execute. `SimulatedBroker` fills at next-bar open with the shared NSE
  cost model; Phase 5 adds Zerodha/Upstox adapters behind the same interface.
- **OMS** (`quantis/oms/`) — order state machine with the TDD's exact status lifecycle
  (`PENDING_RISK → APPROVED → SENT → PARTIALLY_FILLED → FILLED/CANCELLED/ERROR`);
  illegal transitions raise; every transition appends to an `orders.jsonl` audit journal.
- **EMS skeleton** (`quantis/ems/`) — execution algos: immediate and TWAP slicing
  (each child's ADV participation is 1/N of the block's, so impact is measurably lower —
  tested). Refuses any order that isn't risk-APPROVED.
- **Full risk limit set** (`quantis/risk/live.py`) — tiered drawdown response
  (NORMAL → HALVED → FLATTEN → HALTED), circuit breakers (consecutive rejections, feed
  staleness, broker error spike, manual panic button), volatility targeting, and manual
  `reset()` — a halt never re-arms itself.
- **Paper engine** (`quantis/paper/`) — feed → strategy (same code path as backtest) →
  risk gate → OMS → EMS → sim broker, decision at close t / fill at open t+1. Sessions
  persist journal, fills, equity, risk decisions, and a reconciliation report.
- **Reconciliation** — diffs OMS fill-implied positions vs the broker's book
  (the TDD's network-partition safeguard), run at session end and on demand.
- **Backtest parity, tested** — same data, same limits, same first trading day: paper
  and backtest agree to ~0.2% terminal wealth on the controlled test; the CLI `--parity`
  flag reports live tracking divergence on any real run.

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

# 5. Paper trading — replay feed through OMS/EMS/sim-broker with the full risk set
quantis paper --strategy momentum --start 2024-01-01 --parity
quantis paper --strategy ma_crossover --algo twap --slices 4

# 6. Feature store + research workspace
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
3. **Paper trading** — replay/delayed feed, simulated broker adapter, OMS/EMS, full limit set ✅
4. **AI integration** — GBT + ridge signal models, model registry, shadow-mode promotion, LLM copilot ✅
   (deferred to later: deep sequence models — the registry/promotion pipeline is model-agnostic)
5. Live trading — broker connectors (Zerodha/Upstox), reconciliation, circuit breakers, SEBI audit tagging
6. Institutional — multi-asset, multi-tenant RBAC, strategy marketplace, white-label API

Full design: `docs/` (Quantis TDD PDF).
