# Quantis

AI quantitative research and execution platform for Indian equities and derivatives.
This repo is **Phase 1 (MVP)** of the Quantis TDD: the research loop — data → features →
strategy → risk-gated backtest → report — built so that every later phase (paper trading,
AI signals, live execution) extends it without a rewrite.

## What's in the MVP

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
pip install -e ".[data,dev]"

# 1. Ingest data (real NSE via Yahoo, or --source synthetic for offline)
quantis ingest --source yahoo --start 2018-01-01
quantis ingest --source synthetic            # offline alternative

# 2. Run a risk-gated backtest
quantis backtest --strategy momentum --start 2019-01-01
quantis backtest --strategy ma_crossover --param fast=10 --param slow=100

# 3. Parameter sweep (vectorized), then validate the winner in the event engine
quantis sweep --strategy ma_crossover --grid fast=10,20,50 --grid slow=50,100,200

quantis list
pytest                                        # cost model, risk rules, look-ahead checks
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

1. **MVP (this repo)** — data, features, backtester, strategy templates, risk limits ✅
2. Research platform — feature store (Feast), walk-forward validation, MLflow, research UI
3. Paper trading — real-time feed, simulated broker adapter, OMS/EMS skeleton, full limit set
4. AI integration — GBT + sequence models, model registry, shadow-mode promotion, LLM copilot
5. Live trading — broker connectors (Zerodha/Upstox), reconciliation, circuit breakers, SEBI audit tagging
6. Institutional — multi-asset, multi-tenant RBAC, strategy marketplace, white-label API

Full design: `docs/` (Quantis TDD PDF).
