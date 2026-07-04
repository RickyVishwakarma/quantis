"""Experiment tracking (TDD: experiment tracking as a first-class citizen).

Every backtest / sweep / walk-forward is an experiment run with params,
metrics, and artifacts. Backend is MLflow when installed (`pip install
"quantis[research]"`), else a local JSONL registry — the research loop
never breaks because a tracking server is missing.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def _scrub(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        if isinstance(v, float) and not np.isfinite(v):
            v = None
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
    return out


class LocalTracker:
    """Append-only JSONL registry at runs/experiments.jsonl."""

    backend = "local"

    def __init__(self, root: str | Path = "runs"):
        self.path = Path(root) / "experiments.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log_run(self, name: str, params: dict, metrics: dict,
                artifacts_dir: str | Path | None = None,
                tags: dict | None = None) -> str:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        record = {
            "run_id": run_id,
            "name": name,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "params": _scrub(params),
            "metrics": _scrub(metrics),
            "tags": _scrub(tags or {}),
            "artifacts": str(artifacts_dir) if artifacts_dir else None,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        return run_id

    def list_runs(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class MLflowTracker:
    backend = "mlflow"

    def __init__(self, experiment: str = "quantis"):
        import mlflow  # noqa: F401 — import checked by get_tracker

        self._mlflow = mlflow
        mlflow.set_experiment(experiment)

    def log_run(self, name: str, params: dict, metrics: dict,
                artifacts_dir: str | Path | None = None,
                tags: dict | None = None) -> str:
        mlflow = self._mlflow
        with mlflow.start_run(run_name=name) as run:
            mlflow.log_params(_scrub(params))
            mlflow.log_metrics({k: v for k, v in _scrub(metrics).items()
                                if isinstance(v, (int, float)) and v is not None})
            if tags:
                mlflow.set_tags(_scrub(tags))
            if artifacts_dir and Path(artifacts_dir).exists():
                mlflow.log_artifacts(str(artifacts_dir))
            return run.info.run_id


def get_tracker(root: str | Path = "runs"):
    try:
        return MLflowTracker()
    except Exception:
        return LocalTracker(root)
