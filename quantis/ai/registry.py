"""Model registry with the TDD's stage lifecycle (Part 5 / Appendix A).

    EXPERIMENTAL -> CANDIDATE -> SHADOW -> PRODUCTION -> RETIRED

Rules enforced here, not by convention:
  - stage transitions must follow the lifecycle (no skipping to PRODUCTION)
  - promotion to PRODUCTION requires a human sign-off (``approved_by``) AND
    a shadow report on file — the TDD's "human-approved promotion" gate
  - at most one PRODUCTION model per name; promoting a new one retires
    the incumbent
  - every entry records feature_schema_version so a model can never be
    served features computed under a different schema

Storage is a local JSON registry + pickled artifacts (the MLflow-backed
version is a backend swap; the experiment tracker already logs to MLflow
when installed).
"""

from __future__ import annotations

import json
import pickle
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from ..fstore import FEATURE_SCHEMA_VERSION


class Stage(str, Enum):
    EXPERIMENTAL = "EXPERIMENTAL"
    CANDIDATE = "CANDIDATE"
    SHADOW = "SHADOW"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


TRANSITIONS: dict[Stage, set[Stage]] = {
    Stage.EXPERIMENTAL: {Stage.CANDIDATE, Stage.RETIRED},
    Stage.CANDIDATE: {Stage.SHADOW, Stage.RETIRED},
    Stage.SHADOW: {Stage.PRODUCTION, Stage.RETIRED},
    Stage.PRODUCTION: {Stage.RETIRED},
    Stage.RETIRED: set(),
}


class PromotionError(RuntimeError):
    pass


class ModelRegistry:
    def __init__(self, root: str | Path = "models"):
        self.root = Path(root)
        self.artifacts = self.root / "artifacts"
        self.artifacts.mkdir(parents=True, exist_ok=True)
        self._index = self.root / "registry.json"

    # ------------------------------------------------------------------
    def _load_index(self) -> list[dict]:
        if not self._index.exists():
            return []
        return json.loads(self._index.read_text(encoding="utf-8"))

    def _save_index(self, entries: list[dict]) -> None:
        self._index.write_text(json.dumps(entries, indent=2, default=str),
                               encoding="utf-8")

    # ------------------------------------------------------------------
    def register(
        self,
        name: str,
        model,
        metrics: dict,
        feature_names: list[str],
        label: str,
        signal_bounds: tuple[float, float] | None = None,
        extra: dict | None = None,
    ) -> dict:
        entries = self._load_index()
        version = 1 + max((e["version"] for e in entries if e["name"] == name),
                          default=0)
        model_id = uuid.uuid4().hex[:12]
        with (self.artifacts / f"{model_id}.pkl").open("wb") as f:
            pickle.dump(model, f)
        entry = {
            "model_id": model_id,
            "name": name,
            "version": version,
            "stage": Stage.EXPERIMENTAL.value,
            "metrics": metrics,
            "feature_names": feature_names,
            "label": label,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "signal_bounds": list(signal_bounds) if signal_bounds else None,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "promoted_at": None,
            "approved_by": None,
            "shadow_report": None,
            **(extra or {}),
        }
        entries.append(entry)
        self._save_index(entries)
        return entry

    # ------------------------------------------------------------------
    def get(self, model_id: str) -> dict:
        for e in self._load_index():
            if e["model_id"] == model_id:
                return e
        raise KeyError(f"model {model_id!r} not in registry")

    def resolve(self, ref: str) -> dict:
        """Resolve 'production:<name>' / 'production' / a model_id."""
        if ref.startswith("production"):
            name = ref.split(":", 1)[1] if ":" in ref else None
            prod = [e for e in self._load_index()
                    if e["stage"] == Stage.PRODUCTION.value
                    and (name is None or e["name"] == name)]
            if not prod:
                raise KeyError(
                    f"no PRODUCTION model{f' named {name!r}' if name else ''} in the "
                    "registry — promote one (`quantis ai promote --model <id> --to "
                    "PRODUCTION --approved-by <you>`) or pass an explicit model_id"
                )
            return sorted(prod, key=lambda e: e["promoted_at"] or "")[-1]
        return self.get(ref)

    def load_model(self, ref: str):
        entry = self.resolve(ref)
        with (self.artifacts / f"{entry['model_id']}.pkl").open("rb") as f:
            return entry, pickle.load(f)

    def list_models(self) -> list[dict]:
        return self._load_index()

    # ------------------------------------------------------------------
    def promote(self, model_id: str, to: str | Stage,
                approved_by: str | None = None) -> dict:
        to = Stage(to)
        entries = self._load_index()
        entry = next((e for e in entries if e["model_id"] == model_id), None)
        if entry is None:
            raise KeyError(f"model {model_id!r} not in registry")
        current = Stage(entry["stage"])
        if to not in TRANSITIONS[current]:
            raise PromotionError(
                f"{current.value} -> {to.value} is not a legal transition "
                f"(allowed: {sorted(s.value for s in TRANSITIONS[current])})"
            )
        if to == Stage.PRODUCTION:
            if not approved_by:
                raise PromotionError(
                    "PRODUCTION promotion requires human sign-off (approved_by) "
                    "— the TDD's promotion gate, deliberately not automatable"
                )
            if not entry.get("shadow_report"):
                raise PromotionError(
                    "PRODUCTION promotion requires a shadow report; run "
                    "`quantis ai shadow` first (infer, don't trade)"
                )
            # single production model per name
            for e in entries:
                if (e["name"] == entry["name"] and e["model_id"] != model_id
                        and e["stage"] == Stage.PRODUCTION.value):
                    e["stage"] = Stage.RETIRED.value
        entry["stage"] = to.value
        entry["promoted_at"] = datetime.now(timezone.utc).isoformat()
        if approved_by:
            entry["approved_by"] = approved_by
        self._save_index(entries)
        return entry

    def attach_shadow_report(self, model_id: str, report: dict) -> dict:
        entries = self._load_index()
        entry = next((e for e in entries if e["model_id"] == model_id), None)
        if entry is None:
            raise KeyError(f"model {model_id!r} not in registry")
        entry["shadow_report"] = report
        self._save_index(entries)
        return entry
