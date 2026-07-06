"""Audit service: append-only, hash-chained event log (TDD Part 13).

Every record carries the SHA-256 of the previous record, so the log is
tamper-evident: editing, deleting, or reordering any historical record
breaks the chain at that sequence number and ``verify()`` reports it.
This is the WORM-style trail SEBI record-keeping expects — orders, risk
decisions, breaker events, limit changes, session lifecycle.

The chain survives process restarts (the writer re-seeds from the last
record on disk). JSONL on purpose: greppable, appendable, no infra.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64


def _record_hash(record: dict) -> str:
    material = json.dumps(
        {k: v for k, v in record.items() if k != "hash"},
        sort_keys=True, default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._seq = 0
        self._prev_hash = GENESIS
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._seq = rec["seq"]
                    self._prev_hash = rec["hash"]

    # ------------------------------------------------------------------
    def append(self, event_type: str, payload: dict) -> dict:
        self._seq += 1
        record = {
            "seq": self._seq,
            "ts": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "payload": payload,
            "prev_hash": self._prev_hash,
        }
        record["hash"] = _record_hash(record)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
        self._prev_hash = record["hash"]
        return record

    # ------------------------------------------------------------------
    def records(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in
                self.path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def verify(self) -> tuple[bool, int | None]:
        """Walk the chain; returns (ok, first_bad_seq)."""
        prev = GENESIS
        for rec in self.records():
            if rec.get("prev_hash") != prev or _record_hash(rec) != rec.get("hash"):
                return False, rec.get("seq")
            prev = rec["hash"]
        return True, None
