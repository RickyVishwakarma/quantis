import json

from quantis.audit import AuditLog


def test_append_and_verify(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("session_start", {"strategy": "momentum"})
    log.append("risk_decision", {"outcome": "APPROVE", "symbol": "TCS"})
    log.append("fill", {"qty": 10, "price": 3500.0})
    ok, bad = log.verify()
    assert ok and bad is None
    recs = log.records()
    assert [r["seq"] for r in recs] == [1, 2, 3]
    assert recs[1]["prev_hash"] == recs[0]["hash"]


def test_chain_survives_restart(tmp_path):
    path = tmp_path / "audit.jsonl"
    AuditLog(path).append("a", {})
    log2 = AuditLog(path)          # new writer re-seeds from disk
    log2.append("b", {})
    ok, _ = log2.verify()
    assert ok
    assert [r["seq"] for r in log2.records()] == [1, 2]


def test_tampering_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(4):
        log.append("risk_decision", {"i": i, "outcome": "APPROVE"})

    lines = path.read_text(encoding="utf-8").splitlines()
    rec = json.loads(lines[2])
    rec["payload"]["outcome"] = "REJECT"          # rewrite history
    lines[2] = json.dumps(rec)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    ok, bad = AuditLog(path).verify()
    assert not ok and bad == 3


def test_deleting_a_record_breaks_the_chain(tmp_path):
    path = tmp_path / "audit.jsonl"
    log = AuditLog(path)
    for i in range(3):
        log.append("fill", {"i": i})
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2]]) + "\n", encoding="utf-8")
    ok, bad = AuditLog(path).verify()
    assert not ok and bad == 3
