import json

from quantis.research.tracking import LocalTracker, get_tracker


def test_local_tracker_appends_jsonl(tmp_path):
    t = LocalTracker(tmp_path)
    rid1 = t.log_run("momentum", {"top_n": 10}, {"sharpe": 1.2},
                     tags={"engine": "event"})
    rid2 = t.log_run("momentum", {"top_n": 5}, {"sharpe": float("nan")})
    assert rid1 != rid2
    runs = t.list_runs()
    assert len(runs) == 2
    assert runs[0]["params"]["top_n"] == 10
    assert runs[1]["metrics"]["sharpe"] is None      # NaN scrubbed for JSON


def test_local_tracker_scrubs_non_scalar(tmp_path):
    t = LocalTracker(tmp_path)
    t.log_run("x", {"grid": [1, 2], "ok": 3}, {"m": 1.0})
    run = t.list_runs()[0]
    assert "grid" not in run["params"] and run["params"]["ok"] == 3


def test_get_tracker_always_returns_working_backend(tmp_path):
    tracker = get_tracker(tmp_path)
    assert tracker.backend in ("mlflow", "local")
    rid = tracker.log_run("smoke", {"a": 1}, {"sharpe": 0.5})
    assert rid
