import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from quantis.api import create_app  # noqa: E402
from quantis.data.ingest import generate_synthetic  # noqa: E402
from quantis.data.store import BarLake  # noqa: E402

SYMS = ["RELIANCE", "TCS", "INFY", "HDFCBANK", "ITC"]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    root = tmp_path_factory.mktemp("api")
    lake = BarLake(root / "lake")
    lake.save_bars(generate_synthetic(SYMS, start="2021-01-01", end="2023-12-31", seed=9))
    app = create_app(lake_root=str(root / "lake"), runs_root=str(root / "runs"),
                     paper_root=str(root / "paper_sessions"))
    return TestClient(app)


def test_strategies_endpoint(client):
    names = {s["name"] for s in client.get("/v1/strategies").json()}
    assert {"momentum", "ma_crossover", "mean_reversion"} <= names


def test_lake_endpoint(client):
    lake = client.get("/v1/lake").json()
    assert lake["n_symbols"] == len(SYMS)
    assert lake["start"] and lake["end"]


def test_backtest_run_and_registry(client):
    res = client.post("/v1/backtests", json={
        "strategy": "ma_crossover",
        "params": {"fast": 10, "slow": 50},
        "capital": 1_000_000,
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert "sharpe" in body["metrics"]

    runs = client.get("/v1/runs").json()
    assert any(r["run_id"] == body["run_id"] for r in runs)

    detail = client.get(f"/v1/runs/{body['run_id']}").json()
    assert len(detail["equity"]["ts"]) == len(detail["equity"]["value"]) > 100
    assert detail["risk"].get("evaluated", 0) > 0


def test_paper_replay_and_registry(client):
    res = client.post("/v1/paper/replay", json={
        "strategy": "ma_crossover",
        "params": {"fast": 10, "slow": 50},
        "capital": 1_000_000,
        "warmup": 210,
    })
    assert res.status_code == 200, res.text
    body = res.json()
    assert "sharpe" in body["metrics"]
    assert body["risk_status"]["breaker_tripped"] is False
    assert "clean" in body["reconciliation"]

    sessions = client.get("/v1/paper/sessions").json()
    assert any(s["session_id"] == body["session_id"] for s in sessions)

    detail = client.get(f"/v1/paper/sessions/{body['session_id']}").json()
    assert len(detail["equity"]["ts"]) > 100
    assert detail["strategy"] == "ma_crossover"


def test_unknown_strategy_422(client):
    assert client.post("/v1/backtests", json={"strategy": "nope"}).status_code == 422


def test_missing_run_404(client):
    assert client.get("/v1/runs/doesnotexist").status_code == 404
