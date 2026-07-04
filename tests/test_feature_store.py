import pandas as pd
import pytest

from quantis.data.ingest import generate_synthetic
from quantis.data.store import to_wide
from quantis.features import compute_features
from quantis.fstore import FEATURE_SCHEMA_VERSION, FeatureStore

SYMS = ["RELIANCE", "TCS", "INFY"]


@pytest.fixture(scope="module")
def wide():
    bars = generate_synthetic(SYMS, start="2022-01-01", end="2023-06-30", seed=5)
    return to_wide(bars)


@pytest.fixture()
def fs(tmp_path, wide):
    store = FeatureStore(tmp_path / "fstore")
    store.materialize(wide)
    return store


def test_roundtrip_matches_panel(fs, wide):
    panel = compute_features(wide)
    stored = fs.get_wide("rsi_14", symbols=SYMS)
    direct = panel["rsi_14"].dropna(how="all")
    common = stored.index.intersection(direct.index)
    pd.testing.assert_frame_equal(
        stored.loc[common, SYMS], direct.loc[common, SYMS],
        check_names=False, check_freq=False,
    )


def test_asof_never_returns_future(fs, wide):
    panel = compute_features(wide)
    rsi = panel["rsi_14"]
    dates = rsi.dropna(how="all").index
    mid = dates[len(dates) // 2]
    # as_of one day BEFORE `mid` must return the value from a bar < mid
    day_before = mid - pd.Timedelta(days=1)
    snap = fs.get_asof(["rsi_14"], SYMS, as_of=day_before)
    prior_dates = dates[dates <= day_before]
    expected = rsi.loc[prior_dates[-1]]
    for sym in SYMS:
        assert snap.loc[sym, "rsi_14"] == pytest.approx(expected[sym])
        # and it must NOT equal mid's value unless coincidentally identical
        assert snap.loc[sym, "rsi_14"] != pytest.approx(rsi.loc[mid, sym]) or \
            expected[sym] == pytest.approx(rsi.loc[mid, sym])


def test_asof_before_history_is_empty(fs):
    snap = fs.get_asof(["rsi_14"], SYMS, as_of="2010-01-01")
    assert snap["rsi_14"].isna().all()


def test_schema_version_isolation(fs, wide):
    fs.materialize(wide, schema_version="experimental")
    v1 = fs.get_wide("ret_1d", schema_version=FEATURE_SCHEMA_VERSION)
    exp = fs.get_wide("ret_1d", schema_version="experimental")
    assert len(v1) > 0 and len(exp) > 0
    with pytest.raises(FileNotFoundError):
        fs.get_wide("nonexistent_feature")


def test_training_frame_label_is_strictly_future(fs, wide):
    close = wide["close"]
    tf = fs.training_frame(["ret_1d", "rsi_14"], symbols=SYMS,
                           label_horizon=5, close=close)
    tf = tf.dropna(subset=["label_fwd_ret_5d"])
    row = tf.iloc[len(tf) // 2]
    t = pd.Timestamp(row["ts"])
    sym = row["instrument_id"]
    idx = close.index.get_loc(t)
    expected = close.iloc[idx + 5][sym] / close.iloc[idx][sym] - 1
    assert row["label_fwd_ret_5d"] == pytest.approx(expected)
