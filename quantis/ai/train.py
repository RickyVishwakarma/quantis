"""Training pipeline (TDD Part 4 AI loop: features -> train -> eval -> register).

Point-in-time discipline:
  - features come from the feature store's training_frame (row t uses
    only data through close t; label is the strictly-future fwd return)
  - the train/validation split is BY DATE, never shuffled — shuffling
    would leak future cross-sectional structure into training
  - the candidate must beat a naive baseline (ret_21d as the predictor)
    on validation IC before it can be promoted to CANDIDATE

Evaluation metrics (stored on the registry entry):
  ic                 mean per-date Spearman rank correlation (pred vs label)
  hit_rate           sign agreement on validation rows
  top_bottom_spread  mean label of top decile preds minus bottom decile
  baseline_ic        same IC for the naive momentum predictor
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..fstore import FeatureStore
from .models import make_model
from .registry import ModelRegistry, Stage

DEFAULT_FEATURES = [
    "ret_5d", "ret_21d", "mom_63", "mom_126_21",
    "px_vs_sma_200", "rsi_14", "zscore_20", "vol_21", "atr_pct",
]


def _per_date_ic(frame: pd.DataFrame, pred_col: str, label_col: str) -> float:
    ics = []
    for _, grp in frame.groupby("ts"):
        if len(grp) >= 5:
            ic = grp[pred_col].corr(grp[label_col], method="spearman")
            if pd.notna(ic):
                ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def _top_bottom_spread(frame: pd.DataFrame, pred_col: str, label_col: str) -> float:
    spreads = []
    for _, grp in frame.groupby("ts"):
        if len(grp) >= 10:
            k = max(len(grp) // 10, 1)
            ranked = grp.sort_values(pred_col)
            spreads.append(ranked[label_col].tail(k).mean()
                           - ranked[label_col].head(k).mean())
    return float(np.mean(spreads)) if spreads else float("nan")


def train_and_register(
    wide: dict[str, pd.DataFrame],
    model_type: str = "ridge",
    feature_names: list[str] | None = None,
    label_horizon: int = 5,
    train_frac: float = 0.75,
    name: str | None = None,
    fstore_root: str = "data/feature_store",
    registry_root: str = "models",
) -> dict:
    feature_names = feature_names or DEFAULT_FEATURES
    name = name or f"{model_type}_fwd{label_horizon}d"

    fs = FeatureStore(fstore_root)
    fs.materialize(wide)
    tf = (
        fs.training_frame(feature_names, label_horizon=label_horizon,
                          close=wide["close"])
        .dropna()
        .sort_values("ts")
        .reset_index(drop=True)
    )
    label_col = f"label_fwd_ret_{label_horizon}d"
    if len(tf) < 500:
        raise ValueError(f"only {len(tf)} training rows — need more history")

    # Date-based split (no shuffle: time must not leak)
    dates = tf["ts"].unique()
    split_date = dates[int(len(dates) * train_frac)]
    train = tf[tf["ts"] < split_date]
    val = tf[tf["ts"] >= split_date].copy()

    model = make_model(model_type, feature_names)
    model.fit(train[feature_names].to_numpy(), train[label_col].to_numpy())

    val["pred"] = model.predict(val[feature_names].to_numpy())
    ic = _per_date_ic(val, "pred", label_col)
    hit_rate = float((np.sign(val["pred"]) == np.sign(val[label_col])).mean())
    spread = _top_bottom_spread(val, "pred", label_col)
    baseline_ic = _per_date_ic(val.rename(columns={"ret_21d": "naive"}),
                               "naive", label_col) if "ret_21d" in feature_names \
        else float("nan")

    train_preds = model.predict(train[feature_names].to_numpy())
    signal_bounds = (float(np.percentile(train_preds, 0.5)),
                     float(np.percentile(train_preds, 99.5)))

    metrics = {
        "ic": round(ic, 4),
        "hit_rate": round(hit_rate, 4),
        "top_bottom_spread": round(spread, 5),
        "baseline_ic": round(baseline_ic, 4) if not np.isnan(baseline_ic) else None,
        "n_train_rows": len(train),
        "n_val_rows": len(val),
        "val_start": str(pd.Timestamp(split_date).date()),
    }

    registry = ModelRegistry(registry_root)
    entry = registry.register(
        name=name, model=model, metrics=metrics,
        feature_names=feature_names, label=label_col,
        signal_bounds=signal_bounds,
        extra={"model_type": model_type, "label_horizon": label_horizon},
    )

    beats_baseline = ic > 0 and (np.isnan(baseline_ic) or ic > baseline_ic)
    if beats_baseline:
        entry = registry.promote(entry["model_id"], Stage.CANDIDATE)
    return entry
