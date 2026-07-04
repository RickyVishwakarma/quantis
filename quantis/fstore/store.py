"""Materialized, versioned, point-in-time feature store.

Implements the TDD's feature-store contract at MVP scale using the exact
offline-table schema from Appendix A:

    (instrument_id, feature_name, as_of_ts, value, schema_version)

``as_of_ts`` is when the value became KNOWABLE (close of bar t), so an
as-of query can never surface a future value. Storage is one Parquet
file per feature under ``data/feature_store/`` — the same schema Feast
would serve, so swapping in Feast + Redis for online serving in Phase 3
is a backend change, not an API change.

Why not Feast today: until there is a live path (Phase 3), the only
consumer is offline research; a purpose-built as-of join over Parquet
keeps zero infra while matching Feast's offline semantics.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..features import compute_features

# Bump when a feature definition changes meaning; models record the
# version they trained on (TDD models.feature_schema_version).
FEATURE_SCHEMA_VERSION = "1"

_TABLE_COLUMNS = ["instrument_id", "feature_name", "as_of_ts", "value", "schema_version"]


class FeatureStore:
    def __init__(self, root: str | Path = "data/feature_store"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, feature_name: str) -> Path:
        return self.root / f"{feature_name}.parquet"

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------
    def materialize(
        self,
        wide: dict[str, pd.DataFrame],
        schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> dict[str, int]:
        """Compute the registered feature set and upsert into the store."""
        panel = compute_features(wide)
        counts: dict[str, int] = {}
        for name in panel.names():
            frame = panel[name].copy()
            frame.index.name, frame.columns.name = "as_of_ts", "instrument_id"
            long = frame.stack().rename("value").reset_index()
            long["schema_version"] = schema_version
            long["feature_name"] = name
            long = long[_TABLE_COLUMNS].dropna(subset=["value"])

            path = self._path(name)
            if path.exists():
                old = pd.read_parquet(path)
                long = pd.concat([old, long], ignore_index=True).drop_duplicates(
                    subset=["instrument_id", "feature_name", "as_of_ts", "schema_version"],
                    keep="last",
                )
            long = long.sort_values(["instrument_id", "as_of_ts"]).reset_index(drop=True)
            long.to_parquet(path, index=False)
            counts[name] = len(long)
        return counts

    # ------------------------------------------------------------------
    # Read paths
    # ------------------------------------------------------------------
    def available_features(self) -> list[str]:
        return sorted(p.stem for p in self.root.glob("*.parquet"))

    def _load(
        self,
        feature_name: str,
        schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> pd.DataFrame:
        path = self._path(feature_name)
        if not path.exists():
            raise FileNotFoundError(
                f"Feature {feature_name!r} not materialized. Run `quantis materialize`."
            )
        df = pd.read_parquet(path)
        return df[df["schema_version"] == schema_version]

    def get_wide(
        self,
        feature_name: str,
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> pd.DataFrame:
        """Read a feature back as a wide date-by-symbol frame."""
        df = self._load(feature_name, schema_version)
        if symbols:
            df = df[df["instrument_id"].isin(symbols)]
        if start:
            df = df[df["as_of_ts"] >= pd.Timestamp(start)]
        if end:
            df = df[df["as_of_ts"] <= pd.Timestamp(end)]
        return df.pivot(index="as_of_ts", columns="instrument_id", values="value").sort_index()

    def get_asof(
        self,
        feature_names: list[str],
        symbols: list[str],
        as_of: str | pd.Timestamp,
        schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> pd.DataFrame:
        """Point-in-time snapshot: latest value with as_of_ts <= as_of.

        This is THE query that prevents look-ahead bias in training-set
        construction — it structurally cannot return a future row.
        """
        as_of = pd.Timestamp(as_of)
        out = {}
        for name in feature_names:
            df = self._load(name, schema_version)
            df = df[df["instrument_id"].isin(symbols) & (df["as_of_ts"] <= as_of)]
            latest = (
                df.sort_values("as_of_ts")
                .groupby("instrument_id")
                .last()["value"]
            )
            out[name] = latest
        return pd.DataFrame(out).reindex(symbols)

    def training_frame(
        self,
        feature_names: list[str],
        symbols: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        label_horizon: int | None = None,
        close: pd.DataFrame | None = None,
        schema_version: str = FEATURE_SCHEMA_VERSION,
    ) -> pd.DataFrame:
        """Tidy (ts, symbol) frame of features, optionally with a forward-
        return label for supervised training (Phase 4 prep).

        The label at row t is the close-to-close return from t to t+h —
        strictly future information relative to every feature at t, which
        is exactly what a prediction target must be.
        """
        frames = []
        for name in feature_names:
            w = self.get_wide(name, symbols, start, end, schema_version)
            frames.append(w.stack().rename(name))
        out = pd.concat(frames, axis=1)
        out.index.names = ["ts", "instrument_id"]

        if label_horizon is not None:
            if close is None:
                raise ValueError("label_horizon requires the close price frame")
            fwd = close.pct_change(label_horizon).shift(-label_horizon)
            out[f"label_fwd_ret_{label_horizon}d"] = fwd.stack().reindex(out.index)
        return out.reset_index()
