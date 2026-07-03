"""Local data lake: one Parquet file of daily bars per symbol.

Mirrors the TDD's data-plane split at MVP scale — Parquet is the lake
format from day one, so promoting to S3 + TimescaleDB later is a change
of location, not of schema. Schema (long format):

    symbol, ts, open, high, low, close, volume
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BAR_COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume"]


class BarLake:
    def __init__(self, root: str | Path = "data/lake"):
        self.root = Path(root)
        self.bars_dir = self.root / "bars"
        self.bars_dir.mkdir(parents=True, exist_ok=True)

    def save_bars(self, df: pd.DataFrame) -> int:
        """Upsert daily bars (long format) into per-symbol Parquet files."""
        missing = set(BAR_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Bar frame missing columns: {sorted(missing)}")
        n = 0
        for symbol, grp in df.groupby("symbol"):
            path = self.bars_dir / f"{symbol}.parquet"
            grp = grp[BAR_COLUMNS].copy()
            if path.exists():
                old = pd.read_parquet(path)
                grp = pd.concat([old, grp], ignore_index=True)
                grp = grp.drop_duplicates(subset=["symbol", "ts"], keep="last")
            grp = grp.sort_values("ts").reset_index(drop=True)
            grp.to_parquet(path, index=False)
            n += len(grp)
        return n

    def load_bars(
        self,
        symbols: list[str],
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        frames = []
        for symbol in symbols:
            path = self.bars_dir / f"{symbol}.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path)
            if start:
                df = df[df["ts"] >= pd.Timestamp(start)]
            if end:
                df = df[df["ts"] <= pd.Timestamp(end)]
            frames.append(df)
        if not frames:
            raise FileNotFoundError(
                f"No bars in lake at {self.bars_dir}. Run `quantis ingest` first."
            )
        return pd.concat(frames, ignore_index=True).sort_values(["ts", "symbol"])

    def available_symbols(self) -> list[str]:
        return sorted(p.stem for p in self.bars_dir.glob("*.parquet"))


def to_wide(bars: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Pivot long bars into wide date-by-symbol frames used by the engines."""
    out = {}
    for field in ["open", "high", "low", "close", "volume"]:
        out[field] = bars.pivot(index="ts", columns="symbol", values=field).sort_index()
    return out
