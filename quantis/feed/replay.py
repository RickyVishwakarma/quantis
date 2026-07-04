"""Market data feeds for paper trading.

``ReplayFeed`` re-plays lake history bar by bar — the TDD's "replayable
log" idea at MVP scale. It drives tests, demos, and the backtest-parity
check. A delayed live feed (Yahoo polling) lives in ``delayed.py``;
both yield the same ``Bar`` so the paper engine is source-agnostic.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd


@dataclass
class Bar:
    """One period's snapshot across the whole universe (wide row)."""
    ts: pd.Timestamp
    open: pd.Series      # symbol -> price
    high: pd.Series
    low: pd.Series
    close: pd.Series
    volume: pd.Series


class MarketDataFeed(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[Bar]: ...

    def staleness_seconds(self) -> float:
        """Seconds since the last bar arrived (0 for replay)."""
        return 0.0


class ReplayFeed(MarketDataFeed):
    def __init__(self, wide: dict[str, pd.DataFrame],
                 start: str | None = None, end: str | None = None,
                 delay_secs: float = 0.0):
        self.wide = wide
        idx = wide["close"].index
        if start:
            idx = idx[idx >= pd.Timestamp(start)]
        if end:
            idx = idx[idx <= pd.Timestamp(end)]
        self.index = idx
        self.delay_secs = delay_secs

    def __iter__(self) -> Iterator[Bar]:
        for ts in self.index:
            if self.delay_secs:
                time.sleep(self.delay_secs)
            yield Bar(
                ts=ts,
                open=self.wide["open"].loc[ts],
                high=self.wide["high"].loc[ts],
                low=self.wide["low"].loc[ts],
                close=self.wide["close"].loc[ts],
                volume=self.wide["volume"].loc[ts],
            )
