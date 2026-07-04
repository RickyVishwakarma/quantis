"""Delayed live feed: polls Yahoo Finance for new daily bars.

Suitable for EOD-decision paper trading (decide after close, fill next
open) without an NSE real-time data license — the TDD's cost mitigation
of starting on free/broker feeds. Yields a Bar whenever a new trading
day appears; ``staleness_seconds`` drives the feed-staleness circuit
breaker in ``LiveRiskManager``.

Experimental: exercised manually, not in CI (network dependency).
"""

from __future__ import annotations

import time
from collections.abc import Iterator

import pandas as pd

from ..data.ingest import fetch_yahoo
from .replay import Bar, MarketDataFeed


class DelayedYahooFeed(MarketDataFeed):
    def __init__(self, symbols: list[str], poll_secs: float = 300.0,
                 max_polls: int | None = None):
        self.symbols = symbols
        self.poll_secs = poll_secs
        self.max_polls = max_polls
        self._last_bar_wall: float = time.time()
        self._last_ts: pd.Timestamp | None = None

    def staleness_seconds(self) -> float:
        return time.time() - self._last_bar_wall

    def _fetch_latest(self) -> pd.DataFrame:
        start = (pd.Timestamp.now() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
        return fetch_yahoo(self.symbols, start=start, end=None)

    def __iter__(self) -> Iterator[Bar]:
        polls = 0
        while self.max_polls is None or polls < self.max_polls:
            polls += 1
            try:
                bars = self._fetch_latest()
                latest_ts = bars["ts"].max()
                if self._last_ts is None or latest_ts > self._last_ts:
                    self._last_ts = latest_ts
                    self._last_bar_wall = time.time()
                    day = bars[bars["ts"] == latest_ts].set_index("symbol")
                    yield Bar(ts=latest_ts,
                              open=day["open"], high=day["high"], low=day["low"],
                              close=day["close"], volume=day["volume"])
            except Exception:
                pass                                   # staleness clock keeps running
            time.sleep(self.poll_secs)
