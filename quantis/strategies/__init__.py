"""Strategy plug-in registry.

Strategies register themselves by name; the CLI and both backtest engines
look them up here. Adding a strategy = one module + one @register line.
"""

from __future__ import annotations

from .base import Strategy

_REGISTRY: dict[str, type[Strategy]] = {}


def register(cls: type[Strategy]) -> type[Strategy]:
    _REGISTRY[cls.name] = cls
    return cls


def get(name: str) -> type[Strategy]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown strategy {name!r}. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


# Import built-ins so they self-register
from . import ai_signal, momentum, ma_crossover, mean_reversion  # noqa: E402,F401
