"""Signal models (TDD Part 5: many small, individually-understood signals).

Every model implements the same tiny contract:

    fit(X, y)                    train on point-in-time feature rows
    predict(X) -> np.ndarray     cross-sectional score (higher = better)
    attribution(x) -> dict       per-feature contribution for one row —
                                 the explainability the TDD requires on
                                 every AI-produced signal

``RidgeSignalModel`` is the dependency-free baseline (closed-form ridge,
exactly attributable: prediction == sum of contributions). ``GBTSignalModel``
wraps LightGBM when installed (attribution via its built-in SHAP-style
pred_contrib). Both pickle cleanly for the registry.
"""

from __future__ import annotations

import numpy as np


class RidgeSignalModel:
    model_type = "ridge"

    def __init__(self, feature_names: list[str], alpha: float = 10.0):
        self.feature_names = feature_names
        self.alpha = alpha
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean_) / self.std_

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeSignalModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0
        Xs = self._standardize(X)
        n_f = Xs.shape[1]
        self.intercept_ = float(y.mean())
        yc = y - self.intercept_
        self.coef_ = np.linalg.solve(
            Xs.T @ Xs + self.alpha * np.eye(n_f), Xs.T @ yc
        )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._standardize(np.asarray(X, dtype=float)) @ self.coef_ + self.intercept_

    def attribution(self, x: np.ndarray) -> dict[str, float]:
        xs = self._standardize(np.asarray(x, dtype=float).reshape(1, -1))[0]
        contrib = xs * self.coef_
        return dict(sorted(
            zip(self.feature_names, contrib.round(6)),
            key=lambda kv: -abs(kv[1]),
        ))


class GBTSignalModel:
    model_type = "gbt"

    def __init__(self, feature_names: list[str], **lgb_params):
        import lightgbm  # noqa: F401 — fail fast if unavailable

        self.feature_names = feature_names
        self.params = {
            "objective": "regression",
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.05,
            "min_child_samples": 50,
            "verbosity": -1,
            **lgb_params,
        }
        self._model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GBTSignalModel":
        import lightgbm as lgb

        self._model = lgb.LGBMRegressor(**self.params)
        self._model.fit(np.asarray(X, dtype=float), np.asarray(y, dtype=float),
                        feature_name=self.feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict(np.asarray(X, dtype=float))

    def attribution(self, x: np.ndarray) -> dict[str, float]:
        contrib = self._model.predict(
            np.asarray(x, dtype=float).reshape(1, -1), pred_contrib=True
        )[0][:-1]  # last element is the expected value (bias)
        return dict(sorted(
            zip(self.feature_names, contrib.round(6)),
            key=lambda kv: -abs(kv[1]),
        ))


def make_model(model_type: str, feature_names: list[str], **kw):
    if model_type == "ridge":
        return RidgeSignalModel(feature_names, **kw)
    if model_type == "gbt":
        try:
            return GBTSignalModel(feature_names, **kw)
        except ImportError:
            raise ImportError(
                "lightgbm is not installed — `pip install \"quantis[ai]\"` "
                "or use --model ridge"
            )
    raise ValueError(f"unknown model type {model_type!r} (ridge | gbt)")
