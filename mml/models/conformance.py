"""Shared contract checks every model adapter (built-in or third-party) must pass."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mml.contracts.errors import ConfigError
from mml.contracts.models import ModelSpec
from mml.contracts.tasks import TaskType


def _toy(task: TaskType, n: int = 120, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)})
    if task is TaskType.REGRESSION:
        y = pd.Series(X.a * 2 + rng.normal(scale=0.1, size=n))
    else:
        k = 2 if task is TaskType.BINARY else 3
        y = pd.Series(rng.integers(0, k, n))
    return X, y


def check_conformance(spec: ModelSpec) -> None:
    """Raise ConfigError when the adapter violates its declared capabilities."""
    if not spec.capabilities.tasks:
        raise ConfigError(f"{spec.name}: declares no supported tasks")
    for task in spec.capabilities.tasks:
        X, y = _toy(task)
        est = spec.factory(task, 0)
        for method in ("fit", "predict", "get_params", "set_params"):
            if not hasattr(est, method):
                raise ConfigError(f"{spec.name}: estimator lacks '{method}'")
        est.fit(X, y)
        pred = np.asarray(est.predict(X))
        if pred.shape[0] != len(X):
            raise ConfigError(f"{spec.name}: predict returned wrong length")
        if task.is_classification and spec.capabilities.probabilities:
            if not hasattr(est, "predict_proba"):
                raise ConfigError(f"{spec.name}: declares probabilities but lacks predict_proba")
            proba = np.asarray(est.predict_proba(X))
            if proba.shape[0] != len(X) or not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
                raise ConfigError(f"{spec.name}: predict_proba rows must sum to 1")
        a, b = spec.factory(task, 7), spec.factory(task, 7)
        if a.get_params().keys() != b.get_params().keys():
            raise ConfigError(f"{spec.name}: factory is not deterministic in parameters")
