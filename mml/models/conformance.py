"""Shared contract checks every model adapter must pass."""

import random
from typing import Any

import numpy as np
import pandas as pd

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks

ROWS = 120
ABSENT = object()


def toy_problem(
    task: tasks.TaskType, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series]:
    """Builds a small learnable problem for `task`."""
    rng = np.random.default_rng(seed)
    features = pd.DataFrame(
        {"a": rng.normal(size=ROWS), "b": rng.normal(size=ROWS)}
    )
    if task is tasks.TaskType.REGRESSION:
        target = features["a"] * 2 + rng.normal(scale=0.1, size=ROWS)
    else:
        classes = 2 if task is tasks.TaskType.BINARY else 3
        target = pd.Series(rng.integers(0, classes, ROWS))
    return features, target


def applied_value(params: dict[str, Any], key: str) -> Any:
    """Finds `key` among estimator params, including wrapped estimators.

    Args:
        params: Result of `get_params(deep=True)`.
        key: Hyperparameter name from the search space.

    Returns:
        The configured value, or a sentinel when the key is absent.
    """
    if key in params:
        return params[key]
    nested = [v for k, v in params.items() if k.endswith(f"__{key}")]
    return nested[0] if nested else ABSENT


def check_conformance(adapter: models.ModelAdapter) -> None:
    """Verifies an adapter honours its declared capabilities.

    Args:
        adapter: Adapter under test.

    Raises:
        ConfigError: On the first violated expectation.
    """
    name = adapter.name
    if not adapter.capabilities.tasks:
        raise errors.ConfigError(f"{name}: declares no supported tasks")
    for task in adapter.capabilities.tasks:
        features, target = toy_problem(task)
        estimator = adapter.build(task, 0)
        for method in ("fit", "predict", "get_params", "set_params"):
            if not hasattr(estimator, method):
                raise errors.ConfigError(f"{name}: estimator lacks '{method}'")
        estimator.fit(features, target)
        if len(np.asarray(estimator.predict(features))) != len(features):
            raise errors.ConfigError(f"{name}: predict returned wrong length")
        if task.is_classification and adapter.capabilities.probabilities:
            if not hasattr(estimator, "predict_proba"):
                raise errors.ConfigError(
                    f"{name}: declares probabilities but lacks predict_proba"
                )
            proba = np.asarray(estimator.predict_proba(features))
            if not np.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
                raise errors.ConfigError(
                    f"{name}: predict_proba rows must sum to 1"
                )
        space = adapter.search_space
        if space is not None:
            sampled = space.sample(random.Random(0))
            tuned = adapter.build(task, 0, sampled)
            applied = tuned.get_params(deep=True)
            for key, value in sampled.items():
                nested = [
                    v for k, v in applied.items() if k.endswith(f"__{key}")
                ]
                if applied.get(key, *nested[:1]) != value:
                    raise errors.ConfigError(
                        f"{name}: search parameter '{key}' not applied by build"
                    )
            tuned.fit(features, target)
        first, second = adapter.build(task, 7), adapter.build(task, 7)
        if first.get_params().keys() != second.get_params().keys():
            raise errors.ConfigError(f"{name}: build is not deterministic")
