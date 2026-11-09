"""Shared contract checks every model adapter must pass."""

import io
import random

import joblib
import numpy
import pandas
from scipy import sparse

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks

ROWS = 120


def toy_problem(
    task: tasks.Task, seed: int = 0
) -> tuple[pandas.DataFrame, pandas.Series | pandas.DataFrame]:
    """Builds a small learnable problem for `task`.

    Args:
        task: Task type the problem is shaped for.
        seed: Seed for the random generator.

    Returns:
        A pair of 120 rows of two numeric features ("a", "b") and a target
        suited to `task`: a series for single-target tasks, or a two-column
        frame ("first", "second") for multilabel, multioutput and forecast.
    """
    rng = numpy.random.default_rng(seed)
    features = pandas.DataFrame(
        {"a": rng.normal(size=ROWS), "b": rng.normal(size=ROWS)}
    )
    target: pandas.Series | pandas.DataFrame
    if task is tasks.Task.SURVIVAL:
        target = pandas.Series(rng.exponential(1.0, ROWS) + 0.01)
    elif task in (
        tasks.Task.MULTILABEL,
        tasks.Task.MULTIOUTPUT,
        tasks.Task.FORECAST,
    ):
        target = pandas.DataFrame(
            {
                "first": rng.integers(0, 2, ROWS),
                "second": rng.integers(0, 2, ROWS),
            }
        ).astype(float)
    elif task is tasks.Task.UPLIFT:
        target = pandas.Series(rng.integers(0, 2, ROWS))
    elif task is tasks.Task.RANKING:
        target = pandas.Series(
            numpy.clip(numpy.round(features["a"] + rng.normal(size=ROWS)), 0, 2)
        ).astype(float)
    elif task is tasks.Task.ORDINAL:
        target = pandas.Series(
            numpy.clip(
                numpy.round(features["a"] + rng.normal(size=ROWS)), -1, 1
            )
            + 1
        ).astype(float)
    elif task is tasks.Task.REGRESSION:
        target = features["a"] * 2 + rng.normal(scale=0.1, size=ROWS)
    else:
        classes = 2 if task is tasks.Task.BINARY else 3
        target = pandas.Series(rng.integers(0, classes, ROWS))
    return features, target


def check_declared_capabilities(
    adapter: models.Model,
    task: tasks.Task,
    features: pandas.DataFrame,
    target: pandas.Series | pandas.DataFrame,
) -> None:
    """Verifies the capabilities the pipeline relies on actually work.

    The pipeline skips imputation for models declaring `handles_missing`,
    emits sparse matrices for `sparse_input`, marks columns categorical for
    `native_categorical` and persists with joblib for `serializable`, so each
    declaration is exercised here.

    Args:
        adapter: Adapter under test.
        task: Task type to build the estimator for.
        features: Toy features from `toy_problem`.
        target: Toy target from `toy_problem`.

    Raises:
        ConfigError: If a declared capability is not honoured.
    """
    name, caps = adapter.name, adapter.capabilities
    if caps.handles_missing:
        holes = features.copy()
        holes.iloc[::7, 0] = numpy.nan
        fitted = adapter.build(task, 0, adapter.conformance_params).fit(
            holes, target
        )
        if len(numpy.asarray(fitted.predict(holes))) != len(holes):
            raise errors.ConfigError(f"{name}: failed on missing values")
    if caps.sparse_input:
        matrix = sparse.csr_matrix(features.to_numpy())
        fitted = adapter.build(task, 0, adapter.conformance_params).fit(
            matrix, target
        )
        if len(numpy.asarray(fitted.predict(matrix))) != len(features):
            raise errors.ConfigError(f"{name}: failed on sparse input")
    if caps.native_categorical:
        extra = adapter.categorical_params([0])
        if not extra:
            raise errors.ConfigError(
                f"{name}: declares native_categorical but categorical_params "
                "returns nothing"
            )
        codes = features.copy()
        codes["a"] = numpy.arange(len(codes)) % 5
        params = {**(adapter.conformance_params or {}), **extra}
        fitted = adapter.build(task, 0, params).fit(codes, target)
        if len(numpy.asarray(fitted.predict(codes))) != len(codes):
            raise errors.ConfigError(f"{name}: failed with native categoricals")
    if caps.serializable:
        fitted = adapter.build(task, 0, adapter.conformance_params).fit(
            features, target
        )
        buffer = io.BytesIO()
        joblib.dump(fitted, buffer)
        buffer.seek(0)
        restored = joblib.load(buffer)
        if not numpy.array_equal(
            numpy.asarray(restored.predict(features)),
            numpy.asarray(fitted.predict(features)),
        ):
            raise errors.ConfigError(
                f"{name}: declares serializable but predictions changed "
                "after a save / load round trip"
            )


def check_conformance(adapter: models.Model) -> None:
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
        estimator = adapter.build(task, 0, adapter.conformance_params)
        for method in ("fit", "predict", "get_params", "set_params"):
            if not hasattr(estimator, method):
                raise errors.ConfigError(f"{name}: estimator lacks '{method}'")
        estimator.fit(features, target)
        if len(numpy.asarray(estimator.predict(features))) != len(features):
            raise errors.ConfigError(f"{name}: predict returned wrong length")
        if task.is_classification and adapter.capabilities.probabilities:
            if not hasattr(estimator, "predict_proba"):
                raise errors.ConfigError(
                    f"{name}: declares probabilities but lacks predict_proba"
                )
            proba = numpy.asarray(estimator.predict_proba(features))
            if not numpy.allclose(proba.sum(axis=1), 1.0, atol=1e-6):
                raise errors.ConfigError(
                    f"{name}: predict_proba rows must sum to 1"
                )
        space = adapter.search_space
        if space is not None:
            sampled = space.sample(random.Random(0))  # nosec B311 - seeded, not security
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
        check_declared_capabilities(adapter, task, features, target)
        first = adapter.build(task, 7, adapter.conformance_params)
        second = adapter.build(task, 7, adapter.conformance_params)
        if first.get_params().keys() != second.get_params().keys():
            raise errors.ConfigError(f"{name}: build is not deterministic")
