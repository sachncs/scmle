"""Tests for column kinds."""

import numpy as np
import pandas as pd
import pytest

from mml.contracts import tasks
from mml.data import columns


def test_kind_detection():
    frame = pd.DataFrame(
        {
            "n": [1.0, 2.0],
            "b": [True, False],
            "s": ["a", "b"],
            "t": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        }
    )
    kinds = {c: columns.kind_of(frame[c]).name for c in frame}
    assert kinds == {
        "n": "numeric",
        "b": "numeric",
        "s": "categorical",
        "t": "datetime",
    }


def test_statistics_per_kind():
    numeric = columns.NumericKind().statistics(pd.Series([1.0, 3.0, np.nan]))
    assert numeric["mean"] == 2.0 and numeric["missing_rate"] == pytest.approx(
        1 / 3
    )
    assert "mean" not in columns.NumericKind().statistics(
        pd.Series([True, False])
    )
    top = columns.CategoricalKind().statistics(pd.Series(list("aab")))["top"]
    assert top == {"a": 2, "b": 1}


def test_exclusion_rules():
    ids = pd.Series([f"u{i}" for i in range(50)])
    assert "id-like" in columns.CategoricalKind().exclusion_reason(ids)
    assert columns.CategoricalKind().exclusion_reason(pd.Series(list("ab" * 25))) is None
    assert "datetime" in columns.DatetimeKind().exclusion_reason(pd.Series([1]))
    assert columns.NumericKind().exclusion_reason(pd.Series([1])) is None


def test_transformers_exist_only_for_feature_kinds():
    assert columns.NumericKind().transformer(scale=True) is not None
    assert columns.CategoricalKind().transformer(scale=False) is not None
    assert columns.DatetimeKind().transformer(scale=True) is None


def test_scale_flag_controls_scaler():
    scaled = columns.NumericKind().transformer(scale=True)
    unscaled = columns.NumericKind().transformer(scale=False)
    assert "scale" in scaled.named_steps and "scale" not in unscaled.named_steps


def test_association_detects_perfect_predictors():
    rng = np.random.default_rng(0)
    target = pd.Series(rng.integers(0, 2, 200))
    leak = target * 5.0
    noise = pd.Series(rng.normal(size=200))
    kind = columns.NumericKind()
    clf = tasks.TaskType.BINARY
    assert kind.association(leak, target, clf) == pytest.approx(1.0)
    assert kind.association(noise, target, clf) < 0.5
    cat = columns.CategoricalKind()
    labels = target.map({0: "no", 1: "yes"})
    assert cat.association(labels, target, clf) == pytest.approx(1.0)
    assert cat.association(labels, target, tasks.TaskType.REGRESSION) == 0.0


def test_regression_association_uses_rank_correlation():
    x = pd.Series(np.arange(100.0))
    assert columns.NumericKind().association(
        x, x**3, tasks.TaskType.REGRESSION
    ) == pytest.approx(1.0)


def test_association_with_too_few_rows_is_zero():
    kind = columns.NumericKind()
    short = pd.Series([1.0, 2.0, 3.0])
    assert kind.association(short, pd.Series([0, 1, 0]), tasks.TaskType.BINARY) == 0.0
