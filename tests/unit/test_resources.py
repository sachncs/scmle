"""Tests for resource measurement and limits."""

import numpy as np
import pandas as pd
import pytest
from sklearn import dummy
from sklearn import ensemble
from sklearn import linear_model

from mml.evaluation import resources


def fitted(model, rows=300):
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    target = pd.Series((features.a > 0).astype(int))
    return model.fit(features, target), features


def test_bigger_models_are_larger_and_slower():
    small, features = fitted(linear_model.LogisticRegression())
    large = fitted(
        ensemble.RandomForestClassifier(n_estimators=300, random_state=0)
    )[0]
    tiny = resources.measure(small, features)
    big = resources.measure(large, features)
    assert big.model_mb > 10 * tiny.model_mb
    assert big.p95_latency_ms > tiny.p95_latency_ms
    assert (
        tiny.rows_timed == resources.LATENCY_ROWS and "python" in tiny.machine
    )


def test_measurement_does_not_modify_the_model_or_data():
    model, features = fitted(dummy.DummyClassifier())
    before = features.copy()
    resources.measure(model, features)
    pd.testing.assert_frame_equal(features, before)


def test_limits_report_each_violation():
    usage = resources.ResourceUsage(5.0, 20.0, 30, "m")
    assert resources.ResourceLimits().violations(usage) == []
    assert not resources.ResourceLimits().active
    limits = resources.ResourceLimits(max_model_mb=1.0, max_p95_latency_ms=10.0)
    broken = limits.violations(usage)
    assert limits.active and len(broken) == 2
    assert "size" in broken[0] and "latency" in broken[1]
    assert resources.ResourceLimits(max_model_mb=10.0).violations(usage) == []


def test_usage_serializes():
    usage = resources.ResourceUsage(1.5, 2.5, 30, "m")
    assert usage.to_dict() == {
        "model_mb": 1.5,
        "p95_latency_ms": 2.5,
        "rows_timed": 30,
        "machine": "m",
    }
    assert pytest.approx(usage.model_mb) == 1.5
