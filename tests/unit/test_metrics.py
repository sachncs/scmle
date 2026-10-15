import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier, DummyRegressor

from mml.contracts.errors import ConfigError
from mml.contracts.tasks import TaskType
from mml.evaluation.metrics import METRICS, resolve_metric, score_estimator


def test_defaults_and_alias():
    assert resolve_metric(None, TaskType.BINARY).name == "roc_auc"
    assert resolve_metric("roc_auc", TaskType.MULTICLASS).name == "roc_auc_ovr"
    assert resolve_metric(None, TaskType.REGRESSION).name == "rmse"


def test_incompatible_and_unknown():
    with pytest.raises(ConfigError):
        resolve_metric("pr_auc", TaskType.MULTICLASS)
    with pytest.raises(ConfigError):
        resolve_metric("nope", TaskType.BINARY)


def test_orientation_natural_units():
    assert METRICS["rmse"].to_natural(-2.0) == 2.0
    assert METRICS["roc_auc"].to_natural(0.8) == 0.8


def test_score_estimator_values():
    X = pd.DataFrame({"a": np.arange(40.0)})
    y = pd.Series([0, 1] * 20)
    clf = DummyClassifier(strategy="prior").fit(X, y)
    s = score_estimator(clf, X, y, TaskType.BINARY)
    assert s["roc_auc"] == 0.5
    assert s["log_loss"] == pytest.approx(0.6931, abs=1e-3)  # positive loss
    yr = pd.Series(np.arange(40.0))
    reg = DummyRegressor().fit(X, yr)
    r = score_estimator(reg, X, yr, TaskType.REGRESSION)
    assert r["rmse"] > 0 and r["mae"] > 0 and r["r2"] == pytest.approx(0.0, abs=1e-9)
