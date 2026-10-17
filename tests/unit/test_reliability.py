"""Calibration error for binary and multiclass probabilities."""

import numpy as np
import pandas as pd
import pytest
from sklearn import base
from sklearn import linear_model

from mml.artifacts import bundle as bundle_module
from mml.contracts import data as data_contracts
from mml.contracts import tasks
from mml.data.curation import preprocessing
from mml.evaluation import reliability
from mml.models import registry as registry_module


def test_a_perfectly_confident_correct_model_has_no_error():
    probabilities = np.array([[1.0, 0.0], [0.0, 1.0]])

    error, curve = reliability.top_label_curve(probabilities, [0, 1], [0, 1])

    assert error == 0.0
    assert curve[0]["accuracy"] == 1.0


def test_overconfidence_is_measured_as_the_gap_to_accuracy():
    probabilities = np.tile([0.9, 0.1], (10, 1))
    labels = [0] * 5 + [1] * 5  # right half the time at 0.9 confidence

    error = reliability.top_label_curve(probabilities, labels, [0, 1])[0]

    assert error == pytest.approx(0.4)


class Overconfident(base.ClassifierMixin, base.BaseEstimator):
    """Logistic regression whose log-odds are multiplied by a constant."""

    def __init__(self, sharpen: float = 6.0) -> None:
        self.sharpen = sharpen

    def fit(self, features, target):
        self.inner_ = linear_model.LogisticRegression(max_iter=500).fit(
            features, target
        )
        self.classes_ = self.inner_.classes_
        return self

    def predict_proba(self, features):
        scores = self.inner_.decision_function(features) * self.sharpen
        scores = scores - scores.max(axis=1, keepdims=True)
        weights = np.exp(scores)
        return weights / weights.sum(axis=1, keepdims=True)

    def predict(self, features):
        return self.classes_[self.predict_proba(features).argmax(axis=1)]


def test_calibrating_a_multiclass_bundle_lowers_its_calibration_error():
    rng = np.random.default_rng(0)
    features = pd.DataFrame(
        {"a": rng.normal(size=900), "b": rng.normal(size=900)}
    )
    scores = np.column_stack(
        [features.a, features.b, -features.a - features.b]
    ) + rng.normal(0, 1.5, (900, 3))
    target = pd.Series(scores.argmax(axis=1))
    plan = data_contracts.FeaturePlan(("a", "b"), ())
    pipe = preprocessing.build_pipeline(
        plan,
        registry_module.ModelRegistry.with_builtins().get(
            "logistic_regression"
        ),
        tasks.TaskType.MULTICLASS,
        0,
    )
    pipe.steps[-1] = ("model", Overconfident())
    pipe.fit(features, target)
    bundle = bundle_module.ModelBundle.create(
        pipe, plan, features, "y", "multiclass_classification", "log_loss", "x"
    )

    report = bundle.calibrate(features, target, method="isotonic")

    assert report["ece_after"] < report["ece_before"]
    probabilities = bundle.predict_proba(features)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-9)
    assert list(bundle.classes) == [0, 1, 2]
