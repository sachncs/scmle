"""Dirichlet calibration maps log-probabilities through a fitted softmax."""

import numpy
import pytest
from sklearn import linear_model

from scmle.evaluation import dirichlet


class Fixed:
    """A stand-in classifier that returns given probabilities."""

    def __init__(self, probabilities):
        self.probabilities = numpy.asarray(probabilities, dtype=float)

    def predict_proba(self, features):
        return self.probabilities[: len(features)]


def test_the_floor_is_one_in_a_million():
    assert dirichlet.FLOOR == 1e-6


def test_a_featurisation_is_the_log_of_clipped_probabilities():
    calibrator = dirichlet.DirichletCalibrator(
        Fixed([[0.0, 1.0], [0.25, 0.75]])
    )
    result = calibrator.featurise(numpy.zeros((2, 1)))
    expected = numpy.log([[1e-6, 1.0], [0.25, 0.75]])
    numpy.testing.assert_allclose(result, expected)


def test_a_featurisation_clips_values_above_one_to_zero_log():
    calibrator = dirichlet.DirichletCalibrator(Fixed([[1.5, 1e-9]]))
    result = calibrator.featurise(numpy.zeros((1, 1)))
    numpy.testing.assert_allclose(result, [[0.0, numpy.log(1e-6)]])


def test_a_calibrator_keeps_its_settings():
    model = Fixed([[0.5, 0.5]])
    calibrator = dirichlet.DirichletCalibrator(model, regularization=3.0)
    assert calibrator.estimator is model
    assert calibrator.regularization == 3.0
    assert dirichlet.DirichletCalibrator(model).regularization == 1.0


def sample(rows=300, seed=0):
    rng = numpy.random.default_rng(seed)
    labels = rng.integers(0, 3, rows)
    raw = rng.dirichlet([1.0, 1.0, 1.0], rows)
    raw[numpy.arange(rows), labels] += 1.0
    probabilities = raw / raw.sum(axis=1, keepdims=True)
    return probabilities, numpy.array(["a", "b", "c"])[labels]


def test_a_fit_matches_a_logistic_regression_with_inverse_regularization():
    probabilities, labels = sample()
    features = numpy.zeros((len(labels), 1))
    calibrator = dirichlet.DirichletCalibrator(
        Fixed(probabilities), regularization=4.0
    ).fit(features, labels)
    reference = linear_model.LogisticRegression(C=0.25, max_iter=1000).fit(
        numpy.log(numpy.clip(probabilities, 1e-6, 1.0)), labels
    )
    numpy.testing.assert_allclose(
        calibrator.predict_proba(features),
        reference.predict_proba(numpy.log(probabilities)),
        rtol=1e-6,
    )
    assert list(calibrator.classes_) == ["a", "b", "c"]
    assert calibrator.map_.C == 0.25
    assert calibrator.map_.max_iter == 1000


def test_a_zero_regularization_uses_the_tiny_floor():
    probabilities, labels = sample(rows=120)
    features = numpy.zeros((len(labels), 1))
    calibrator = dirichlet.DirichletCalibrator(
        Fixed(probabilities), regularization=0.0
    ).fit(features, labels)
    assert pytest.approx(1e9) == calibrator.map_.C


def test_a_prediction_is_the_class_of_the_largest_calibrated_probability():
    probabilities, labels = sample()
    features = numpy.zeros((len(labels), 1))
    calibrator = dirichlet.DirichletCalibrator(Fixed(probabilities)).fit(
        features, labels
    )
    calibrated = calibrator.predict_proba(features)
    numpy.testing.assert_allclose(calibrated.sum(axis=1), 1.0)
    expected = calibrator.classes_[calibrated.argmax(axis=1)]
    assert list(calibrator.predict(features)) == list(expected)
    assert (calibrator.predict(features) == labels).mean() > 0.6
