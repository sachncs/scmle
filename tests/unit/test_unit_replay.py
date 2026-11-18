"""Replayed estimators hand back stored predictions."""

import numpy
import pytest
from sklearn import base

from scmle.contracts import errors
from scmle.evaluation import replay


def test_a_replay_returns_its_predictions_whatever_rows_it_is_given():
    predictions = numpy.array([1, 0, 1])
    model = replay.Replay(predictions)
    assert model.predict(None) is predictions
    assert model.predict(numpy.zeros((9, 2))) is predictions
    assert model.classes_ is None
    assert model.probabilities is None


def test_a_replay_returns_its_probabilities_and_classes():
    probabilities = numpy.array([[0.2, 0.8]])
    classes = numpy.array(["no", "yes"])
    model = replay.Replay(numpy.array(["yes"]), probabilities, classes)
    assert model.predict_proba(None) is probabilities
    assert model.classes_ is classes


def test_a_replay_without_probabilities_raises_an_input_error():
    model = replay.Replay(numpy.array([1.0]))
    with pytest.raises(errors.InputError, match="no probabilities were stored"):
        model.predict_proba(None)


def test_the_classifier_replay_is_a_classifier_and_the_other_a_regressor():
    assert base.is_classifier(replay.ClassifierPredictions(numpy.array([1])))
    assert not base.is_regressor(replay.ClassifierPredictions(numpy.array([1])))
    assert base.is_regressor(replay.Replayed(numpy.array([1.0])))
    assert not base.is_classifier(replay.Replayed(numpy.array([1.0])))


def test_a_classifier_replay_scores_its_stored_predictions():
    model = replay.ClassifierPredictions(numpy.array([1, 0, 1, 1]))
    assert model.score(None, numpy.array([1, 0, 0, 1])) == 0.75


def test_a_regressor_replay_scores_its_stored_values():
    model = replay.Replayed(numpy.array([1.0, 2.0, 3.0]))
    assert model.score(None, numpy.array([1.0, 2.0, 3.0])) == 1.0
