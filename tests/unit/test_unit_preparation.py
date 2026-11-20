"""Behavioural tests for forecasting tables and the side-column wrappers."""

import numpy
import pandas
import pytest
from sklearn import base
from sklearn import compose
from sklearn import pipeline

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.preparation import forecasters
from scmle.preparation import forecasting
from scmle.preparation import survival
from scmle.preparation import uplift


def series(**kwargs):
    arguments = {
        "uri": "m",
        "target": "v",
        "value_column": "v",
        "horizon": 2,
        "time_column": "t",
    } | kwargs
    return tasks.Dataset(**arguments)


def test_augment_adds_future_targets_within_each_entity():
    frame = pandas.DataFrame(
        {
            "g": ["b", "a", "a", "b", "a", "b"],
            "t": [1, 3, 1, 2, 2, 3],
            "v": [10.0, 3.0, 1.0, 20.0, 2.0, 30.0],
        }
    )
    table = forecasting.augment(frame, series(group_column="g"))
    assert table["g"].tolist() == ["a", "b"]
    assert table["v"].tolist() == [1.0, 10.0]
    assert table["v__h1"].tolist() == [2.0, 20.0]
    assert table["v__h2"].tolist() == [3.0, 30.0]
    assert list(table.index) == [0, 1]
    assert frame["t"].tolist() == [1, 3, 1, 2, 2, 3]


def test_augment_treats_an_ungrouped_frame_as_one_series():
    frame = pandas.DataFrame({"t": [3, 1, 2, 4], "v": [3.0, 1.0, 2.0, 4.0]})
    table = forecasting.augment(frame, series(horizon=1))
    assert table["v"].tolist() == [1.0, 2.0, 3.0]
    assert table["v__h1"].tolist() == [2.0, 3.0, 4.0]


def test_augment_keeps_ties_in_their_original_order():
    frame = pandas.DataFrame({"t": [1, 1, 2], "v": [5.0, 6.0, 7.0]})
    table = forecasting.augment(frame, series(horizon=1))
    assert table["v"].tolist() == [5.0, 6.0]
    assert table["v__h1"].tolist() == [6.0, 7.0]


def test_augment_needs_a_value_column_and_horizon():
    frame = pandas.DataFrame({"t": [1], "v": [1.0]})
    with pytest.raises(errors.InputError, match="value_column and horizon"):
        forecasting.augment(frame, tasks.Dataset(uri="m", target="v"))


def test_the_naive_forecast_repeats_the_current_value():
    naive = forecasters.Naive(
        target_columns=("v__h1", "v__h2", "v__h3"), value_column="v"
    )
    assert naive.fit(None, None) is naive
    predicted = naive.predict(pandas.DataFrame({"v": [1, "x", 2.5]}))
    assert predicted.shape == (3, 3)
    assert predicted[0].tolist() == [1.0, 1.0, 1.0]
    assert numpy.isnan(predicted[1]).all()
    assert predicted[2].tolist() == [2.5, 2.5, 2.5]
    assert naive.estimator is None


def test_a_forecaster_remembers_its_value_column():
    model = forecasters.Forecaster("estimator", ("v__h1",), value_column="v")
    assert (model.estimator, model.target_columns, model.value_column) == (
        "estimator",
        ("v__h1",),
        "v",
    )
    assert forecasters.Forecaster("e", ()).value_column == ""


class Recorder(base.BaseEstimator):
    """A final step that records what it was fitted on."""

    def fit(self, features, target, *extra, event=None):
        self.features_ = numpy.asarray(features)
        self.target_ = numpy.asarray(target)
        self.extra_ = extra
        self.event_ = event
        return self

    def predict(self, features):
        return numpy.asarray(features).sum(axis=1)

    def predict_survival(self, features, times):
        rows = numpy.asarray(features).sum(axis=1)
        return numpy.outer(rows, numpy.asarray(times, dtype=float))


def chain():
    select = compose.ColumnTransformer([("x", "passthrough", ["x"])])
    return pipeline.Pipeline([("select", select), ("model", Recorder())])


def test_the_survival_wrapper_hands_events_only_to_the_model():
    frame = pandas.DataFrame({"x": [1.0, 2.0, 3.0], "e": [1, 0, 1]})
    template = chain()
    fitted = survival.Estimator(template, "e").fit(frame, [5, 6, 7])
    model = fitted.model_
    assert model.features_.tolist() == [[1.0], [2.0], [3.0]]
    assert model.target_.tolist() == [5.0, 6.0, 7.0]
    assert model.target_.dtype == float
    assert model.event_.tolist() == [1.0, 0.0, 1.0]
    assert model is not template.steps[-1][1]
    assert not hasattr(template.steps[-1][1], "features_")
    unseen = pandas.DataFrame({"x": [4.0]})
    assert fitted.predict(unseen).tolist() == [4.0]
    assert fitted.predict_survival(unseen, [1, 2]).tolist() == [[4.0, 8.0]]


def test_the_uplift_wrapper_hands_treatment_only_to_the_model():
    frame = pandas.DataFrame({"x": [1.0, 2.0], "arm": [1.0, 0.0]})
    fitted = uplift.Estimator(chain(), "arm").fit(frame, [0, 1])
    model = fitted.model_
    assert model.features_.tolist() == [[1.0], [2.0]]
    assert model.target_.tolist() == [0, 1]
    assert model.extra_[0].tolist() == [1, 0]
    assert model.extra_[0].dtype.kind == "i"
    assert fitted.predict(pandas.DataFrame({"x": [7.0]})).tolist() == [7.0]
    assert fitted.treatment_column == "arm"
