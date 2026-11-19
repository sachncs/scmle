"""What the ranking wrapper hands to each step of its pipeline."""

import numpy
import pandas
from sklearn import base
from sklearn import pipeline

import scmle.preparation.ranking


class Keep(base.TransformerMixin, base.BaseEstimator):
    """Passes the frame through and remembers what it was fitted on."""

    def fit(self, features, target=None):
        self.columns_ = list(features.columns)
        self.target_ = None if target is None else numpy.asarray(target)
        return self

    def transform(self, features):
        return features.drop(columns="query").to_numpy(dtype=float)


class Pass(base.TransformerMixin, base.BaseEstimator):
    """Leaves its input as it is."""

    def fit(self, features, target=None):
        self.width_ = numpy.shape(features)[1]
        return self

    def transform(self, features):
        return features


class Rank(base.BaseEstimator):
    """A ranker that scores by the first feature and remembers its fit."""

    def fit(self, features, target, group):
        self.features_ = numpy.asarray(features)
        self.target_ = numpy.asarray(target)
        self.group_ = list(group)
        return self

    def predict(self, features):
        return numpy.asarray(features)[:, 0]


def fitted(frame, target, steps):
    estimator = pipeline.Pipeline(steps)
    return scmle.preparation.ranking.Ranker(estimator, "query").fit(
        frame, target
    )


def test_rows_go_to_the_ranker_in_text_order_of_queries_and_stably():
    size = 64
    query = numpy.where(numpy.arange(size) % 2 == 0, 9, 10)
    frame = pandas.DataFrame(
        {"query": query, "row": numpy.arange(size, dtype=float)},
        index=numpy.arange(size) * 3 + 7,
    )
    target = numpy.arange(size) % 4

    ranker = fitted(frame, target, [("keep", Keep()), ("rank", Rank())])

    rows = ranker.model_.features_[:, 0]
    first = numpy.arange(1, size, 2, dtype=float)
    second = numpy.arange(0, size, 2, dtype=float)
    assert rows.tolist() == [*first.tolist(), *second.tolist()]
    assert ranker.model_.group_ == [size // 2, size // 2]
    assert ranker.model_.target_.tolist() == (target[rows.astype(int)]).tolist()


def test_preprocessing_sees_the_columns_and_the_sorted_relevance():
    frame = pandas.DataFrame(
        {"query": [2, 1, 2, 1], "row": [0.0, 1.0, 2.0, 3.0]}, index=[5, 6, 7, 8]
    )
    target = numpy.array([3, 2, 1, 0])

    ranker = fitted(frame, target, [("keep", Keep()), ("rank", Rank())])

    assert ranker.preprocessing_.steps[0][1].columns_ == ["query", "row"]
    assert ranker.preprocessing_.steps[0][1].target_.tolist() == [2, 0, 3, 1]


def test_the_last_step_of_a_longer_pipeline_is_the_ranker():
    frame = pandas.DataFrame({"query": [1, 1, 2], "row": [0.0, 1.0, 2.0]})
    steps = [("keep", Keep()), ("pass", Pass()), ("rank", Rank())]

    ranker = fitted(frame, numpy.array([1, 0, 1]), steps)

    assert ranker.model_.group_ == [2, 1]
    assert ranker.predict(frame).tolist() == [0.0, 1.0, 2.0]
