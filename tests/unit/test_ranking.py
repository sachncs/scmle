"""Ranking metrics and estimators."""

import numpy
import pandas
import pytest
from sklearn import metrics

from scmle.evaluation import ranking
from scmle.models.ranking import estimators
from scmle.models.ranking import wrapper


def corpus(queries=60, per_query=12, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "query": numpy.repeat(numpy.arange(queries), per_query),
            "signal": rng.normal(size=queries * per_query),
            "noise": rng.normal(size=queries * per_query),
        }
    )
    grade = numpy.clip(
        numpy.round(frame.signal + rng.normal(0, 0.5, len(frame))), 0, 3
    )
    return frame, grade.astype(int)


def test_ndcg_matches_scikit_learn_on_one_query():
    truth = numpy.array([3, 2, 3, 0, 1, 2])
    scores = numpy.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7])

    ours = ranking.mean_ndcg(truth, scores, numpy.zeros(6), cutoff=5)

    expected = metrics.ndcg_score(truth[None], scores[None], k=5)
    assert ours == pytest.approx(expected)


def test_a_perfect_order_scores_one_and_a_reversed_one_less():
    truth = numpy.array([3, 2, 1, 0, 3, 2, 1, 0])
    queries = numpy.array([0] * 4 + [1] * 4)

    perfect = ranking.mean_ndcg(truth, truth.astype(float), queries)
    reversed_order = ranking.mean_ndcg(truth, -truth.astype(float), queries)

    assert perfect == pytest.approx(1.0) and reversed_order < perfect


def test_queries_with_nothing_relevant_are_skipped():
    truth = numpy.array([0, 0, 0, 1, 0, 0])
    scores = numpy.array([1.0, 2.0, 3.0, 5.0, 1.0, 2.0])
    queries = numpy.array([0, 0, 0, 1, 1, 1])

    assert ranking.mean_ndcg(truth, scores, queries) == pytest.approx(1.0)


def test_average_precision_counts_hits_by_position():
    truth = numpy.array([1, 0, 1, 0])
    scores = numpy.array([4.0, 3.0, 2.0, 1.0])

    value = ranking.mean_average_precision(truth, scores, numpy.zeros(4))

    assert value == pytest.approx((1 / 1 + 2 / 3) / 2)


def test_the_wrapper_hands_the_ranker_contiguous_query_sizes():
    frame, grade = corpus()
    shuffled = frame.sample(frac=1.0, random_state=1).reset_index(drop=True)
    labels = grade[frame.sample(frac=1.0, random_state=1).index]
    seen = {}

    class Spy(estimators.PointwiseRanker):
        def fit(self, features, target, group=None):
            seen["group"] = list(group)
            return super().fit(features, target, group)

    from sklearn import compose
    from sklearn import pipeline
    from sklearn import preprocessing

    steps = pipeline.Pipeline(
        [
            (
                "prep",
                compose.ColumnTransformer(
                    [("x", preprocessing.StandardScaler(), ["signal", "noise"])]
                ),
            ),
            ("model", Spy()),
        ]
    )
    model = wrapper.QueryAwareEstimator(steps, "query").fit(shuffled, labels)

    assert sum(seen["group"]) == len(shuffled)
    assert sorted(seen["group"]) == [12] * 60
    assert len(model.predict(shuffled)) == len(shuffled)


@pytest.mark.parametrize("library", ["lightgbm", "xgboost"])
def test_library_rankers_learn_the_ordering(library):
    pytest.importorskip(library)
    frame, grade = corpus()
    features = frame[["signal", "noise"]].to_numpy()
    sizes = [12] * 60

    model = estimators.LibraryRanker(library).fit(features, grade, group=sizes)

    value = ranking.mean_ndcg(
        grade, model.predict(features), frame["query"], cutoff=3
    )
    chance = ranking.mean_ndcg(
        grade,
        numpy.random.default_rng(0).normal(size=len(grade)),
        frame["query"],
        cutoff=3,
    )
    assert value > chance + 0.05


def test_reciprocal_rank_is_one_over_the_first_relevant_position():
    truth = numpy.array([0, 0, 1, 0, 1, 0, 0, 0])
    scores = numpy.array([4.0, 3.0, 2.0, 1.0, 4.0, 3.0, 2.0, 1.0])
    queries = numpy.array([0] * 4 + [1] * 4)

    value = ranking.mean_reciprocal_rank(truth, scores, queries)

    assert value == pytest.approx((1 / 3 + 1 / 1) / 2)


def test_precision_and_recall_at_k_count_hits_in_the_top():
    truth = numpy.array([1, 0, 1, 0, 1])
    scores = numpy.array([5.0, 4.0, 3.0, 2.0, 1.0])
    queries = numpy.zeros(5)

    assert ranking.precision_at(truth, scores, queries, cutoff=2) == 0.5
    assert ranking.recall_at(truth, scores, queries, cutoff=3) == pytest.approx(
        2 / 3
    )
