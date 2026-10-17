"""Ranking metrics and estimators."""

import numpy as np
import pandas as pd
import pytest
from sklearn import metrics as sk_metrics

from mml.evaluation import ranking
from mml.models.ranking import estimators
from mml.models.ranking import wrapper


def corpus(queries=60, per_query=12, seed=0):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "query": np.repeat(np.arange(queries), per_query),
            "signal": rng.normal(size=queries * per_query),
            "noise": rng.normal(size=queries * per_query),
        }
    )
    grade = np.clip(
        np.round(frame.signal + rng.normal(0, 0.5, len(frame))), 0, 3
    )
    return frame, grade.astype(int)


def test_ndcg_matches_scikit_learn_on_one_query():
    truth = np.array([3, 2, 3, 0, 1, 2])
    scores = np.array([0.1, 0.9, 0.8, 0.2, 0.3, 0.7])

    ours = ranking.mean_ndcg(truth, scores, np.zeros(6), cutoff=5)

    expected = sk_metrics.ndcg_score(truth[None], scores[None], k=5)
    assert ours == pytest.approx(expected)


def test_a_perfect_order_scores_one_and_a_reversed_one_less():
    truth = np.array([3, 2, 1, 0, 3, 2, 1, 0])
    queries = np.array([0] * 4 + [1] * 4)

    perfect = ranking.mean_ndcg(truth, truth.astype(float), queries)
    reversed_order = ranking.mean_ndcg(truth, -truth.astype(float), queries)

    assert perfect == pytest.approx(1.0) and reversed_order < perfect


def test_queries_with_nothing_relevant_are_skipped():
    truth = np.array([0, 0, 0, 1, 0, 0])
    scores = np.array([1.0, 2.0, 3.0, 5.0, 1.0, 2.0])
    queries = np.array([0, 0, 0, 1, 1, 1])

    assert ranking.mean_ndcg(truth, scores, queries) == pytest.approx(1.0)


def test_average_precision_counts_hits_by_position():
    truth = np.array([1, 0, 1, 0])
    scores = np.array([4.0, 3.0, 2.0, 1.0])

    value = ranking.mean_average_precision(truth, scores, np.zeros(4))

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
        np.random.default_rng(0).normal(size=len(grade)),
        frame["query"],
        cutoff=3,
    )
    assert value > chance + 0.05
