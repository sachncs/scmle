"""Documents are ranked within queries, and queries never straddle the split."""

import json

import numpy as np
import pandas as pd
import pytest
import support

from scmle.contracts import errors


def corpus(queries=150, per_query=15, seed=0):
    rng = np.random.default_rng(seed)
    rows = queries * per_query
    frame = pd.DataFrame(
        {
            "query": np.repeat([f"q{i}" for i in range(queries)], per_query),
            "match": rng.normal(size=rows),
            "price": rng.normal(size=rows),
        }
    )
    frame["grade"] = np.clip(
        np.round(1.2 * frame.match + rng.normal(0, 0.6, rows)), 0, 3
    ).astype(int)
    return frame


def run(tmp_path, frame, **kwargs):
    return support.make_project(
        frame,
        "grade",
        task="ranking",
        query_column="query",
        time_budget=300,
        output_dir=tmp_path,
        **kwargs,
    ).run()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run(tmp_path_factory.mktemp("ranking"), corpus())


def test_no_query_appears_in_both_development_and_locked_rows(result):
    frame = corpus()
    manifest = json.loads(
        (result.run_dir / "validation_manifest.json").read_text()
    )

    development = set(frame["query"].iloc[manifest["train_ids"]])
    locked = set(frame["query"].iloc[manifest["test_ids"]])

    assert result.evaluation["split_strategy"] == "group"
    assert locked and not development & locked


def test_the_selected_ranker_beats_a_constant_ordering(result):
    assert result.evaluation["test_metrics"]["ndcg"] > 0.6
    baseline = next(t for t in result.trials if t.model_name == "dummy")
    assert result.evaluation["selected_cv_score"] > baseline.score_mean


def test_the_bundle_orders_rows_within_each_query(result):
    frame = corpus(5, 10, seed=3).drop(columns="grade")

    ranked = result.bundle.rank(frame, "query")

    assert {"score", "rank"} <= set(ranked.columns)
    for query in ranked["query"].unique():
        block = ranked[ranked["query"] == query]
        assert list(block["rank"]) == list(range(1, len(block) + 1))
        assert block["score"].is_monotonic_decreasing
    assert "query" not in result.bundle.features


def test_non_integer_relevance_is_refused(tmp_path):
    frame = corpus(40, 10)
    frame["grade"] = frame["grade"] + 0.5

    with pytest.raises(errors.BlockedError, match="whole numbers"):
        run(tmp_path, frame)


def test_ranking_needs_a_query_column_and_others_forbid_one(tmp_path):
    with pytest.raises(ValueError, match="query_column"):
        support.make_project(
            corpus(10, 5), "grade", task="ranking", output_dir=tmp_path
        )
    with pytest.raises(ValueError, match="query_column"):
        support.make_project(
            corpus(10, 5),
            "grade",
            task="regression",
            query_column="query",
            output_dir=tmp_path,
        )
