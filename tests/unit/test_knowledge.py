"""Tests for dataset fingerprints and the meta-learner."""

import dataclasses

import numpy
import pandas
import pytest

from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.data.profiling import profiler
from scmle.data.readers import tabular
from scmle.intelligence import knowledge


def profile_of(frame):
    spec = tasks.ProjectSpec(data=tasks.DataSpec(uri="m", target="y"))
    manifest = tabular.build_manifest(frame, "y")
    return profiler.DatasetProfiler(spec).profile(
        frame, manifest, tasks.TaskType.BINARY
    )


def run_record(
    model, fp, dataset="d", data_hash="h", task="binary_classification"
):
    return experiments.HistoricalRun(
        f"run_{model}_{dataset}",
        dataset,
        data_hash,
        task,
        "roc_auc",
        fp,
        model,
        "default",
        0.9,
    )


def fp(**overrides):
    base = dict.fromkeys(knowledge.FIELDS, 1.0)
    return {**base, **overrides}


def test_fingerprint_describes_shape_types_and_balance():
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {
            "a": rng.exponential(size=1000),
            "b": rng.normal(size=1000),
            "c": rng.choice(list("xyz"), 1000),
        }
    )
    frame["y"] = (rng.random(1000) < 0.1).astype(int)
    result = knowledge.fingerprint(profile_of(frame), tasks.TaskType.BINARY)
    assert set(result) == set(knowledge.FIELDS)
    assert result["log_rows"] == pytest.approx(3.0)
    assert result["categorical_share"] == pytest.approx(1 / 3)
    assert result["log_imbalance"] > 0.8 and result["mean_abs_skew"] > 0.3


def test_fingerprint_has_a_defined_missing_rate_without_features():
    """An empty feature plan remains finite and warning-free."""
    profile = profile_of(
        pandas.DataFrame({"feature": [1.0, 2.0, 3.0, 4.0], "y": [0, 1, 0, 1]})
    )
    empty_plan = dataclasses.replace(
        profile.feature_plan, numeric=(), categorical=(), text=()
    )
    profile = dataclasses.replace(profile, feature_plan=empty_plan)

    result = knowledge.fingerprint(profile, tasks.TaskType.BINARY)

    assert result["missing_rate"] == 0.0


def test_nearest_datasets_vote_for_their_winners():
    history = [
        run_record("random_forest", fp(log_rows=5.0), "big1", "h1"),
        run_record("lightgbm", fp(log_rows=5.1), "big2", "h2"),
        run_record("ridge", fp(log_rows=1.0), "small", "h3"),
    ]
    learner = knowledge.MetaLearner(history, neighbors=2)
    near_big = learner.recommend(
        fp(log_rows=5.0), tasks.TaskType.BINARY, "new", "hn"
    )
    assert [pair[0] for pair in near_big] == ["random_forest", "lightgbm"]
    near_small = knowledge.MetaLearner(history, neighbors=1).recommend(
        fp(log_rows=1.1), tasks.TaskType.BINARY, "new", "hn"
    )
    assert near_small[0][0] == "ridge"


def test_same_dataset_and_other_tasks_are_excluded():
    history = [
        run_record("random_forest", fp(), "self", "same"),
        run_record("ridge", fp(), "other", "o", task="regression"),
    ]
    learner = knowledge.MetaLearner(history)
    assert (
        learner.recommend(fp(), tasks.TaskType.BINARY, "self", "different-hash")
        == []
    )
    assert learner.recommend(fp(), tasks.TaskType.BINARY, "new", "same") == []
    others = learner.recommend(fp(), tasks.TaskType.BINARY, "new", "hn")
    assert [model for model, weight in others] == ["random_forest"]  # not ridge


def test_weights_accumulate_and_empty_history_is_safe():
    history = [run_record("lightgbm", fp(), f"d{i}", f"h{i}") for i in range(3)]
    ((model, weight),) = knowledge.MetaLearner(history).recommend(
        fp(), tasks.TaskType.BINARY, "n", "x"
    )
    assert model == "lightgbm" and weight == pytest.approx(3.0)
    assert (
        knowledge.MetaLearner([]).recommend(
            fp(), tasks.TaskType.BINARY, "n", "x"
        )
        == []
    )
