"""Tests for the SCMLE side of the comparison harness (no network)."""

import numpy as np
import pandas as pd
import pytest

from benchmarks.sota import records
from benchmarks.sota import scmle_runner
from benchmarks.sota import split


def frames(rows=160):
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(rows, 3)), columns=["a", "b", "c"])
    data["y"] = np.where(data["a"] + rng.normal(size=rows) > 0, "p", "n")
    return (
        data.iloc[:120].reset_index(drop=True),
        data.iloc[120:],
        "y",
        "binary",
    )


def test_holdout_tasks_are_refused_by_default():
    holdout = split.assign(split.load_tasks())["holdout"]

    with pytest.raises(PermissionError, match="sealed"):
        scmle_runner.check_task_ids([holdout[0]])


def test_a_mixed_request_is_refused_as_a_whole():
    assignment = split.assign(split.load_tasks())

    with pytest.raises(PermissionError):
        scmle_runner.check_task_ids(
            [assignment["development"][0], assignment["holdout"][0]]
        )


def test_the_flag_opens_the_holdout_and_development_always_runs():
    assignment = split.assign(split.load_tasks())

    scmle_runner.check_task_ids(assignment["development"])
    scmle_runner.check_task_ids(assignment["holdout"], open_holdout=True)


def test_unknown_tasks_are_refused():
    with pytest.raises(ValueError, match="not in the protocol"):
        scmle_runner.check_task_ids([1])


def test_the_seed_is_deterministic_and_depends_on_the_split():
    assert scmle_runner.seed_for(1, 0, 0) == scmle_runner.seed_for(1, 0, 0)
    assert scmle_runner.seed_for(1, 0, 0) != scmle_runner.seed_for(1, 1, 0)
    assert 0 <= scmle_runner.seed_for(1, 2, 3) < 2**31


def test_a_run_becomes_a_record_in_the_results_file(tmp_path):
    record = scmle_runner.run_frames(7, 0, 1, frames(), 15, tmp_path / "runs")
    output = tmp_path / "out.jsonl"
    records.append(output, record)

    (loaded,) = records.read(output)
    assert loaded.status == "ok", loaded.error
    assert loaded.system == "scmle"
    assert (loaded.task_id, loaded.fold, loaded.repeat) == (7, 0, 1)
    assert loaded.metric == "roc_auc"
    assert 0.5 < loaded.score <= 1.0
    assert loaded.seconds > 0


def test_a_crash_is_a_failed_record_not_an_exception(tmp_path):
    train, test, _, kind = frames()

    record = scmle_runner.run_frames(
        7, 0, 0, (train, test, "missing", kind), 15, tmp_path
    )

    assert record.status == "failed"
    assert record.score is None
    assert record.error
