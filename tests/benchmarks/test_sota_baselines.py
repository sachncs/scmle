"""Tests for the reference-system driver (no baseline is ever imported)."""

import json
import pathlib
import sys

import pytest

from benchmarks.sota import baselines
from benchmarks.sota import records


def test_a_holdout_task_is_refused_by_default():
    sealed = baselines.sets()["holdout"][0]

    with pytest.raises(baselines.HoldoutRefusedError):
        baselines.check_tasks([sealed])


def test_a_mixed_request_is_refused_as_a_whole():
    assignment = baselines.sets()
    request = [assignment["development"][0], assignment["holdout"][0]]

    with pytest.raises(baselines.HoldoutRefusedError):
        baselines.check_tasks(request)


def test_the_explicit_flag_opens_the_holdout():
    assignment = baselines.sets()

    baselines.check_tasks(assignment["holdout"], open_holdout=True)


def test_development_tasks_pass_and_unknown_ones_are_rejected():
    baselines.check_tasks(baselines.sets()["development"])
    with pytest.raises(ValueError, match="not protocol tasks"):
        baselines.check_tasks([1])


def test_the_driver_command_refuses_a_holdout_task_before_running(
    tmp_path: pathlib.Path,
):
    sealed = baselines.sets()["holdout"][0]
    output = tmp_path / "out.jsonl"

    code = baselines.main(
        [
            "run",
            "--python",
            sys.executable,
            "--tasks",
            str(sealed),
            "--output",
            str(output),
        ]
    )

    assert code == 2
    assert not output.exists()


def test_the_lite_tier_runs_one_split_and_the_full_tier_six():
    assert baselines.splits(lite=True) == [(0, 0)]
    assert len(baselines.splits(lite=False)) == 6


def test_a_worker_that_leaves_no_result_becomes_a_failed_record(
    tmp_path: pathlib.Path,
):
    outcome = baselines.read_outcome(tmp_path / "missing.json", 1, "boom")
    job = baselines.Job("lightgbm", 60, 363696, 0, 0)

    record = baselines.to_record(job, "roc_auc", outcome, 3.0)

    assert record.status == "failed"
    assert record.score is None
    assert "boom" in record.error


def test_an_ok_outcome_without_a_score_is_not_trusted(tmp_path: pathlib.Path):
    path = tmp_path / "result.json"
    path.write_text(json.dumps({"status": "ok"}), encoding="utf-8")
    job = baselines.Job("flaml", 60, 363696, 0, 0)

    record = baselines.to_record(
        job, "roc_auc", baselines.read_outcome(path, 0, ""), 1.0
    )

    assert record.status == "failed"


def test_a_hung_worker_is_killed_and_recorded_as_a_timeout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(baselines, "MIN_GRACE_S", 1.0)
    monkeypatch.setattr(baselines, "GRACE_FRACTION", 0.0)
    sleeper = tmp_path / "python"
    sleeper.write_text("#!/bin/sh\nsleep 30\n", encoding="utf-8")
    sleeper.chmod(0o755)
    job = baselines.Job("lightgbm", 1, 363696, 0, 0)

    outcome, elapsed = baselines.execute(sleeper, job, cpus=1)

    assert outcome.status == "timeout"
    assert elapsed < 15


def test_resuming_skips_what_is_already_recorded(tmp_path: pathlib.Path):
    path = tmp_path / "out.jsonl"
    records.append(
        path,
        records.RunRecord(
            "lightgbm", 60, 363696, 0, 0, "roc_auc", None, 1.0, "failed", "x"
        ),
    )

    assert ("lightgbm", 60, 363696, 0, 0) in baselines.done(path)


def test_more_than_twelve_cores_is_refused(tmp_path: pathlib.Path):
    task = baselines.sets()["development"][0]
    args = baselines.build_parser().parse_args(
        [
            "run",
            "--python",
            "p",
            "--tasks",
            str(task),
            "--output",
            str(tmp_path / "o.jsonl"),
            "--cpus",
            "13",
        ]
    )

    with pytest.raises(ValueError, match="--cpus"):
        baselines.run_all(args)
