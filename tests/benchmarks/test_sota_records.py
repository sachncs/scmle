"""Tests for the shared result record of the comparison harness."""

import pytest

from benchmarks.sota import records


def record(**changes):
    fields = {
        "system": "scmle",
        "budget_s": 60,
        "task_id": 363612,
        "fold": 0,
        "repeat": 0,
        "metric": "rmse",
        "score": 1.5,
        "seconds": 41.2,
    }
    fields.update(changes)
    return records.RunRecord(**fields)


def test_records_round_trip_through_a_json_lines_file(tmp_path):
    path = tmp_path / "results.jsonl"
    first, second = record(), record(task_id=363613, score=0.9)

    records.append(path, first)
    records.append(path, second)

    assert records.read(path) == [first, second]


def test_a_failure_is_a_record_without_a_score(tmp_path):
    path = tmp_path / "results.jsonl"
    failed = record(score=None, status="failed", error="out of memory")

    records.append(path, failed)

    assert records.read(path)[0].status == "failed"
    assert records.read(path)[0].score is None


def test_an_unknown_status_is_rejected():
    with pytest.raises(ValueError, match="unknown status"):
        record(status="succeeded")


def test_an_ok_record_must_carry_a_score():
    with pytest.raises(ValueError, match="needs a score"):
        record(score=None)
