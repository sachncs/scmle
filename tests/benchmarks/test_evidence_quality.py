"""Claim C4 evidence: the product against a competent practitioner."""

import json
import math

import pytest

from benchmarks import acceptance
from benchmarks.evidence import custody
from benchmarks.evidence import quality
from benchmarks.evidence import refusal
from benchmarks.sota import scmle_runner
from scmle.contracts import errors


def kind(name: str, gaps: list[float]) -> dict:
    """One problem type decided from its gaps, as `measure` stores it."""
    rows = [{"gap": g, "product": {"score": 0.0}} for g in gaps]
    return quality.tally(name, rows)


def document(gaps: dict[str, list[float]] | None = None) -> dict:
    """A current document with the given gaps per problem type."""
    variables = acceptance.load()
    gaps = gaps or dict.fromkeys(quality.TASKS, [0.0] * 5)
    return {
        "median_max": variables[quality.MEDIAN].value,
        "upper_max": variables[quality.UPPER].value,
        "code": quality.digest(),
        "data": quality.files(),
        "design": {"budget_seconds": quality.BUDGET},
        "kinds": [kind(n, g) for n, g in gaps.items()],
    }


def test_the_minimum_follows_from_alpha():
    assert quality.minimum() == 5
    assert all(len(ids) >= quality.minimum() for ids in quality.TASKS.values())


def test_the_gap_is_normalised_by_the_trivial_predictor():
    assert quality.gap("binary", 0.9, 0.8, 0.5) == pytest.approx(0.25)
    assert quality.gap("regression", 1.0, 2.0, 3.0) == pytest.approx(0.5)
    assert quality.gap("binary", 0.8, 0.9, 0.5) == pytest.approx(-0.25)


def test_a_failure_scores_the_trivial_predictor():
    assert quality.gap("binary", 0.9, None, 0.5) == pytest.approx(1.0)
    assert quality.gap("multiclass", None, 0.5, 1.0) == pytest.approx(-1.0)
    assert quality.gap("binary", 0.4, 0.3, 0.5) == 0.0


def test_an_unknown_problem_type_is_refused():
    with pytest.raises(errors.InputError):
        quality.gap("ranking", 0.9, 0.8, 0.5)


def test_small_gaps_still_leave_the_row_not_measured():
    text, verdict = quality.summary(document())
    assert all(k["passed"] for k in document()["kinds"])
    assert verdict == "not measured"
    assert "other data kinds not measured" in text


def test_a_large_gap_fails():
    # Enforces docs/preregistration/c4.md:50.
    gaps = dict.fromkeys(quality.TASKS, [0.0] * 5)
    gaps["multiclass"] = [0.1] * 5
    assert quality.summary(document(gaps))[1] == "fail"


def test_a_good_median_with_a_wide_bound_fails():
    gaps = dict.fromkeys(quality.TASKS, [0.0] * 5)
    gaps["binary"] = [0.0, 0.0, 0.01, 0.3, 0.4]
    found = document(gaps)
    binary = next(k for k in found["kinds"] if k["kind"] == "binary")
    assert binary["median"] <= acceptance.load()[quality.MEDIAN].value
    assert not binary["passed"]


def test_too_few_tasks_is_not_measured():
    gaps = dict.fromkeys(quality.TASKS, [0.0] * 4)
    issues = quality.problems(document(gaps))
    assert any("none can pass below 5" in i for i in issues)
    assert quality.summary(document(gaps))[1] == "not measured"


def test_stale_code_is_not_measured():
    found = document()
    found["code"] = "0" * 64
    assert "quality: measured with other code" in quality.problems(found)


def test_the_stored_measurement_is_current():
    assert quality.problems() == []


def test_every_kind_holds_both_deciding_splits():
    # Custody needs `minimum()` tasks of each kind in each of validation
    # and acceptance; a table another set placed first keeps its split.
    sets = custody.catalog()
    placed = custody.assign(sets)["quality"]
    each = next(s for s in sets if s.name == "quality")
    for ids in quality.TASKS.values():
        assert len(ids) >= 2 * quality.minimum()
        splits = [placed[str(t)] for t in ids]
        assert splits.count(custody.VALIDATION) >= quality.minimum()
        assert splits.count(custody.ACCEPTANCE) >= quality.minimum()
    assert {i.stratum for i in each.items} == set(quality.TASKS)


def test_every_task_is_pinned_with_its_licence_and_table():
    tasks = [t for ids in quality.TASKS.values() for t in ids]
    assert len(tasks) == len(set(tasks))
    assert set(quality.PINS) == set(tasks) == set(quality.TABLES)
    for task, (digest, licence) in quality.PINS.items():
        assert len(digest) == 64
        assert licence and licence != "NA"
        meta = json.loads((quality.DATA / f"{task}.json").read_text("utf-8"))
        assert meta["sha256"] == digest
        assert meta["licence"] == licence
        assert quality.TABLES[task] == meta["dataset"]
    assert set(quality.SAME) <= set(quality.TABLES.values())


def test_no_task_is_sealed_or_a_listed_copy():
    split = json.loads(quality.SPLIT.read_text("utf-8"))
    tasks = {t for ids in quality.TASKS.values() for t in ids}
    assert not tasks & set(split["holdout"])
    assert not set(quality.TABLES.values()) & set(refusal.COPIES)


def test_a_table_shared_with_another_set_keeps_its_custody_key():
    each = next(s for s in custody.catalog() if s.name == "quality")
    keys = {i.name: i.source for i in each.items}
    for task, table in quality.TABLES.items():
        expected = custody.origin(quality.SAME.get(table, table))
        assert keys[str(task)] == expected


def test_a_mismatched_file_is_refused(monkeypatch, tmp_path):
    task = quality.TASKS["binary"][0]
    monkeypatch.setitem(quality.PINS, task, ("0" * 64, "Public"))
    with pytest.raises(errors.InputError):
        quality.verify(task, b"not the pinned file", "Public")


def test_the_run_count_comes_from_alpha():
    alpha = acceptance.load()["alpha"].value
    runs = quality.runs()
    assert runs == 6
    assert 2 * 0.5**runs <= alpha < 2 * 0.5 ** (runs - 1)
    assert runs == math.ceil(1 + math.log2(1 / alpha))


def test_the_first_run_is_the_registered_one_and_the_rest_differ():
    task = quality.TASKS["binary"][0]
    seeds = [quality.seed(task, r) for r in range(quality.runs())]
    assert seeds[0] == scmle_runner.seed_for(task, 0, 0)
    assert len(set(seeds)) == len(seeds)


def test_the_noise_is_reported_but_the_first_run_decides():
    row = {
        "gap": 0.0,
        "product": {"score": 0.0},
        "repeats": [{"gap": g} for g in (0.1, -0.1, 0.0, 0.2, 0.0)],
        "spread": 0.3,
    }
    rows = [row] * 5
    found = quality.tally("binary", rows)
    assert found["median"] == 0.0
    assert found["passed"]
    assert found["noise"]["spread"] == pytest.approx(0.3)
    assert found["noise"]["medians"][1] == pytest.approx(0.1)
    assert len(found["noise"]["medians"]) == quality.runs()
