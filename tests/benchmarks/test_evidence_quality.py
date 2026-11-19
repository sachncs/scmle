"""Claim C4 evidence: the product against a competent practitioner."""

import pytest

from benchmarks import acceptance
from benchmarks.evidence import quality
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
