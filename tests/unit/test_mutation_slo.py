"""The SLO contract's statuses, refusals and records, word for word."""

import dataclasses

import pytest

from scmle.contracts import errors
from scmle.contracts import observability
from scmle.monitoring import slo
from tests.unit.test_slo import metric
from tests.unit.test_slo import target

NOW = "2026-10-20T12:00:00+00:00"
EARLIER = "2026-10-19T12:00:00+00:00"


def assessment(**changes) -> slo.Assessment:
    values = {
        "target": target(),
        "observed_at": NOW,
        "status": slo.Status.PASS,
        "sample_count": 2,
        "in_window_count": 2,
        "observed_maximum": 100.0,
        "error_budget_remaining": 0.5,
        "reason": "fine",
    }
    values.update(changes)
    return slo.Assessment(**values)


def test_require_text_refuses_a_non_string_and_a_blank() -> None:
    for value in (5, "  "):
        with pytest.raises(errors.InputError, match=r"^x must not be empty$"):
            slo.require_text(value, "x")
    assert slo.require_text(" y ", "x") == "y"


def test_a_target_reads_its_kind_from_text_and_records_every_field() -> None:
    found = dataclasses.replace(target(), kind="latency", maximum=0.0)
    assert found.kind is observability.Indicator.LATENCY
    assert found.to_dict() == {
        "name": "prediction-latency",
        "metric": "p95_latency_ms",
        "kind": "latency",
        "maximum": 0.0,
        "window": 7,
        "samples": 2,
        "owner": "ml-on-call",
        "runbook": "https://runbooks.example/prediction-latency",
        "route": "on-call",
    }


def test_a_blank_runbook_is_refused_by_name() -> None:
    with pytest.raises(errors.InputError, match=r"^runbook must not be empty$"):
        dataclasses.replace(target(), runbook=" ")


def test_a_negative_maximum_is_refused_word_for_word() -> None:
    with pytest.raises(
        errors.InputError, match=r"^maximum must be finite and non-negative$"
    ):
        dataclasses.replace(target(), maximum=-1.0)


def test_the_window_ignores_samples_of_another_kind() -> None:
    other = dataclasses.replace(
        metric(900.0, NOW), kind=observability.Indicator.ERROR
    )
    found = slo.evaluate(
        target(),
        (metric(100.0, EARLIER), metric(200.0, NOW), other),
        observed_at=NOW,
    )
    assert found.status is slo.Status.PASS
    assert found.sample_count == 2
    assert found.observed_maximum == 200.0


def test_a_window_above_the_maximum_is_breached() -> None:
    found = slo.evaluate(
        target(),
        (metric(100.0, EARLIER), metric(300.0, NOW)),
        observed_at=NOW,
    )
    assert found.status is slo.Status.BREACHED
    assert found.error_budget_remaining == 0.0
    assert found.reason == "maximum 300 exceeds 250"


def test_too_few_samples_keep_the_evaluation_time() -> None:
    found = slo.evaluate(target(), (metric(100.0, NOW),), observed_at=NOW)
    assert found.status is slo.Status.INSUFFICIENT_DATA
    assert found.observed_at == NOW
    assert (found.sample_count, found.in_window_count) == (1, 1)


def test_a_future_sample_invalidates_the_window() -> None:
    found = slo.evaluate(
        target(),
        (metric(100.0, EARLIER), metric(100.0, "2026-10-21T12:00:00+00:00")),
        observed_at=NOW,
    )
    assert found.status is slo.Status.INVALID
    assert found.observed_maximum is None
    assert (found.sample_count, found.in_window_count) == (2, 1)


def test_an_assessment_reads_its_status_from_text() -> None:
    assert assessment(status="breached").status is slo.Status.BREACHED


def test_an_assessment_record_names_its_observed_maximum() -> None:
    record = assessment().to_dict()
    assert record["maximum"] == 100.0
    assert record["remaining"] == 0.5


@pytest.mark.parametrize(
    ("changes", "kind", "message"),
    [
        ({"target": "slo"}, errors.KindError, r"^target must be Slo$"),
        (
            {"sample_count": -1},
            errors.InputError,
            r"^sample counts must not be negative$",
        ),
        (
            {"in_window_count": 3},
            errors.InputError,
            r"^in_window_count cannot exceed sample_count$",
        ),
        (
            {"observed_maximum": -1.0},
            errors.InputError,
            r"^observed_maximum must be finite and non-negative$",
        ),
        (
            {"error_budget_remaining": 1.5},
            errors.InputError,
            r"^error_budget_remaining must be between 0 and 1$",
        ),
        ({"reason": " "}, errors.InputError, r"^reason must not be empty$"),
    ],
)
def test_an_assessment_refuses_word_for_word(changes, kind, message) -> None:
    with pytest.raises(kind, match=message):
        assessment(**changes)
