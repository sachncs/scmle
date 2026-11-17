"""Claim C6 evidence: summary and problems on small synthetic documents."""

import copy
from typing import Any

import pytest

from benchmarks.evidence import reproducibility
from scmle.contracts import errors


def task(
    differing: list[str] | None = None,
    relative: float = 0.0,
    untraced: list[str] | None = None,
    auc: float = 0.9,
) -> dict[str, Any]:
    return {
        "selected": ["m", "m"],
        "metrics": {"roc_auc": auc},
        "gap": {"absolute": relative, "relative": relative},
        "members": 3,
        "differing": differing or [],
        "hashes": {},
        "untraced": untraced or [],
        "seconds": [1.0, 1.0],
        "threads": [],
    }


def record(**tasks: dict[str, Any]) -> dict[str, Any]:
    return {
        "machine": {"python": "3.14.0", "platform": "p"},
        "seed": reproducibility.SEED,
        "rows": reproducibility.ROWS,
        "tasks": tasks or {"binary": task(), "regression": task()},
        "seconds": 1.0,
    }


def document(**platforms: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "claim": "C6",
        "tolerance": {
            "name": reproducibility.TOLERANCE,
            "value": reproducibility.tolerance(),
        },
        "code": reproducibility.fingerprint(),
        "platforms": platforms or {"here": record()},
    }


def test_one_identical_platform_is_not_measured_across_platforms():
    value, verdict = reproducibility.summary(document())
    assert verdict == "not measured"
    assert value.startswith("this platform only")
    assert "2/2 tasks byte-identical" in value
    assert "cross-platform not measured" in value


def test_a_differing_member_fails_and_is_named():
    found = document(here=record(binary=task(differing=["pipeline.joblib"])))
    value, verdict = reproducibility.summary(found)
    assert verdict == "fail"
    assert "binary differs in pipeline.joblib" in value


def test_a_metric_gap_beyond_the_tolerance_fails():
    beyond = 10 * reproducibility.tolerance()
    found = document(here=record(binary=task(relative=beyond)))
    assert reproducibility.summary(found)[1] == "fail"
    within = reproducibility.tolerance() / 10
    found = document(here=record(binary=task(relative=within)))
    assert reproducibility.summary(found)[1] == "not measured"


def test_a_missing_trace_item_fails():
    found = document(here=record(binary=task(untraced=["commit"])))
    value, verdict = reproducibility.summary(found)
    assert verdict == "fail"
    assert "manifest lacks commit" in value


def test_two_agreeing_platforms_pass_and_disagreeing_fail():
    found = document(one=record(), two=record())
    assert reproducibility.summary(found)[1] == "pass"
    apart = record(
        binary=task(auc=0.9 * (1 + 10 * reproducibility.tolerance())),
        regression=task(),
    )
    found = document(one=record(), two=apart)
    assert reproducibility.summary(found)[1] == "fail"


def test_stale_or_malformed_documents_are_problems_not_verdicts():
    stale = document()
    stale["code"] = "0" * 64
    assert reproducibility.problems(stale) == [
        "reproducibility: measurements are for other code"
    ]
    assert reproducibility.summary(stale)[1] == "not measured"
    moved = copy.deepcopy(document())
    moved["tolerance"]["value"] = 1.0
    assert len(reproducibility.problems(moved)) == 1
    assert reproducibility.problems({"code": "x"}) == [
        "reproducibility: the document has no platform records"
    ]


def test_metric_gaps_are_relative_to_the_larger_magnitude():
    gap = reproducibility.differ({"a": 2.0, "b": 0.0}, {"a": 1.0, "b": 0.0})
    assert gap == {"absolute": 1.0, "relative": 0.5}
    with pytest.raises(errors.InputError):
        reproducibility.differ({"a": 1.0}, {"b": 1.0})


def test_trace_lists_the_items_a_manifest_does_not_carry():
    assert reproducibility.trace({"environment": {"numpy": "2"}}) == [
        "data hash",
        "configuration",
        "commit",
    ]


def test_an_unknown_task_kind_is_refused():
    with pytest.raises(errors.InputError):
        reproducibility.generate("ranking")


def test_the_stored_measurements_are_for_the_current_code():
    assert reproducibility.problems() == []
