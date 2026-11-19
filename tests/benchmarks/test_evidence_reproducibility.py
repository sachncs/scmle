"""Claim C6 evidence: summary and problems on small synthetic documents."""

import copy
from typing import Any

import pytest

from benchmarks import sources
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


@pytest.mark.parametrize(
    "module",
    [
        # Training and model code the measured project runs, not only the
        # bundle modules: a change to either makes the result stale.
        "scmle.pipeline.engine",
        "scmle.models.ensembles",
        "scmle.artifacts.bundle.seal",
    ],
)
def test_a_change_to_the_code_it_runs_makes_the_result_stale(
    monkeypatch, tmp_path, module
):
    current = document()
    assert reproducibility.problems(current) == []
    original = sources.locate(module)
    copy = tmp_path / "copy.py"
    copy.write_text(original.read_text() + "\nEDITED = 1\n")
    real = sources.locate

    def edited(target, root=sources.ROOT):
        return copy if target == module else real(target, root)

    monkeypatch.setattr(sources, "locate", edited)
    assert reproducibility.problems(current) == [
        "reproducibility: measurements are for other code"
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


def test_the_interaction_task_is_pinned_on_every_platform(tmp_path):
    # Measured bit for bit on macOS arm64 and on x86-64; before the
    # classification trees were grown on the symmetric criterion, x86-64
    # gave a roc_auc of 0.867630700778643 and arm64 0.8672135706340378.
    found = reproducibility.train("interaction", tmp_path)
    assert found["selected"] == "random_forest"
    assert found["metrics"] == {
        "roc_auc": 0.8670745272525028,
        "pr_auc": 0.8182773493133442,
        "log_loss": 0.4343082532185374,
        "f1": 0.8571428571428571,
        "balanced_accuracy": 0.8492769744160178,
        "accuracy": 0.85,
    }


def test_an_unknown_task_kind_is_refused():
    with pytest.raises(errors.InputError):
        reproducibility.generate("ranking")


def test_the_stored_measurements_are_for_the_current_code():
    assert reproducibility.problems() == []


def test_the_interaction_bundle_is_byte_identical_over_two_processes(
    tmp_path,
):
    # Two fresh processes build the interaction bundle; every member,
    # pipeline.joblib included, must hash the same (claim C6).
    found = reproducibility.compare("interaction", tmp_path)
    assert found["differing"] == []
    assert found["gap"]["absolute"] == 0.0
