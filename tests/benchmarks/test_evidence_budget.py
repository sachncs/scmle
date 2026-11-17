"""Budget compliance and kill-and-resume evidence for claim C7."""

import json

import pytest

from benchmarks import acceptance
from benchmarks.evidence import budget
from scmle.contracts import errors


def minimum() -> int:
    variables = acceptance.load()
    rule = variables[budget.VARIABLE].rule
    assert isinstance(rule, acceptance.Floor)
    return rule.minimum(variables["alpha"].value)


def document(hits: int, trials: int, same: int = 3, cases: int = 3) -> dict:
    return {
        "settings": budget.settings(),
        "compliance": {"hits": hits, "trials": trials},
        "resume": {"identical": same, "trials": cases},
        "code": budget.current(),
    }


def test_the_floor_minimum_is_the_one_the_thresholds_file_states():
    # the derivation in the thresholds file says 299 runs pass at all
    assert minimum() == 299


def test_too_few_runs_is_not_measured_and_says_how_many_more():
    value, verdict = budget.summary(document(20, 20))
    assert verdict == "not measured"
    assert f"{minimum() - 20} more runs needed" in value


def test_every_run_within_budget_at_the_minimum_passes():
    assert budget.summary(document(minimum(), minimum()))[1] == "pass"


def test_one_run_over_at_the_minimum_fails():
    assert budget.summary(document(minimum() - 1, minimum()))[1] == "fail"


def test_a_resume_that_differs_fails_whatever_the_runs():
    value, verdict = budget.summary(document(minimum(), minimum(), same=2))
    assert verdict == "fail"
    assert "identical in 2 of 3" in value
    assert budget.summary(document(5, 5, same=2))[1] == "fail"


def test_no_kill_case_cannot_pass():
    assert budget.summary(document(minimum(), minimum(), 0, 0))[1] == "fail"


def test_malformed_documents_are_refused():
    with pytest.raises(errors.InputError):
        budget.summary({"compliance": {}})
    with pytest.raises(errors.InputError):
        budget.summary(document(5, 4))


def test_the_grace_is_the_product_hard_ceiling_grace():
    assert budget.grace(20) == pytest.approx(2.0)
    assert budget.grace(1) == pytest.approx(1.0)


def test_differences_name_each_differing_field():
    found = budget.differences({"a": 1, "b": [1]}, {"a": 1, "b": [2], "c": 0})
    assert found == ["b: [1] != [2]", "c: None != 0"]
    assert budget.differences({"a": 1}, {"a": 1}) == []


def test_nan_scores_compare_equal_after_cleaning():
    nan = float("nan")
    assert budget.clean({"s": [nan, 1.0]}) == {"s": [None, 1.0]}


def test_problems_flag_other_code_and_other_settings():
    assert budget.problems(document(1, 1)) == []
    stale = document(1, 1) | {"code": "0" * 64}
    assert budget.problems(stale) == ["budget: measurements are for other code"]
    moved = document(1, 1)
    moved["settings"] = {**moved["settings"], "budget_seconds": 1}
    assert budget.problems(moved) == [
        "budget: measurements are for other settings"
    ]


def test_an_unknown_child_mode_is_refused():
    with pytest.raises(errors.InputError):
        budget.child(["walk"])


def test_the_stored_measurements_are_for_the_current_code():
    assert budget.problems() == []
    stored = json.loads(budget.PATH.read_text(encoding="utf-8"))
    assert stored["compliance"]["trials"] == len(stored["compliance"]["runs"])
    assert stored["resume"]["trials"] == len(budget.KILL_POINTS)
