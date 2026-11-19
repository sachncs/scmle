"""Budget compliance and kill-and-resume evidence for claim C7."""

import json

import pytest

from benchmarks import acceptance
from benchmarks.evidence import budget
from benchmarks.sota import scmle_runner
from scmle.contracts import errors
from scmle.runtime import limits


def minimum() -> int:
    variables = acceptance.load()
    rule = variables[budget.VARIABLE].rule
    assert isinstance(rule, acceptance.Floor)
    return rule.minimum(variables["alpha"].value)


def document(
    hits: int,
    trials: int,
    same: int = 3,
    cases: int = 3,
    real: tuple[int, int] | None = None,
    refused: int = 0,
) -> dict:
    good, runs = (hits, trials) if real is None else real
    return {
        "settings": budget.settings(),
        "practicerule": budget.RULE,
        "compliance": {"hits": hits, "trials": trials},
        "resume": {"identical": same, "trials": cases},
        "real": {
            "hits": good,
            "trials": runs,
            "refused": refused,
            "runs": [],
        },
        "realsettings": budget.realsettings(),
        "code": budget.current(),
        "realcode": budget.realcode(),
    }


def test_the_floor_minimum_is_the_one_the_thresholds_file_states():
    # the derivation in the thresholds file says 299 runs pass at all
    assert minimum() == 299


def test_too_few_runs_is_not_measured_and_says_how_many_more():
    value, verdict = budget.summary(document(20, 20))
    assert verdict == "not measured"
    assert f"{minimum() - 20} more real runs needed" in value


def test_the_practice_project_alone_never_passes():
    full = minimum()
    found = budget.summary(document(full, full, real=(46, 46)))
    assert found[1] == "not measured"
    assert f"{full - 46} more real runs needed" in found[0]
    assert budget.summary(document(full, full, real=(0, 0)))[1] == (
        "not measured"
    )


def test_the_real_family_alone_decides():
    full = minimum()
    real = document(full, full, real=(full - 1, full))
    practice = document(0, full, real=(full, full))
    assert budget.summary(real)[1] == "fail"
    assert budget.summary(practice)[1] == "pass"
    assert "not deciding" in budget.summary(practice)[0]


def test_refusals_are_counted_apart_and_never_pass_or_fail():
    full = minimum()
    value, verdict = budget.summary(
        document(full, full, real=(full - 5, full - 5), refused=5)
    )
    assert verdict == "not measured"
    assert "5 refused before starting" in value
    found = budget.summary(document(full, full, real=(full, full), refused=9))
    assert found[1] == "pass"


def test_practice_runs_under_the_old_rule_are_kept_and_labelled():
    full = minimum()
    old = document(full, full, real=(full, full))
    del old["practicerule"]
    old["settings"] = {"budget_seconds": 20, "grace_seconds": 2.0}
    old["code"] = "0" * 64
    assert budget.problems(old) == []
    value, verdict = budget.summary(old)
    assert verdict == "pass"
    assert "old rule, not deciding" in value
    assert "20 s + 2 s grace" in value


def test_a_run_is_judged_against_the_budget_plus_the_slack():
    limit = budget.allowance(30)
    assert limit == pytest.approx(30 + limits.SLACK_SECONDS)
    assert budget.judge({"seconds": limit, "model": "m"}, limit) == (
        "compliant"
    )
    assert budget.judge({"seconds": limit + 0.01, "model": "m"}, limit) == (
        "over"
    )
    assert budget.judge({"seconds": 1.0, "blocked": "no"}, limit) == "refused"
    assert budget.judge({"timed_out": True}, limit) == "killed"
    assert budget.judge({"timed_out": False, "error": "x"}, limit) == (
        "crashed"
    )


def test_every_pass_runs_a_table_with_its_own_seed():
    seeds = {budget.seeded("quality", 363621, n) for n in range(3)}
    assert len(seeds) == 3
    assert budget.seeded("quality", 363621, 0) == scmle_runner.seed_for(
        363621, 0, 0
    )


def test_a_document_without_the_real_family_is_not_current():
    old = document(1, 1)
    del old["real"]
    assert budget.problems(old) == ["budget: the real family is not measured"]


def test_overruns_are_named_worst_first():
    family = {
        "runs": [
            {"id": 1, "compliant": True, "over_seconds": -3.0},
            {"id": 2, "compliant": False, "over_seconds": 4.5},
            {
                "id": 3,
                "compliant": False,
                "over_seconds": None,
                "timed_out": True,
            },
            {
                "id": 5,
                "compliant": False,
                "over_seconds": None,
                "timed_out": False,
            },
            {"id": 4, "compliant": False, "over_seconds": 60.0, "pass": 2},
            {"id": 6, "compliant": False, "refused": True},
        ]
    }
    assert budget.overruns(family) == (
        "3 killed, 5 crashed, 4/2 +60 s, 2 +4.5 s"
    )
    assert budget.overruns({"runs": []}) == ""


def test_the_budget_doubles_for_every_tenfold_more_cells():
    assert budget.allot(600) == 30
    assert budget.allot(10_000) == 30
    assert budget.allot(10_001) == 60
    assert budget.allot(100_000) == 60
    assert budget.allot(572_385) == 120


def test_every_real_table_loads_with_its_target():
    for source, ident in budget.tables()[:2]:
        frame, target, options = budget.load(source, ident)
        assert target in frame
        assert options["task"] in {"classification", "regression"}
    with pytest.raises(errors.InputError):
        budget.load("elsewhere", 1)


def test_a_run_whose_kill_time_falls_past_the_window_does_not_start():
    found = budget.attempt(1, "refusal", 61, budget.ROOT, 0.0)
    assert found["started"] is False
    assert found["seed"] == budget.seeded("refusal", 61, 1)
    assert "run_seconds" not in found


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


def test_the_allowance_has_no_grace_share():
    assert budget.allowance(20) == pytest.approx(20 + limits.SLACK_SECONDS)
    assert budget.allowance(240) - 240 == pytest.approx(limits.SLACK_SECONDS)


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


def test_the_real_fingerprint_covers_the_product_code(tmp_path):
    for name in budget.PRODUCT:
        folder = tmp_path / name
        folder.mkdir()
        (folder / "unit.py").write_text("x = 1\n", encoding="utf-8")
    before = budget.product(tmp_path)
    for name in budget.PRODUCT:
        (tmp_path / name / "unit.py").write_text("x = 2\n", encoding="utf-8")
        assert budget.product(tmp_path) != before
        (tmp_path / name / "unit.py").write_text("x = 1\n", encoding="utf-8")
    assert budget.product(tmp_path) == before


def test_the_product_packages_include_what_a_run_executes():
    assert {"pipeline", "runtime", "models", "api"} <= set(budget.PRODUCT)
    for name in budget.PRODUCT:
        assert (budget.PACKAGE / name).is_dir()


def test_an_unknown_child_mode_is_refused():
    with pytest.raises(errors.InputError):
        budget.child(["walk"])
    with pytest.raises(errors.InputError):
        budget.table(["quality"])
    with pytest.raises(errors.InputError):
        budget.table(["quality", "1", "d", "30"])


def test_the_stored_measurements_are_for_the_current_code():
    assert budget.problems() == []
    stored = json.loads(budget.PATH.read_text(encoding="utf-8"))
    assert stored["compliance"]["trials"] == len(stored["compliance"]["runs"])
    assert stored["resume"]["trials"] == len(budget.KILL_POINTS)
    real = stored["real"]
    refused = [r for r in real["runs"] if r["refused"]]
    assert real["refused"] == len(refused)
    assert real["trials"] == len(real["runs"]) - len(refused)
    assert real["hits"] == sum(r["compliant"] for r in real["runs"])
    for run in real["runs"]:
        assert run["compliant"] == (run["outcome"] == "compliant")
        if run["compliant"]:
            assert run["run_seconds"] <= run["budget"] + limits.SLACK_SECONDS
