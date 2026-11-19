"""Claim C3 evidence: coverage of the 95 percent metric intervals."""

import json

import pytest

from benchmarks import acceptance
from benchmarks.evidence import coverage
from scmle.contracts import errors


def family(name: str, hits: int, trials: int) -> dict:
    return {"family": name, "hits": hits, "trials": trials}


def document(counts: dict[str, tuple[int, int]] | None = None) -> dict:
    """A current document with the given (hits, trials) per family."""
    planned = coverage.required()[1]
    counts = counts or dict.fromkeys(coverage.FAMILIES, (planned, planned))
    families = [family(n, h, t) for n, (h, t) in counts.items()]
    return {
        "floor": acceptance.load()[coverage.VARIABLE].value,
        "code": coverage.digest(),
        "families": families,
        "pooled": {
            "hits": sum(f["hits"] for f in families),
            "trials": sum(f["trials"] for f in families),
        },
    }


def test_the_trial_count_is_the_planned_sample_from_acceptance():
    least, planned = coverage.required()
    assert least == 42
    assert coverage.TRIALS >= planned >= least


def test_the_worst_of_all_families_has_the_planned_power():
    # Every family must pass, so a nominal interval passes the row with the
    # product of the families' chances; that product, not each factor,
    # must reach `coverage_power`.
    variables = acceptance.load()
    alpha = variables["alpha"].value
    power = variables[coverage.POWER].value
    truth = variables[coverage.ASSUMED].value
    planned = coverage.required()[1]
    each = coverage.rule().chance(truth, planned, alpha)
    assert each ** len(coverage.FAMILIES) >= power
    fewer = coverage.rule().chance(truth, planned - 1, alpha)
    assert fewer ** len(coverage.FAMILIES) < power
    assert planned == coverage.TRIALS


def test_the_row_is_planned_for_joint_power_of_at_least_ninety_percent():
    # A verdict within noise of the floor is decided by chance; the row is
    # planned so a nominal interval passes all three families together at
    # least nine times in ten.
    variables = acceptance.load()
    assert variables[coverage.POWER].value >= 0.90
    assert variables[coverage.ASSUMED].value == 0.95
    assert variables[coverage.POWER].kind == "policy"
    assert coverage.TRIALS == 1678


def test_a_current_document_with_enough_hits_passes():
    text, verdict = coverage.summary(document())
    assert coverage.problems(document()) == []
    assert verdict == "pass"
    assert "pooled" in text


def test_the_worst_family_decides():
    planned = coverage.required()[1]
    counts = dict.fromkeys(coverage.FAMILIES, (planned, planned))
    counts["binary_groups"] = (int(planned * 0.92), planned)
    text, verdict = coverage.summary(document(counts))
    assert verdict == "fail"
    assert "binary_groups" in text


def test_too_few_trials_is_not_measured_even_when_every_trial_hits():
    counts = dict.fromkeys(coverage.FAMILIES, (41, 41))
    issues = coverage.problems(document(counts))
    assert any("none can pass below 42" in i for i in issues)
    assert coverage.summary(document(counts))[1] == "not measured"


def test_fewer_than_the_planned_trials_is_a_problem():
    counts = dict.fromkeys(coverage.FAMILIES, (100, 100))
    issues = coverage.problems(document(counts))
    assert len(issues) == len(coverage.FAMILIES)
    assert all("planned" in i for i in issues)


def test_a_missing_family_is_a_problem():
    planned = coverage.required()[1]
    counts = {"binary_rows": (planned, planned)}
    assert "coverage: family regression_rows not measured" in (
        coverage.problems(document(counts))
    )


def test_stale_code_or_floor_is_not_measured():
    changed = document() | {"code": "0" * 64}
    assert coverage.problems(changed) == ["coverage: measured with other code"]
    assert coverage.summary(changed)[1] == "not measured"
    moved = document() | {"floor": 0.5}
    assert coverage.summary(moved)[1] == "not measured"


def test_an_empty_document_is_not_measured():
    assert coverage.summary({"families": []}) == (
        "no measurement",
        "not measured",
    )


def test_tally_counts_misses_on_each_side_and_missing_intervals():
    outcomes = [
        {"interval": [0.0, 1.0], "truth": 0.5, "hit": True},
        {"interval": [0.6, 1.0], "truth": 0.5, "hit": False},
        {"interval": [0.0, 0.4], "truth": 0.5, "hit": False},
        {"interval": None, "truth": 0.5, "hit": False},
    ]
    counted = coverage.tally("binary_rows", outcomes)
    assert (counted["hits"], counted["trials"]) == (1, 4)
    assert (counted["below"], counted["above"]) == (1, 1)
    assert counted["no_interval"] == 1
    assert counted["passed"] is False


def test_an_unknown_family_is_refused():
    with pytest.raises(errors.InputError):
        coverage.simulate("nothing", None, 10)  # type: ignore[arg-type]


def test_the_stored_coverage_is_for_the_current_code():
    stored = json.loads(coverage.PATH.read_text(encoding="utf-8"))
    assert stored["code"] == coverage.digest()
    assert coverage.problems() == []
    assert coverage.summary()[1] in {"pass", "fail"}


def test_a_change_to_training_or_model_code_makes_check_stale(
    tmp_path, monkeypatch, capsys
):
    for name in coverage.PRODUCT:
        folder = tmp_path / "scmle" / name
        folder.mkdir(parents=True)
        (folder / "unit.py").write_text("x = 1\n", encoding="utf-8")
    monkeypatch.setattr(coverage, "PACKAGE", tmp_path / "scmle")
    monkeypatch.setattr(coverage, "PATH", tmp_path / "coverage.json")
    coverage.PATH.write_text(json.dumps(document()), encoding="utf-8")
    assert coverage.main(["--check"]) == 0
    for name in ("preparation", "models"):
        changed = tmp_path / "scmle" / name / "unit.py"
        changed.write_text("x = 2\n", encoding="utf-8")
        assert coverage.main(["--check"]) == 1
        assert "measured with other code" in capsys.readouterr().out
        changed.write_text("x = 1\n", encoding="utf-8")
    assert coverage.main(["--check"]) == 0


def test_the_fingerprint_covers_the_training_and_model_packages():
    assert {"preparation", "models", "evaluation"} <= set(coverage.PRODUCT)
    for name in coverage.PRODUCT:
        assert (coverage.PACKAGE / name).is_dir()
