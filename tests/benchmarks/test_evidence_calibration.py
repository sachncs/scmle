"""Claim C3 evidence: simulation-based calibration of intervals and tests."""

import json
import pathlib
import re
from typing import Any

import numpy
import pytest
from scipy import stats
import sklearn

from benchmarks import acceptance
from benchmarks.evidence import calibration
from scmle.assurance.admissibility import leakage
from scmle.contracts import errors
from scmle.evaluation import conformal
from scmle.evaluation import coverage

ROOT = pathlib.Path(calibration.__file__).resolve().parents[2]


def document(trials: dict[str, int] | None = None) -> dict[str, Any]:
    """A current document whose checks are all at their nominal rate."""
    level = calibration.level()
    scenarios = []
    for scenario in calibration.SCENARIOS:
        count = (trials or {}).get(scenario.name, calibration.planned(scenario))
        scenarios.append(
            {
                "name": scenario.name,
                "trials": count,
                "checks": [
                    {"field": c.field, "state": "nominal"}
                    for c in scenario.checks
                ],
            }
        )
    return {
        "code": calibration.digest(),
        "family": calibration.family(),
        "level": level,
        "scenarios": scenarios,
        "miscalibrated": [],
    }


def test_scenario_names_are_unique_and_every_check_reads_its_own_field() -> (
    None
):
    names = [s.name for s in calibration.SCENARIOS]
    assert len(names) == len(set(names))
    for scenario in calibration.SCENARIOS:
        fields = [c.field for c in scenario.checks]
        assert len(fields) == len(set(fields))
        assert scenario.location.startswith("scmle/")
        assert ".py:" in scenario.location


def test_the_named_product_surfaces_are_all_calibrated() -> None:
    # The clause names bootstrap, conformal, rd/did/causal intervals,
    # subgroup tests and the leakage permutation tests.
    locations = " ".join(
        [s.location for s in calibration.SCENARIOS]
        + [c.location for s in calibration.SCENARIOS for c in s.checks]
    )
    for part in (
        "evaluation/bootstrap.py:Uncertainty.compute",
        "evaluation/conformal.py:ConformalCalibrator.interval",
        "evaluation/conformal.py:ConformalCalibrator.sets",
        "evaluation/conformal.py:ConformalCalibrator.weighted_interval",
        "evaluation/cv_plus.py:CvPlusEnsemble.interval",
        "evaluation/confidence.py:cv_interval",
        "causal/identification/rd.py:estimate",
        "causal/identification/did.py:estimate",
        "causal/identification/run/adjustment.py:RandomizedEstimator",
        "causal/identification/run/strata.py:stratify",
        "parameter_estimation/estimate.py:estimate_parameters",
        "evaluation/coverage.py:slice_coverage",
        "evaluation/coverage.py:undercovered",
        "admissibility/leakage/__init__.py:probe",
        "admissibility/leakage/missingness.py:missingness_test",
        "admissibility/screen/__init__.py:screen",
        "causal/identification/aipw.py:estimate",
        "causal/identification/iv.py:estimate",
        "causal/identification/iv.py:anderson_rubin",
        "causal/identification/overid.py:sargan",
        "causal/identification/overid.py:hansen_j",
        "causal/identification/synth.py:estimate",
    ):
        assert part in locations, part


def test_the_family_level_is_alpha_over_every_check() -> None:
    alpha = acceptance.load()["alpha"].value
    count = sum(len(s.checks) for s in calibration.SCENARIOS)
    assert calibration.family() == count
    assert calibration.level() == pytest.approx(alpha / count)


def test_a_rate_inside_its_exact_bounds_is_nominal() -> None:
    check = calibration.Rate("covered", 0.9, -1)
    found = check.judge([1.0] * 90 + [0.0] * 10, 0.05)
    assert found["count"] == 90
    assert found["trials"] == 100
    assert found["rate"] == 0.9
    assert found["state"] == "nominal"
    assert found["kind"] == "coverage"
    low, high = stats.beta.ppf([0.025, 0.975], [90, 91], [11, 10])
    assert found["low"] == pytest.approx(low)
    assert found["high"] == pytest.approx(high)


def test_undercoverage_is_liberal_and_overcoverage_conservative() -> None:
    check = calibration.Rate("covered", 0.95, -1)
    assert check.judge([1.0] * 80 + [0.0] * 20, 0.05)["state"] == "liberal"
    assert check.judge([1.0] * 1000, 0.05)["state"] == "conservative"


def test_oversize_is_liberal_and_undersize_conservative() -> None:
    check = calibration.Rate("rejected", 0.05, 1)
    assert check.kind == "size"
    assert check.judge([1.0] * 20 + [0.0] * 80, 0.05)["state"] == "liberal"
    assert check.judge([0.0] * 1000, 0.05)["state"] == "conservative"


def test_an_errored_trial_counts_against_the_product() -> None:
    covering = calibration.Rate("covered", 0.9, -1)
    found = covering.judge([1.0] * 9 + [None], 0.05)
    assert (found["count"], found["errors"]) == (9, 1)
    size = calibration.Rate("rejected", 0.05, 1)
    found = size.judge([0.0] * 9 + [None], 0.05)
    assert (found["count"], found["errors"]) == (1, 1)


def test_ranks_fill_bins_and_a_lopsided_spread_is_nonuniform() -> None:
    check = calibration.Uniform("rank")
    even = [(i + 0.5) / 1000 for i in range(1000)]
    found = check.judge(even, 0.05)
    assert found["bins"] == [100] * calibration.BINS
    assert found["state"] == "nominal"
    assert found["kind"] == "rank"
    lopsided = [0.05] * 300 + even[:700]
    assert check.judge(lopsided, 0.05)["state"] == "nonuniform"
    assert check.judge([*even, None], 0.05)["state"] == "errored"


def test_the_exact_power_matches_a_direct_sum() -> None:
    # Power of the two-sided Clopper-Pearson rule at n = 50, null 0.9,
    # truth 0.85, computed term by term from the product's own bound.
    trials, null, truth, level = 50, 0.9, 0.85, 0.05
    total = 0.0
    for k in range(trials + 1):
        low, high = coverage.clopper_pearson(k, trials, 1 - level)
        if high < null or low > null:
            total += stats.binom.pmf(k, trials, truth)
    assert calibration.chance(null, truth, trials, level) == pytest.approx(
        total
    )


def test_the_planned_count_reaches_the_power_and_halving_it_does_not() -> None:
    power = acceptance.load()["power"].value
    level = calibration.level()
    for check in (
        calibration.Rate("covered", 0.95, -1),
        calibration.Rate("rejected", 0.05, 1),
    ):
        count = check.plan(level)
        truth = check.nominal + check.liberal * calibration.DELTA
        assert calibration.chance(check.nominal, truth, count, level) >= power
        assert (
            calibration.chance(check.nominal, truth, count // 2, level) < power
        )
    rank = calibration.Uniform("rank")
    share = 1 / calibration.BINS
    count = rank.plan(level)
    truth, bin_level = share + calibration.DELTA, level / calibration.BINS
    assert calibration.chance(share, truth, count, bin_level) >= power
    assert calibration.chance(share, truth, count // 2, bin_level) < power


def test_attained_conformal_coverage_is_the_finite_sample_rank() -> None:
    assert calibration.attained(199, 0.9) == 0.9
    assert calibration.attained(40, 0.9) == pytest.approx(37 / 41)
    assert calibration.attained(5, 0.9) == 1.0
    assert calibration.attained(0, 0.9) == 1.0


def test_the_mondrian_nominal_is_the_class_size_mixture() -> None:
    total = 0.0
    for share in calibration.SHARES:
        for rows in range(calibration.CALIBRATION + 1):
            weight = stats.binom.pmf(rows, calibration.CALIBRATION, share)
            total += share * weight * calibration.attained(rows, 0.9)
    assert calibration.mondrian() == pytest.approx(total)
    assert 0.9 < calibration.mondrian() < 0.91


def test_the_holm_nominal_is_the_chance_any_exact_test_rejects() -> None:
    rows, null, tests = 100, 0.9, 4
    cut = 0.05 / tests
    reject = sum(
        stats.binom.pmf(k, rows, null)
        for k in range(rows + 1)
        if stats.binom.cdf(k, rows, null) <= cut
    )
    assert calibration.holm(rows, null, tests) == pytest.approx(
        1 - (1 - reject) ** tests
    )


def test_the_validation_truth_is_the_expected_ols_risk() -> None:
    # E[(y - x'b)^2] = sigma^2 (1 + p / (n - p - 1)) for Gaussian rows.
    rows = calibration.VALIDATION_ROWS * (calibration.FOLDS - 1)
    rows //= calibration.FOLDS
    expected = 1 + calibration.FEATURES / (rows - calibration.FEATURES - 1)
    assert calibration.risk() == pytest.approx(expected)


@pytest.mark.parametrize(
    "name",
    [
        s.name
        for s in calibration.SCENARIOS
        if not s.name.startswith("bootstrap.")
    ],
)
def test_every_scenario_draws_every_field_it_is_judged_on(name: str) -> None:
    found = calibration.run(name, 0)
    assert "error" not in found, found
    scenario = calibration.lookup(name)
    for check in scenario.checks:
        value = found[check.field]
        assert 0.0 <= value <= 1.0
        if isinstance(check, calibration.Rate):
            assert value in (0.0, 1.0)
        else:
            assert value < 1.0
    assert calibration.run(name, 0) == found


def test_a_bootstrap_trial_runs_the_coverage_trial() -> None:
    before = sklearn.get_config()
    found = calibration.run("bootstrap.regression_rows", 0)
    assert set(found) == {"covered"}
    assert found["covered"] in (0.0, 1.0)
    # The bootstrap settings never leak into the next scenario's trial.
    assert sklearn.get_config() == before


def test_a_typed_failure_is_recorded_not_raised(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail(rng: numpy.random.Generator) -> dict[str, float]:
        raise errors.InputError("too few rows")

    scenario = calibration.lookup("conformal.regression")
    broken = calibration.Scenario(
        scenario.name, scenario.location, fail, scenario.checks
    )
    monkeypatch.setattr(calibration, "SCENARIOS", (broken,))
    assert calibration.run(scenario.name, 0) == {
        "error": "InputError: too few rows"
    }


def test_an_unknown_scenario_is_a_typed_error() -> None:
    with pytest.raises(errors.InputError, match="unknown scenario"):
        calibration.lookup("nothing")


def test_measure_builds_a_document_from_a_few_trials() -> None:
    found = calibration.measure(
        trials=4, workers=1, names=("conformal.regression", "did.effect")
    )
    assert found["claim"] == "C3"
    assert found["family"] == calibration.family()
    assert [s["name"] for s in found["scenarios"]] == [
        "conformal.regression",
        "did.effect",
    ]
    for scenario in found["scenarios"]:
        assert scenario["trials"] == 4
        for check in scenario["checks"]:
            assert check["state"] in calibration.STATES
    assert found["code"] == calibration.digest()
    json.dumps(found)


def test_a_full_nominal_document_is_partial_while_surfaces_are_unmeasured() -> (
    None
):
    assert calibration.problems(document()) == []
    text, verdict = calibration.summary(document())
    assert verdict == "partial"
    total = calibration.family()
    left = len(calibration.NOT_MEASURED)
    assert text == (
        f"{total}/{total} checks nominal; {left} product intervals and "
        "tests not measured"
    )


def test_a_full_nominal_document_passes_once_nothing_is_unmeasured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(calibration, "NOT_MEASURED", ())
    text, verdict = calibration.summary(document())
    assert verdict == "pass"
    total = calibration.family()
    assert text == f"{total}/{total} checks nominal"


def test_every_unmeasured_surface_names_an_existing_definition() -> None:
    assert calibration.NOT_MEASURED
    for location, what in calibration.NOT_MEASURED:
        assert what
        path, name = location.split(":")
        source = (ROOT / path).read_text(encoding="utf-8")
        *outer, inner = name.split(".")
        for part in (*outer, inner):
            pattern = rf"^\s*(def|class) {re.escape(part)}\b"
            assert re.search(pattern, source, re.MULTILINE), location


def test_a_miscalibrated_check_fails_and_is_named() -> None:
    found = document()
    first = found["scenarios"][0]
    first["checks"][0]["state"] = "liberal"
    found["miscalibrated"] = [
        {
            "scenario": first["name"],
            "field": first["checks"][0]["field"],
            "state": "liberal",
            "location": "here",
        }
    ]
    text, verdict = calibration.summary(found)
    assert verdict == "fail"
    total = calibration.family()
    field = first["checks"][0]["field"]
    left = len(calibration.NOT_MEASURED)
    assert text == (
        f"{total - 1}/{total} checks nominal; miscalibrated: "
        f"{first['name']}.{field} (liberal); {left} product intervals and "
        "tests not measured"
    )


def test_too_few_trials_stale_code_or_a_missing_scenario_decide_nothing() -> (
    None
):
    name = calibration.SCENARIOS[0].name
    few = document({name: 3})
    least = calibration.planned(calibration.SCENARIOS[0])
    assert calibration.problems(few) == [
        f"calibration: {name} has 3 trials, fewer than the {least} planned"
    ]
    assert calibration.summary(few)[1] == "not measured"
    stale = document()
    stale["code"] = "old"
    assert "calibration: measured with other code" in calibration.problems(
        stale
    )
    missing = document()
    missing["scenarios"] = missing["scenarios"][1:]
    assert f"calibration: scenario {name} not measured" in (
        calibration.problems(missing)
    )


def test_check_reports_a_missing_file(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(calibration, "PATH", tmp_path / "calibration.json")
    assert calibration.main(["--check"]) == 1
    assert "calibration.json is missing" in capsys.readouterr().out
    assert calibration.summary() == ("no measurement", "not measured")


def test_every_check_reaches_the_power_at_its_scenario_planned_count() -> None:
    power = acceptance.load()["power"].value
    level = calibration.level()
    share = 1 / calibration.BINS
    for scenario in calibration.SCENARIOS:
        count = calibration.planned(scenario)
        for check in scenario.checks:
            if isinstance(check, calibration.Rate):
                truth = check.nominal + check.liberal * calibration.DELTA
                found = calibration.chance(check.nominal, truth, count, level)
            else:
                found = calibration.chance(
                    share,
                    share + calibration.DELTA,
                    count,
                    level / calibration.BINS,
                )
            assert found >= power, (scenario.name, check.field)
            assert check.chance(count, level) == found


def test_a_field_a_trial_did_not_report_counts_against_the_product() -> None:
    scenario = calibration.Scenario(
        "absent",
        "scmle/x.py:y",
        calibration.conform,
        (calibration.Rate("rejected", 0.05, 1),),
    )
    found = calibration.tally(scenario, [{}, {"rejected": 0.0}])
    assert found["errors"] == 1
    assert found["messages"] == ["rejected not reported in 1 trials"]
    check = found["checks"][0]
    assert (check["absent"], check["errors"], check["count"]) == (1, 1, 1)
    refused = calibration.tally(
        scenario, [{"error": "InputError: no"}, {"rejected": 0.0}]
    )
    assert refused["errors"] == 1
    assert refused["messages"] == ["InputError: no"]
    assert refused["checks"][0]["absent"] == 0


def test_a_placebo_rank_grid_fills_its_bins_evenly() -> None:
    # 1 - 16/20 rounds to 0.19999999999999996; it still belongs to bin 2.
    donors = calibration.DONORS
    ranks = [
        calibration.rank((1 + k) / (donors + 1)) for k in range(donors + 1)
    ]
    found = calibration.Uniform("rank").judge(ranks, 0.05)
    per = (donors + 1) // calibration.BINS
    assert found["bins"] == [per] * calibration.BINS


def test_the_pool_gives_the_records_of_one_process() -> None:
    names = ("conformal.regression",)
    pooled = calibration.measure(trials=2, workers=2, names=names)
    alone = calibration.measure(trials=2, workers=1, names=names)
    assert pooled["scenarios"] == alone["scenarios"]


def test_write_stores_the_document_and_check_reads_it(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(calibration, "PATH", tmp_path / "calibration.json")
    stored = document()

    def measure(**options: Any) -> dict[str, Any]:
        return stored

    monkeypatch.setattr(calibration, "measure", measure)
    assert calibration.main(["--write"]) == 0
    assert json.loads(calibration.PATH.read_text(encoding="utf-8")) == stored
    assert capsys.readouterr().out == (
        json.dumps(calibration.summary(stored)) + "\n"
    )
    assert calibration.problems() == []
    assert calibration.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "problems": []}


def test_the_conformal_rank_grid_fills_whole_bins() -> None:
    # A discrete rank is uniform on n + 1 values; bins are exact only when
    # the grid divides into them.
    assert (calibration.CALIBRATION + 1) % calibration.BINS == 0
    assert (leakage.CERTIFIED.permutations + 1) % calibration.BINS == 0
    assert conformal.MIN_CALIBRATION_ROWS <= calibration.CALIBRATION
