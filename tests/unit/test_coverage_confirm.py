"""The coverage claim on C: one registered, exact one-sided test."""

import numpy
import pandas
from scipy import stats

from scmle.data.ledger import confirm
from scmle.evaluation import coverage_confirm
from scmle.evaluation import task_gate

CHOICE = coverage_confirm.primary_choice("regression")


def test_the_primary_choice_is_fixed_by_the_task():
    assert CHOICE == {
        "method": "residual",
        "level": 0.9,
        "tolerance": 0.02,
        "alpha": 0.05,
    }
    assert (
        coverage_confirm.primary_choice("binary_classification")["method"]
        == "pooled"
    )
    assert coverage_confirm.primary_choice("ranking") is None


def test_the_p_value_is_the_exact_binomial_tail_above_the_floor():
    decided = coverage_confirm.decide(95, 100, CHOICE)
    assert decided["floor"] == 0.88
    assert decided["p_value"] == stats.binom.sf(94, 100, 0.88)
    assert decided["status"] == "confirmed"
    assert coverage_confirm.decide(90, 100, CHOICE)["status"] == "unverified"


def test_a_shortfall_is_reported_not_confirmed():
    low = coverage_confirm.decide(60, 100, CHOICE)
    assert low["status"] == "unverified" and low["coverage"] == 0.6


def test_few_rows_cannot_confirm():
    assert coverage_confirm.decide(10, 10, CHOICE)["status"] == "unverified"
    assert coverage_confirm.decide(0, 0, CHOICE)["status"] == "unverified"


def test_the_registration_hash_depends_on_the_claim_and_the_choice():
    first = coverage_confirm.registration("a" * 64, CHOICE)
    assert first.claim_id == "coverage_nominal" and first.kind == "coverage"
    assert (
        first.sha256 == coverage_confirm.registration("a" * 64, CHOICE).sha256
    )
    assert (
        first.sha256 != coverage_confirm.registration("b" * 64, CHOICE).sha256
    )
    other = {**CHOICE, "tolerance": 0.05}
    assert first.sha256 != coverage_confirm.registration("a" * 64, other).sha256


class Calibrator:
    kind = "regression"
    scores = tuple(numpy.linspace(0.0, 1.0, 100))


class Band:
    conformal = Calibrator()
    quantile = None

    def predict_interval(self, features, coverage, method):
        centre = features["x"].to_numpy(dtype=float)
        return pandas.DataFrame(
            {
                "prediction": centre,
                "lower": centre - 1.0,
                "upper": centre + 1.0,
            }
        )


def test_the_runner_tests_the_released_rows_once():
    runner = coverage_confirm.build_runner(Band(), "regression", "c" * 64)
    assert runner is not None and len(runner.registered) == 1
    rng = numpy.random.default_rng(0)
    x = pandas.DataFrame({"x": rng.normal(size=400)})
    y = x["x"] + rng.uniform(-1.0, 1.0, 400)  # always inside the band
    ran = runner.run(x, y)
    document = ran.documents[coverage_confirm.DOCUMENT]
    assert ran.statuses == {"coverage_nominal": "confirmed"}
    assert document["covered"] == document["rows"] == 400


def test_no_calibration_or_no_interval_task_gives_no_runner():
    class Bare:
        conformal = None

    assert coverage_confirm.build_runner(Bare(), "regression", "c" * 64) is None
    assert coverage_confirm.build_runner(Band(), "ranking", "c" * 64) is None
    assert (
        coverage_confirm.not_evaluated("x")["status"] == confirm.NOT_EVALUATED
    )


def run_band_rows():
    rng = numpy.random.default_rng(0)
    x = pandas.DataFrame({"x": rng.normal(size=400)})
    y = x["x"] + rng.uniform(-1.0, 1.0, 400)
    runner = coverage_confirm.build_runner(Band(), "regression", "c" * 64)
    return runner.run(x, y)


def test_a_passed_task_type_keeps_the_status_and_records_its_gate():
    document = run_band_rows().documents[coverage_confirm.DOCUMENT]
    assert document["status"] == "confirmed"
    assert document["task_gate"] == {
        "task": "regression",
        "gate": "passed",
        "capped": False,
    }


def test_a_task_type_that_has_not_passed_its_gate_is_capped_at_unverified(
    monkeypatch,
):
    monkeypatch.setattr(task_gate, "status", lambda task: "no_referent")
    ran = run_band_rows()
    document = ran.documents[coverage_confirm.DOCUMENT]
    assert ran.statuses == {"coverage_nominal": "unverified"}
    assert document["status"] == "unverified"
    assert document["task_gate"]["capped"] is True
    assert document["task_gate"]["gate"] == "no_referent"
    assert document["covered"] == document["rows"] == 400


def test_the_cap_never_raises_a_status():
    assert coverage_confirm.capped("regression", "unverified") == "unverified"
    assert coverage_confirm.capped("regression", "confirmed") == "confirmed"
    assert coverage_confirm.capped("ordinal", "confirmed") == "unverified"
