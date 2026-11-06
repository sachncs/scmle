"""The coverage claim on C: one registered, exact one-sided test."""

import numpy as np
import pandas as pd
from scipy import stats

from scmle.data.ledger import confirm
from scmle.evaluation import coverage_confirm

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


class Band:
    conformal = Calibrator()
    quantile = None

    def predict_interval(self, features, coverage, method):
        centre = features["x"].to_numpy(dtype=float)
        return pd.DataFrame({"lower": centre - 1.0, "upper": centre + 1.0})


def test_the_runner_tests_the_released_rows_once():
    runner = coverage_confirm.build_runner(Band(), "regression", "c" * 64)
    assert runner is not None and len(runner.registered) == 1
    rng = np.random.default_rng(0)
    x = pd.DataFrame({"x": rng.normal(size=400)})
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
