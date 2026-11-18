"""Independent reference calculations for the IPW and AIPW effects."""

import numpy
import pytest
from scipy import special

from scmle.evaluation.parameter_estimation import causal
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import linear
from scmle.evaluation.parameter_estimation import references
from scmle.evaluation.parameter_estimation import weighting

CAUSAL = ("consistency", "conditional_exchangeability", "positivity")


def policy(estimator="ipw_ate", **changes):
    values = {
        "estimator": estimator,
        "covariance": "iid",
        "treatment_position": 0,
        "causal_assumptions": CAUSAL,
    }
    values.update(changes)
    return config.Config(**values)


def world(rows=300, seed=5):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    t = (rng.random(rows) < special.expit(0.4 * x)).astype(float)
    y = 0.5 + 1.5 * t + x + rng.normal(scale=0.4, size=rows)
    return numpy.column_stack([t, x]), y


def test_a_bfgs_fit_matches_the_newton_logit():
    matrix, ignored = world()
    design = causal.confound(matrix, 0)
    result = references.bfgs(design, matrix[:, 0], policy())
    newton = linear.logit(design, matrix[:, 0], 100, 1e-12)[0]
    assert result.success
    numpy.testing.assert_allclose(result.x, newton, atol=1e-5)


def test_a_verdict_compares_the_mean_score_with_the_primary_ate():
    scores = numpy.array([1.0, 2.0, 3.0, 6.0])
    ate, variance, method, difference = references.verdict(
        scores, numpy.array([2.5]), policy(), None
    )
    assert ate == 3.0
    assert variance == pytest.approx(
        weighting.dispersion(scores - 3.0, "iid", None)[0]
    )
    assert method == "iid_centered_influence_function"
    assert difference == 0.5


def test_an_ipw_reference_agrees_with_the_primary_fit():
    matrix, y = world()
    ate, ignored, ignored, design = causal.fit_ipw_ate(
        matrix, y, 0, policy(), None
    )
    found = references.reference_ipw_ate(
        design, y, matrix[:, 0], ate, policy(), None
    )
    assert found["method"] == "scipy.optimize.minimize:BFGS_propensity"
    assert found["converged"] is True
    assert found["passed"] is True
    assert found["reference_ate"] == pytest.approx(ate[0], abs=1e-5)
    assert (
        found["variance_status"]
        == "declared_influence_calculation_not_independently_verified"
    )
    off = references.reference_ipw_ate(
        design, y, matrix[:, 0], ate + 1.0, policy(), None
    )
    assert off["passed"] is False
    assert off["max_absolute_difference"] == pytest.approx(1.0, abs=1e-5)


def test_an_aipw_reference_refits_the_nuisances():
    matrix, y = world()
    ate, ignored, ignored, inputs = causal.dr(
        matrix, y, 0, policy("aipw_ate"), None
    )
    found = references.reference_aipw_ate(
        inputs, y, ate, policy("aipw_ate"), None
    )
    assert found["method"] == "scipy.optimize.minimize:BFGS_plus_lstsq"
    assert found["passed"] is True
    assert found["max_absolute_difference"] < 1e-5


def test_a_cross_fitted_reference_records_each_fold():
    matrix, y = world()
    chosen = policy("aipw_ate", cross_fitting_folds=3)
    ate, ignored, ignored, inputs = causal.dr(matrix, y, 0, chosen, None)
    found = references.reference_aipw_ate(inputs, y, ate, chosen, None)
    assert (
        found["method"]
        == "scipy.optimize.minimize:BFGS_plus_lstsq_per_training_fold"
    )
    assert [f["fold"] for f in found["folds"]] == [0, 1, 2]
    assert all(f["converged"] for f in found["folds"])
    assert found["passed"] is True
    scores, record = references.refold(
        inputs, y, chosen, inputs["fold_assignments"] != 0, 0
    )
    assert len(scores) == 100
    assert record["fold"] == 0


def test_a_cross_fitted_reference_rejects_misaligned_folds():
    matrix, y = world()
    chosen = policy("aipw_ate", cross_fitting_folds=2)
    ate, ignored, ignored, inputs = causal.dr(matrix, y, 0, chosen, None)
    broken = {**inputs, "fold_assignments": inputs["fold_assignments"][:-1]}
    with pytest.raises(ValueError, match="fold assignments are invalid"):
        references.reference_cross_fitted_aipw_ate(broken, y, ate, chosen, None)
