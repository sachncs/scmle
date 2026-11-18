"""Linear and logistic likelihoods and their sandwich covariances."""

import numpy
import pytest
from scipy import special
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import linear


def gaussian(rows=80, seed=0):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 2))
    design = numpy.column_stack([numpy.ones(rows), x])
    y = design @ numpy.array([1.0, 2.0, -0.5]) + rng.normal(
        scale=0.3, size=rows
    )
    return design, y


def bernoulli(rows=300, seed=1):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=rows)
    design = numpy.column_stack([numpy.ones(rows), x])
    y = (rng.random(rows) < special.expit(-0.3 + 1.2 * x)).astype(float)
    return design, y


def test_a_regression_solves_the_normal_equations():
    design, y = gaussian()
    coefficients, diagnostics = linear.regress(design, y, 1, 1.0)
    numpy.testing.assert_allclose(
        coefficients, numpy.linalg.lstsq(design, y, rcond=None)[0]
    )
    assert diagnostics == {"converged": True, "iterations": 1}


def test_a_rank_deficient_regression_is_rejected():
    design = numpy.column_stack([numpy.ones(5), numpy.ones(5)])
    with pytest.raises(
        errors.InputError, match="linear design matrix is rank deficient"
    ):
        linear.regress(design, numpy.arange(5.0), 1, 1.0)


def test_a_log_likelihood_is_clipped_bernoulli():
    values = numpy.array([1.0, 0.0, 1.0])
    probabilities = numpy.array([0.8, 0.3, 0.0])
    expected = numpy.log(0.8) + numpy.log(0.7) + numpy.log(1e-15)
    assert linear.loglik(values, probabilities) == pytest.approx(expected)


def test_a_logit_converges_to_the_score_root():
    design, y = bernoulli()
    coefficients, probabilities, diagnostics = linear.logit(
        design, y, 100, 1e-10
    )
    assert numpy.abs(design.T @ (y - probabilities)).max() < 1e-6
    assert diagnostics["converged"] is True
    assert 1 < diagnostics["iterations"] < 100
    assert diagnostics["log_likelihood"] == pytest.approx(
        linear.loglik(y, probabilities)
    )
    numpy.testing.assert_allclose(
        probabilities, special.expit(design @ coefficients)
    )


def test_a_logit_that_cannot_converge_raises():
    design, y = bernoulli()
    with pytest.raises(
        errors.InputError, match="logistic estimation did not converge"
    ):
        linear.logit(design, y, 1, 1e-12)


def test_an_iid_linear_covariance_scales_the_bread():
    design, y = gaussian()
    beta = numpy.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    bread = numpy.linalg.inv(design.T @ design)
    scale = (residuals**2).sum() / (len(y) - 3)
    numpy.testing.assert_allclose(
        linear.covariance_matrix(design, residuals, "iid", None, None),
        scale * bread,
    )


def test_an_hc3_linear_covariance_divides_by_one_minus_leverage():
    design, y = gaussian()
    beta = numpy.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    bread = numpy.linalg.inv(design.T @ design)
    leverage = numpy.einsum("ij,jk,ik->i", design, bread, design)
    scores = design * (residuals / (1 - leverage))[:, None]
    numpy.testing.assert_allclose(
        linear.covariance_matrix(design, residuals, "hc3", None, None),
        bread @ scores.T @ scores @ bread,
    )


def test_a_clustered_covariance_applies_the_small_sample_correction():
    design, y = gaussian(rows=60)
    beta = numpy.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ beta
    clusters = numpy.repeat(numpy.arange(6), 10)
    bread = numpy.linalg.inv(design.T @ design)
    meat = sum(
        numpy.outer(
            design[clusters == g].T @ residuals[clusters == g],
            design[clusters == g].T @ residuals[clusters == g],
        )
        for g in range(6)
    )
    correction = (6 / 5) * (59 / 57)
    numpy.testing.assert_allclose(
        linear.covariance_matrix(
            design, residuals, "clustered", clusters, None
        ),
        correction * bread @ meat @ bread,
    )


def test_a_clustered_covariance_needs_two_labelled_clusters():
    design, y = gaussian(rows=20)
    with pytest.raises(errors.InputError, match="cluster labels are required"):
        linear.covariance_matrix(design, y, "clustered", None, None)
    with pytest.raises(errors.InputError, match="at least two clusters"):
        linear.covariance_matrix(design, y, "clustered", numpy.zeros(20), None)


def test_a_logistic_iid_covariance_is_the_inverse_information():
    design, y = bernoulli()
    ignored, p, ignored = linear.logit(design, y, 100, 1e-10)
    information = design.T @ ((p * (1 - p))[:, None] * design)
    numpy.testing.assert_allclose(
        linear.covariance_matrix(design, y - p, "iid", None, p),
        numpy.linalg.inv(information),
    )
    hc3 = linear.covariance_matrix(design, y - p, "hc3", None, p)
    assert hc3.shape == (2, 2)
    assert numpy.allclose(hc3, hc3.T)


def test_an_ols_fit_reports_t_intervals_and_the_reference():
    design, y = gaussian()
    policy = config.Config(
        estimator="linear", covariance="iid", confidence_level=0.9
    )
    fit = linear.ols(design, y, policy, None, ("intercept", "a", "b"))
    assert fit.names == ("intercept", "a", "b")
    assert fit.likelihood == "gaussian"
    assert fit.pivot == pytest.approx(stats.t.ppf(0.95, 77))
    assert fit.reference["method"] == "numpy.linalg.lstsq"
    assert fit.reference["passed"] is True
    assert fit.reference["max_absolute_difference"] < 1e-9
    residuals = y - design @ fit.coefficients
    assert fit.diagnostics["residual_sum_of_squares"] == pytest.approx(
        (residuals**2).sum()
    )
    assert fit.diagnostics["rmse"] == pytest.approx(
        numpy.sqrt((residuals**2).mean())
    )
    assert fit.diagnostics["degrees_of_freedom"] == 77
    se = numpy.sqrt(numpy.diag(fit.covariance))
    numpy.testing.assert_allclose(
        fit.p_values, 2 * stats.t.sf(numpy.abs(fit.coefficients / se), 77)
    )


def test_a_logistic_fit_reports_normal_intervals_and_the_bfgs_reference():
    design, y = bernoulli()
    policy = config.Config(estimator="logistic", covariance="iid")
    fit = linear.fit_logistic_estimator(
        design, y, policy, None, ("intercept", "x")
    )
    assert fit.likelihood == "bernoulli"
    assert fit.diagnostics["likelihood"] == "bernoulli"
    assert fit.pivot == pytest.approx(stats.norm.ppf(0.975))
    assert fit.reference["method"] == "scipy.optimize.minimize:BFGS"
    assert fit.reference["converged"] is True
    assert fit.reference["passed"] is True
    se = numpy.sqrt(numpy.diag(fit.covariance))
    numpy.testing.assert_allclose(
        fit.p_values, 2 * stats.norm.sf(numpy.abs(fit.coefficients / se))
    )


def test_a_corroboration_fails_for_coefficients_off_the_optimum():
    design, y = bernoulli()
    policy = config.Config(estimator="logistic", covariance="iid")
    found = linear.corroborate(design, y, numpy.array([2.0, -2.0]), policy)
    assert found["passed"] is False
    assert found["max_absolute_difference"] > 0.5
