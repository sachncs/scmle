"""The conjugate Bayesian linear model and its causal variant."""

import numpy
import pytest
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import bayesian
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import declaration

ATE = ("consistency", "conditional_exchangeability", "positivity", "linearity")


def prior(size=3, precision=0.5, shape=2.0, scale=1.0):
    return declaration.Prior(
        (0.0,) * size,
        tuple(
            tuple(precision if i == j else 0.0 for j in range(size))
            for i in range(size)
        ),
        shape,
        scale,
    )


def linear_policy(**changes):
    return config.Config(
        estimator="bayesian_linear", covariance="iid", prior=prior(), **changes
    )


def ate_policy():
    return config.Config(
        estimator="bayesian_ate",
        covariance="iid",
        prior=prior(),
        treatment=0,
        causal_assumptions=ATE,
    )


def world(rows=60, seed=2):
    rng = numpy.random.default_rng(seed)
    t = (rng.random(rows) < 0.5).astype(float)
    x = rng.normal(size=rows)
    matrix = numpy.column_stack([t, x])
    design = numpy.column_stack([numpy.ones(rows), matrix])
    y = 1.0 + 2.0 * t - 0.5 * x + rng.normal(scale=0.4, size=rows)
    return matrix, design, y


def test_a_posterior_is_the_conjugate_update():
    ignored, design, y = world()
    mean, covariance, diagnostics = bayesian.update(design, y, prior())
    precision = 0.5 * numpy.eye(3) + design.T @ design
    expected = numpy.linalg.solve(precision, design.T @ y)
    numpy.testing.assert_allclose(mean, expected)
    shape = 2.0 + 30.0
    scale = 1.0 + 0.5 * (y @ y - expected @ precision @ expected)
    degrees = 2 * shape
    inverse = numpy.linalg.inv(precision)
    numpy.testing.assert_allclose(
        covariance, scale / shape * degrees / (degrees - 2) * inverse
    )
    assert diagnostics["posterior_shape"] == shape
    assert diagnostics["posterior_degrees_of_freedom"] == degrees
    numpy.testing.assert_allclose(
        diagnostics["posterior_scale"], scale / shape * inverse
    )
    assert diagnostics["prior_shape"] == 2.0
    assert diagnostics["prior_scale"] == 1.0
    assert diagnostics["estimator_family"] == "bayesian_conjugate"


def test_a_posterior_rejects_a_mismatched_prior_and_few_degrees():
    ignored, design, y = world()
    with pytest.raises(errors.InputError, match="prior dimension must match"):
        bayesian.update(design, y, prior(size=2))
    with pytest.raises(
        errors.InputError, match="more than two degrees of freedom"
    ):
        bayesian.update(design[:1], y[:1], prior(shape=0.25))
    bayesian.conform(prior(size=3), 3)


def test_credible_intervals_use_the_student_scale():
    ignored, design, y = world()
    mean, covariance, diagnostics = bayesian.update(design, y, prior())
    fit = bayesian.credible(
        ("a", "b", "c"),
        mean,
        covariance,
        diagnostics,
        {},
        linear_policy(confidence=0.9),
    )
    degrees = diagnostics["posterior_degrees_of_freedom"]
    scale = numpy.asarray(diagnostics["posterior_scale"])
    assert fit.pivot == pytest.approx(stats.t.ppf(0.95, degrees))
    numpy.testing.assert_allclose(
        fit.standard_errors, numpy.sqrt(numpy.diag(covariance))
    )
    numpy.testing.assert_allclose(
        fit.interval_errors, numpy.sqrt(numpy.diag(scale))
    )
    numpy.testing.assert_allclose(
        fit.p_values,
        2
        * stats.t.sf(numpy.abs(mean) / numpy.sqrt(numpy.diag(scale)), degrees),
    )
    assert fit.diagnostics["uncertainty"] == "posterior_credible_interval"
    assert fit.diagnostics["credible_interval"]["level"] == 0.9
    assert fit.diagnostics["credible_interval"]["distribution"] == "student_t"


def test_a_dichotomous_check_counts_the_arms():
    matrix, ignored, ignored = world()
    position, treated, control = bayesian.dichotomous(matrix, ate_policy())
    assert position == 0
    assert treated == int(matrix[:, 0].sum())
    assert treated + control == 60
    with pytest.raises(
        errors.InputError, match="treatment must contain only 0 and 1"
    ):
        bayesian.dichotomous(matrix * 2, ate_policy())
    one = matrix.copy()
    one[:, 0] = 0.0
    one[0, 0] = 1.0
    with pytest.raises(
        errors.InputError, match="at least two rows per treatment arm"
    ):
        bayesian.dichotomous(one, ate_policy())


def test_a_bayesian_linear_fit_passes_its_reference():
    matrix, design, y = world()
    estimand = declaration.Estimand(
        "n", "y", "p", "gaussian", ("t", "x"), ("a",)
    )
    fit = bayesian.bayes(
        matrix, design, y, linear_policy(), estimand, ("intercept", "t", "x")
    )
    assert fit.reference["method"] == "scipy.linalg.solve:positive_definite"
    assert fit.reference["passed"] is True
    assert "treatment_name" not in fit.diagnostics


def test_a_bayesian_ate_fit_names_the_treatment():
    matrix, design, y = world()
    estimand = declaration.Estimand(
        "n", "y", "p", "gaussian", ("t", "x"), ("a",)
    )
    fit = bayesian.bayes(
        matrix, design, y, ate_policy(), estimand, ("intercept", "t", "x")
    )
    assert fit.diagnostics["treatment_name"] == "t"
    assert (
        fit.diagnostics["method"] == "normal_inverse_gamma_outcome_regression"
    )
    assert fit.diagnostics["estimator_family"] == "causal_bayesian_gaussian"
    assert len(fit.diagnostics["limitations"]) == 4
    assert (
        fit.reference["method"]
        == "scipy.linalg.solve:positive_definite_causal_gaussian"
    )
    assert fit.reference["reference_treatment_coefficient"] == pytest.approx(
        fit.coefficients[1]
    )
    assert fit.reference["treatment_position"] == 0
    assert fit.reference["passed"] is True
    assert 1.6 < fit.coefficients[1] < 2.4


def test_the_references_fail_for_off_coefficients():
    ignored, design, y = world()
    mean, ignored, ignored = bayesian.update(design, y, prior())
    off = bayesian.reference_bayesian_linear(
        design, y, prior(), mean + 1e-3, linear_policy()
    )
    assert off["passed"] is False
    assert off["max_absolute_difference"] == pytest.approx(1e-3)
    ate = bayesian.reference_bayesian_ate(
        design, y, prior(), mean + 1e-3, ate_policy()
    )
    assert ate["passed"] is False
