"""Continuous-treatment generalized-propensity-weighted dose response."""

import numpy
import pytest
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import continuous
from scmle.evaluation.parameter_estimation import declaration
from scmle.evaluation.parameter_estimation import linear

ASSUMPTIONS = (
    "consistency",
    "conditional_exchangeability",
    "continuous_treatment_positivity",
    "linear_dose_response",
    "normal_conditional_treatment_density",
)


def policy(cap=8.0, covariance="iid"):
    return config.Config(
        estimator="continuous_ipw_ate",
        covariance=covariance,
        cluster="site" if covariance == "clustered" else None,
        treatment=0,
        cap=cap,
        continuous_treatment_assumptions=ASSUMPTIONS,
    )


def world(rows=400, seed=4):
    rng = numpy.random.default_rng(seed)
    age = rng.normal(size=rows)
    dose = 0.6 * age + rng.normal(scale=1.0, size=rows)
    y = 0.8 + 1.75 * dose + 0.45 * age + rng.normal(scale=0.5, size=rows)
    return numpy.column_stack([dose, age]), y


def test_the_capability_texts_are_fixed():
    # Enforces website/src/pages/docs/modeling.astro:431.
    assert continuous.EVIDENCE == (
        "generalized_propensity_density_diagnostics",
        "weight_effective_sample_size",
        "independent_reference_calculation",
    )
    assert len(continuous.CAPABILITY_LIMITS) == 4
    assert continuous.LIMITS[0] == "positivity is diagnosed but not proven"
    assert len(continuous.LIMITS) == 5


def test_a_treatment_density_is_the_least_squares_fit():
    matrix, ignored = world()
    mean, residuals, scale = continuous.fit_treatment_density(
        matrix[:, 0], matrix[:, 1:]
    )
    design = numpy.column_stack([numpy.ones(400), matrix[:, 1]])
    beta = numpy.linalg.lstsq(design, matrix[:, 0], rcond=None)[0]
    numpy.testing.assert_allclose(mean, design @ beta)
    numpy.testing.assert_allclose(residuals, matrix[:, 0] - design @ beta)
    assert scale == pytest.approx(numpy.sqrt((residuals**2).sum() / 398))


def test_a_treatment_density_rejects_degenerate_designs():
    with pytest.raises(
        errors.InputError, match="density design matrix is rank deficient"
    ):
        continuous.fit_treatment_density(numpy.arange(4.0), numpy.ones((4, 1)))
    with pytest.raises(
        errors.InputError, match="needs residual degrees of freedom"
    ):
        continuous.fit_treatment_density(
            numpy.array([1.0, 2.0]), numpy.array([[0.0], [1.0]])
        )
    exact = numpy.arange(6.0)
    with pytest.raises(errors.InputError, match="non-positive residual scale"):
        continuous.fit_treatment_density(2 * exact, exact[:, None])


def test_weights_are_marginal_over_conditional_densities():
    matrix, ignored = world()
    weights, overlap, degrees, scale = continuous.weigh(matrix, 0)
    mean, ignored, ignored = continuous.fit_treatment_density(
        matrix[:, 0], matrix[:, 1:]
    )
    expected = stats.norm.pdf(
        matrix[:, 0], matrix[:, 0].mean(), matrix[:, 0].std(ddof=1)
    ) / stats.norm.pdf(matrix[:, 0], mean, scale)
    numpy.testing.assert_allclose(weights, expected, rtol=1e-9)
    assert degrees == 398
    assert overlap["status"] == "diagnosed_not_proven"
    assert overlap["unique_treatment_values"] == 400
    assert overlap["marginal_treatment_sd"] == pytest.approx(
        matrix[:, 0].std(ddof=1)
    )
    assert len(overlap["conditional_standardized_residual_quantiles"]) == 4


def test_weights_need_four_levels_and_variance():
    matrix = numpy.column_stack([[1.0, 2.0, 3.0, 1.0, 2.0], numpy.arange(5.0)])
    with pytest.raises(
        errors.InputError, match="at least four distinct treatment values"
    ):
        continuous.weigh(matrix, 0)


def test_a_cap_limits_weights_and_records_them():
    # Enforces website/src/pages/docs/modeling.astro:431.
    raw = numpy.array([0.5, 1.0, 10.0, 2.0])
    weights, record = continuous.cap(raw, policy(cap=4.0))
    assert weights.tolist() == [0.5, 1.0, 4.0, 2.0]
    assert record["raw_max"] == 10.0
    assert record["capped_max"] == 4.0
    assert record["cap"] == 4.0
    assert record["fraction_capped"] == 0.25
    assert record["effective_sample_size"] == pytest.approx(7.5**2 / 21.25)
    assert record["effective_sample_fraction"] == pytest.approx(
        7.5**2 / 21.25 / 4
    )
    assert (
        record["normalization"]
        == "stabilized_marginal_over_conditional_density"
    )


def test_a_cap_rejects_a_single_dominant_weight():
    with pytest.raises(
        errors.InputError, match="fewer than two effective observations"
    ):
        continuous.cap(numpy.array([1e-9, 1e-9, 5.0]), policy(cap=8.0))


def test_a_dose_regression_is_weighted_least_squares():
    treatment = numpy.array([0.0, 1.0, 2.0, 3.0, 4.0])
    values = numpy.array([1.0, 3.2, 4.9, 7.1, 9.0])
    weights = numpy.array([1.0, 2.0, 1.0, 0.5, 1.0])
    coefficients, covariance = continuous.dose(
        treatment, values, weights, policy(), None
    )
    design = numpy.column_stack([numpy.ones(5), treatment])
    expected = numpy.linalg.solve(
        design.T @ (weights[:, None] * design), design.T @ (weights * values)
    )
    numpy.testing.assert_allclose(coefficients, expected)
    wd = design * numpy.sqrt(weights)[:, None]
    wr = (values - design @ expected) * numpy.sqrt(weights)
    numpy.testing.assert_allclose(
        covariance, linear.covariance_matrix(wd, wr, "iid", None, None)
    )
    with pytest.raises(
        errors.InputError,
        match="weighted dose-response design is rank deficient",
    ):
        continuous.dose(numpy.ones(5), values, weights, policy(), None)


def test_a_continuous_fit_recovers_the_slope_and_agrees_with_its_reference():
    matrix, y = world()
    estimand = declaration.Estimand(
        "n", "y", "p", "gaussian", ("dose", "age"), ("x",)
    )
    fit = continuous.reweight(matrix, y, policy(), estimand, None)
    assert fit.names == ("intercept", "dose_dose_response_slope")
    assert 1.6 < fit.coefficients[1] < 1.95
    assert fit.reference["passed"] is True
    assert fit.reference["clustered_reference"] is False
    assert fit.reference["reference_dose_response_slope"] == pytest.approx(
        fit.coefficients[1]
    )
    assert fit.diagnostics["treatment_name"] == "dose"
    assert (
        fit.diagnostics["uncertainty_method"] == "weighted_gaussian_regression"
    )
    assert fit.diagnostics["capability"]["assumptions"] == list(ASSUMPTIONS)
    assert fit.pivot == pytest.approx(stats.norm.ppf(0.975))


def test_a_clustered_continuous_fit_reports_cluster_uncertainty():
    matrix, y = world()
    clusters = numpy.repeat(numpy.arange(20), 20)
    coefficients, ignored, diagnostics = continuous.dosage(
        matrix, y, 0, policy(covariance="clustered"), clusters
    )
    assert (
        diagnostics["uncertainty_method"]
        == "clustered_weighted_gaussian_regression"
    )
    assert diagnostics["cluster_uncertainty"]["cluster_count"] == 20
    found = continuous.recompute(matrix, y, 0, policy(), coefficients, clusters)
    assert found["clustered_reference"] is True
    assert found["passed"] is True
    off = continuous.recompute(matrix, y, 0, policy(), coefficients + 1.0, None)
    assert off["passed"] is False
    assert off["max_absolute_difference"] == pytest.approx(1.0, abs=1e-6)
