"""IPW and AIPW treatment effects on a confounded binary treatment."""

import numpy
import pytest
from scipy import special
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import causal
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import crossfitting
from scmle.evaluation.parameter_estimation import weighting

CAUSAL = ("consistency", "conditional_exchangeability", "positivity")


def policy(estimator="ipw_ate", **changes):
    values = {
        "estimator": estimator,
        "covariance": "iid",
        "treatment": 0,
        "causal_assumptions": CAUSAL,
    }
    values.update(changes)
    return config.Config(**values)


def world(rows=400, seed=3):
    rng = numpy.random.default_rng(seed)
    age = rng.normal(size=rows)
    t = (rng.random(rows) < special.expit(0.2 + 0.8 * age)).astype(float)
    y = 1.0 + 2.0 * t + 0.5 * age + rng.normal(scale=0.5, size=rows)
    return numpy.column_stack([t, age]), y


def test_arms_count_the_binary_treatment():
    matrix = numpy.array([[1, 0], [1, 0], [0, 0], [0, 0], [0, 0.0]])
    position, treatment, treated, control = causal.arms(matrix, 0, "x")
    assert (position, treated, control) == (0, 2, 3)
    assert treatment.tolist() == [1, 1, 0, 0, 0]


@pytest.mark.parametrize(
    ("matrix", "position", "message"),
    [
        (numpy.zeros((4, 1)), None, "x requires treatment_position"),
        (
            numpy.array([[0.0], [2.0], [1.0], [1.0]]),
            0,
            "x treatment must contain only 0 and 1",
        ),
        (
            numpy.array([[0.0], [1.0], [1.0], [1.0]]),
            0,
            "x needs at least two rows per treatment arm",
        ),
    ],
)
def test_arms_reject_bad_treatments(matrix, position, message):
    with pytest.raises(errors.InputError, match=message):
        causal.arms(matrix, position, "x")


def test_a_confounder_design_drops_the_treatment_and_adds_an_intercept():
    matrix = numpy.array([[9.0, 1.0, 2.0], [8.0, 3.0, 4.0]])
    assert causal.confound(matrix, 0).tolist() == [
        [1.0, 1.0, 2.0],
        [1.0, 3.0, 4.0],
    ]
    assert causal.confound(matrix, 2).tolist() == [
        [1.0, 9.0, 1.0],
        [1.0, 8.0, 3.0],
    ]


def test_a_positivity_record_counts_clipped_rows():
    raw = numpy.array([0.001, 0.5, 0.999, 0.3])
    clipped = numpy.clip(raw, 0.01, 0.99)
    assert causal.positivity(raw, clipped, 0.01) == {
        "status": "diagnosed_not_proven",
        "raw_min": 0.001,
        "raw_max": 0.999,
        "clipped_min": 0.01,
        "clipped_max": 0.99,
        "clip_threshold": 0.01,
        "fraction_clipped": 0.5,
    }


def test_an_ipw_fit_is_the_hajek_estimate_at_the_fitted_propensity():
    matrix, y = world()
    ate, variance, diagnostics, design = causal.fit_ipw_ate(
        matrix, y, 0, policy(), None
    )
    assert design.tolist() == causal.confound(matrix, 0).tolist()
    propensity, ignored, ignored = crossfitting.propensities(
        design, matrix[:, 0], design, policy()
    )
    expected, expected_variance, ignored = weighting.hajek(
        matrix[:, 0], y, propensity
    )
    assert ate.tolist() == pytest.approx([expected])
    assert variance.shape == (1, 1)
    assert variance[0, 0] == pytest.approx(expected_variance)
    assert diagnostics["method"] == "hajek_ipw"
    assert diagnostics["estimator_family"] == "causal_ipw"
    assert (
        diagnostics["uncertainty_method"]
        == "influence_function_with_estimated_propensity"
    )
    assert diagnostics["treated_rows"] + diagnostics["control_rows"] == 400
    assert diagnostics["propensity_model"]["method"] == "logistic_mle"
    assert diagnostics["influence_function"]["variance"] == pytest.approx(
        expected_variance
    )
    assert 1.7 < ate[0] < 2.3


def test_a_clustered_ipw_fit_adds_cluster_diagnostics():
    matrix, y = world()
    clusters = numpy.repeat(numpy.arange(20), 20)
    ignored, ignored, diagnostics, ignored = causal.fit_ipw_ate(
        matrix, y, 0, policy(covariance="clustered", cluster="c"), clusters
    )
    assert (
        diagnostics["uncertainty_method"]
        == "clustered_centered_influence_function"
    )
    assert diagnostics["cluster_count"] == 20
    assert diagnostics["cluster_sizes"] == [20] * 20


def test_an_aipw_fit_averages_the_augmented_score():
    matrix, y = world()
    ate, variance, diagnostics, inputs = causal.dr(
        matrix, y, 0, policy("aipw_ate"), None
    )
    fitted = crossfitting.nuisance(
        inputs["nuisance_design"], inputs["treatment"], y, policy("aipw_ate")
    )
    score = weighting.aipw(
        fitted.treated,
        fitted.control,
        inputs["treatment"],
        y,
        fitted.propensity,
    )
    assert ate[0] == pytest.approx(score.mean())
    numpy.testing.assert_allclose(inputs["outcome_treated"], fitted.treated)
    assert diagnostics["method"] == "augmented_ipw"
    assert (
        diagnostics["uncertainty_method"]
        == "influence_function_with_fitted_nuisance"
    )
    assert diagnostics["cross_fitting"]["status"] == "disabled"
    assert inputs["fold_assignments"].tolist() == [0] * 400
    assert 1.8 < ate[0] < 2.2
    assert variance[0, 0] == diagnostics["influence_function"]["variance"]


def test_a_cross_fitted_aipw_fit_names_its_uncertainty():
    matrix, y = world()
    ignored, ignored, diagnostics, inputs = causal.dr(
        matrix, y, 0, policy("aipw_ate", folds=2), None
    )
    assert (
        diagnostics["uncertainty_method"]
        == "influence_function_with_cross_fitted_nuisance"
    )
    assert inputs["fold_assignments"].tolist()[:4] == [0, 1, 0, 1]


def test_a_propensity_estimator_reports_the_ate_with_normal_p_values():
    matrix, y = world()
    for estimator, method in (
        ("ipw_ate", "scipy.optimize.minimize:BFGS_propensity"),
        ("aipw_ate", "scipy.optimize.minimize:BFGS_plus_lstsq"),
    ):
        fit = causal.balance(matrix, y, policy(estimator), None)
        assert fit.names == ("ate",)
        assert fit.likelihood == "gaussian"
        assert fit.diagnostics["likelihood"] == "gaussian"
        assert fit.pivot == pytest.approx(stats.norm.ppf(0.975))
        se = numpy.sqrt(fit.covariance[0, 0])
        assert fit.p_values[0] == pytest.approx(
            2 * stats.norm.sf(abs(fit.coefficients[0]) / se)
        )
        assert fit.reference["method"] == method
        assert fit.reference["passed"] is True
