"""Hierarchical linear models: moments, GLS, REML and their references."""

import math
from types import SimpleNamespace

import numpy
import pytest
from scipy import linalg
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import audit
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import hierarchical
from scmle.evaluation.parameter_estimation import mixed
from scmle.evaluation.parameter_estimation import reml

INTERCEPT = (
    "random_intercept",
    "cluster_independence",
    "random_effect_exogeneity",
    "constant_residual_variance",
)


def policy(method="moment_gls", slope=None):
    assumptions = INTERCEPT if slope is None else (*INTERCEPT, "random_slope")
    return config.Config(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster="site",
        hierarchical_assumptions=assumptions,
        hierarchy=method,
        slope=slope,
    )


def grouped(clusters=12, size=15, seed=4):
    rng = numpy.random.default_rng(seed)
    labels = numpy.repeat(numpy.arange(clusters), size)
    x = rng.normal(size=clusters * size)
    a = rng.normal(scale=0.8, size=clusters)[labels]
    b = rng.normal(scale=0.4, size=clusters)[labels]
    y = 1.0 + 2.0 * x + a + b * x + rng.normal(scale=0.5, size=len(x))
    design = numpy.column_stack([numpy.ones(len(x)), x])
    return design, y, labels


# mixed


def test_a_projection_zeroes_negative_eigenvalues():
    found = mixed.project(numpy.array([[1.0, 2.0], [2.0, 1.0]]))
    numpy.testing.assert_allclose(found, [[1.5, 1.5], [1.5, 1.5]])
    numpy.testing.assert_allclose(
        mixed.project(numpy.diag([2.0, 1.0])), numpy.diag([2.0, 1.0])
    )
    numpy.testing.assert_allclose(
        mixed.project(numpy.array([[1.0, 0.0], [2.0, 1.0]])),
        [[1.0, 1.0], [1.0, 1.0]],
    )


def test_distinct_labels_keep_first_appearance():
    assert mixed.distinct(numpy.array([3, 1, 3, 2])) == (3, 1, 2)
    lists = numpy.empty(2, dtype=object)
    lists[0], lists[1] = [1], [2]
    with pytest.raises(
        errors.InputError, match="hierarchical cluster labels must be hashable"
    ):
        mixed.distinct(lists)


def test_columns_and_marginal_covariance_follow_the_slope_feature():
    group = numpy.array([[1.0, 5.0, 2.0], [1.0, 6.0, 3.0]])
    block = mixed.columns(group, 1)
    assert block.tolist() == [[1.0, 2.0], [1.0, 3.0]]
    cov = numpy.array([[1.0, 0.5], [0.5, 2.0]])
    numpy.testing.assert_allclose(
        mixed.marginal(block, 0.25, cov),
        0.25 * numpy.eye(2) + block @ cov @ block.T,
    )
    numpy.testing.assert_allclose(
        mixed.precisions(block, 0.25, cov),
        numpy.linalg.inv(mixed.marginal(block, 0.25, cov)),
    )


def test_a_precision_of_a_singular_block_is_rejected():
    block = numpy.array([[1.0, 1.0], [1.0, 1.0]])
    with pytest.raises(errors.InputError, match="not positive definite"):
        mixed.precisions(block, 0.0, numpy.zeros((2, 2)))


def test_a_random_intercept_gls_matches_the_dense_system():
    design, y, labels = grouped()
    names = mixed.distinct(labels)
    information, rhs = mixed.gls(design, y, labels, names, 0.25, 0.64)
    dense_info = numpy.zeros((2, 2))
    dense_rhs = numpy.zeros(2)
    for label in names:
        g = labels == label
        precision = numpy.linalg.inv(
            0.25 * numpy.eye(g.sum()) + 0.64 * numpy.ones((g.sum(), g.sum()))
        )
        dense_info += design[g].T @ precision @ design[g]
        dense_rhs += design[g].T @ precision @ y[g]
    numpy.testing.assert_allclose(information, dense_info)
    numpy.testing.assert_allclose(rhs, dense_rhs)
    plain, ignored = mixed.gls(design, y, labels, names, 0.5, 0.0)
    numpy.testing.assert_allclose(plain, design.T @ design / 0.5)


def test_a_random_slope_gls_matches_the_block_precisions():
    design, y, labels = grouped()
    names = mixed.distinct(labels)
    cov = numpy.array([[0.6, 0.1], [0.1, 0.2]])
    information, rhs = mixed.assemble(design, y, labels, names, 0, 0.25, cov)
    expected = numpy.zeros((2, 2))
    for label in names:
        g = labels == label
        p = mixed.precisions(mixed.columns(design[g], 0), 0.25, cov)
        expected += design[g].T @ p @ design[g]
    numpy.testing.assert_allclose(information, expected)
    assert rhs.shape == (2,)


def test_a_solve_returns_coefficients_and_inverse():
    coefficients, covariance = mixed.solve(
        numpy.diag([2.0, 4.0]), numpy.array([2.0, 2.0]), "m"
    )
    assert coefficients.tolist() == [1.0, 0.5]
    assert covariance.tolist() == [[0.5, 0.0], [0.0, 0.25]]
    with pytest.raises(errors.InputError, match=r"^custom$"):
        mixed.solve(numpy.zeros((2, 2)), numpy.zeros(2), "custom")


def test_a_settle_scores_only_positive_definite_points():
    found = mixed.settle(numpy.diag([2.0, 3.0]), numpy.array([2.0, 3.0]))
    assert found[0] == pytest.approx(math.log(6.0))
    assert found[1].tolist() == [1.0, 1.0]
    assert mixed.settle(numpy.diag([-1.0, 1.0]), numpy.zeros(2)) is None
    assert mixed.settle(numpy.zeros((2, 2)), numpy.zeros(2)) is None


def test_a_profile_is_half_the_reml_parts():
    design = numpy.ones((5, 2))
    expected = 0.5 * (1.0 + 2.0 + 3.0 + 3 * math.log(2 * math.pi))
    assert mixed.profile(
        design, numpy.zeros(5), 1.0, 2.0, 3.0
    ) == pytest.approx(expected)


# hierarchical


def test_a_start_checks_the_clusters_and_design():
    design, y, labels = grouped()
    names, residuals = hierarchical.start(design, y, labels)
    assert names == tuple(range(12))
    beta = numpy.linalg.lstsq(design, y, rcond=None)[0]
    numpy.testing.assert_allclose(residuals, y - design @ beta)
    with pytest.raises(errors.InputError, match="at least two clusters"):
        hierarchical.start(design, y, numpy.zeros(len(y)))
    with pytest.raises(errors.InputError, match="rank deficient"):
        hierarchical.start(
            numpy.column_stack([design[:, 0], design[:, 0]]), y, labels
        )


def test_moments_estimate_within_and_between_variance():
    labels = numpy.array([0, 0, 1, 1, 2, 2])
    residuals = numpy.array([1.0, 3.0, -1.0, -3.0, 0.0, 0.0])
    within, between = hierarchical.moments(labels, (0, 1, 2), residuals)
    assert within == pytest.approx((2 + 2 + 0) / 3)
    means = [2.0, -2.0, 0.0]
    assert between == pytest.approx(
        max(0.0, numpy.var(means, ddof=1) - within * 0.5)
    )
    floored = hierarchical.moments(
        labels, (0, 1, 2), numpy.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    )
    assert floored[1] == 0.0
    with pytest.raises(
        errors.InputError, match="positive within-cluster variance"
    ):
        hierarchical.moments(
            labels, (0, 1, 2), numpy.array([1.0, 1.0, 2.0, 2.0, 3.0, 3.0])
        )


def test_slopes_need_within_cluster_variation():
    design = numpy.column_stack([numpy.ones(4), [1.0, 1.0, 2.0, 3.0]])
    with pytest.raises(
        errors.InputError,
        match="within-cluster variation for the declared random slope",
    ):
        hierarchical.slopes(
            design, numpy.array([0, 0, 1, 1]), (0, 1), numpy.zeros(4), 0
        )


def test_a_moment_fit_matches_the_explicit_crosscheck():
    design, y, labels = grouped()
    fit = hierarchical.nest(design, y, policy(), labels, ("intercept", "x"))
    assert (
        fit.reference["method"]
        == "scipy.linalg.solve:explicit_random_intercept_covariance"
    )
    assert fit.reference["passed"] is True
    assert fit.diagnostics["degrees_of_freedom"] == 11
    assert fit.pivot == pytest.approx(stats.t.ppf(0.975, 11))
    assert (
        fit.diagnostics["uncertainty_method"]
        == "conditional_on_moment_variance_components"
    )
    assert 1.6 < fit.coefficients[1] < 2.4
    se = numpy.sqrt(numpy.diag(fit.covariance))
    numpy.testing.assert_allclose(
        fit.p_values, 2 * stats.t.sf(numpy.abs(fit.coefficients / se), 11)
    )


def test_a_hierarchical_fit_needs_cluster_labels():
    design, y, ignored = grouped()
    with pytest.raises(errors.InputError, match="requires cluster labels"):
        hierarchical.nest(design, y, policy(), None, ("i", "x"))


def test_a_reml_fit_passes_its_dense_reference():
    design, y, labels = grouped()
    fit = hierarchical.nest(
        design, y, policy("reml"), labels, ("intercept", "x")
    )
    assert fit.reference["passed"] is True
    assert fit.reference["method"] == (
        "scipy.linalg.solve:dense_random_intercept_covariance"
    )
    diagnostics = fit.diagnostics
    assert diagnostics["method"] == "bounded_reml_gls"
    assert (
        diagnostics["variance_component_method"] == "profile_reml_with_l_bfgs_b"
    )
    within = diagnostics["residual_variance"]
    between = diagnostics["random_intercept_variance"]
    assert diagnostics["intraclass_correlation"] == pytest.approx(
        between / (between + within)
    )
    names = mixed.distinct(labels)
    point = numpy.log([within, between])
    assert diagnostics["reml_objective"] == pytest.approx(
        reml.deviance(point, design, y, labels, names)
    )
    for step in (numpy.array([0.05, 0.0]), numpy.array([0.0, 0.05])):
        assert (
            reml.deviance(point + step, design, y, labels, names)
            > (diagnostics["reml_objective"])
        )
    information, rhs = mixed.gls(design, y, labels, names, within, between)
    numpy.testing.assert_allclose(
        fit.coefficients, numpy.linalg.solve(information, rhs), rtol=1e-8
    )


def test_random_slope_fits_pass_their_references():
    design, y, labels = grouped()
    for method in ("moment_gls", "reml"):
        fit = hierarchical.nest(
            design, y, policy(method, slope=0), labels, ("intercept", "x")
        )
        assert fit.reference["passed"] is True, (method, fit.reference)
        assert 1.5 < fit.coefficients[1] < 2.5


def test_a_random_slope_fit_needs_a_declared_position():
    design, y, labels = grouped()
    with pytest.raises(
        errors.InputError, match="random slope position must be declared"
    ):
        hierarchical.fit_hierarchical_random_slope(design, y, labels, policy())


# reml


def test_a_random_slope_covariance_uses_the_scaled_tanh_correlation():
    found = reml.decode(numpy.array([9.0, 4.0, 1.0]), 0.5)
    rho = 0.999999 * math.tanh(0.5)
    numpy.testing.assert_allclose(found, [[4.0, rho * 2.0], [rho * 2.0, 1.0]])


def test_the_reml_objectives_check_their_dimensions():
    design, y, labels = grouped()
    names = mixed.distinct(labels)
    with pytest.raises(
        errors.InputError, match="needs residual and random-intercept variances"
    ):
        reml.deviance(numpy.zeros(3), design, y, labels, names)
    with pytest.raises(
        errors.InputError, match="four bounded covariance parameters"
    ):
        reml.score(numpy.zeros(3), design, y, labels, names, 0)
    with numpy.errstate(over="ignore"):
        assert (
            reml.deviance(numpy.array([1000.0, 0.0]), design, y, labels, names)
            == math.inf
        )
        assert (
            reml.score(
                numpy.array([1000.0, 0.0, 0.0, 0.0]),
                design,
                y,
                labels,
                names,
                0,
            )
            == math.inf
        )


def test_a_deviance_matches_the_dense_reml_objective():
    design, y, labels = grouped()
    names = mixed.distinct(labels)
    point = numpy.log([0.3, 0.5])
    value = reml.deviance(point, design, y, labels, names)
    blocks = [
        0.3 * numpy.eye((labels == g).sum())
        + 0.5 * numpy.ones(((labels == g).sum(),) * 2)
        for g in names
    ]
    covariance = linalg.block_diag(*blocks)
    precision = numpy.linalg.inv(covariance)
    information = design.T @ precision @ design
    beta = numpy.linalg.solve(information, design.T @ precision @ y)
    ignored, objective_gap = audit.dense(design, y, blocks, beta, value)
    assert objective_gap < 1e-8


def result(**changes):
    values = {
        "success": True,
        "status": 0,
        "nit": 3,
        "nfev": 5,
        "x": numpy.array([0.0, 0.0]),
        "fun": 1.0,
        "jac": numpy.array([0.0, 0.0]),
        "message": "ok",
    }
    values.update(changes)
    return SimpleNamespace(**values)


BOUNDS = ((-1.0, 1.0), (-1.0, 1.0))


def test_an_optimizer_result_is_accepted_when_valid():
    assert reml.vet(result(), BOUNDS) is None


@pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
        (
            {"success": False, "message": "boom"},
            errors.InputError,
            "did not converge: boom",
        ),
        ({"status": True}, ValueError, "invalid status"),
        ({"status": None}, ValueError, "invalid status"),
        ({"nit": -1}, ValueError, "invalid iteration diagnostics"),
        ({"nfev": True}, ValueError, "invalid iteration diagnostics"),
        (
            {"x": numpy.array([0.0])},
            errors.InputError,
            "invalid variance components",
        ),
        (
            {"x": numpy.array([0.0, numpy.nan])},
            errors.InputError,
            "invalid variance components",
        ),
        (
            {"x": numpy.array([0.0, 1.5])},
            ValueError,
            "out-of-bounds components",
        ),
        (
            {"x": numpy.array([-1.5, 0.0])},
            ValueError,
            "out-of-bounds components",
        ),
        ({"fun": math.inf}, errors.InputError, "invalid objective"),
        (
            {"jac": numpy.array([0.0])},
            errors.InputError,
            "invalid gradient diagnostics",
        ),
        (
            {"jac": numpy.array([0.0, math.nan])},
            errors.InputError,
            "invalid gradient diagnostics",
        ),
    ],
)
def test_an_invalid_optimizer_result_is_rejected(changes, error, message):
    with pytest.raises(error, match=message):
        reml.vet(result(**changes), BOUNDS)


# audit


def test_a_crosscheck_fails_for_off_coefficients():
    design, y, labels = grouped()
    names = mixed.distinct(labels)
    information, rhs = mixed.gls(design, y, labels, names, 0.25, 0.64)
    beta = numpy.linalg.solve(information, rhs)
    good = audit.crosscheck(design, y, labels, 0.25, 0.64, beta, policy())
    assert good["passed"] is True
    assert good["max_absolute_difference"] < 1e-9
    bad = audit.crosscheck(design, y, labels, 0.25, 0.64, beta + 1e-3, policy())
    assert bad["passed"] is False
    assert bad["max_absolute_difference"] == pytest.approx(1e-3)


def test_a_dense_check_rejects_negative_definite_information():
    design = numpy.ones((2, 1))
    with pytest.raises((errors.InputError, numpy.linalg.LinAlgError)):
        audit.dense(
            design, numpy.zeros(2), [-numpy.eye(2)], numpy.zeros(1), 0.0
        )
