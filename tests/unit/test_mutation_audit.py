"""Exact records and limits of the dense hierarchical references."""

import numpy

from scmle.evaluation.parameter_estimation import audit
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import mixed

ASSUMPTIONS = (
    "random_intercept",
    "cluster_independence",
    "random_effect_exogeneity",
    "constant_residual_variance",
)


def policy(slope=None):
    extra = () if slope is None else ("random_slope",)
    return config.Config(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster="site",
        hierarchical_assumptions=(*ASSUMPTIONS, *extra),
        hierarchy="reml",
        slope=slope,
    )


def grouped(clusters=6, size=8, seed=4):
    rng = numpy.random.default_rng(seed)
    labels = numpy.repeat(numpy.arange(clusters), size)
    x = rng.normal(size=clusters * size)
    a = rng.normal(scale=0.8, size=clusters)[labels]
    y = 1.0 + 2.0 * x + a + rng.normal(scale=0.5, size=len(x))
    design = numpy.column_stack([numpy.ones(len(x)), x])
    return design, y, labels


def test_the_random_slope_reference_names_its_method():
    design, y, labels = grouped()
    covariance = numpy.array([[0.6, 0.1], [0.1, 0.2]])

    found = audit.reference_hierarchical_random_slope(
        design, y, labels, 0, 0.25, covariance, numpy.zeros(2), policy(0)
    )

    assert found["method"] == (
        "scipy.linalg.solve:dense_random_intercept_slope_covariance"
    )
    assert found["random_slope_position"] == 0
    assert found["random_effect_covariance"] == covariance.tolist()


def test_the_reml_objective_must_agree_within_ten_tolerances():
    design, y, labels = grouped()
    rules = policy()
    names = mixed.distinct(labels)
    information, rhs = mixed.gls(design, y, labels, names, 0.25, 0.64)
    beta = numpy.linalg.solve(information, rhs)
    blocks = [
        0.25 * numpy.eye(int((labels == k).sum()))
        + 0.64 * numpy.ones(((labels == k).sum(),) * 2)
        for k in names
    ]
    gap = audit.dense(design, y, blocks, beta, 0.0)[1]
    truth = gap if audit.dense(design, y, blocks, beta, gap)[1] < gap else -gap

    def check(offset):
        return audit.reference_hierarchical_reml(
            design, y, labels, 0.25, 0.64, beta, truth + offset, rules
        )

    tolerance = rules.reference
    assert check(9.0 * tolerance)["passed"] is True
    assert check(10.5 * tolerance)["passed"] is False
