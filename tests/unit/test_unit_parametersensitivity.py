"""Prior, shift and bootstrap sensitivity for parameter estimation."""

import hashlib

import numpy
import pytest
from scipy import special

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import declaration
from scmle.evaluation.parameter_estimation import linear
from scmle.evaluation.parameter_estimation import resampling
from scmle.evaluation.parameter_estimation import sensitivity

CAUSAL = ("consistency", "conditional_exchangeability", "positivity")


def prior(size=3):
    return declaration.Prior(
        (0.0,) * size,
        tuple(tuple(float(i == j) for j in range(size)) for i in range(size)),
    )


def estimand(features=("a", "b")):
    return declaration.Estimand("n", "y", "p", "gaussian", features, ("x",))


def world(rows=80, seed=1):
    rng = numpy.random.default_rng(seed)
    x = rng.normal(size=(rows, 2))
    y = 1.0 + x @ numpy.array([2.0, -1.0]) + rng.normal(scale=0.3, size=rows)
    return x, numpy.column_stack([numpy.ones(rows), x]), y


def test_a_reprior_refits_under_weaker_and_stronger_precision():
    ignored, design, y = world()
    base = numpy.linalg.solve(numpy.eye(3) + design.T @ design, design.T @ y)
    policy = config.Config(
        estimator="bayesian_linear", covariance="iid", prior=prior()
    )
    found = sensitivity.reprior(prior(), design, y, base, policy)
    weak = numpy.linalg.solve(
        0.01 * numpy.eye(3) + design.T @ design, design.T @ y
    )
    assert found["status"] == "evaluated"
    assert found["method"] == "precision_scale_sensitivity"
    assert [a["name"] for a in found["alternatives"]] == [
        "weakened_precision",
        "strengthened_precision",
    ]
    assert found["alternatives"][0][
        "max_absolute_coefficient_change"
    ] == pytest.approx(numpy.abs(weak - base).max())
    assert "treatment_coefficient_change" not in found["alternatives"][0]


def test_a_reprior_of_a_causal_model_reports_the_treatment_change():
    ignored, design, y = world()
    base = numpy.zeros(3)
    policy = config.Config(
        estimator="bayesian_ate",
        covariance="iid",
        prior=prior(),
        treatment=1,
        causal_assumptions=(*CAUSAL, "linearity"),
    )
    found = sensitivity.reprior(prior(), design, y, base, policy)
    strong = numpy.linalg.solve(
        100 * numpy.eye(3) + design.T @ design, design.T @ y
    )
    assert found["alternatives"][1][
        "treatment_coefficient_change"
    ] == pytest.approx(strong[2])


def test_a_transport_reports_standardised_mean_differences():
    matrix = numpy.array([[0.0, 1.0], [2.0, 1.0], [4.0, 1.0]])
    reference = numpy.array([[2.0, 1.0], [4.0, 1.0], [6.0, 1.0]])
    found = sensitivity.transport(matrix, reference)
    pooled = numpy.sqrt((matrix[:, 0].var() + reference[:, 0].var()) / 2)
    assert found["status"] == "evaluated"
    assert found["standardized_mean_difference"] == pytest.approx(
        [2.0 / pooled, 0.0]
    )
    assert found["max_standardized_mean_difference"] == pytest.approx(
        2.0 / pooled
    )


def test_a_transport_rejects_bad_references():
    matrix = numpy.zeros((3, 2))
    with pytest.raises(
        errors.InputError, match=r"shift_reference must have shape \(rows, 2\)"
    ):
        sensitivity.transport(matrix, numpy.zeros((3, 1)))
    with pytest.raises(errors.InputError, match="only finite values"):
        sensitivity.transport(
            matrix, numpy.array([[numpy.nan, 0.0], [0, 0], [0, 0]])
        )


def test_a_sensitivity_report_without_extras_records_what_was_not_run():
    x, design, y = world()
    policy = config.Config(estimator="linear")
    found = sensitivity.perturb(
        x, y, policy, design, numpy.zeros(3), None, estimand(), None
    )
    assert found == {
        "prior": {
            "status": "not_applicable",
            "reason": "current estimators are frequentist and use no prior",
        },
        "missingness": {"status": "evaluated", "policy": "reject", "rows": 80},
        "sampling": {"status": "not_requested", "bootstrap_replicates": 0},
        "distribution_shift": {"status": "not_available"},
    }


def test_a_row_bootstrap_matches_its_seeded_draws():
    x, design, y = world()
    policy = config.Config(estimator="linear", replicates=5, seed=7)
    found = sensitivity.perturb(
        x, y, policy, design, numpy.zeros(3), x, estimand(), None
    )
    rng = numpy.random.default_rng(7)
    estimates = []
    for ignored in range(5):
        rows = rng.integers(0, 80, size=80)
        estimates.append(linear.regress(design[rows], y[rows], 1, 1e-8)[0])
    assert found["sampling"]["status"] == "available"
    assert found["sampling"]["bootstrap_intervals"] == resampling.intervals(
        estimates, 0.95
    )
    assert (
        found["distribution_shift"]["max_standardized_mean_difference"] == 0.0
    )


def test_a_logistic_bootstrap_uses_the_logit():
    rng = numpy.random.default_rng(3)
    x = rng.normal(size=(150, 2))
    y = (rng.random(150) < special.expit(x[:, 0])).astype(float)
    design = numpy.column_stack([numpy.ones(150), x])
    policy = config.Config(estimator="logistic", replicates=3)
    found = sensitivity.perturb(
        x, y, policy, design, numpy.zeros(3), None, estimand(), None
    )
    assert len(found["sampling"]["bootstrap_intervals"]) == 2
    assert len(found["sampling"]["bootstrap_intervals"][0]) == 3


def test_a_hierarchical_bootstrap_needs_clusters():
    x, design, y = world()
    policy = config.Config(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster="c",
        replicates=2,
        hierarchical_assumptions=(
            "random_intercept",
            "cluster_independence",
            "random_effect_exogeneity",
            "constant_residual_variance",
        ),
    )
    with pytest.raises(
        errors.InputError, match="hierarchical_linear requires cluster labels"
    ):
        sensitivity.sample(
            {"sampling": {}},
            (x, y, design, numpy.zeros(3)),
            estimand(),
            policy,
            None,
        )


# resampling


def test_the_slope_limits_are_fixed():
    assert resampling.SLOPE_LIMITS[1] == "only one random slope is supported"
    assert len(resampling.SLOPE_LIMITS) == 5


def test_a_sample_digest_hashes_little_endian_int64():
    indices = numpy.array([3, 1, 2])
    assert (
        resampling.digest(indices)
        == hashlib.sha256(
            numpy.array([3, 1, 2], dtype="<i8").tobytes()
        ).hexdigest()
    )
    assert resampling.digest([1, 2, 3]) != resampling.digest(indices)


def test_intervals_are_per_parameter_percentiles():
    estimates = [numpy.array([float(i), -float(i)]) for i in range(101)]
    low, high = resampling.intervals(estimates, 0.9)
    assert low == pytest.approx([5.0, -95.0])
    assert high == pytest.approx([95.0, -5.0])


def test_checks_count_successes_and_failures():
    assert resampling.checks([1, 2], [0]) == {
        "successful": 2,
        "failed": 1,
        "requirement": "every accepted replicate passes its independent point-estimate reference check",
    }


def test_a_causal_row_draw_resamples_rows():
    rng = numpy.random.default_rng(0)
    matrix = numpy.arange(10.0).reshape(5, 2)
    values = numpy.arange(5.0)
    sampled, outcome, clusters, record = resampling.draw(
        matrix, values, None, rng
    )
    indices = numpy.random.default_rng(0).integers(0, 5, size=5)
    assert outcome.tolist() == values[indices].tolist()
    assert sampled.tolist() == matrix[indices].tolist()
    assert clusters is None
    assert record == {
        "sampled_rows": 5,
        "source_unit_count": 5,
        "sample_digest": resampling.digest(indices),
    }


def test_a_clustered_draw_gives_fresh_identities():
    labels = numpy.array(["a", "a", "b", "c"], dtype=object)
    values = numpy.array([1.0, 2.0, 3.0, 4.0])
    matrix = values[:, None]
    ignored, outcome, clusters, record = resampling.draw(
        matrix, values, labels, numpy.random.default_rng(1)
    )
    picks = numpy.random.default_rng(1).integers(0, 3, size=3)
    groups = {0: [1.0, 2.0], 1: [3.0], 2: [4.0]}
    assert outcome.tolist() == [v for p in picks for v in groups[p]]
    assert sorted(set(clusters.tolist())) == [0, 1, 2]
    assert record["sampled_units"] == 3
    assert record["source_unit_count"] == 3
    assert record["sample_digest"] == resampling.digest(picks)
    ignored, ignored, ignored, cloned = resampling.clone(
        matrix, values, labels, numpy.random.default_rng(1)
    )
    assert cloned["fresh_cluster_identity_digest"] == resampling.digest(
        numpy.arange(3, dtype=numpy.int64)
    )


def test_draws_need_two_hashable_clusters():
    rng = numpy.random.default_rng(0)
    values = numpy.zeros(2)
    with pytest.raises(errors.InputError, match="at least two source clusters"):
        resampling.draw(values[:, None], values, numpy.array([1, 1]), rng)
    with pytest.raises(
        errors.InputError,
        match="hierarchical bootstrap needs at least two source clusters",
    ):
        resampling.clone(values[:, None], values, numpy.array([1, 1]), rng)
    lists = numpy.empty(2, dtype=object)
    lists[0], lists[1] = [1], [2]
    with pytest.raises(
        errors.InputError, match="hashable for clustered bootstrap"
    ):
        resampling.draw(values[:, None], values, lists, rng)
    with pytest.raises(
        errors.InputError, match="must be hashable for bootstrap"
    ):
        resampling.clone(values[:, None], values, lists, rng)


def causal_world(rows=200, seed=2):
    rng = numpy.random.default_rng(seed)
    age = rng.normal(size=rows)
    t = (rng.random(rows) < special.expit(0.5 * age)).astype(float)
    y = 1.0 + 2.0 * t + age + rng.normal(scale=0.5, size=rows)
    return numpy.column_stack([t, age]), y


def test_a_causal_bootstrap_accepts_reference_checked_replicates():
    matrix, y = causal_world()
    policy = config.Config(
        estimator="ipw_ate",
        covariance="iid",
        treatment=0,
        causal_assumptions=CAUSAL,
        replicates=4,
        seed=3,
    )
    found = resampling.causal_bootstrap_sensitivity(
        matrix, y, estimand(("t", "age")), policy, numpy.zeros(1), None
    )
    assert found["status"] == "available"
    assert found["successful_replicates"] == 4
    assert found["failed_replicates"] == 0
    assert found["resampling_unit"] == "row"
    assert found["source_unit_count"] == 200
    assert [d["replicate"] for d in found["draws"]] == [0, 1, 2, 3]
    assert found["draws"][0]["resampling_unit"] == "row"
    assert (
        1.5
        < found["bootstrap_intervals"][0][0]
        <= found["bootstrap_intervals"][1][0]
        < 2.5
    )


def test_a_replication_with_too_few_successes_raises():
    matrix, y = causal_world()
    policy = config.Config(
        estimator="ipw_ate",
        covariance="iid",
        treatment=0,
        causal_assumptions=CAUSAL,
        replicates=2,
    )

    def bad_draw():
        return matrix[:3], y[:3], None, {}

    with pytest.raises(
        errors.InputError,
        match="causal bootstrap needs at least two successful replicates; received 0 of 2",
    ):
        resampling.replicate(
            bad_draw,
            {"resampling_unit": "row"},
            (estimand(("t", "age")), policy, numpy.zeros(1)),
            "causal",
        )


def test_a_rerun_rejects_a_shape_mismatch():
    matrix, y = causal_world()
    policy = config.Config(
        estimator="ipw_ate",
        covariance="iid",
        treatment=0,
        causal_assumptions=CAUSAL,
    )
    estimate = resampling.rerun(
        matrix, y, estimand(("t", "age")), policy, None, numpy.zeros(1)
    )
    assert estimate.shape == (1,)
    with pytest.raises(ValueError, match="unexpected parameter shape"):
        resampling.rerun(
            matrix, y, estimand(("t", "age")), policy, None, numpy.zeros(2)
        )
