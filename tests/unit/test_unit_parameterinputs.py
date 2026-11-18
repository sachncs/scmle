"""Input checks, missing values and fingerprints for parameter estimation."""

import hashlib
import json

import numpy
import pytest

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation import config
from scmle.evaluation.parameter_estimation import declaration
from scmle.evaluation.parameter_estimation import inputs

CAUSAL = ("consistency", "conditional_exchangeability", "positivity")
HIERARCHICAL = (
    "random_intercept",
    "cluster_independence",
    "random_effect_exogeneity",
    "constant_residual_variance",
)


def estimand(likelihood="gaussian"):
    return declaration.Estimand("n", "y", "p", likelihood, ("a", "b"), ("x",))


def ipw(**changes):
    values = {
        "estimator": "ipw_ate",
        "covariance": "iid",
        "treatment_position": 0,
        "causal_assumptions": CAUSAL,
    }
    values.update(changes)
    return config.Config(**values)


MATRIX = numpy.zeros((5, 2))
VALUES = numpy.zeros(5)


def test_an_admission_accepts_matching_inputs():
    assert (
        inputs.admit(
            MATRIX, VALUES, None, config.Config(estimator="linear"), estimand()
        )
        is None
    )


@pytest.mark.parametrize(
    ("values", "clusters", "policy", "likelihood", "message"),
    [
        (
            numpy.zeros(4),
            None,
            config.Config(estimator="linear"),
            "gaussian",
            "features and outcome must have the same row count",
        ),
        (
            VALUES,
            None,
            config.Config(
                estimator="linear", covariance="clustered", cluster_name="c"
            ),
            "gaussian",
            "clustered covariance needs cluster labels",
        ),
        (
            VALUES,
            None,
            config.Config(
                estimator="hierarchical_linear",
                covariance="iid",
                cluster_name="c",
                hierarchical_assumptions=HIERARCHICAL,
            ),
            "gaussian",
            "hierarchical_linear requires cluster labels",
        ),
        (
            VALUES,
            numpy.zeros(5),
            ipw(),
            "gaussian",
            "ipw_ate cluster labels require covariance='clustered'",
        ),
        (
            VALUES,
            None,
            config.Config(estimator="logistic"),
            "gaussian",
            "estimand likelihood does not match estimator: expected bernoulli, got gaussian",
        ),
        (
            VALUES,
            None,
            config.Config(estimator="linear"),
            "bernoulli",
            "expected gaussian, got bernoulli",
        ),
    ],
)
def test_an_admission_rejects_mismatched_inputs(
    values, clusters, policy, likelihood, message
):
    with pytest.raises(errors.InputError, match=message):
        inputs.admit(MATRIX, values, clusters, policy, estimand(likelihood))


def test_an_admission_rejects_a_random_slope_outside_the_features():
    policy = config.Config(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster_name="c",
        hierarchical_assumptions=(*HIERARCHICAL, "random_slope"),
        random_slope_position=2,
    )
    with pytest.raises(
        errors.InputError, match="random_slope_position is outside"
    ):
        inputs.admit(MATRIX, VALUES, numpy.zeros(5), policy, estimand())
    inside = config.Config(**{**fields(policy), "random_slope_position": 1})
    assert (
        inputs.admit(MATRIX, VALUES, numpy.zeros(5), inside, estimand()) is None
    )


def fields(policy):
    return {
        "estimator": policy.estimator,
        "covariance": policy.covariance,
        "cluster_name": policy.cluster_name,
        "hierarchical_assumptions": policy.hierarchical_assumptions,
        "random_slope_position": policy.random_slope_position,
    }


def test_a_screen_rejects_bad_logistic_outcomes_and_positions():
    with pytest.raises(
        errors.InputError, match="logistic outcome must contain only 0 and 1"
    ):
        inputs.screen(
            MATRIX,
            numpy.array([0, 1, 2, 0, 1.0]),
            None,
            config.Config(estimator="logistic"),
        )
    inputs.screen(
        MATRIX,
        numpy.array([0, 1, 1, 0, 1.0]),
        None,
        config.Config(estimator="logistic"),
    )
    with pytest.raises(
        errors.InputError, match="treatment_position is outside"
    ):
        inputs.screen(MATRIX, VALUES, None, ipw(treatment_position=2))
    inputs.screen(MATRIX, VALUES, None, ipw(treatment_position=1))
    prior = declaration.BayesianPrior(
        (0.0,) * 3,
        tuple(tuple(float(i == j) for j in range(3)) for i in range(3)),
    )
    with pytest.raises(
        errors.InputError,
        match="bayesian_linear estimation does not accept clusters",
    ):
        inputs.screen(
            MATRIX,
            VALUES,
            [1] * 5,
            config.Config(
                estimator="bayesian_linear", prior=prior, covariance="iid"
            ),
        )


def test_a_fingerprint_binds_shape_values_and_clusters():
    base = inputs.raw_input_fingerprint([[1, 2], [3, 4]], [1, 2])
    assert base == inputs.raw_input_fingerprint(
        numpy.array([[1.0, 2.0], [3.0, 4.0]]), [[1.0], [2.0]]
    )
    assert base != inputs.raw_input_fingerprint([[1, 2, 3, 4]], [1, 2])
    assert base != inputs.raw_input_fingerprint([[1, 2], [3, 5]], [1, 2])
    assert base != inputs.raw_input_fingerprint(
        [[1, 2], [3, 4]], [1, 2], ["a", "b"]
    )
    assert inputs.raw_input_fingerprint([[1, 2], [3, 4]], [1, 2], [1, 2]) != (
        inputs.raw_input_fingerprint([[1, 2], [3, 4]], [1, 2], ["1", "2"])
    )


def test_an_absorption_of_numbers_matches_the_documented_layout():
    digest = hashlib.sha256()
    inputs.absorb(digest, "x", numpy.array([1.0, 2.0]))
    expected = hashlib.sha256()
    descriptor = json.dumps(
        {"name": "x", "shape": [2], "dtype": "float64"},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    expected.update(len(descriptor).to_bytes(8, "big"))
    expected.update(descriptor)
    payload = numpy.array([1.0, 2.0], dtype="<f8").tobytes()
    expected.update(len(payload).to_bytes(8, "big"))
    expected.update(payload)
    assert digest.hexdigest() == expected.hexdigest()


def test_a_matrix_validation_needs_the_shape_and_enough_rows():
    inputs.validate_matrix(numpy.zeros((3, 2)), 2, "m")
    with pytest.raises(
        errors.InputError, match=r"m must have shape \(rows, 3\)"
    ):
        inputs.validate_matrix(numpy.zeros((3, 2)), 3, "m")
    with pytest.raises(errors.InputError, match="must have shape"):
        inputs.validate_matrix(numpy.zeros(3), 1, "m")
    with pytest.raises(errors.InputError, match="more rows than parameters"):
        inputs.validate_matrix(numpy.zeros((2, 2)), 2, "m")


def test_a_cluster_validation_checks_shape_and_length():
    assert inputs.validate_clusters(None, 3) is None
    found = inputs.validate_clusters([1, "a", 2], 3)
    assert found.dtype == object and list(found) == [1, "a", 2]
    with pytest.raises(
        errors.InputError, match=r"clusters must have shape \(rows,\)"
    ):
        inputs.validate_clusters([[1], [2]], 2)
    with pytest.raises(errors.InputError, match="same row count as outcome"):
        inputs.validate_clusters([1, 2], 3)


def test_a_condition_report_classifies_the_covariance():
    found = inputs.condition(numpy.diag([4.0, 1.0]), 2)
    assert found == {
        "status": "identified",
        "dimension": 2,
        "rank": 2,
        "condition_number": 4.0,
        "condition_number_threshold": 1e12,
    }
    singular = inputs.condition(numpy.array([[1.0, 1.0], [1.0, 1.0]]), 2)
    assert singular["status"] == "rank_deficient"
    assert singular["rank"] == 1
    ill = inputs.condition(numpy.diag([1.0, 1e-13]), 2)
    assert ill["status"] == "ill_conditioned"
    assert ill["rank"] == 2
    zero = inputs.condition(numpy.zeros((2, 2)), 2)
    assert zero["condition_number"] is None
    assert zero["rank"] == 0


def test_a_condition_report_rejects_bad_matrices():
    with pytest.raises(
        errors.InputError, match=r"covariance must have shape \(3, 3\)"
    ):
        inputs.condition(numpy.eye(2), 3)
    with pytest.raises(errors.InputError, match="finite values"):
        inputs.condition(numpy.array([[numpy.nan, 0], [0, 1.0]]), 2)
    with pytest.raises(errors.InputError, match="must be symmetric"):
        inputs.condition(numpy.array([[1.0, 0.5], [0.0, 1.0]]), 2)


def test_a_preparation_drops_incomplete_rows_under_complete_case():
    matrix = numpy.array([[1.0], [numpy.nan], [3.0], [4.0], [5.0], [6.0]])
    values = numpy.array([1.0, 2.0, numpy.inf, 4.0, 5.0, 6.0])
    clusters = ["a", "b", "c", None, "e", "f"]
    kept, outcome, labels, dropped = inputs.prepare(
        matrix, values, "complete_case", clusters
    )
    assert dropped == 3
    assert kept.tolist() == [[1.0], [5.0], [6.0]]
    assert outcome.tolist() == [1.0, 5.0, 6.0]
    assert list(labels) == ["a", "e", "f"]


def test_a_preparation_rejects_incomplete_rows_or_too_few_rows():
    matrix = numpy.array([[1.0], [numpy.nan], [3.0], [4.0]])
    with pytest.raises(errors.InputError, match="require an explicit policy"):
        inputs.prepare(matrix, numpy.ones(4), "reject", None)
    with pytest.raises(
        errors.InputError, match="complete rows are insufficient"
    ):
        inputs.prepare(numpy.ones((2, 1)), numpy.ones(2), "reject", None)
    kept = inputs.prepare(numpy.ones((3, 1)), numpy.ones(3), "reject", None)
    assert kept[2] is None and kept[3] == 0
    with pytest.raises(
        errors.InputError, match="cluster labels must be scalar"
    ):
        inputs.prepare(
            numpy.ones((3, 1)),
            numpy.ones(3),
            "reject",
            [[1, 2], [3, 4], [5, 6]],
        )


def test_a_clustering_report_counts_sizes_in_first_seen_order():
    found = inputs.clustering(numpy.array(["b", "a", "b", "c"], dtype=object))
    assert found == {
        "method": "finite_sample_cluster_robust_sandwich",
        "cluster_count": 3,
        "cluster_sizes": [2, 1, 1],
        "minimum_cluster_size": 1,
        "maximum_cluster_size": 2,
        "singleton_cluster_fraction": pytest.approx(2 / 3),
        "independent_sampling_unit": "declared_cluster_label",
        "small_cluster_caution": True,
    }
    many = inputs.clustering(numpy.repeat(numpy.arange(30), 2))
    assert many["small_cluster_caution"] is False
    assert many["singleton_cluster_fraction"] == 0.0


def test_a_clustering_report_needs_two_hashable_clusters():
    with pytest.raises(errors.InputError, match="cluster labels are required"):
        inputs.clustering(None)
    with pytest.raises(errors.InputError, match="at least two clusters"):
        inputs.clustering(numpy.array([1, 1]))
    lists = numpy.empty(2, dtype=object)
    lists[0], lists[1] = [1], [2]
    with pytest.raises(errors.InputError, match="hashable scalar values"):
        inputs.clustering(lists)
