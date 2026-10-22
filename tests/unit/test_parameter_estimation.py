"""First-class parameter-estimation contracts and uncertainty evidence."""

import json

import numpy as np
import pytest

from scml.evaluation.parameter_estimation import EstimandSpec
from scml.evaluation.parameter_estimation import estimate_parameters
from scml.evaluation.parameter_estimation import EstimationPolicy
from scml.evaluation.parameter_estimation import EstimationResult
from scml.evaluation.parameter_estimation import ParameterEstimate


def linear_estimand() -> EstimandSpec:
    """Return a declared Gaussian estimand for test fixtures."""
    return EstimandSpec(
        name="mean_outcome_difference",
        outcome="outcome",
        population="eligible observations in the 2026 study window",
        likelihood="gaussian",
        feature_names=("treatment", "age"),
        assumptions=(
            "the observed covariates are sufficient for the declared model",
            "the observation errors are independent across declared clusters",
        ),
    )


def test_linear_estimation_records_robust_uncertainty_and_reference():
    """A known linear relationship produces auditable coefficient evidence."""
    rng = np.random.default_rng(7)
    features = rng.normal(size=(180, 2))
    outcome = 1.5 + 2.0 * features[:, 0] - 0.75 * features[:, 1]
    outcome += rng.normal(scale=0.2, size=len(outcome))

    result = estimate_parameters(
        features,
        outcome,
        linear_estimand(),
        EstimationPolicy(estimator="linear", covariance="hc3"),
    )

    assert result.parameters[0].name == "intercept"
    assert result.parameters[1].estimate == pytest.approx(2.0, abs=0.08)
    assert result.parameters[2].estimate == pytest.approx(-0.75, abs=0.08)
    assert result.reference["passed"] is True
    assert result.diagnostics["likelihood"] == "gaussian"
    assert result.parameters[1].interval[0] < result.parameters[1].estimate


def test_clustered_covariance_and_shift_diagnostics_are_explicit():
    """Cluster and target-population policies cannot disappear into a score."""
    rng = np.random.default_rng(11)
    features = rng.normal(size=(120, 2))
    clusters = np.repeat(np.arange(12), 10)
    outcome = 0.5 + features[:, 0] + features[:, 1] + rng.normal(size=120)

    result = estimate_parameters(
        features,
        outcome,
        linear_estimand(),
        EstimationPolicy(
            estimator="linear",
            covariance="clustered",
            cluster_name="site_id",
            bootstrap_replicates=8,
            seed=3,
        ),
        clusters=clusters,
        shift_reference=features + np.array([2.0, 0.0]),
    )

    assert result.sensitivity["sampling"]["status"] == "available"
    assert result.sensitivity["distribution_shift"]["status"] == "evaluated"
    assert (
        result.sensitivity["distribution_shift"][
            "max_standardized_mean_difference"
        ]
        > 1
    )
    json.dumps(result.to_dict())


def test_logistic_estimation_has_independent_optimization_check():
    """A Bernoulli likelihood reports convergence and a second calculation."""
    rng = np.random.default_rng(4)
    features = rng.normal(size=(500, 2))
    logits = -0.3 + 1.1 * features[:, 0] - 0.8 * features[:, 1]
    outcome = rng.binomial(1, 1.0 / (1.0 + np.exp(-logits)))
    estimand = EstimandSpec(
        name="conversion_log_odds",
        outcome="converted",
        population="eligible visitors in the observation window",
        likelihood="bernoulli",
        feature_names=("signal_a", "signal_b"),
        assumptions=("rows are conditionally independent",),
    )

    result = estimate_parameters(
        features,
        outcome,
        estimand,
        EstimationPolicy(estimator="logistic", covariance="iid"),
    )

    assert result.diagnostics["converged"] is True
    assert result.reference["passed"] is True
    assert result.parameters[1].estimate == pytest.approx(1.1, abs=0.25)


def test_complete_case_is_explicit_and_reject_is_fail_closed():
    """Missingness must be a declared policy, never an accidental drop."""
    features = np.array([[0.0], [1.0], [np.nan], [3.0], [4.0], [5.0]])
    outcome = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    estimand = EstimandSpec(
        name="slope",
        outcome="outcome",
        population="all rows in the source snapshot",
        likelihood="gaussian",
        feature_names=("feature",),
        assumptions=("linear conditional mean",),
    )

    with pytest.raises(ValueError, match="explicit policy"):
        estimate_parameters(
            features,
            outcome,
            estimand,
            EstimationPolicy(estimator="linear"),
        )

    result = estimate_parameters(
        features,
        outcome,
        estimand,
        EstimationPolicy(estimator="linear", missing="complete_case"),
    )
    assert result.rows_input == 6
    assert result.rows_used == 5
    assert result.rows_dropped == 1


def test_declared_likelihood_must_match_estimation_method():
    """A report cannot silently relabel the scientific likelihood."""
    estimand = EstimandSpec(
        name="conversion",
        outcome="converted",
        population="eligible visitors",
        likelihood="gaussian",
        feature_names=("signal",),
        assumptions=("rows are conditionally independent",),
    )

    with pytest.raises(ValueError, match="expected bernoulli"):
        estimate_parameters(
            np.ones((12, 1)),
            np.zeros(12),
            estimand,
            EstimationPolicy(estimator="logistic"),
        )


def test_report_persistence_is_atomic_and_digest_bound(tmp_path):
    """A saved report can be reloaded and tampering fails closed."""
    rng = np.random.default_rng(12)
    features = rng.normal(size=(40, 1))
    outcome = 2.0 + features[:, 0]
    estimand = EstimandSpec(
        name="slope",
        outcome="outcome",
        population="test rows",
        likelihood="gaussian",
        feature_names=("feature",),
        assumptions=("linear conditional mean",),
    )
    result = estimate_parameters(
        features,
        outcome,
        estimand,
        EstimationPolicy(estimator="linear"),
    )
    path = tmp_path / "estimation.json"

    digest = result.write(path)

    restored = EstimationResult.read(path)
    assert digest == result.digest() == restored.digest()
    path.write_text(
        path.read_text().replace('"rows_used": 40', '"rows_used": 39')
    )
    with pytest.raises(ValueError, match="digest mismatch"):
        EstimationResult.read(path)


def test_parameter_report_rejects_invalid_scientific_evidence():
    """Malformed estimates cannot enter the evidence contract."""
    with pytest.raises(ValueError, match="standard_error"):
        ParameterEstimate("slope", 1.0, -0.1, (0.5, 1.5), 0.01)

    with pytest.raises(ValueError, match="interval must be ordered"):
        ParameterEstimate("slope", 1.0, 0.1, (1.5, 0.5), 0.01)


def test_parameter_report_rejects_inconsistent_shape_and_row_accounting():
    """Covariance shape and complete-case accounting are release invariants."""
    estimand = EstimandSpec(
        name="slope",
        outcome="outcome",
        population="test rows",
        likelihood="gaussian",
        feature_names=("feature",),
        assumptions=("linear conditional mean",),
    )
    policy = EstimationPolicy(estimator="linear")
    parameter = ParameterEstimate("intercept", 1.0, 0.1, (0.8, 1.2), 0.01)

    with pytest.raises(ValueError, match="covariance must be square"):
        EstimationResult(
            estimand,
            policy,
            (parameter,),
            ((1.0, 0.0), (0.0, 1.0)),
            2,
            2,
            0,
            {},
            {},
            {},
        )

    with pytest.raises(ValueError, match="row accounting"):
        EstimationResult(
            estimand,
            policy,
            (parameter,),
            ((1.0,),),
            2,
            1,
            0,
            {},
            {},
            {},
        )
