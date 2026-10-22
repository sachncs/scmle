"""First-class parameter-estimation contracts and uncertainty evidence."""

import json

import numpy as np
import pytest

from scml.evaluation.parameter_estimation import BayesianPrior
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


def causal_estimand() -> EstimandSpec:
    """Return a continuous-outcome estimand with a declared treatment column."""
    return EstimandSpec(
        name="average_treatment_effect",
        outcome="outcome",
        population="eligible observations in the 2026 study window",
        likelihood="gaussian",
        feature_names=("treatment", "age", "risk_score"),
        assumptions=(
            "the treatment and outcome are measured in the same observation",
            "the declared covariates precede treatment assignment",
        ),
    )


def hierarchical_estimand() -> EstimandSpec:
    """Return a Gaussian estimand for grouped observations."""
    return EstimandSpec(
        name="grouped_outcome_slope",
        outcome="outcome",
        population="observations nested in declared sites",
        likelihood="gaussian",
        feature_names=("signal",),
        assumptions=(
            "the random intercept is independent of the signal",
            "sites are independent sampling units",
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


def test_estimation_fingerprint_binds_raw_feature_outcome_and_clusters():
    """Raw inputs have stable fingerprints before missing-row filtering."""
    values = np.arange(15, dtype=float)
    features = np.column_stack((values, values**2))
    outcome = 1.0 + features[:, 0] - 0.5 * features[:, 1]
    clusters = np.repeat(("a", "b", "c"), 5)
    policy = EstimationPolicy(
        estimator="linear", covariance="clustered", cluster_name="site"
    )

    first = estimate_parameters(
        features, outcome, linear_estimand(), policy, clusters=clusters
    )
    second = estimate_parameters(
        features.copy(),
        outcome.copy(),
        linear_estimand(),
        policy,
        clusters=clusters.copy(),
    )
    changed = estimate_parameters(
        features,
        outcome,
        linear_estimand(),
        policy,
        clusters=np.roll(clusters, 1),
    )

    assert first.raw_input_fingerprint == second.raw_input_fingerprint
    assert first.raw_input_fingerprint != changed.raw_input_fingerprint
    assert len(first.raw_input_fingerprint) == 64
    assert first.to_dict()["raw_input_fingerprint"] == (
        first.raw_input_fingerprint
    )


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


def test_bayesian_linear_estimation_reports_posterior_evidence():
    """A declared conjugate prior produces auditable posterior evidence."""
    rng = np.random.default_rng(19)
    features = rng.normal(size=(180, 2))
    outcome = 1.25 + 1.8 * features[:, 0] - 0.6 * features[:, 1]
    outcome += rng.normal(scale=0.35, size=len(outcome))
    prior = BayesianPrior(
        mean=(0.0, 0.0, 0.0),
        precision=((0.01, 0.0, 0.0), (0.0, 0.01, 0.0), (0.0, 0.0, 0.01)),
    )

    result = estimate_parameters(
        features,
        outcome,
        linear_estimand(),
        EstimationPolicy(
            estimator="bayesian_linear", covariance="iid", prior=prior
        ),
    )

    assert result.parameters[1].estimate == pytest.approx(1.8, abs=0.12)
    assert result.parameters[2].estimate == pytest.approx(-0.6, abs=0.12)
    assert result.parameters[1].interval[0] < result.parameters[1].estimate
    assert result.parameters[1].interval[1] > result.parameters[1].estimate
    assert result.diagnostics["estimator_family"] == "bayesian_conjugate"
    assert result.diagnostics["uncertainty"] == "posterior_credible_interval"
    assert result.sensitivity["prior"]["status"] == "evaluated"
    assert result.reference["passed"] is True
    json.dumps(result.to_dict())


def test_bayesian_prior_round_trip_preserves_digest_contract(tmp_path):
    """A Bayesian report reloads with its prior and digest unchanged."""
    features = np.arange(30, dtype=float).reshape(15, 2)
    outcome = 1.0 + 0.5 * features[:, 0] - 0.25 * features[:, 1]
    prior = BayesianPrior(
        mean=(0.0, 0.0, 0.0),
        precision=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        shape=3.0,
        scale=2.0,
    )
    result = estimate_parameters(
        features,
        outcome,
        linear_estimand(),
        EstimationPolicy(
            estimator="bayesian_linear", covariance="iid", prior=prior
        ),
    )
    path = tmp_path / "bayesian-estimation.json"

    digest = result.write(path)
    restored = EstimationResult.read(path)

    assert restored.policy.prior == prior
    assert restored.digest() == digest


def test_bayesian_estimation_fails_closed_for_unsupported_policies():
    """Bayesian evidence does not silently claim robust or clustered errors."""
    with pytest.raises(ValueError, match="requires an explicit prior"):
        EstimationPolicy(estimator="bayesian_linear")

    prior = BayesianPrior(
        mean=(0.0, 0.0, 0.0),
        precision=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
    )
    with pytest.raises(ValueError, match="only iid covariance"):
        EstimationPolicy(
            estimator="bayesian_linear", covariance="hc3", prior=prior
        )

    with pytest.raises(
        ValueError, match="bootstrap sensitivity is unsupported"
    ):
        EstimationPolicy(
            estimator="bayesian_linear",
            covariance="iid",
            prior=prior,
            bootstrap_replicates=2,
        )

    with pytest.raises(ValueError, match="only by bayesian_linear"):
        EstimationPolicy(estimator="linear", prior=prior)


def test_ipw_ate_requires_declared_assumptions_and_reports_overlap():
    """IPW exposes treatment assignment evidence without claiming proof."""
    rng = np.random.default_rng(23)
    rows = 500
    age = rng.normal(size=rows)
    risk_score = rng.normal(size=rows)
    treatment_probability = 1.0 / (
        1.0 + np.exp(-(-0.2 + 0.8 * age - 0.4 * risk_score))
    )
    treatment = rng.binomial(1, treatment_probability)
    outcome = 1.0 + 2.0 * treatment + 0.5 * age - 0.25 * risk_score
    outcome += rng.normal(scale=0.8, size=rows)
    features = np.column_stack((treatment, age, risk_score))
    policy = EstimationPolicy(
        estimator="ipw_ate",
        covariance="iid",
        treatment_position=0,
        propensity_clip=0.02,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )

    result = estimate_parameters(features, outcome, causal_estimand(), policy)

    assert result.parameters[0].name == "ate"
    assert result.parameters[0].estimate == pytest.approx(2.0, abs=0.35)
    assert result.diagnostics["estimator_family"] == "causal_ipw"
    assert result.diagnostics["method"] == "hajek_ipw"
    assert (
        result.diagnostics["propensity_estimation_uncertainty"]
        == "not_propagated"
    )
    assert result.diagnostics["positivity"]["status"] == "diagnosed_not_proven"
    assert result.diagnostics["effective_sample_size"]["treated"] > 2
    assert result.reference["passed"] is True
    json.dumps(result.to_dict())


def test_ipw_ate_fails_closed_without_all_causal_assumptions():
    """The causal estimator cannot be selected with underspecified claims."""
    with pytest.raises(ValueError, match="requires consistency"):
        EstimationPolicy(
            estimator="ipw_ate",
            covariance="iid",
            treatment_position=0,
            causal_assumptions=("positivity",),
        )


def test_hierarchical_linear_reports_random_intercept_variance():
    """Grouped GLS reports its variance components and reference evidence."""
    rng = np.random.default_rng(29)
    cluster_count = 16
    rows_per_cluster = 18
    clusters = np.repeat(np.arange(cluster_count), rows_per_cluster)
    signal = rng.normal(size=len(clusters))
    random_intercepts = rng.normal(scale=1.1, size=cluster_count)
    outcome = (
        1.0
        + 0.85 * signal
        + random_intercepts[clusters]
        + rng.normal(scale=0.45, size=len(clusters))
    )
    assumptions = (
        "random_intercept",
        "cluster_independence",
        "random_effect_exogeneity",
        "constant_residual_variance",
    )
    policy = EstimationPolicy(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster_name="site_id",
        hierarchical_assumptions=assumptions,
    )

    result = estimate_parameters(
        signal.reshape(-1, 1),
        outcome,
        hierarchical_estimand(),
        policy,
        clusters=clusters,
    )

    assert result.parameters[1].estimate == pytest.approx(0.85, abs=0.12)
    assert result.diagnostics["estimator_family"] == (
        "hierarchical_random_intercept"
    )
    assert result.diagnostics["cluster_count"] == cluster_count
    assert result.diagnostics["random_intercept_variance"] >= 0.0
    assert 0.0 <= result.diagnostics["intraclass_correlation"] <= 1.0
    assert (
        result.diagnostics["variance_component_uncertainty"] == "not_propagated"
    )
    assert result.reference["passed"] is True
    json.dumps(result.to_dict())


def test_hierarchical_linear_requires_groups_and_rejects_robust_policy():
    """Unsupported grouped covariance claims fail before estimation."""
    assumptions = (
        "random_intercept",
        "cluster_independence",
        "random_effect_exogeneity",
        "constant_residual_variance",
    )
    with pytest.raises(ValueError, match="supplies its own covariance"):
        EstimationPolicy(
            estimator="hierarchical_linear",
            covariance="hc3",
            cluster_name="site_id",
            hierarchical_assumptions=assumptions,
        )

    policy = EstimationPolicy(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster_name="site_id",
        hierarchical_assumptions=assumptions,
    )
    features = np.arange(20, dtype=float).reshape(10, 2)
    outcome = np.arange(10, dtype=float)
    with pytest.raises(ValueError, match="requires cluster labels"):
        estimate_parameters(
            features,
            outcome,
            linear_estimand(),
            policy,
        )


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
            "0" * 64,
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
            "0" * 64,
        )
