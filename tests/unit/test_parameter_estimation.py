"""First-class parameter-estimation contracts and uncertainty evidence."""

import json
from types import SimpleNamespace
from typing import Any
from typing import Literal

import numpy as np
import pytest

from scml.evaluation import parameter_estimation as estimation_module
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


def clustered_causal_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed clustered observations for causal uncertainty tests."""
    rng = np.random.default_rng(303)
    cluster_count = 20
    rows_per_cluster = 12
    clusters = np.repeat(np.arange(cluster_count), rows_per_cluster)
    age = rng.normal(size=len(clusters))
    risk_score = rng.normal(size=len(clusters))
    treatment_probability = 1.0 / (
        1.0 + np.exp(-(-0.25 + 0.65 * age - 0.35 * risk_score))
    )
    treatment = rng.binomial(1, treatment_probability)
    outcome = (
        0.75
        + 1.8 * treatment
        + 0.4 * age
        - 0.2 * risk_score
        + rng.normal(scale=0.7, size=len(clusters))
    )
    return np.column_stack((treatment, age, risk_score)), outcome, clusters


def clustered_causal_policy(
    estimator: Literal["ipw_ate", "aipw_ate"],
) -> EstimationPolicy:
    """Return a project-serializable clustered causal policy."""
    return EstimationPolicy(
        estimator=estimator,
        covariance="clustered",
        cluster_name="site_id",
        treatment_position=0,
        propensity_clip=0.02,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )


def continuous_treatment_estimand() -> EstimandSpec:
    """Return a declared continuous-treatment dose-response estimand."""
    return EstimandSpec(
        name="continuous_treatment_dose_response",
        outcome="outcome",
        population="eligible observations in the 2026 study window",
        likelihood="gaussian",
        feature_names=("dose", "age", "risk_score"),
        assumptions=(
            "the treatment and outcome are measured in the same observation",
            "the declared covariates precede treatment assignment",
        ),
    )


def continuous_treatment_policy(
    covariance: Literal["iid", "clustered"] = "iid",
) -> EstimationPolicy:
    """Return the fully identified bounded continuous-treatment policy."""
    return EstimationPolicy(
        estimator="continuous_ipw_ate",
        covariance=covariance,
        cluster_name="site_id" if covariance == "clustered" else None,
        treatment_position=0,
        continuous_weight_cap=8.0,
        continuous_treatment_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "continuous_treatment_positivity",
            "linear_dose_response",
            "normal_conditional_treatment_density",
        ),
    )


def continuous_treatment_fixture(
    cluster_count: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Return a confounded continuous-treatment fixture with known slope."""
    rng = np.random.default_rng(414)
    rows = 720 if cluster_count == 1 else cluster_count * 24
    age = rng.normal(size=rows)
    risk_score = rng.normal(size=rows)
    dose = 0.6 * age - 0.45 * risk_score + rng.normal(scale=1.1, size=rows)
    outcome = (
        0.8
        + 1.75 * dose
        + 0.45 * age
        - 0.25 * risk_score
        + rng.normal(scale=0.7, size=rows)
    )
    clusters = (
        None
        if cluster_count == 1
        else np.repeat(np.arange(cluster_count), rows // cluster_count)
    )
    return np.column_stack((dose, age, risk_score)), outcome, clusters


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
    covariance_conditioning = result.diagnostics["covariance_conditioning"]
    assert covariance_conditioning["status"] == "identified"
    assert covariance_conditioning["rank"] == 3
    assert covariance_conditioning["condition_number"] < 1e12
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


def test_bayesian_ate_reports_causal_posterior_evidence():
    """A Gaussian outcome model identifies a declared binary-treatment ATE."""
    rng = np.random.default_rng(121)
    rows = 420
    age = rng.normal(size=rows)
    risk_score = rng.normal(size=rows)
    treatment_probability = 1.0 / (
        1.0 + np.exp(-(-0.3 + 0.7 * age - 0.4 * risk_score))
    )
    treatment = rng.binomial(1, treatment_probability)
    outcome = (
        0.8
        + 2.25 * treatment
        + 0.6 * age
        - 0.3 * risk_score
        + rng.normal(scale=0.4, size=rows)
    )
    features = np.column_stack((treatment, age, risk_score))
    prior = BayesianPrior(
        mean=(0.0, 0.0, 0.0, 0.0),
        precision=tuple(
            tuple(0.01 if row == column else 0.0 for column in range(4))
            for row in range(4)
        ),
    )
    policy = EstimationPolicy(
        estimator="bayesian_ate",
        covariance="iid",
        prior=prior,
        treatment_position=0,
        causal_assumptions=(
            "linearity",
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )

    result = estimate_parameters(features, outcome, causal_estimand(), policy)

    assert result.parameters[1].name == "treatment"
    assert result.parameters[1].estimate == pytest.approx(2.25, abs=0.16)
    assert result.parameters[1].interval[0] < result.parameters[1].estimate
    assert result.parameters[1].interval[1] > result.parameters[1].estimate
    assert result.diagnostics["estimator_family"] == (
        "causal_bayesian_gaussian"
    )
    assert result.diagnostics["method"] == (
        "normal_inverse_gamma_outcome_regression"
    )
    assert result.diagnostics["uncertainty"] == ("posterior_credible_interval")
    assert result.diagnostics["posterior_covariance"][1][1] > 0.0
    assert result.diagnostics["credible_interval"]["distribution"] == (
        "student_t"
    )
    assert result.sensitivity["prior"]["status"] == "evaluated"
    assert (
        "treatment_coefficient_change"
        in result.sensitivity["prior"]["alternatives"][0]
    )
    assert result.reference["passed"] is True
    assert any(
        "time-varying treatment effects" in limitation
        for limitation in result.diagnostics["limitations"]
    )
    assert any(
        "nuisance-model uncertainty" in limitation
        for limitation in result.diagnostics["limitations"]
    )
    json.dumps(result.to_dict())


def test_bayesian_ate_validates_causal_policy_and_prior_shape():
    """Causal Bayesian estimation fails closed at its scientific boundary."""
    with pytest.raises(ValueError, match="requires an explicit prior"):
        EstimationPolicy(
            estimator="bayesian_ate",
            treatment_position=0,
            causal_assumptions=(
                "linearity",
                "consistency",
                "conditional_exchangeability",
                "positivity",
            ),
        )

    prior = BayesianPrior(
        mean=(0.0, 0.0),
        precision=((1.0, 0.0), (0.0, 1.0)),
    )
    policy_values = {
        "estimator": "bayesian_ate",
        "covariance": "iid",
        "prior": prior,
        "treatment_position": 0,
        "causal_assumptions": (
            "linearity",
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    }
    with pytest.raises(ValueError, match="prior dimension"):
        estimate_parameters(
            np.column_stack(
                (
                    np.tile((0.0, 1.0), 6),
                    np.arange(12, dtype=float),
                    np.arange(12, dtype=float) ** 2,
                )
            ),
            np.arange(12, dtype=float),
            causal_estimand(),
            EstimationPolicy(**policy_values),
        )

    with pytest.raises(ValueError, match="requires consistency"):
        EstimationPolicy(
            estimator="bayesian_ate",
            covariance="iid",
            prior=BayesianPrior(
                mean=(0.0, 0.0, 0.0, 0.0),
                precision=tuple(
                    tuple(1.0 if row == column else 0.0 for column in range(4))
                    for row in range(4)
                ),
            ),
            treatment_position=0,
            causal_assumptions=("positivity",),
        )

    with pytest.raises(ValueError, match="only iid covariance"):
        EstimationPolicy(
            estimator="bayesian_ate",
            covariance="hc3",
            prior=policy_values["prior"],
            treatment_position=0,
            causal_assumptions=(
                "linearity",
                "consistency",
                "conditional_exchangeability",
                "positivity",
            ),
        )

    with pytest.raises(ValueError, match="does not support clustered"):
        EstimationPolicy(
            estimator="bayesian_ate",
            covariance="iid",
            cluster_name="site_id",
            prior=BayesianPrior(
                mean=(0.0, 0.0, 0.0, 0.0),
                precision=tuple(
                    tuple(1.0 if row == column else 0.0 for column in range(4))
                    for row in range(4)
                ),
            ),
            treatment_position=0,
            causal_assumptions=(
                "linearity",
                "consistency",
                "conditional_exchangeability",
                "positivity",
            ),
        )


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


def test_continuous_ipw_ate_reports_gps_overlap_and_effect():
    """Continuous treatment reports a weighted marginal dose-response slope."""
    features, outcome, unused_clusters = continuous_treatment_fixture()
    del unused_clusters
    policy = continuous_treatment_policy()

    first = estimate_parameters(
        features, outcome, continuous_treatment_estimand(), policy
    )
    second = estimate_parameters(
        features.copy(), outcome.copy(), continuous_treatment_estimand(), policy
    )

    assert first.to_dict() == second.to_dict()
    assert first.parameters[1].name == "dose_dose_response_slope"
    assert first.parameters[1].estimate == pytest.approx(1.75, abs=0.18)
    assert first.diagnostics["estimator_family"] == "causal_continuous_ipw"
    assert first.diagnostics["method"] == (
        "stabilized_generalized_propensity_weighted_regression"
    )
    assert first.diagnostics["overlap"]["status"] == "diagnosed_not_proven"
    assert first.diagnostics["weights"]["cap"] == 8.0
    assert 0.0 <= first.diagnostics["weights"]["fraction_capped"] <= 1.0
    assert first.diagnostics["weights"]["effective_sample_size"] > 2.0
    assert first.diagnostics["capability"]["status"] == "experimental"
    assert first.reference["passed"] is True
    json.dumps(first.to_dict())


def test_continuous_ipw_ate_supports_clustered_uncertainty():
    """Continuous-treatment covariance names its independent sampling units."""
    features, outcome, clusters = continuous_treatment_fixture(cluster_count=24)
    policy = continuous_treatment_policy("clustered")

    result = estimate_parameters(
        features,
        outcome,
        continuous_treatment_estimand(),
        policy,
        clusters=clusters,
    )

    cluster_evidence = result.diagnostics["cluster_uncertainty"]
    assert cluster_evidence["method"] == (
        "finite_sample_cluster_robust_sandwich"
    )
    assert cluster_evidence["cluster_count"] == 24
    assert cluster_evidence["minimum_cluster_size"] == 24
    assert result.reference["clustered_reference"] is True
    assert result.reference["passed"] is True


def test_continuous_ipw_ate_requires_declared_density_assumptions():
    """A continuous-treatment claim cannot reuse the binary policy default."""
    with pytest.raises(ValueError, match="continuous_treatment_positivity"):
        EstimationPolicy(
            estimator="continuous_ipw_ate",
            covariance="iid",
            treatment_position=0,
        )

    features, outcome, unused_clusters = continuous_treatment_fixture()
    del unused_clusters
    with pytest.raises(ValueError, match="at least four distinct"):
        estimate_parameters(
            np.column_stack(
                (
                    np.tile((0.0, 1.0), len(features) // 2),
                    features[:, 1],
                    features[:, 2],
                )
            ),
            outcome,
            continuous_treatment_estimand(),
            continuous_treatment_policy(),
        )


def test_continuous_policy_round_trip_preserves_explicit_capability():
    """Serialized continuous policy evidence reconstructs without defaults."""
    policy = continuous_treatment_policy()
    restored = EstimationPolicy(**policy.to_dict())

    assert restored == policy
    assert restored.to_dict()["continuous_weight_cap"] == 8.0
    assert restored.to_dict()["continuous_treatment_assumptions"] == (
        "consistency",
        "conditional_exchangeability",
        "continuous_treatment_positivity",
        "linear_dose_response",
        "normal_conditional_treatment_density",
    )


def test_aipw_ate_reports_augmented_score_and_reference_evidence():
    """AIPW combines propensity and outcome nuisance evidence."""
    rng = np.random.default_rng(24)
    rows = 600
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
        estimator="aipw_ate",
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
    assert result.diagnostics["estimator_family"] == "causal_augmented_ipw"
    assert result.diagnostics["method"] == "augmented_ipw"
    assert result.diagnostics["cross_fitting"]["status"] == "disabled"
    assert result.diagnostics["cross_fitting"]["fold_count"] == 1
    assert any(
        "same rows used for scoring" in limitation
        for limitation in result.diagnostics["cross_fitting"]["limitations"]
    )
    assert result.diagnostics["positivity"]["status"] == "diagnosed_not_proven"
    assert result.reference["passed"] is True
    json.dumps(result.to_dict())


def test_ipw_ate_clustered_influence_uncertainty_is_deterministic():
    """IPW uses centered cluster sums and records finite-sample evidence."""
    features, outcome, clusters = clustered_causal_fixture()
    policy = clustered_causal_policy("ipw_ate")

    first = estimate_parameters(
        features, outcome, causal_estimand(), policy, clusters=clusters
    )
    second = estimate_parameters(
        features, outcome, causal_estimand(), policy, clusters=clusters
    )
    iid_policy_values = policy.to_dict()
    iid_policy_values["covariance"] = "iid"
    iid = estimate_parameters(
        features,
        outcome,
        causal_estimand(),
        EstimationPolicy(**iid_policy_values),
    )

    assert first.to_dict() == second.to_dict()
    assert first.parameters[0].estimate == pytest.approx(1.8, abs=0.35)
    assert first.diagnostics["uncertainty_method"] == (
        "clustered_centered_influence_function"
    )
    assert first.diagnostics["cluster_count"] == 20
    assert first.diagnostics["cluster_sizes"] == [12] * 20
    assert first.covariance[0][0] != pytest.approx(iid.covariance[0][0])
    assert first.reference["passed"] is True
    assert first.reference["variance_method"] == (
        "clustered_centered_influence_function"
    )
    assert first.reference["variance_status"].startswith("declared_")


def test_aipw_ate_clustered_influence_uncertainty_is_serializable(tmp_path):
    """AIPW persists clustered uncertainty and project policy fields."""
    features, outcome, clusters = clustered_causal_fixture()
    policy = clustered_causal_policy("aipw_ate")
    result = estimate_parameters(
        features, outcome, causal_estimand(), policy, clusters=clusters
    )

    restored_policy = EstimationPolicy(**policy.to_dict())
    assert restored_policy == policy
    assert result.diagnostics["uncertainty_method"] == (
        "clustered_centered_influence_function"
    )
    assert result.diagnostics["cluster_count"] == 20
    assert result.diagnostics["cluster_sizes"] == [12] * 20
    assert result.reference["passed"] is True
    assert result.reference["variance_method"] == (
        "clustered_centered_influence_function"
    )
    json.dumps(result.to_dict())
    path = tmp_path / "clustered-aipw.json"
    result.write(path)
    restored = EstimationResult.read(path)
    assert restored.to_dict() == result.to_dict()


def test_causal_clustered_uncertainty_rejects_hc3_and_one_cluster():
    """Causal uncertainty fails clearly for unsupported covariance claims."""
    with pytest.raises(ValueError, match="hc3 is unsupported"):
        EstimationPolicy(
            estimator="ipw_ate",
            covariance="hc3",
            treatment_position=0,
            causal_assumptions=(
                "consistency",
                "conditional_exchangeability",
                "positivity",
            ),
        )

    features, outcome, clusters = clustered_causal_fixture()
    with pytest.raises(ValueError, match="at least two clusters"):
        estimate_parameters(
            features,
            outcome,
            causal_estimand(),
            clustered_causal_policy("aipw_ate"),
            clusters=np.zeros(len(clusters), dtype=int),
        )


def test_aipw_ate_cross_fitting_is_deterministic_and_auditable():
    """Cross-fitting uses held-out scores and records every fold contract."""
    rng = np.random.default_rng(24)
    rows = 600
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
        estimator="aipw_ate",
        covariance="iid",
        treatment_position=0,
        propensity_clip=0.02,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
        cross_fitting_folds=5,
    )

    first = estimate_parameters(features, outcome, causal_estimand(), policy)
    second = estimate_parameters(features, outcome, causal_estimand(), policy)
    cross_fitting = first.diagnostics["cross_fitting"]

    assert first.parameters[0].estimate == pytest.approx(2.0, abs=0.45)
    assert first.parameters[0].estimate == second.parameters[0].estimate
    assert cross_fitting["status"] == "enabled"
    assert cross_fitting["fold_count"] == 5
    assert cross_fitting["assignment_method"] == "row_index_modulo"
    assert cross_fitting["fold_assignments"] == [
        index % 5 for index in range(rows)
    ]
    assert all(
        support["training_treated_rows"] >= 2
        and support["training_control_rows"] >= 2
        and support["validation_treated_rows"] >= 1
        and support["validation_control_rows"] >= 1
        for support in cross_fitting["support"]
    )
    assert len(cross_fitting["nuisance_fits"]) == 5
    assert (
        first.diagnostics["uncertainty_method"]
        == "influence_function_with_cross_fitted_nuisance"
    )
    assert first.reference["method"].endswith("per_training_fold")
    assert first.reference["passed"] is True
    assert any(
        "not Bayesian" in limitation and "time-varying" in limitation
        for limitation in cross_fitting["limitations"]
    )
    assert first.to_dict()["policy"]["cross_fitting_folds"] == 5
    json.dumps(first.to_dict())


@pytest.mark.parametrize("estimator", ["ipw_ate", "aipw_ate"])
def test_causal_row_bootstrap_is_reproducible_and_reports_intervals(
    estimator: Literal["ipw_ate", "aipw_ate"],
):
    """Causal row bootstrap retains assumptions and reference evidence."""
    fixture = clustered_causal_fixture()
    features, outcome = fixture[0], fixture[1]
    policy = EstimationPolicy(
        estimator=estimator,
        covariance="iid",
        treatment_position=0,
        propensity_clip=0.02,
        bootstrap_replicates=4,
        seed=31,
        cross_fitting_folds=3 if estimator == "aipw_ate" else 1,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )

    first = estimate_parameters(features, outcome, causal_estimand(), policy)
    second = estimate_parameters(features, outcome, causal_estimand(), policy)
    sampling = first.sensitivity["sampling"]

    assert first.to_dict() == second.to_dict()
    assert sampling["status"] == "available"
    assert sampling["method"] == "nonparametric_percentile_bootstrap"
    assert sampling["resampling_unit"] == "row"
    assert sampling["source_unit_count"] == len(outcome)
    assert sampling["successful_replicates"] == 4
    assert sampling["failed_replicates"] == 0
    assert sampling["reference_checks"]["successful"] == 4
    assert len(sampling["draws"]) == 4
    assert all(
        len(interval) == 1 for interval in sampling["bootstrap_intervals"]
    )
    assert (
        sampling["bootstrap_intervals"][0][0]
        <= sampling["bootstrap_intervals"][1][0]
    )
    assert first.policy.causal_assumptions == (
        "consistency",
        "conditional_exchangeability",
        "positivity",
    )


@pytest.mark.parametrize("estimator", ["ipw_ate", "aipw_ate"])
def test_causal_cluster_bootstrap_resamples_whole_clusters(
    estimator: Literal["ipw_ate", "aipw_ate"],
):
    """Clustered causal bootstrap records cluster-unit and covariance evidence."""
    features, outcome, clusters = clustered_causal_fixture()
    policy = clustered_causal_policy(estimator)
    policy = EstimationPolicy(
        **{
            **policy.to_dict(),
            "bootstrap_replicates": 4,
            "seed": 47,
        }
    )

    result = estimate_parameters(
        features,
        outcome,
        causal_estimand(),
        policy,
        clusters=clusters,
    )
    sampling = result.sensitivity["sampling"]

    assert sampling["status"] == "available"
    assert sampling["resampling_unit"] == "cluster"
    assert sampling["source_unit_count"] == 20
    assert sampling["source_row_count"] == len(outcome)
    assert sampling["successful_replicates"] == 4
    assert sampling["failed_replicates"] == 0
    assert all(draw["sampled_units"] == 20 for draw in sampling["draws"])
    assert result.diagnostics["uncertainty_method"] == (
        "clustered_centered_influence_function"
    )
    assert result.reference["passed"] is True


def test_causal_bootstrap_records_failed_replicates(
    monkeypatch: pytest.MonkeyPatch,
):
    """An invalid replicate is recorded and excluded from intervals."""
    fixture = clustered_causal_fixture()
    features, outcome = fixture[0], fixture[1]
    policy = EstimationPolicy(
        estimator="ipw_ate",
        covariance="iid",
        treatment_position=0,
        propensity_clip=0.02,
        bootstrap_replicates=4,
        seed=53,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )
    original_fit = estimation_module.fit_ipw_ate
    calls = 0

    def fail_one_replicate(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("forced invalid bootstrap replicate")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(estimation_module, "fit_ipw_ate", fail_one_replicate)
    result = estimate_parameters(features, outcome, causal_estimand(), policy)
    sampling = result.sensitivity["sampling"]

    assert sampling["status"] == "partial"
    assert sampling["successful_replicates"] == 3
    assert sampling["failed_replicates"] == 1
    assert sampling["failures"] == [
        {
            "replicate": 0,
            "resampling_unit": "row",
            "error_type": "ValueError",
            "message": "forced invalid bootstrap replicate",
        }
    ]
    assert sampling["reference_checks"] == {
        "successful": 3,
        "failed": 1,
        "requirement": "every accepted replicate passes its independent "
        "point-estimate reference check",
    }


def test_causal_bootstrap_fails_closed_when_too_few_replicates_succeed(
    monkeypatch: pytest.MonkeyPatch,
):
    """Bootstrap intervals are not emitted when evidence is insufficient."""
    fixture = clustered_causal_fixture()
    features, outcome = fixture[0], fixture[1]
    policy = EstimationPolicy(
        estimator="aipw_ate",
        covariance="iid",
        treatment_position=0,
        propensity_clip=0.02,
        bootstrap_replicates=3,
        seed=59,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )
    original_fit = estimation_module.fit_aipw_ate
    calls = 0

    def fail_bootstrap(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise ValueError("forced bootstrap failure")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(estimation_module, "fit_aipw_ate", fail_bootstrap)
    with pytest.raises(
        ValueError,
        match="at least two successful replicates; received 0 of 3",
    ):
        estimate_parameters(features, outcome, causal_estimand(), policy)


def test_aipw_cross_fitting_validates_policy_and_fold_support():
    """Cross-fitting fails closed for unsupported folds and treatment support."""
    with pytest.raises(ValueError, match="positive integer"):
        EstimationPolicy(
            estimator="aipw_ate",
            treatment_position=0,
            cross_fitting_folds=0,
            causal_assumptions=(
                "consistency",
                "conditional_exchangeability",
                "positivity",
            ),
        )
    with pytest.raises(ValueError, match="only by aipw_ate"):
        EstimationPolicy(estimator="linear", cross_fitting_folds=2)

    treatment = np.tile((0.0, 1.0), 4)
    signal = np.arange(8, dtype=float)
    features = np.column_stack((treatment, signal, signal**2))
    outcome = signal
    policy = EstimationPolicy(
        estimator="aipw_ate",
        covariance="iid",
        treatment_position=0,
        cross_fitting_folds=4,
        causal_assumptions=(
            "consistency",
            "conditional_exchangeability",
            "positivity",
        ),
    )

    with pytest.raises(ValueError, match="both treatment arms"):
        estimate_parameters(features, outcome, causal_estimand(), policy)


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
    assert result.policy.hierarchical_method == "moment_gls"
    assert result.reference["passed"] is True
    json.dumps(result.to_dict())


def make_hierarchical_reml_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Return deterministic grouped data with a non-zero random intercept."""
    rng = np.random.default_rng(41)
    clusters = np.repeat(np.arange(8), 12)
    signal = rng.normal(size=len(clusters))
    random_intercepts = rng.normal(scale=0.8, size=8)
    outcome = (
        1.2
        + 0.65 * signal
        + random_intercepts[clusters]
        + rng.normal(scale=0.35, size=len(clusters))
    )
    return signal.reshape(-1, 1), outcome, clusters


def make_hierarchical_reml_policy() -> EstimationPolicy:
    """Return the bounded REML policy used by grouped estimator tests."""
    return EstimationPolicy(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster_name="site_id",
        hierarchical_assumptions=(
            "random_intercept",
            "cluster_independence",
            "random_effect_exogeneity",
            "constant_residual_variance",
        ),
        hierarchical_method="reml",
    )


def test_hierarchical_linear_selects_bounded_reml_method():
    """REML is explicit and records its bounded optimizer contract."""
    features, outcome, clusters = make_hierarchical_reml_fixture()
    result = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        make_hierarchical_reml_policy(),
        clusters=clusters,
    )

    assert result.policy.hierarchical_method == "reml"
    assert result.diagnostics["method"] == "bounded_reml_gls"
    assert result.diagnostics["optimizer"]["method"] == "L-BFGS-B"
    assert result.diagnostics["optimizer"]["success"] is True
    assert result.diagnostics["variance_component_uncertainty"] == (
        "not_propagated"
    )
    assert any(
        "random slopes" in limitation
        for limitation in result.diagnostics["limitations"]
    )
    assert result.reference["passed"] is True


def make_hierarchical_random_slope_fixture() -> tuple[
    np.ndarray, np.ndarray, np.ndarray
]:
    """Return deterministic data with a declared random signal slope."""
    rng = np.random.default_rng(151)
    clusters = np.repeat(np.arange(14), 20)
    signal = rng.normal(size=len(clusters))
    random_intercepts = rng.normal(scale=0.75, size=14)
    random_slopes = rng.normal(scale=0.35, size=14)
    outcome = (
        1.1
        + 0.7 * signal
        + random_intercepts[clusters]
        + random_slopes[clusters] * signal
        + rng.normal(scale=0.3, size=len(clusters))
    )
    return signal.reshape(-1, 1), outcome, clusters


def make_hierarchical_random_slope_policy(
    method: Literal["moment_gls", "reml"] = "moment_gls",
) -> EstimationPolicy:
    """Return a policy declaring one random slope for the signal feature."""
    return EstimationPolicy(
        estimator="hierarchical_linear",
        covariance="iid",
        cluster_name="site_id",
        hierarchical_assumptions=(
            "random_intercept",
            "random_slope",
            "cluster_independence",
            "random_effect_exogeneity",
            "constant_residual_variance",
        ),
        hierarchical_method=method,
        random_slope_position=0,
    )


def test_hierarchical_linear_supports_one_declared_random_slope():
    """Moment GLS records the declared slope covariance and limitations."""
    features, outcome, clusters = make_hierarchical_random_slope_fixture()
    result = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        make_hierarchical_random_slope_policy(),
        clusters=clusters,
    )

    assert result.parameters[1].estimate == pytest.approx(0.7, abs=0.15)
    assert result.policy.random_slope_position == 0
    assert result.diagnostics["estimator_family"] == (
        "hierarchical_random_intercept_slope"
    )
    assert result.diagnostics["random_slope_position"] == 0
    assert result.diagnostics["random_slope_variance"] >= 0.0
    assert len(result.diagnostics["random_effect_covariance"]) == 2
    assert result.diagnostics["variance_component_uncertainty"] == (
        "not_propagated"
    )
    assert (
        "only one random slope is supported"
        in result.diagnostics["limitations"]
    )
    assert (
        "fully Bayesian multilevel inference is unsupported"
        in result.diagnostics["limitations"]
    )
    assert result.reference["method"].endswith(
        "dense_random_intercept_slope_covariance"
    )
    assert result.reference["passed"] is True
    serialized = json.loads(json.dumps(result.to_dict()))
    restored = EstimationResult.from_dict(serialized)
    assert restored.to_dict() == result.to_dict()


def test_hierarchical_linear_random_slope_supports_bounded_reml():
    """Bounded REML handles the one-slope covariance block independently."""
    features, outcome, clusters = make_hierarchical_random_slope_fixture()
    result = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        make_hierarchical_random_slope_policy("reml"),
        clusters=clusters,
    )

    assert result.diagnostics["method"] == "bounded_reml_gls"
    assert result.diagnostics["optimizer"]["success"] is True
    assert result.diagnostics["optimizer"]["correlation_bounds"] == [
        -0.999999,
        0.999999,
    ]
    assert result.reference["passed"] is True
    assert result.reference["objective_difference"] < 1e-5


def test_hierarchical_random_slope_policy_is_strictly_declared():
    """Slope position cannot be silently broadened or left out of bounds."""
    with pytest.raises(ValueError, match="non-negative integer"):
        EstimationPolicy(
            estimator="hierarchical_linear",
            cluster_name="site_id",
            random_slope_position=True,
        )
    with pytest.raises(ValueError, match="only by hierarchical_linear"):
        EstimationPolicy(estimator="linear", random_slope_position=0)
    features, outcome, clusters = make_hierarchical_random_slope_fixture()
    with pytest.raises(ValueError, match="outside the feature matrix"):
        estimate_parameters(
            features,
            outcome,
            hierarchical_estimand(),
            EstimationPolicy(
                **{
                    **make_hierarchical_random_slope_policy().to_dict(),
                    "random_slope_position": 1,
                }
            ),
            clusters=clusters,
        )


def test_hierarchical_linear_reml_is_deterministic():
    """Identical inputs and policy produce identical REML evidence."""
    features, outcome, clusters = make_hierarchical_reml_fixture()
    first = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        make_hierarchical_reml_policy(),
        clusters=clusters,
    )
    second = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        make_hierarchical_reml_policy(),
        clusters=clusters,
    )

    assert first.to_dict() == second.to_dict()


@pytest.mark.parametrize("method", ["moment_gls", "reml"])
def test_hierarchical_bootstrap_is_deterministic_and_reference_checked(
    method: Literal["moment_gls", "reml"],
):
    """Grouped bootstrap preserves the declared method and evidence contract."""
    features, outcome, clusters = make_hierarchical_reml_fixture()
    base_policy = make_hierarchical_reml_policy()
    policy = EstimationPolicy(
        **{
            **base_policy.to_dict(),
            "hierarchical_method": method,
            "bootstrap_replicates": 4,
            "seed": 107,
        }
    )

    first = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        policy,
        clusters=clusters,
    )
    second = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        policy,
        clusters=clusters,
    )
    sampling = first.sensitivity["sampling"]

    assert first.to_dict() == second.to_dict()
    assert sampling["status"] == "available"
    assert sampling["method"] == "nonparametric_percentile_bootstrap"
    assert sampling["resampling_unit"] == "cluster"
    assert sampling["estimator_method"] == method
    assert sampling["source_unit_count"] == 8
    assert sampling["source_row_count"] == len(outcome)
    assert sampling["cluster_identity_policy"] == (
        "fresh_identity_per_selected_cluster"
    )
    assert sampling["successful_replicates"] == 4
    assert sampling["failed_replicates"] == 0
    assert sampling["reference_checks"] == {
        "successful": 4,
        "failed": 0,
        "requirement": "every accepted replicate passes its independent "
        "point-estimate reference check",
    }
    assert all(draw["sampled_units"] == 8 for draw in sampling["draws"])
    assert all(len(draw["sample_digest"]) == 64 for draw in sampling["draws"])
    assert all(
        len(draw["fresh_cluster_identity_digest"]) == 64
        for draw in sampling["draws"]
    )
    assert all(draw["estimator_method"] == method for draw in sampling["draws"])
    assert len(sampling["bootstrap_intervals"]) == 2
    assert all(
        len(interval) == len(first.parameters)
        for interval in sampling["bootstrap_intervals"]
    )
    assert (
        sampling["bootstrap_intervals"][0] <= sampling["bootstrap_intervals"][1]
    )
    assert sampling["limitations"] == [
        "variance-component uncertainty is not propagated",
        "random slopes are unsupported",
        "nested effects are unsupported",
        "crossed effects are unsupported",
    ]


def test_hierarchical_bootstrap_records_failed_replicates(
    monkeypatch: pytest.MonkeyPatch,
):
    """Invalid grouped draws are recorded and excluded from intervals."""
    features, outcome, clusters = make_hierarchical_reml_fixture()
    policy = EstimationPolicy(
        **{
            **make_hierarchical_reml_policy().to_dict(),
            "bootstrap_replicates": 4,
            "seed": 109,
        }
    )
    original_fit = estimation_module.fit_hierarchical_linear
    calls = 0

    def fail_one_replicate(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("forced invalid hierarchical bootstrap replicate")
        return original_fit(*args, **kwargs)

    monkeypatch.setattr(
        estimation_module, "fit_hierarchical_linear", fail_one_replicate
    )
    result = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        policy,
        clusters=clusters,
    )
    sampling = result.sensitivity["sampling"]

    assert sampling["status"] == "partial"
    assert sampling["successful_replicates"] == 3
    assert sampling["failed_replicates"] == 1
    assert sampling["failures"][0]["replicate"] == 0
    assert sampling["failures"][0]["error_type"] == "ValueError"
    assert sampling["failures"][0]["message"] == (
        "forced invalid hierarchical bootstrap replicate"
    )
    assert sampling["reference_checks"]["successful"] == 3
    assert sampling["reference_checks"]["failed"] == 1


def test_hierarchical_linear_rejects_invalid_reml_optimizer_state(
    monkeypatch: pytest.MonkeyPatch,
):
    """A failed optimizer cannot be converted into a release artifact."""
    features, outcome, clusters = make_hierarchical_reml_fixture()

    def return_failed_optimizer(*args: object, **kwargs: object) -> object:
        del args, kwargs
        return SimpleNamespace(
            success=False,
            status=2,
            message="forced optimizer failure",
            x=np.array([0.0, 0.0]),
            fun=1.0,
            jac=np.array([0.0, 0.0]),
            nit=1,
            nfev=1,
        )

    monkeypatch.setattr(
        "scml.evaluation.parameter_estimation.optimize.minimize",
        return_failed_optimizer,
    )
    with pytest.raises(ValueError, match="REML optimizer did not converge"):
        estimate_parameters(
            features,
            outcome,
            hierarchical_estimand(),
            make_hierarchical_reml_policy(),
            clusters=clusters,
        )


def test_hierarchical_linear_reml_serializes_without_losing_policy(
    tmp_path,
):
    """A REML report round-trips its method and bounded policy."""
    features, outcome, clusters = make_hierarchical_reml_fixture()
    result = estimate_parameters(
        features,
        outcome,
        hierarchical_estimand(),
        make_hierarchical_reml_policy(),
        clusters=clusters,
    )
    path = tmp_path / "scml-reml-estimation-report.json"
    result.write(path)
    restored = EstimationResult.read(path)

    assert restored.to_dict() == result.to_dict()
    assert restored.policy.hierarchical_method == "reml"
    assert restored.policy.reml_variance_bounds == (
        1e-12,
        1e12,
    )


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


def test_estimation_validates_vector_and_cluster_shapes_before_fitting():
    """Malformed row shapes fail before a model can consume ambiguous data."""
    signal = np.arange(6, dtype=float)
    features = np.column_stack((signal, signal**2))
    outcome = np.arange(6, dtype=float)
    policy = EstimationPolicy(estimator="linear")

    with pytest.raises(ValueError, match=r"outcome must have shape \(rows,\)"):
        estimate_parameters(
            features,
            outcome.reshape(-1, 1),
            linear_estimand(),
            policy,
        )

    with pytest.raises(ValueError, match=r"clusters must have shape \(rows,\)"):
        estimate_parameters(
            features,
            outcome,
            linear_estimand(),
            policy,
            clusters=np.zeros((6, 1)),
        )


def test_covariance_conditioning_is_reported_and_nonfinite_shift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
):
    """Release evidence records covariance identifiability and finite shifts."""
    signal = np.arange(6, dtype=float)
    features = np.column_stack((signal, signal**2))
    outcome = 1.0 + features[:, 0] - 0.5 * features[:, 1]

    def return_zero_covariance(
        *arguments: object, **keywords: object
    ) -> np.ndarray:
        del arguments, keywords
        return np.zeros((3, 3))

    monkeypatch.setattr(
        "scml.evaluation.parameter_estimation.covariance_matrix",
        return_zero_covariance,
    )
    result = estimate_parameters(
        features,
        outcome,
        linear_estimand(),
        EstimationPolicy(estimator="linear"),
    )

    covariance_conditioning = result.diagnostics["covariance_conditioning"]
    assert covariance_conditioning["status"] == "rank_deficient"
    assert covariance_conditioning["rank"] == 0
    assert covariance_conditioning["condition_number"] is None

    with pytest.raises(ValueError, match="shift_reference must contain only"):
        estimate_parameters(
            features,
            outcome,
            linear_estimand(),
            EstimationPolicy(estimator="linear"),
            shift_reference=np.array([[0.0, 1.0], [np.nan, 2.0], [2.0, 3.0]]),
        )


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
