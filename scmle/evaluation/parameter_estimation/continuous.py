"""The continuous-treatment generalized-propensity-weighted effect.

Calls written `parameter_estimation.<name>(...)` look the function up on
the package when they run, so a test that patches the package attribute
reaches them as it did when the package was one module.
"""

from collections.abc import Mapping
import math
from typing import Any

import numpy
from scipy import linalg
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation import parameter_estimation
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import (
    MIN_EFFECTIVE_SAMPLE_SIZE,
)
from scmle.evaluation.parameter_estimation.declaration import (
    MIN_TREATMENT_LEVELS,
)
from scmle.evaluation.parameter_estimation.declaration import SCALE_FLOOR
from scmle.evaluation.parameter_estimation.inputs import clustering
from scmle.evaluation.parameter_estimation.report import Fit

EVIDENCE = (
    "generalized_propensity_density_diagnostics",
    "weight_effective_sample_size",
    "independent_reference_calculation",
)
CAPABILITY_LIMITS = (
    "normal conditional treatment density is required",
    "linear dose-response is required",
    "nuisance-model uncertainty is not propagated",
    ("nonlinear dose-response and continuous-treatment AIPW are unsupported"),
)
LIMITS = (
    "positivity is diagnosed but not proven",
    "normal conditional treatment density is imposed",
    "linear dose-response is imposed",
    "nuisance-model uncertainty is not propagated",
    ("nonlinear dose-response and continuous-treatment AIPW are unsupported"),
)


def fit_continuous_estimator(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    estimand: Estimand,
    cluster_values: numpy.ndarray | None,
) -> Fit:
    """Fits the continuous-treatment inverse-probability-weighted effect.

    Args:
        matrix: Feature matrix of the kept rows.
        values: Outcome vector.
        policy: Estimator policy with a treatment position.
        estimand: Scientific estimand.
        cluster_values: Cluster labels of the kept rows, or None.

    Returns:
        The fitted quantities for the intercept and the dose-response slope.

    Raises:
        ValueError: If the policy has no treatment position.
    """
    if policy.treatment_position is None:
        raise errors.InputError(
            "continuous_ipw_ate requires treatment_position"
        )
    coefficients, covariance, diagnostics = dosage(
        matrix,
        values,
        policy.treatment_position,
        policy,
        cluster_values,
    )
    reference = reference_continuous_ipw_effect(
        matrix,
        values,
        policy.treatment_position,
        policy,
        coefficients,
        cluster_values,
    )
    treatment_name = estimand.feature_names[policy.treatment_position]
    names = ("intercept", f"{treatment_name}_dose_response_slope")
    likelihood = "gaussian"
    pivot = stats.norm.ppf(0.5 + policy.confidence_level / 2.0)
    p_values = 2.0 * stats.norm.sf(
        numpy.abs(coefficients)
        / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
    )
    diagnostics = {
        **diagnostics,
        "likelihood": likelihood,
        "treatment_name": treatment_name,
        "treatment_position": policy.treatment_position,
        "estimand_interpretation": (
            "the slope is the marginal outcome change for a one-unit "
            "increase in continuous treatment under the declared "
            "linear dose-response and generalized-propensity-density "
            "assumptions"
        ),
    }
    return Fit(
        names,
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def fit_treatment_density(
    treatment: numpy.ndarray, confounders: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """Fits the conditional Gaussian density of a continuous treatment.

    Args:
        treatment: The continuous treatment values.
        confounders: The non-treatment feature columns.

    Returns:
        The conditional mean, the residuals and the residual scale of the
        linear treatment model.

    Raises:
        ValueError: If the density design is rank deficient, the model has no
            residual degrees of freedom, or the residual scale is not
            positive.
    """
    density_design = numpy.column_stack(
        (numpy.ones(len(confounders)), confounders)
    )
    if numpy.linalg.matrix_rank(density_design) != density_design.shape[1]:
        raise errors.InputError(
            "continuous treatment density design matrix is rank deficient"
        )
    density_coefficients, residual_sum, rank, singular_values = (
        numpy.linalg.lstsq(density_design, treatment, rcond=None)
    )
    del residual_sum, singular_values
    if rank != density_design.shape[1]:
        raise errors.InputError(
            "continuous treatment density model is not identifiable"
        )
    conditional_mean = density_design @ density_coefficients
    density_residuals = treatment - conditional_mean
    residual_degrees = len(treatment) - density_design.shape[1]
    if residual_degrees < 1:
        raise errors.InputError(
            "continuous treatment density model needs residual degrees of "
            "freedom"
        )
    conditional_scale = float(
        numpy.sqrt(numpy.sum(density_residuals**2) / residual_degrees)
    )
    if not math.isfinite(conditional_scale) or conditional_scale <= SCALE_FLOOR:
        raise errors.InputError(
            "continuous treatment density has non-positive residual scale"
        )
    return conditional_mean, density_residuals, conditional_scale


def dosage(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Estimates a linear continuous-treatment effect with GPS weights.

    The generalized propensity score is a conditional Gaussian treatment
    density estimated from the declared non-treatment features. Stabilized
    density-ratio weights then define a weighted marginal dose-response
    regression. The reported treatment slope is a one-unit marginal effect,
    not a claim about an arbitrary nonlinear dose-response curve. The weights
    are capped at `policy.continuous_weight_cap`.

    Args:
        matrix: Complete numeric feature matrix.
        values: Complete numeric outcomes.
        treatment_position: Column containing the continuous treatment.
        policy: Supplies the weight cap and the covariance kind.
        clusters: Optional independent sampling-unit labels.

    Returns:
        The weighted intercept and dose-response slope, their covariance
        matrix, and a diagnostics dictionary.

    Raises:
        ValueError: If treatment support, density fitting, or weighted
            regression is not identifiable.
    """
    treatment = matrix[:, treatment_position]
    raw_weights, overlap, residual_degrees, conditional_scale = weigh(
        matrix, treatment_position
    )
    weights, weight_diagnostics = cap(raw_weights, policy)
    coefficients, covariance = dose(
        treatment, values, weights, policy, clusters
    )
    diagnostics: dict[str, Any] = {
        "converged": True,
        "iterations": 1,
        "estimator_family": "causal_continuous_ipw",
        "method": "stabilized_generalized_propensity_weighted_regression",
        "uncertainty_method": (
            "weighted_gaussian_regression"
            if policy.covariance == "iid"
            else "clustered_weighted_gaussian_regression"
        ),
        "nuisance_model_uncertainty": "not_propagated",
        "treatment_density_model": {
            "method": "ordinary_least_squares_gaussian_density",
            "conditional_mean_features": "all_declared_non_treatment_features",
            "residual_degrees_of_freedom": residual_degrees,
            "residual_scale": conditional_scale,
        },
        "overlap": overlap,
        "weights": weight_diagnostics,
        "capability": {
            "status": "experimental",
            "name": "continuous_treatment_effect",
            "contract": "linear_marginal_dose_response",
            "assumptions": list(policy.continuous_treatment_assumptions),
            "evidence": list(EVIDENCE),
            "limitations": list(CAPABILITY_LIMITS),
        },
        "limitations": list(LIMITS),
    }
    if policy.covariance == "clustered":
        diagnostics["cluster_uncertainty"] = clustering(clusters)
    return coefficients, covariance, diagnostics


def weigh(
    matrix: numpy.ndarray, treatment_position: int
) -> tuple[numpy.ndarray, Mapping[str, Any], int, float]:
    """Computes the uncapped stabilized density-ratio weights.

    The generalized propensity score is a conditional Gaussian treatment
    density estimated from the non-treatment features; each weight is the
    marginal treatment density over the conditional one.

    Args:
        matrix: Complete numeric feature matrix.
        treatment_position: Column containing the continuous treatment.

    Returns:
        The uncapped weights, the overlap record, the residual degrees of
        freedom and the residual scale of the treatment density model.

    Raises:
        ValueError: If the treatment has fewer than four distinct values or
            no variance, or the weights are not finite and positive.  Errors
            from the density fit propagate.
    """
    treatment = matrix[:, treatment_position]
    unique_treatment = numpy.unique(treatment)
    if len(unique_treatment) < MIN_TREATMENT_LEVELS:
        raise errors.InputError(
            "continuous_ipw_ate needs at least four distinct treatment values"
        )
    marginal_scale = float(numpy.std(treatment, ddof=1))
    if not math.isfinite(marginal_scale) or marginal_scale <= SCALE_FLOOR:
        raise errors.InputError(
            "continuous_ipw_ate needs positive treatment variance"
        )
    confounders = numpy.delete(matrix, treatment_position, axis=1)
    conditional_mean, density_residuals, conditional_scale = (
        fit_treatment_density(treatment, confounders)
    )
    residual_degrees = len(treatment) - (1 + confounders.shape[1])
    log_marginal_density = stats.norm.logpdf(
        treatment, loc=float(numpy.mean(treatment)), scale=marginal_scale
    )
    log_conditional_density = stats.norm.logpdf(
        treatment, loc=conditional_mean, scale=conditional_scale
    )
    log_weight = log_marginal_density - log_conditional_density
    raw_weights = numpy.exp(numpy.clip(log_weight, -745.0, 709.0))
    if not numpy.isfinite(raw_weights).all() or numpy.any(raw_weights <= 0.0):
        raise errors.InputError(
            "continuous treatment density weights are invalid"
        )
    overlap = {
        "status": "diagnosed_not_proven",
        "treatment_min": float(numpy.min(treatment)),
        "treatment_max": float(numpy.max(treatment)),
        "unique_treatment_values": len(unique_treatment),
        "marginal_treatment_mean": float(numpy.mean(treatment)),
        "marginal_treatment_sd": marginal_scale,
        "conditional_treatment_residual_sd": conditional_scale,
        "conditional_standardized_residual_quantiles": [
            float(item)
            for item in numpy.quantile(
                numpy.abs(density_residuals) / conditional_scale,
                (0.5, 0.9, 0.99, 1.0),
            )
        ],
        "conditional_density_min": float(
            numpy.min(numpy.exp(log_conditional_density))
        ),
        "conditional_density_max": float(
            numpy.max(numpy.exp(log_conditional_density))
        ),
    }
    return raw_weights, overlap, residual_degrees, conditional_scale


def cap(
    raw_weights: numpy.ndarray, policy: Config
) -> tuple[numpy.ndarray, Mapping[str, Any]]:
    """Caps the weights and checks their effective sample size.

    Args:
        raw_weights: Uncapped stabilized density-ratio weights.
        policy: Supplies the weight cap.

    Returns:
        The capped weights and the weight record.

    Raises:
        ValueError: If the capped weights leave fewer than two effective
            observations.
    """
    weights = numpy.minimum(raw_weights, policy.continuous_weight_cap)
    capped = raw_weights > policy.continuous_weight_cap
    effective_sample_size = float(
        numpy.sum(weights) ** 2 / numpy.sum(weights**2)
    )
    if effective_sample_size < MIN_EFFECTIVE_SAMPLE_SIZE:
        raise errors.InputError(
            "continuous treatment weights leave fewer than two effective "
            "observations"
        )
    record = {
        "raw_min": float(numpy.min(raw_weights)),
        "raw_max": float(numpy.max(raw_weights)),
        "capped_min": float(numpy.min(weights)),
        "capped_max": float(numpy.max(weights)),
        "cap": policy.continuous_weight_cap,
        "fraction_capped": float(numpy.mean(capped)),
        "quantiles": [
            float(item)
            for item in numpy.quantile(weights, (0.0, 0.5, 0.9, 0.99, 1.0))
        ],
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": effective_sample_size / len(weights),
        "normalization": "stabilized_marginal_over_conditional_density",
    }
    return weights, record


def dose(
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    weights: numpy.ndarray,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Fits the weighted marginal dose-response regression.

    Args:
        treatment: The continuous treatment values.
        values: Complete numeric outcomes.
        weights: Capped stabilized density-ratio weights.
        policy: Supplies the covariance kind.
        clusters: Optional independent sampling-unit labels.

    Returns:
        The weighted intercept and dose-response slope and their covariance
        matrix.

    Raises:
        ValueError: If the weighted design is rank deficient; the
            covariance calculation raises its own errors.
    """
    dose_design = numpy.column_stack((numpy.ones(len(treatment)), treatment))
    weighted_design = dose_design * numpy.sqrt(weights)[:, None]
    weighted_outcome = values * numpy.sqrt(weights)
    if numpy.linalg.matrix_rank(weighted_design) != weighted_design.shape[1]:
        raise errors.InputError(
            "weighted dose-response design is rank deficient"
        )
    coefficients = numpy.linalg.solve(
        weighted_design.T @ weighted_design,
        weighted_design.T @ weighted_outcome,
    )
    weighted_residuals = (values - dose_design @ coefficients) * numpy.sqrt(
        weights
    )
    covariance = parameter_estimation.covariance_matrix(  # type: ignore[attr-defined]
        weighted_design,
        weighted_residuals,
        policy.covariance,
        clusters,
        None,
    )
    return coefficients, covariance


def reference_continuous_ipw_effect(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int,
    policy: Config,
    coefficients: numpy.ndarray,
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Recomputes the GPS-weighted dose-response with dense linear algebra.

    This reference intentionally uses a separate least-squares and normal
    equation calculation from :func:`dosage`. It verifies
    the point estimate, while the primary fit separately records covariance
    and overlap evidence.

    Args:
        matrix: Complete numeric feature matrix.
        values: Complete numeric outcomes.
        treatment_position: Column containing the continuous treatment.
        policy: Supplies the weight cap and the reference tolerance.
        coefficients: Intercept and slope from the primary fit.
        clusters: Optional cluster labels; only their presence is recorded.

    Returns:
        A dictionary with the method name, the largest absolute difference
        from `coefficients`, the reference slope, whether clusters were
        supplied, and ``passed``, True when the difference is within ten
        times the policy's reference tolerance.  The dense solver raises
        its own error if the weighted system is singular or not positive
        definite.
    """
    treatment = matrix[:, treatment_position]
    confounders = numpy.delete(matrix, treatment_position, axis=1)
    density_design = numpy.column_stack(
        (numpy.ones(len(confounders)), confounders)
    )
    density_coefficients = numpy.linalg.lstsq(
        density_design, treatment, rcond=None
    )[0]
    conditional_mean = density_design @ density_coefficients
    residual_degrees = len(treatment) - density_design.shape[1]
    conditional_scale = float(
        numpy.sqrt(
            numpy.sum((treatment - conditional_mean) ** 2)
            / max(residual_degrees, 1)
        )
    )
    marginal_scale = float(numpy.std(treatment, ddof=1))
    log_weights = stats.norm.logpdf(
        treatment, numpy.mean(treatment), marginal_scale
    ) - stats.norm.logpdf(treatment, conditional_mean, conditional_scale)
    weights = numpy.minimum(
        numpy.exp(numpy.clip(log_weights, -745.0, 709.0)),
        policy.continuous_weight_cap,
    )
    dose_design = numpy.column_stack((numpy.ones(len(treatment)), treatment))
    information = dose_design.T @ (weights[:, None] * dose_design)
    right_hand_side = dose_design.T @ (weights * values)
    reference = linalg.solve(information, right_hand_side, assume_a="pos")
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:dense_stabilized_gps_regression",
        "max_absolute_difference": difference,
        "reference_dose_response_slope": float(reference[1]),
        "clustered_reference": clusters is not None,
        "passed": difference <= policy.reference_tolerance * 10,
    }
