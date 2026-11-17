"""Inverse-probability-weighted and augmented IPW treatment effects.

Calls written `parameter_estimation.<name>(...)` look the function up on
the package when they run, so a test that patches the package attribute
reaches them as it did when the package was one module.
"""

from typing import Any

import numpy
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation import parameter_estimation
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.crossfitting import Nuisance
from scmle.evaluation.parameter_estimation.crossfitting import nuisance
from scmle.evaluation.parameter_estimation.crossfitting import propensities
from scmle.evaluation.parameter_estimation.declaration import CLIP_TOLERANCE
from scmle.evaluation.parameter_estimation.declaration import MIN_ROWS_PER_ARM
from scmle.evaluation.parameter_estimation.references import reference_aipw_ate
from scmle.evaluation.parameter_estimation.references import reference_ipw_ate
from scmle.evaluation.parameter_estimation.report import Fit
from scmle.evaluation.parameter_estimation.weighting import aipw
from scmle.evaluation.parameter_estimation.weighting import dispersion
from scmle.evaluation.parameter_estimation.weighting import hajek


def fit_propensity_estimator(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    cluster_values: numpy.ndarray | None,
) -> Fit:
    """Fits the inverse-probability-weighted or doubly robust effect.

    Args:
        matrix: Feature matrix of the kept rows.
        values: Outcome vector.
        policy: Estimator policy, `ipw_ate` or `aipw_ate`.
        cluster_values: Cluster labels of the kept rows, or None.

    Returns:
        The fitted quantities for the single parameter `ate`.
    """
    if policy.estimator == "ipw_ate":
        coefficients, covariance, diagnostics, propensity_design = (
            parameter_estimation.fit_ipw_ate(  # type: ignore[attr-defined]
                matrix,
                values,
                policy.treatment_position,
                policy,
                cluster_values,
            )
        )
        reference = reference_ipw_ate(
            propensity_design,
            values,
            matrix[:, policy.treatment_position],
            coefficients,
            policy,
            cluster_values,
        )
    else:
        (
            coefficients,
            covariance,
            diagnostics,
            aipw_inputs,
        ) = parameter_estimation.dr(  # type: ignore[attr-defined]
            matrix,
            values,
            policy.treatment_position,
            policy,
            cluster_values,
        )
        reference = reference_aipw_ate(
            aipw_inputs,
            values,
            coefficients,
            policy,
            cluster_values,
        )
    likelihood = "gaussian"
    pivot = stats.norm.ppf(0.5 + policy.confidence_level / 2.0)
    p_values = 2.0 * stats.norm.sf(
        numpy.abs(coefficients)
        / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
    )
    diagnostics = {**diagnostics, "likelihood": likelihood}
    return Fit(
        ("ate",),
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def arms(
    matrix: numpy.ndarray, treatment_position: int | None, estimator: str
) -> tuple[int, numpy.ndarray, int, int]:
    """Checks the binary treatment of a propensity-weighted estimator.

    Args:
        matrix: Complete numeric feature matrix, including the treatment.
        treatment_position: Column of the 0/1 treatment.
        estimator: Estimator name used in the error messages.

    Returns:
        The treatment position, the treatment column, and the numbers of
        treated and of control rows.

    Raises:
        ValueError: If `treatment_position` is None, the treatment is not
            0/1, or either arm has fewer than two rows.
    """
    if treatment_position is None:
        raise errors.InputError(f"{estimator} requires treatment_position")
    treatment = matrix[:, treatment_position]
    if not set(numpy.unique(treatment)).issubset({0.0, 1.0}):
        raise errors.InputError(
            f"{estimator} treatment must contain only 0 and 1"
        )
    treated_count = int(numpy.sum(treatment == 1.0))
    control_count = int(numpy.sum(treatment == 0.0))
    if treated_count < MIN_ROWS_PER_ARM or control_count < MIN_ROWS_PER_ARM:
        raise errors.InputError(
            f"{estimator} needs at least two rows per treatment arm"
        )
    return treatment_position, treatment, treated_count, control_count


def confound(matrix: numpy.ndarray, treatment_position: int) -> numpy.ndarray:
    """Returns the non-treatment features behind an intercept column.

    Args:
        matrix: Complete numeric feature matrix, including the treatment.
        treatment_position: Column of the treatment, left out.

    Returns:
        The design matrix of the propensity and outcome nuisance models.
    """
    confounders = numpy.delete(matrix, treatment_position, axis=1)
    design: numpy.ndarray = numpy.column_stack(
        (numpy.ones(confounders.shape[0]), confounders)
    )
    return design


def positivity(
    raw_propensity: numpy.ndarray, propensity: numpy.ndarray, clip: float
) -> dict[str, str | float]:
    """Records the propensity range before and after clipping.

    Args:
        raw_propensity: Unclipped propensity score of every row.
        propensity: Clipped propensity score of every row.
        clip: The clip threshold of the policy.

    Returns:
        The positivity record: diagnosed, not proven.
    """
    clipped = numpy.abs(raw_propensity - propensity) > CLIP_TOLERANCE
    return {
        "status": "diagnosed_not_proven",
        "raw_min": float(numpy.min(raw_propensity)),
        "raw_max": float(numpy.max(raw_propensity)),
        "clipped_min": float(numpy.min(propensity)),
        "clipped_max": float(numpy.max(propensity)),
        "clip_threshold": clip,
        "fraction_clipped": float(numpy.mean(clipped)),
    }


def fit_ipw_ate(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int | None,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, Any], numpy.ndarray]:
    """Estimates a binary-treatment Hájek inverse-probability ATE.

    Args:
        matrix: Complete numeric feature matrix, including the treatment.
        values: Complete numeric outcomes.
        treatment_position: Column of the 0/1 treatment.
        policy: Supplies the propensity clip, iteration limit, tolerance and
            covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        The one-element ATE vector, its 1x1 variance matrix, a diagnostics
        dictionary, and the propensity design matrix (intercept plus the
        non-treatment features).

    Raises:
        ValueError: If `treatment_position` is None, the treatment is not
            0/1, or either arm has fewer than two rows.  Errors from the
            propensity fit and variance calculation also propagate.
    """
    position, treatment, treated_count, control_count = arms(
        matrix, treatment_position, "ipw_ate"
    )
    propensity_design = confound(matrix, position)
    propensity, raw_propensity, propensity_diagnostics = propensities(
        propensity_design, treatment, propensity_design, policy
    )
    ate, variance, evidence = hajek(
        treatment,
        values,
        propensity,
        clusters,
        policy.covariance,
    )
    diagnostics = {
        "converged": True,
        "iterations": propensity_diagnostics["iterations"],
        "estimator_family": "causal_ipw",
        "method": "hajek_ipw",
        "uncertainty_method": (
            "influence_function_with_estimated_propensity"
            if policy.covariance == "iid"
            else evidence["uncertainty_method"]
        ),
        "propensity_estimation_uncertainty": "not_propagated",
        "treatment_position": position,
        "treated_rows": treated_count,
        "control_rows": control_count,
        "propensity_model": {
            "method": "logistic_mle",
            "converged": propensity_diagnostics["converged"],
            "iterations": propensity_diagnostics["iterations"],
            "log_likelihood": propensity_diagnostics["log_likelihood"],
        },
        "positivity": positivity(
            raw_propensity, propensity, policy.propensity_clip
        ),
        "effective_sample_size": evidence["effective_sample_size"],
        "treatment_means": evidence["treatment_means"],
        "influence_function": {
            "method": "centered_ipw_influence_function",
            "variance": variance,
            "variance_method": evidence["uncertainty_method"],
        },
    }
    if policy.covariance == "clustered":
        diagnostics.update(evidence["uncertainty"]["cluster_diagnostics"])
    return (
        numpy.asarray([ate]),
        numpy.asarray([[variance]]),
        diagnostics,
        propensity_design,
    )


def dr(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int | None,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[
    numpy.ndarray, numpy.ndarray, dict[str, Any], dict[str, numpy.ndarray]
]:
    """Estimates a binary-treatment ATE with an augmented IPW score.

    With one fold, nuisance models are fitted on the full sample and scored
    on the same rows. With multiple folds, each nuisance
    model is fitted only on the corresponding training rows and its score is
    evaluated on held-out rows.

    Args:
        matrix: Complete numeric feature matrix, including the treatment.
        values: Complete numeric outcomes.
        treatment_position: Column of the 0/1 treatment.
        policy: Supplies the number of cross-fitting folds, the propensity
            clip, iteration limit, tolerance and covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        The one-element ATE vector, its 1x1 variance matrix, a diagnostics
        dictionary, and a dictionary of the arrays the reference calculation
        needs (nuisance design, treatment, predicted control and treated
        outcomes, and fold assignments).

    Raises:
        ValueError: If `treatment_position` is None, the treatment is not
            0/1, or either arm has fewer than two rows.  Errors from the fold
            assignment, support check, nuisance fits and variance calculation
            also propagate.
    """
    position, treatment, treated_count, control_count = arms(
        matrix, treatment_position, "aipw_ate"
    )
    nuisance_design = confound(matrix, position)
    fitted = nuisance(nuisance_design, treatment, values, policy)
    ate, variance, diagnostics = augment(
        fitted,
        treatment,
        values,
        policy,
        clusters,
        (position, treated_count, control_count),
    )
    return (
        numpy.asarray([ate]),
        numpy.asarray([[variance]]),
        diagnostics,
        {
            "nuisance_design": nuisance_design,
            "treatment": treatment,
            "outcome_control": fitted.control,
            "outcome_treated": fitted.treated,
            "fold_assignments": fitted.folds,
        },
    )


def augment(
    fitted: Nuisance,
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    clusters: numpy.ndarray | None,
    counts: tuple[int, int, int],
) -> tuple[float, float, dict[str, Any]]:
    """Scores the augmented IPW estimate and records its evidence.

    Args:
        fitted: The nuisance predictions for every row.
        treatment: 0/1 treatment of every row.
        values: Outcome of every row.
        policy: Supplies the folds, propensity clip and covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.
        counts: The treatment position and the treated and control rows.

    Returns:
        The ATE, its variance, and the diagnostics dictionary.  Errors from
        the variance calculation propagate.
    """
    score = aipw(
        fitted.treated, fitted.control, treatment, values, fitted.propensity
    )
    ate = float(numpy.mean(score))
    influence = score - ate
    variance, uncertainty = dispersion(influence, policy.covariance, clusters)
    position, treated_count, control_count = counts
    diagnostics = {
        "converged": True,
        "iterations": fitted.diagnostics["propensity_iterations"],
        "estimator_family": "causal_augmented_ipw",
        "method": "augmented_ipw",
        "uncertainty_method": (
            (
                "influence_function_with_cross_fitted_nuisance"
                if policy.cross_fitting_folds > 1
                else "influence_function_with_fitted_nuisance"
            )
            if policy.covariance == "iid"
            else uncertainty["method"]
        ),
        "cross_fitting": fitted.crossfitting,
        "propensity_estimation_uncertainty": "not_propagated",
        "outcome_model_uncertainty": "not_propagated",
        "treatment_position": position,
        "treated_rows": treated_count,
        "control_rows": control_count,
        **fitted.records(),
        "positivity": positivity(
            fitted.raw, fitted.propensity, policy.propensity_clip
        ),
        "influence_function": {
            "method": "centered_augmented_ipw_score",
            "variance": variance,
            "variance_method": uncertainty["method"],
        },
    }
    if policy.covariance == "clustered":
        diagnostics.update(uncertainty["cluster_diagnostics"])
    return ate, variance, diagnostics
