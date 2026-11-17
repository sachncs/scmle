"""Prior, missingness, shift and sampling sensitivity records."""

from collections.abc import MutableMapping
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.bayesian import conform
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import BayesianPrior
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.inputs import validate_matrix
from scmle.evaluation.parameter_estimation.linear import logit
from scmle.evaluation.parameter_estimation.linear import regress
from scmle.evaluation.parameter_estimation.resampling import (
    causal_bootstrap_sensitivity,
)
from scmle.evaluation.parameter_estimation.resampling import (
    hierarchical_bootstrap_sensitivity,
)
from scmle.evaluation.parameter_estimation.resampling import intervals


def reprior(
    prior: BayesianPrior,
    design: numpy.ndarray,
    values: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Refits a Bayesian model under weaker and stronger prior precision.

    Args:
        prior: Bayesian prior of the fit.
        design: Design matrix including the intercept column.
        values: Complete outcomes used by the fit.
        coefficients: Point-estimate coefficient vector.
        policy: Supplies the estimator and treatment position.

    Returns:
        The evaluated prior section with the coefficient change under each
        alternative precision.

    Raises:
        ValueError: If the Bayesian causal estimator has no treatment
            position.
    """
    conform(prior, design.shape[1])
    baseline = numpy.asarray(coefficients, dtype=float)
    alternatives = []
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    for label, precision_factor in (
        ("weakened_precision", 0.01),
        ("strengthened_precision", 100.0),
    ):
        alternative_precision = prior_precision * precision_factor
        alternative = numpy.linalg.solve(
            alternative_precision + design.T @ design,
            alternative_precision @ prior_mean + design.T @ values,
        )
        alternative_record = {
            "name": label,
            "max_absolute_coefficient_change": float(
                numpy.max(numpy.abs(alternative - baseline))
            ),
        }
        if policy.estimator == "bayesian_ate":
            if policy.treatment_position is None:
                raise errors.InputError(
                    "bayesian_ate requires treatment_position"
                )
            target_index = policy.treatment_position + 1
            alternative_record["treatment_coefficient_change"] = float(
                alternative[target_index] - baseline[target_index]
            )
        alternatives.append(alternative_record)
    return {
        "status": "evaluated",
        "method": "precision_scale_sensitivity",
        "alternatives": alternatives,
    }


def transport(matrix: numpy.ndarray, shift_reference: Any) -> dict[str, Any]:
    """Compares the fitted rows with a target-population feature matrix.

    Args:
        matrix: Complete feature rows used by the fit.
        shift_reference: Feature matrix from the target population.

    Returns:
        The evaluated distribution-shift section with standardized mean
        differences.

    Raises:
        ValueError: If `shift_reference` has the wrong shape or holds
            non-finite values.
    """
    reference = numpy.asarray(shift_reference, dtype=float)
    validate_matrix(reference, matrix.shape[1], "shift_reference")
    if not numpy.isfinite(reference).all():
        raise errors.InputError(
            "shift_reference must contain only finite values"
        )
    pooled = numpy.sqrt((matrix.var(axis=0) + reference.var(axis=0)) / 2.0)
    difference = numpy.abs(reference.mean(axis=0) - matrix.mean(axis=0)) / (
        numpy.maximum(pooled, 1e-12)
    )
    return {
        "status": "evaluated",
        "standardized_mean_difference": [float(item) for item in difference],
        "max_standardized_mean_difference": float(numpy.max(difference)),
    }


def sensitivity_report(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    design: numpy.ndarray,
    coefficients: numpy.ndarray,
    shift_reference: Any | None,
    estimand: Estimand,
    clusters: numpy.ndarray | None,
    prior: BayesianPrior | None = None,
) -> dict[str, Any]:
    """Records declared sensitivity coverage without inventing causal claims.

    Args:
        matrix: Complete feature rows used by the fit.
        values: Complete outcomes used by the fit.
        policy: Supplies the estimator, missing policy, seed, confidence
            level, bootstrap replicate count and treatment position.
        design: Design matrix including the intercept column.
        coefficients: Point-estimate coefficient vector.
        shift_reference: Optional feature matrix from a target population.
        estimand: Scientific definition of the estimand.
        clusters: Optional cluster labels of the rows used.
        prior: Bayesian prior, when the estimator is Bayesian.

    Returns:
        A dictionary with the sections ``prior``, ``missingness``,
        ``sampling`` and ``distribution_shift``, each recording whether it
        was evaluated and its evidence.

    Raises:
        ValueError: If the Bayesian causal estimator has no treatment
            position, if `shift_reference` has the wrong shape or holds
            non-finite values, if a hierarchical bootstrap is requested
            without cluster labels, or if a causal or hierarchical bootstrap
            yields fewer than two valid replicates.
    """
    report: dict[str, Any] = {
        "prior": {
            "status": "not_applicable",
            "reason": "current estimators are frequentist and use no prior",
        },
        "missingness": {
            "status": "evaluated",
            "policy": policy.missing,
            "rows": len(values),
        },
        "sampling": {
            "status": (
                "available" if policy.bootstrap_replicates else "not_requested"
            ),
            "bootstrap_replicates": policy.bootstrap_replicates,
        },
        "distribution_shift": {
            "status": (
                "not_available" if shift_reference is None else "evaluated"
            )
        },
    }
    if prior is not None:
        report["prior"] = reprior(prior, design, values, coefficients, policy)
    if shift_reference is not None:
        report["distribution_shift"] = transport(matrix, shift_reference)
    sample(
        report,
        (matrix, values, design, coefficients),
        estimand,
        policy,
        clusters,
    )
    return report


def sample(
    report: MutableMapping[str, Any],
    data: tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray],
    estimand: Estimand,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> None:
    """Fills the sampling section of a sensitivity report.

    Causal and hierarchical estimators get their reference-checked
    bootstraps; the linear and logistic estimators get a row bootstrap.

    Args:
        report: The sensitivity report; its ``sampling`` section is set.
        data: The rows, outcomes, design matrix and point-estimate
            coefficients of the fit.
        estimand: Scientific definition of the estimand.
        policy: Supplies the estimator, seed, confidence level, replicate
            count and iteration settings.
        clusters: Optional cluster labels of the rows used.

    Raises:
        ValueError: If a hierarchical bootstrap is requested without cluster
            labels, or a causal or hierarchical bootstrap yields fewer than
            two valid replicates.
    """
    matrix, values, design, coefficients = data
    if not policy.bootstrap_replicates:
        return
    if policy.estimator in {"ipw_ate", "aipw_ate", "continuous_ipw_ate"}:
        report["sampling"] = causal_bootstrap_sensitivity(
            matrix, values, estimand, policy, coefficients, clusters
        )
        return
    if policy.estimator == "hierarchical_linear":
        if clusters is None:
            raise errors.InputError(
                "hierarchical_linear requires cluster labels"
            )
        report["sampling"] = hierarchical_bootstrap_sensitivity(
            matrix, values, estimand, policy, coefficients, clusters
        )
        return
    rng = numpy.random.default_rng(policy.seed)
    estimates: list[numpy.ndarray] = []
    bootstrap_replicate = 0
    while bootstrap_replicate < policy.bootstrap_replicates:
        rows = rng.integers(0, len(values), size=len(values))
        if policy.estimator == "linear":
            linear_fit = regress(design[rows], values[rows], 1, 1e-8)
            estimate = linear_fit[0]
        else:
            logistic_fit = logit(
                design[rows],
                values[rows],
                policy.max_iterations,
                policy.tolerance,
            )
            estimate = logistic_fit[0]
        estimates.append(estimate)
        bootstrap_replicate += 1
    report["sampling"]["bootstrap_intervals"] = intervals(
        estimates, policy.confidence_level
    )
