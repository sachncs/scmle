"""The conjugate Bayesian linear model and its causal variant."""

from collections.abc import Mapping
import math
from typing import Any

import numpy
from scipy import linalg
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import MIN_ROWS_PER_ARM
from scmle.evaluation.parameter_estimation.declaration import (
    MIN_VARIANCE_DEGREES,
)
from scmle.evaluation.parameter_estimation.declaration import Prior
from scmle.evaluation.parameter_estimation.report import Fit


def dichotomous(matrix: numpy.ndarray, policy: Config) -> tuple[int, int, int]:
    """Checks the binary treatment of the Bayesian average treatment effect.

    Args:
        matrix: Feature matrix of the kept rows.
        policy: Estimator policy with a treatment position.

    Returns:
        The treatment position, the number of treated rows and the number of
        control rows.

    Raises:
        ValueError: If the treatment position is missing, the treatment is
            not 0 or 1, or an arm has fewer than two rows.
    """
    treatment_position = policy.treatment
    if treatment_position is None:
        raise errors.InputError("bayesian_ate requires treatment_position")
    treatment_position = int(treatment_position)
    treatment = matrix[:, treatment_position]
    if not set(numpy.unique(treatment)).issubset({0.0, 1.0}):
        raise errors.InputError(
            "bayesian_ate treatment must contain only 0 and 1"
        )
    treated_rows = int(numpy.sum(treatment == 1.0))
    control_rows = int(numpy.sum(treatment == 0.0))
    if treated_rows < MIN_ROWS_PER_ARM or control_rows < MIN_ROWS_PER_ARM:
        raise errors.InputError(
            "bayesian_ate needs at least two rows per treatment arm"
        )
    return treatment_position, treated_rows, control_rows


def bayes(
    matrix: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    estimand: Estimand,
    names: tuple[str, ...],
) -> Fit:
    """Fits the conjugate Bayesian linear model or its causal variant.

    Args:
        matrix: Feature matrix of the kept rows.
        design: Design matrix with an intercept column.
        values: Outcome vector.
        policy: Estimator policy with an explicit prior.
        estimand: Scientific estimand.
        names: Parameter names.

    Returns:
        The fitted quantities, with posterior standard errors and interval
        scales.

    Raises:
        ValueError: If the policy has no prior, or the treatment of a
            Bayesian average treatment effect is invalid.
    """
    if policy.prior is None:
        raise errors.InputError(
            f"{policy.estimator} estimation requires an explicit prior"
        )
    treatment_position, treated_rows, control_rows = 0, 0, 0
    if policy.estimator == "bayesian_ate":
        treatment_position, treated_rows, control_rows = dichotomous(
            matrix, policy
        )
    coefficients, covariance, diagnostics = update(design, values, policy.prior)
    if policy.estimator == "bayesian_ate":
        reference = reference_bayesian_ate(
            design,
            values,
            policy.prior,
            coefficients,
            policy,
        )
        diagnostics = {
            **diagnostics,
            "estimator_family": "causal_bayesian_gaussian",
            "method": "normal_inverse_gamma_outcome_regression",
            "treatment_position": treatment_position,
            "treatment_name": estimand.features[treatment_position],
            "treated_rows": treated_rows,
            "control_rows": control_rows,
            "identification": (
                "the treatment coefficient is the ATE under the "
                "declared linearity, consistency, conditional "
                "exchangeability, and positivity assumptions"
            ),
            "limitations": [
                "time-varying treatment effects are unsupported",
                (
                    "propensity or other nuisance-model uncertainty is not "
                    "propagated"
                ),
                "clustered covariance is unsupported",
                "fully Bayesian multilevel models are unsupported",
            ],
        }
    else:
        reference = reference_bayesian_linear(
            design, values, policy.prior, coefficients, policy
        )
    return credible(
        names, coefficients, covariance, diagnostics, reference, policy
    )


def credible(
    names: tuple[str, ...],
    coefficients: numpy.ndarray,
    covariance: numpy.ndarray,
    diagnostics: Mapping[str, Any],
    reference: dict[str, Any],
    policy: Config,
) -> Fit:
    """Builds the posterior credible-interval evidence of a Bayesian fit.

    Args:
        names: Parameter names.
        coefficients: Posterior mean of the coefficients.
        covariance: Posterior covariance of the coefficients.
        diagnostics: Posterior diagnostics with the scale matrix and degrees
            of freedom.
        reference: Independent reference calculation.
        policy: Supplies the confidence level.

    Returns:
        The fitted quantities, with posterior standard errors and interval
        scales.
    """
    likelihood = "gaussian"
    posterior_scale = numpy.asarray(diagnostics["posterior_scale"])
    posterior_degrees = diagnostics["posterior_degrees_of_freedom"]
    pivot = stats.t.ppf(0.5 + policy.confidence / 2.0, posterior_degrees)
    p_values = 2.0 * stats.t.sf(
        numpy.abs(coefficients)
        / numpy.maximum(numpy.sqrt(numpy.diag(posterior_scale)), 1e-15),
        posterior_degrees,
    )
    standard_errors = numpy.sqrt(numpy.maximum(numpy.diag(covariance), 0.0))
    interval_errors = numpy.sqrt(
        numpy.maximum(numpy.diag(posterior_scale), 0.0)
    )
    diagnostics = {
        **diagnostics,
        "likelihood": likelihood,
        "uncertainty": "posterior_credible_interval",
        "p_value_semantics": "two-sided posterior tail probability",
        "posterior_covariance": covariance.tolist(),
        "credible_interval": {
            "level": policy.confidence,
            "distribution": "student_t",
            "degrees_of_freedom": posterior_degrees,
            "scale_matrix": posterior_scale.tolist(),
        },
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
        standard_errors,
        interval_errors,
    )


def update(
    design: numpy.ndarray,
    values: numpy.ndarray,
    prior: Prior,
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Computes a conjugate Gaussian posterior from an explicit prior.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        prior: Normal-inverse-gamma prior over the coefficients.

    Returns:
        The posterior mean of the coefficients, their posterior covariance
        (the covariance of the marginal Student-t), and a diagnostics
        dictionary that includes the posterior scale matrix and degrees of
        freedom.

    Raises:
        ValueError: If the prior dimension does not match the design, the
            posterior precision is not positive definite, the posterior
            variance scale is invalid, or the posterior has two or fewer
            degrees of freedom.
    """
    conform(prior, design.shape[1])
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    try:
        factor = linalg.cho_factor(posterior_precision, check_finite=True)
        posterior_mean = linalg.cho_solve(
            factor, prior_precision @ prior_mean + design.T @ values
        )
        posterior_precision_inverse = linalg.cho_solve(
            factor, numpy.eye(design.shape[1])
        )
    except numpy.linalg.LinAlgError as error:
        raise errors.InputError(
            "Bayesian posterior precision is not identifiable"
        ) from error
    posterior_shape = prior.shape + len(values) / 2.0
    posterior_scale = prior.scale + 0.5 * (
        values @ values
        + prior_mean @ prior_precision @ prior_mean
        - posterior_mean @ posterior_precision @ posterior_mean
    )
    if not math.isfinite(posterior_scale) or posterior_scale <= 0.0:
        raise errors.InputError("Bayesian posterior variance scale is invalid")
    posterior_degrees = 2.0 * posterior_shape
    if posterior_degrees <= MIN_VARIANCE_DEGREES:
        raise errors.InputError(
            "Bayesian posterior needs more than two degrees of freedom"
        )
    posterior_covariance = (
        posterior_scale
        / posterior_shape
        * (posterior_degrees / (posterior_degrees - 2.0))
        * posterior_precision_inverse
    )
    posterior_scale_matrix = (
        posterior_scale / posterior_shape
    ) * posterior_precision_inverse
    return (
        posterior_mean,
        posterior_covariance,
        {
            "converged": True,
            "iterations": 1,
            "estimator_family": "bayesian_conjugate",
            "prior_shape": prior.shape,
            "prior_scale": prior.scale,
            "posterior_shape": posterior_shape,
            "posterior_scale": posterior_scale_matrix.tolist(),
            "posterior_degrees_of_freedom": posterior_degrees,
        },
    )


def conform(prior: Prior, dimension: int) -> None:
    """Validates prior vector and matrix dimensions against a design.

    Args:
        prior: Prior to check.
        dimension: Number of design columns, including the intercept.

    Raises:
        ValueError: If the prior mean is not of length `dimension` or the
            precision is not `dimension` by `dimension`.
    """
    mean_shape = numpy.asarray(prior.mean, dtype=float).shape
    precision_shape = numpy.asarray(prior.precision, dtype=float).shape
    expected_shape = (dimension, dimension)
    if mean_shape != (dimension,) or precision_shape != expected_shape:
        raise errors.InputError(
            "Bayesian prior dimension must match intercept and features"
        )


def reference_bayesian_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    prior: Prior,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Verifies the conjugate posterior mean with an independent solver.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        prior: Normal-inverse-gamma prior used by the primary fit.
        coefficients: Posterior mean from the primary fit.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute difference
        from `coefficients`, and ``passed``, True when it is within the
        tolerance.
    """
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    reference = linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + design.T @ values,
        assume_a="pos",
    )
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:positive_definite",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference,
    }


def reference_bayesian_ate(
    design: numpy.ndarray,
    values: numpy.ndarray,
    prior: Prior,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Independently verifies the causal posterior mean and ATE coefficient.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        prior: Normal-inverse-gamma prior used by the primary fit.
        coefficients: Posterior mean from the primary fit.
        policy: Supplies the treatment position and reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute difference
        from `coefficients`, the reference treatment coefficient, the
        treatment position, and ``passed``, True when the difference is within
        the tolerance.

    Raises:
        ValueError: If the prior dimension does not match the design or the
            policy has no treatment position.
    """
    conform(prior, design.shape[1])
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    reference = linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + design.T @ values,
        assume_a="pos",
    )
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    treatment_index = policy.treatment
    if treatment_index is None:
        raise errors.InputError("bayesian_ate requires treatment_position")
    coefficient_index = treatment_index + 1
    return {
        "method": "scipy.linalg.solve:positive_definite_causal_gaussian",
        "max_absolute_difference": difference,
        "reference_treatment_coefficient": float(reference[coefficient_index]),
        "treatment_position": treatment_index,
        "passed": difference <= policy.reference,
    }
