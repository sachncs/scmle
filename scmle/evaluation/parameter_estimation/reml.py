"""Bounded profile REML for random-intercept and random-slope models."""

from collections.abc import Callable
from collections.abc import Mapping
import math
from typing import Any

import numpy
from scipy import linalg
from scipy import optimize

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import INTERCEPT_LIMITS
from scmle.evaluation.parameter_estimation.mixed import columns
from scmle.evaluation.parameter_estimation.mixed import gls
from scmle.evaluation.parameter_estimation.mixed import (
    hierarchical_random_slope_gls_system,
)
from scmle.evaluation.parameter_estimation.mixed import intercept
from scmle.evaluation.parameter_estimation.mixed import marginal
from scmle.evaluation.parameter_estimation.mixed import profile
from scmle.evaluation.parameter_estimation.mixed import settle
from scmle.evaluation.parameter_estimation.mixed import slope


def optimise(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    random_slope_position: int,
    initial_residual_variance: float,
    initial_random_effect_covariance: numpy.ndarray,
    cluster_sizes: list[int],
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits the one-random-slope model with bounded profile REML.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.
        initial_residual_variance: Moment estimate of the residual variance,
            the starting point of the search.
        initial_random_effect_covariance: Moment estimate of the 2x2
            random-effect covariance, the starting point of the search.
        cluster_sizes: Number of rows in each cluster.
        policy: Supplies the variance bounds, iteration limit and tolerance.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary, the residual variance, the random-intercept variance,
        and the 2x2 random-effect covariance matrix at the REML optimum.

    Raises:
        ValueError: If the optimizer result is rejected (see
            `validate_reml_optimizer_result`), the GLS system is not
            identifiable, or the objective rejects a covariance block that
            is not positive definite.
    """
    lower_bound, upper_bound = policy.reml_variance_bounds
    initial_parameters, log_bounds = begin(
        initial_residual_variance,
        initial_random_effect_covariance,
        policy.reml_variance_bounds,
    )
    optimizer_result = search(
        random_slope_reml_profile_objective,
        initial_parameters,
        (design, values, clusters, labels, random_slope_position),
        log_bounds,
        policy,
    )
    point = numpy.asarray(optimizer_result.x, dtype=float)
    variance_components = numpy.exp(point[:3])
    residual_variance = float(variance_components[0])
    random_effect_covariance = random_slope_covariance_from_reml_point(
        variance_components, float(point[3])
    )
    return slope(
        design,
        values,
        clusters,
        (labels, random_slope_position, cluster_sizes),
        (residual_variance, random_effect_covariance),
        "bounded_reml_gls",
        {
            "reml_objective": float(optimizer_result.fun),
            "optimizer": {
                **record(optimizer_result, lower_bound, upper_bound),
                "correlation_bounds": [-0.999999, 0.999999],
            },
        },
    )


def begin(
    initial_residual_variance: float,
    initial_random_effect_covariance: numpy.ndarray,
    bounds: tuple[float, float],
) -> tuple[numpy.ndarray, tuple[tuple[float, float], ...]]:
    """Places the random-slope REML search's start inside its bounds.

    Args:
        initial_residual_variance: Moment estimate of the residual variance.
        initial_random_effect_covariance: Moment estimate of the 2x2
            random-effect covariance.
        bounds: Lower and upper bound of the variance components.

    Returns:
        The starting point (the logs of the three variances and the
        correlation parameter) and the bounds of each parameter.  A
        variance at or below zero starts at the residual variance.
    """
    lower_bound, upper_bound = bounds
    intercept_variance = max(float(initial_random_effect_covariance[0, 0]), 0.0)
    slope_variance = max(float(initial_random_effect_covariance[1, 1]), 0.0)
    if intercept_variance <= 0.0:
        intercept_variance = initial_residual_variance
    if slope_variance <= 0.0:
        slope_variance = initial_residual_variance
    initial_variances = numpy.clip(
        numpy.asarray(
            [
                initial_residual_variance,
                intercept_variance,
                slope_variance,
            ],
            dtype=float,
        ),
        lower_bound,
        upper_bound,
    )
    scale = math.sqrt(intercept_variance * slope_variance)
    initial_correlation = (
        float(initial_random_effect_covariance[0, 1]) / scale
        if scale > 0.0
        else 0.0
    )
    initial_correlation = float(numpy.clip(initial_correlation, -0.999, 0.999))
    initial_parameters = numpy.concatenate(
        (numpy.log(initial_variances), [numpy.arctanh(initial_correlation)])
    )
    log_bounds = (
        (math.log(lower_bound), math.log(upper_bound)),
        (math.log(lower_bound), math.log(upper_bound)),
        (math.log(lower_bound), math.log(upper_bound)),
        (-8.0, 8.0),
    )
    return initial_parameters, log_bounds


def search(
    objective: Callable[..., float],
    start: numpy.ndarray,
    arguments: tuple[Any, ...],
    log_bounds: tuple[tuple[float, float], ...],
    policy: Config,
) -> optimize.OptimizeResult:
    """Minimizes a REML objective with bounded L-BFGS-B and checks the result.

    Args:
        objective: Negative profile REML objective.
        start: Starting point inside the bounds.
        arguments: Further arguments of the objective.
        log_bounds: Lower and upper bound of each parameter.
        policy: Supplies the iteration limit and tolerance.

    Returns:
        The optimizer result.  A result that cannot support a release
        artifact raises the ValueError of `validate_reml_optimizer_result`.
    """
    optimizer_result = optimize.minimize(
        objective,
        start,
        args=arguments,
        method="L-BFGS-B",
        bounds=log_bounds,
        options={
            "maxiter": policy.max_iterations,
            "ftol": policy.tolerance,
            "gtol": policy.tolerance,
        },
    )
    validate_reml_optimizer_result(
        optimizer_result, log_bounds, dimension=len(log_bounds)
    )
    return optimizer_result


def record(
    optimizer_result: optimize.OptimizeResult,
    lower_bound: float,
    upper_bound: float,
) -> Mapping[str, Any]:
    """Returns the auditable record of a REML optimizer run.

    Args:
        optimizer_result: Result returned by the optimizer.
        lower_bound: Lower bound of the variance components.
        upper_bound: Upper bound of the variance components.

    Returns:
        The method, success, status, message, function evaluations,
        gradient norm and variance bounds.
    """
    jacobian = numpy.asarray(optimizer_result.jac, dtype=float)
    return {
        "method": "L-BFGS-B",
        "success": bool(optimizer_result.success),
        "status": int(optimizer_result.status),
        "message": str(optimizer_result.message),
        "function_evaluations": int(optimizer_result.nfev),
        "gradient_norm": float(numpy.linalg.norm(jacobian)),
        "variance_bounds": [lower_bound, upper_bound],
    }


def random_slope_covariance_from_reml_point(
    variance_components: numpy.ndarray,
    correlation_parameter: float,
) -> numpy.ndarray:
    """Constructs a random-effect covariance from a REML parameter point.

    Args:
        variance_components: Array of the residual, random-intercept and
            random-slope variances, in that order.
        correlation_parameter: Unbounded parameter whose scaled hyperbolic
            tangent gives the intercept-slope correlation (at most 0.999999
            in absolute value).

    Returns:
        The 2x2 covariance matrix of the random intercept and slope.
    """
    correlation = 0.999999 * math.tanh(correlation_parameter)
    intercept_variance, slope_variance = variance_components[1:]
    covariance = correlation * math.sqrt(intercept_variance * slope_variance)
    return numpy.asarray(
        [
            [intercept_variance, covariance],
            [covariance, slope_variance],
        ],
        dtype=float,
    )


def random_slope_reml_profile_objective(
    parameters: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    random_slope_position: int,
) -> float:
    """Returns the negative profile REML objective for one random slope.

    Args:
        parameters: Four values: the logs of the residual, random-intercept
            and random-slope variances, then the correlation parameter.
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.

    Returns:
        The objective value to minimize, or infinity when the parameters
        give a variance that is not positive and finite, the information
        matrix or a cluster covariance has a non-positive determinant, or the
        GLS system is singular.

    Raises:
        ValueError: If `parameters` does not have exactly four values, or if
            the GLS system building rejects a covariance block (see
            `precisions`).
    """
    point = numpy.asarray(parameters, dtype=float)
    if point.shape != (4,):
        raise errors.InputError(
            "random-slope REML needs four bounded covariance parameters"
        )
    variance_components = numpy.exp(point[:3])
    if not numpy.isfinite(variance_components).all() or numpy.any(
        variance_components <= 0.0
    ):
        return float("inf")
    random_effect_covariance = random_slope_covariance_from_reml_point(
        variance_components, float(point[3])
    )
    residual_variance = float(variance_components[0])
    information, right_hand_side = hierarchical_random_slope_gls_system(
        design,
        values,
        clusters,
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
    )
    settled = settle(information, right_hand_side)
    if settled is None:
        return float("inf")
    logdet_information, coefficients = settled
    residuals = values - design @ coefficients
    quadratic = 0.0
    logdet_covariance = 0.0
    for label in labels:
        group = clusters == label
        covariance = marginal(
            columns(design[group], random_slope_position),
            residual_variance,
            random_effect_covariance,
        )
        sign_covariance, logdet = numpy.linalg.slogdet(covariance)
        if sign_covariance <= 0.0:
            return float("inf")
        precision = linalg.solve(covariance, numpy.eye(covariance.shape[0]))
        group_residuals = residuals[group]
        quadratic += float(group_residuals @ precision @ group_residuals)
        logdet_covariance += float(logdet)
    return profile(
        design, values, logdet_covariance, logdet_information, quadratic
    )


def fit_hierarchical_reml(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    initial_residual_variance: float,
    initial_random_intercept_variance: float,
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits a bounded random-intercept model with profile REML.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        initial_residual_variance: Moment estimate of the residual variance,
            the starting point of the search.
        initial_random_intercept_variance: Moment estimate of the
            random-intercept variance, the starting point of the search.
        policy: Supplies the variance bounds, iteration limit and tolerance.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary including the optimizer record, the residual variance, the
        random-intercept variance, and the 2x2 random-effect covariance
        (random intercept only; other entries zero).

    Raises:
        ValueError: If the optimizer result is rejected (see
            `validate_reml_optimizer_result`) or the GLS system is not
            identifiable.
    """
    lower_bound, upper_bound = policy.reml_variance_bounds
    initial_variances = numpy.clip(
        numpy.asarray(
            [initial_residual_variance, initial_random_intercept_variance],
            dtype=float,
        ),
        lower_bound,
        upper_bound,
    )
    log_bounds = ((math.log(lower_bound), math.log(upper_bound)),) * 2
    optimizer_result = search(
        deviance,
        numpy.log(initial_variances),
        (design, values, clusters, labels),
        log_bounds,
        policy,
    )
    variance_components = numpy.exp(
        numpy.asarray(optimizer_result.x, dtype=float)
    )
    return intercept(
        design,
        values,
        clusters,
        labels,
        (float(variance_components[0]), float(variance_components[1])),
        (
            int(optimizer_result.nit),
            "bounded_reml_gls",
            "profile_reml_with_l_bfgs_b",
        ),
        {
            "variance_component_uncertainty": "not_propagated",
            "limitations": list(INTERCEPT_LIMITS),
            "reml_objective": float(optimizer_result.fun),
            "optimizer": record(optimizer_result, lower_bound, upper_bound),
        },
    )


def deviance(
    log_variances: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
) -> float:
    """Returns the negative profile REML objective in log-variance space.

    Args:
        log_variances: The logs of the residual and random-intercept
            variances.
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.

    Returns:
        The objective value to minimize, or infinity when a variance is not
        positive and finite, the information matrix has a non-positive
        determinant, or the GLS system is singular.

    Raises:
        ValueError: If `log_variances` does not have exactly two values.
    """
    if numpy.asarray(log_variances).shape != (2,):
        raise errors.InputError(
            "REML needs residual and random-intercept variances"
        )
    variances = numpy.exp(numpy.asarray(log_variances, dtype=float))
    if not numpy.isfinite(variances).all() or numpy.any(variances <= 0.0):
        return float("inf")
    residual_variance, random_intercept_variance = variances
    information, right_hand_side = gls(
        design,
        values,
        clusters,
        labels,
        float(residual_variance),
        float(random_intercept_variance),
    )
    settled = settle(information, right_hand_side)
    if settled is None:
        return float("inf")
    logdet_information, coefficients = settled
    residuals = values - design @ coefficients
    quadratic = 0.0
    logdet_covariance = 0.0
    for label in labels:
        group = clusters == label
        count = int(numpy.sum(group))
        group_residuals = residuals[group]
        group_sum = float(numpy.sum(group_residuals))
        quadratic += float(
            numpy.sum(group_residuals**2) / residual_variance
            - random_intercept_variance
            * group_sum**2
            / (
                residual_variance
                * (residual_variance + count * random_intercept_variance)
            )
        )
        logdet_covariance += (count - 1) * math.log(
            residual_variance
        ) + math.log(residual_variance + count * random_intercept_variance)
    return profile(
        design, values, logdet_covariance, logdet_information, quadratic
    )


def validate_reml_optimizer_result(
    optimizer_result: optimize.OptimizeResult,
    log_bounds: tuple[tuple[float, float], ...],
    dimension: int = 2,
) -> None:
    """Rejects an optimizer result that cannot support a release artifact.

    Args:
        optimizer_result: Result returned by the optimizer.
        log_bounds: Lower and upper bound of each optimized parameter.
        dimension: Expected number of optimized parameters.

    Raises:
        ValueError: If the optimizer did not report success; the status,
            iteration count or evaluation count is invalid; the solution is
            not finite, has the wrong length or lies outside the bounds; the
            objective is not finite; or the gradient is not finite or has
            the wrong length.
    """
    if not bool(getattr(optimizer_result, "success", False)):
        raise errors.InputError(
            "REML optimizer did not converge: "
            f"{getattr(optimizer_result, 'message', 'unknown failure')}"
        )
    status = getattr(optimizer_result, "status", None)
    if isinstance(status, bool) or not isinstance(status, (int, numpy.integer)):
        raise ValueError("REML optimizer returned an invalid status")
    iterations = getattr(optimizer_result, "nit", None)
    evaluations = getattr(optimizer_result, "nfev", None)
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, numpy.integer))
        or iterations < 0
        or isinstance(evaluations, bool)
        or not isinstance(evaluations, (int, numpy.integer))
        or evaluations < 0
    ):
        raise ValueError(
            "REML optimizer returned invalid iteration diagnostics"
        )
    point = numpy.asarray(getattr(optimizer_result, "x", ()), dtype=float)
    if point.shape != (dimension,) or not numpy.isfinite(point).all():
        raise errors.InputError(
            "REML optimizer returned invalid variance components"
        )
    for value, bounds in zip(point, log_bounds, strict=True):
        if value < bounds[0] or value > bounds[1]:
            raise ValueError("REML optimizer returned out-of-bounds components")
    objective = float(getattr(optimizer_result, "fun", float("nan")))
    if not math.isfinite(objective):
        raise errors.InputError("REML optimizer returned an invalid objective")
    gradient = numpy.asarray(getattr(optimizer_result, "jac", ()), dtype=float)
    if gradient.shape != (dimension,) or not numpy.isfinite(gradient).all():
        raise errors.InputError(
            "REML optimizer returned invalid gradient diagnostics"
        )
