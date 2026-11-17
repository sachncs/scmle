"""Independent dense reference calculations for hierarchical models."""

import math
from typing import Any
from typing import cast

import numpy
from scipy import linalg

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.mixed import columns
from scmle.evaluation.parameter_estimation.mixed import distinct
from scmle.evaluation.parameter_estimation.mixed import marginal


def crosscheck(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    residual_variance: float,
    random_intercept_variance: float,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Verifies hierarchical GLS with explicit group covariance matrices.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        residual_variance: Residual variance of the primary fit.
        random_intercept_variance: Random-intercept variance of the primary
            fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute coefficient
        difference, and ``passed``, True when it is within the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable. A dense system
            that is not positive definite raises `LinAlgError`, a subclass.
    """
    labels = distinct(clusters)
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        count = int(numpy.sum(group))
        covariance = residual_variance * numpy.eye(count) + (
            random_intercept_variance * numpy.ones((count, count))
        )
        precision = linalg.solve(covariance, numpy.eye(count), assume_a="pos")
        group_design = design[group]
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ values[group]
    reference = linalg.solve(information, right_hand_side, assume_a="pos")
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:explicit_random_intercept_covariance",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_hierarchical_reml(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    residual_variance: float,
    random_intercept_variance: float,
    coefficients: numpy.ndarray,
    objective: float,
    policy: Config,
) -> dict[str, Any]:
    """Verifies REML GLS with an independent dense covariance calculation.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        residual_variance: REML residual variance of the primary fit.
        random_intercept_variance: REML random-intercept variance of the
            primary fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        objective: Negative REML objective at the primary optimum.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute coefficient
        difference, the absolute objective difference, and ``passed``, True
        when the coefficients agree within the tolerance and the objectives
        within ten times the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable, or if the dense
            covariance or information matrix is not positive definite
            (scipy raises `LinAlgError`, a subclass, before the
            determinant sign check can).
    """
    covariance_blocks = []
    for label in distinct(clusters):
        count = int(numpy.sum(clusters == label))
        covariance_blocks.append(
            residual_variance * numpy.eye(count)
            + random_intercept_variance * numpy.ones((count, count))
        )
    coefficient_difference, objective_difference = dense(
        design, values, covariance_blocks, coefficients, objective
    )
    return {
        "method": "scipy.linalg.solve:dense_random_intercept_covariance",
        "max_absolute_difference": coefficient_difference,
        "objective_difference": objective_difference,
        "passed": bool(
            coefficient_difference <= policy.reference_tolerance
            and objective_difference <= policy.reference_tolerance * 10
        ),
    }


def reference_hierarchical_random_slope(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Verifies one-random-slope GLS with an independent dense calculation.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: Residual variance of the primary fit.
        random_effect_covariance: 2x2 covariance of the random intercept and
            slope from the primary fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the slope position, the largest
        absolute coefficient difference, the random-effect covariance, and
        ``passed``, True when the difference is within the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable. A dense system
            that is not positive definite raises `LinAlgError`, a subclass.
    """
    labels = distinct(clusters)
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        group_design = design[group]
        count = group_design.shape[0]
        covariance = marginal(
            columns(group_design, random_slope_position),
            residual_variance,
            random_effect_covariance,
        )
        precision = linalg.solve(covariance, numpy.eye(count), assume_a="pos")
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ values[group]
    reference = linalg.solve(information, right_hand_side, assume_a="pos")
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:dense_random_intercept_slope_covariance",
        "random_slope_position": random_slope_position,
        "max_absolute_difference": difference,
        "random_effect_covariance": random_effect_covariance.tolist(),
        "passed": difference <= policy.reference_tolerance,
    }


def reference_hierarchical_random_slope_reml(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
    coefficients: numpy.ndarray,
    objective: float,
    policy: Config,
) -> dict[str, Any]:
    """Verifies random-slope REML with independent dense block algebra.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: REML residual variance of the primary fit.
        random_effect_covariance: 2x2 REML covariance of the random
            intercept and slope from the primary fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        objective: Negative REML objective at the primary optimum.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the slope position, the largest
        absolute coefficient difference, the absolute objective difference,
        the random-effect covariance, and ``passed``, True when the
        coefficients agree within the tolerance and the objectives within
        ten times the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable, or if the dense
            covariance or information matrix is not positive definite
            (scipy raises `LinAlgError`, a subclass, before the
            determinant sign check can).
    """
    covariance_blocks = [
        marginal(
            columns(design[clusters == label], random_slope_position),
            residual_variance,
            random_effect_covariance,
        )
        for label in distinct(clusters)
    ]
    coefficient_difference, objective_difference = dense(
        design, values, covariance_blocks, coefficients, objective
    )
    return {
        "method": "scipy.linalg.solve:dense_random_intercept_slope_covariance",
        "random_slope_position": random_slope_position,
        "max_absolute_difference": coefficient_difference,
        "objective_difference": objective_difference,
        "random_effect_covariance": random_effect_covariance.tolist(),
        "passed": bool(
            coefficient_difference <= policy.reference_tolerance
            and objective_difference <= policy.reference_tolerance * 10
        ),
    }


def dense(
    design: numpy.ndarray,
    values: numpy.ndarray,
    covariance_blocks: list[numpy.ndarray],
    coefficients: numpy.ndarray,
    objective: float,
) -> tuple[float, float]:
    """Recomputes GLS and the REML objective from a dense covariance.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        covariance_blocks: Marginal covariance of each cluster, in label
            order.
        coefficients: Fixed-effect coefficients from the primary fit.
        objective: Negative REML objective at the primary optimum.

    Returns:
        The largest absolute coefficient difference and the absolute
        objective difference from the primary fit.

    Raises:
        ValueError: If the dense covariance or information matrix is not
            positive definite (scipy raises `LinAlgError`, a subclass,
            before the determinant sign check can).
    """
    covariance = cast("numpy.ndarray", linalg.block_diag(*covariance_blocks))
    precision = linalg.solve(covariance, numpy.eye(len(values)), assume_a="pos")
    # scipy.linalg.solve is typed as returning a generic array; the product
    # is an ndarray at runtime.
    information = cast("numpy.ndarray", design.T @ precision @ design)
    right_hand_side = design.T @ precision @ values
    reference_coefficients = linalg.solve(
        information, right_hand_side, assume_a="pos"
    )
    residuals = values - design @ reference_coefficients
    sign_covariance, logdet_covariance = numpy.linalg.slogdet(covariance)
    sign_information, logdet_information = numpy.linalg.slogdet(information)
    if sign_covariance <= 0.0 or sign_information <= 0.0:
        raise errors.InputError(
            "reference REML covariance is not positive definite"
        )
    reference_objective = 0.5 * (
        logdet_covariance
        + logdet_information
        + float(residuals @ precision @ residuals)
        + (len(values) - design.shape[1]) * math.log(2.0 * math.pi)
    )
    coefficient_difference = float(
        numpy.max(numpy.abs(reference_coefficients - coefficients))
    )
    return coefficient_difference, float(abs(reference_objective - objective))
