"""Generalized least-squares systems and diagnostics for random effects."""

from collections.abc import Mapping
import math
from typing import Any

import numpy
from scipy import linalg

from scmle.contracts import errors


def project(
    covariance: numpy.ndarray,
) -> numpy.ndarray:
    """Projects a moment covariance to the positive-semidefinite cone.

    Args:
        covariance: Square moment estimate of the random-effect covariance.

    Returns:
        The symmetrized matrix with negative eigenvalues replaced by zero.
    """
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = numpy.linalg.eigh(symmetric)
    projected = eigenvectors @ numpy.diag(numpy.maximum(eigenvalues, 0.0))
    projected = projected @ eigenvectors.T
    symmetrized: numpy.ndarray = (projected + projected.T) / 2.0
    return symmetrized


def assemble(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Builds the GLS system for one random slope and intercept.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: Residual variance.
        random_effect_covariance: 2x2 covariance of the random intercept and
            slope.

    Returns:
        The GLS information matrix and right-hand-side vector summed over
        clusters.  A cluster whose marginal covariance is not positive
        definite raises the ValueError of
        `precisions`.
    """
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        group_design = design[group]
        random_effect_design = columns(group_design, random_slope_position)
        precision = precisions(
            random_effect_design,
            residual_variance,
            random_effect_covariance,
        )
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ values[group]
    return information, right_hand_side


def precisions(
    random_effect_design: numpy.ndarray,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
) -> numpy.ndarray:
    """Returns dense precision for one random-effect covariance block.

    Args:
        random_effect_design: Intercept and slope columns for one cluster.
        residual_variance: Residual variance.
        random_effect_covariance: 2x2 covariance of the random effects.

    Returns:
        The inverse of the cluster's marginal covariance matrix.

    Raises:
        ValueError: If the marginal covariance is not positive definite.
    """
    count = random_effect_design.shape[0]
    covariance = marginal(
        random_effect_design, residual_variance, random_effect_covariance
    )
    try:
        precision: numpy.ndarray = linalg.solve(
            covariance, numpy.eye(count), assume_a="pos"
        )
    except numpy.linalg.LinAlgError as error:
        raise errors.InputError(
            "hierarchical random-effect covariance is not positive definite"
        ) from error
    return precision


def diagnose(
    labels: tuple[Any, ...],
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
    cluster_sizes: list[int],
    method: str,
) -> dict[str, Any]:
    """Builds auditable diagnostics for the one-random-slope model.

    Args:
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: Residual variance.
        random_effect_covariance: 2x2 covariance of the random intercept and
            slope.
        cluster_sizes: Number of rows in each cluster.
        method: ``two_stage_moment_gls`` or the REML method name; it selects
            the recorded variance-component method.

    Returns:
        A dictionary of variance components, correlation, cluster-size
        range, status flags and stated limitations.
    """
    random_intercept_variance = float(random_effect_covariance[0, 0])
    random_slope_variance = float(random_effect_covariance[1, 1])
    covariance = float(random_effect_covariance[0, 1])
    denominator = math.sqrt(
        max(random_intercept_variance * random_slope_variance, 0.0)
    )
    correlation = covariance / denominator if denominator > 0.0 else 0.0
    total_variance = residual_variance + random_intercept_variance
    return {
        "converged": True,
        "iterations": 2,
        "estimator_family": "hierarchical_random_intercept_slope",
        "method": method,
        "cluster_count": len(labels),
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "random_slope_position": random_slope_position,
        "residual_variance": residual_variance,
        "random_intercept_variance": random_intercept_variance,
        "random_slope_variance": random_slope_variance,
        "random_intercept_slope_covariance": covariance,
        "random_effect_correlation": correlation,
        "random_effect_covariance": random_effect_covariance.tolist(),
        "intraclass_correlation": random_intercept_variance / total_variance,
        "random_intercept_status": (
            "estimated" if random_intercept_variance > 0.0 else "boundary_zero"
        ),
        "random_slope_status": (
            "estimated" if random_slope_variance > 0.0 else "boundary_zero"
        ),
        "variance_component_method": (
            "moment_estimate_from_initial_ols_random_effects"
            if method == "two_stage_moment_gls"
            else "profile_reml_with_l_bfgs_b"
        ),
        "uncertainty_method": "conditional_on_variance_components",
        "variance_component_uncertainty": "not_propagated",
        "limitations": [
            "only one random slope is supported",
            "nested effects are unsupported",
            "crossed effects are unsupported",
            "fully Bayesian multilevel inference is unsupported",
            "variance-component uncertainty is not propagated",
        ],
        "observed_random_effect_structure": "intercept_and_declared_slope",
        "random_effect_covariance_evidence": {
            "positive_semidefinite_projection": True,
            "covariance_is_conditional": True,
        },
    }


def gls(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    residual_variance: float,
    random_intercept_variance: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Builds the block-diagonal precision system for random intercepts.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        residual_variance: Residual variance.
        random_intercept_variance: Random-intercept variance; zero gives
            independent rows.

    Returns:
        The GLS information matrix and right-hand-side vector summed over
        clusters.
    """
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        group_design = design[group]
        group_values = values[group]
        count = group_design.shape[0]
        precision = numpy.eye(count) / residual_variance
        if random_intercept_variance > 0.0:
            precision -= (
                random_intercept_variance
                / (
                    residual_variance
                    * (residual_variance + count * random_intercept_variance)
                )
            ) * numpy.ones((count, count))
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ group_values
    return information, right_hand_side


def distinct(clusters: numpy.ndarray) -> tuple[Any, ...]:
    """Returns the cluster labels in order of first appearance.

    Args:
        clusters: Cluster label of every row.

    Returns:
        The distinct labels.

    Raises:
        ValueError: If the labels are not hashable.
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise errors.InputError(
            "hierarchical cluster labels must be hashable"
        ) from error
    return labels


def columns(
    group_design: numpy.ndarray, random_slope_position: int
) -> numpy.ndarray:
    """Returns one cluster's random-intercept and random-slope columns.

    Args:
        group_design: Design rows of the cluster, intercept first.
        random_slope_position: Feature index carrying the random slope.

    Returns:
        A column of ones beside the slope feature's column.
    """
    random_effect_design: numpy.ndarray = numpy.column_stack(
        (
            numpy.ones(group_design.shape[0]),
            group_design[:, random_slope_position + 1],
        )
    )
    return random_effect_design


def marginal(
    random_effect_design: numpy.ndarray,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
) -> numpy.ndarray:
    """Returns one cluster's marginal covariance under random effects.

    Args:
        random_effect_design: Intercept and slope columns for one cluster.
        residual_variance: Residual variance.
        random_effect_covariance: 2x2 covariance of the random effects.

    Returns:
        The residual variance on the diagonal plus the random-effect
        covariance seen through the cluster's design.
    """
    count = random_effect_design.shape[0]
    covariance: numpy.ndarray = (
        residual_variance * numpy.eye(count)
        + random_effect_design
        @ random_effect_covariance
        @ random_effect_design.T
    )
    return covariance


def solve(
    information: numpy.ndarray, right_hand_side: numpy.ndarray, message: str
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Solves a GLS system for the fixed effects and their covariance.

    Args:
        information: GLS information matrix.
        right_hand_side: GLS right-hand-side vector.
        message: Error message for a singular system.

    Returns:
        The fixed-effect coefficients and their covariance matrix.

    Raises:
        ValueError: If the system is singular.
    """
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
        covariance = numpy.linalg.solve(
            information, numpy.eye(information.shape[0])
        )
    except numpy.linalg.LinAlgError as error:
        raise errors.InputError(message) from error
    return coefficients, covariance


def settle(
    information: numpy.ndarray, right_hand_side: numpy.ndarray
) -> tuple[float, numpy.ndarray] | None:
    """Solves a REML objective's GLS system, if the point admits one.

    Args:
        information: GLS information matrix at the point.
        right_hand_side: GLS right-hand-side vector at the point.

    Returns:
        The log-determinant of the information matrix and the fixed
        effects, or None when the determinant is not positive and finite or
        the system is singular, so the objective scores the point infinite.
    """
    sign, logdet_information = numpy.linalg.slogdet(information)
    if sign <= 0.0 or not math.isfinite(float(logdet_information)):
        return None
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
    except numpy.linalg.LinAlgError:
        # handled: singular point scored inf; optimizer rejects it
        return None
    return float(logdet_information), coefficients


def profile(
    design: numpy.ndarray,
    values: numpy.ndarray,
    logdet_covariance: float,
    logdet_information: float,
    quadratic: float,
) -> float:
    """Returns the negative profile REML objective from its parts.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        logdet_covariance: Log-determinant of the marginal covariance.
        logdet_information: Log-determinant of the GLS information matrix.
        quadratic: Residual quadratic form under the marginal precision.

    Returns:
        Half the sum of the log-determinants, the quadratic form and the
        normal constant over the residual degrees of freedom.
    """
    observations = len(values)
    parameters = design.shape[1]
    return float(
        0.5
        * (
            logdet_covariance
            + logdet_information
            + quadratic
            + (observations - parameters) * math.log(2.0 * math.pi)
        )
    )


def slope(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    structure: tuple[tuple[Any, ...], int, list[int]],
    components: tuple[float, numpy.ndarray],
    method: str,
    extra: Mapping[str, Any],
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits the fixed effects of the one-random-slope model and reports them.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        structure: The distinct labels, the feature index carrying the
            random slope, and the number of rows in each cluster.
        components: The residual variance and the 2x2 random-effect
            covariance.
        method: ``two_stage_moment_gls`` or ``bounded_reml_gls``.
        extra: Records added to the diagnostics, such as the REML optimizer.

    Returns:
        The fixed-effect coefficients, their covariance, the diagnostics,
        the residual variance, the random-intercept variance and the
        random-effect covariance.

    Raises:
        ValueError: If the GLS system is singular or a cluster's marginal
            covariance is not positive definite.
    """
    labels, random_slope_position, cluster_sizes = structure
    residual_variance, random_effect_covariance = components
    information, right_hand_side = assemble(
        design,
        values,
        clusters,
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
    )
    coefficients, covariance = solve(
        information,
        right_hand_side,
        "hierarchical GLS system is not identifiable"
        if method == "two_stage_moment_gls"
        else "hierarchical REML GLS system is not identifiable",
    )
    diagnostics = diagnose(
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
        cluster_sizes,
        method=method,
    )
    diagnostics.update(extra)
    return (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        float(random_effect_covariance[0, 0]),
        random_effect_covariance,
    )


def intercept(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    components: tuple[float, float],
    method: tuple[int, str, str],
    extra: Mapping[str, Any],
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits the fixed effects of the random-intercept model and reports them.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        components: The residual and random-intercept variances.
        method: The iteration count, the method name (``two_stage_moment_gls``
            or ``bounded_reml_gls``) and the variance-component method.
        extra: Records appended to the diagnostics.

    Returns:
        The fixed-effect coefficients, their covariance, the diagnostics,
        the residual variance, the random-intercept variance and the 2x2
        random-effect covariance (random intercept only).

    Raises:
        ValueError: If the GLS system is singular.
    """
    residual_variance, random_intercept_variance = components
    iterations, name, component_method = method
    information, right_hand_side = gls(
        design, values, clusters, labels, *components
    )
    coefficients, covariance = solve(
        information,
        right_hand_side,
        "hierarchical GLS system is not identifiable"
        if name == "two_stage_moment_gls"
        else "hierarchical REML GLS system is not identifiable",
    )
    cluster_sizes, effect = shrink(
        design, values, clusters, labels, coefficients, components
    )
    total_variance = residual_variance + random_intercept_variance
    diagnostics = {
        "converged": True,
        "iterations": iterations,
        "estimator_family": "hierarchical_random_intercept",
        "method": name,
        "cluster_count": len(labels),
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "residual_variance": residual_variance,
        "random_intercept_variance": random_intercept_variance,
        "intraclass_correlation": random_intercept_variance / total_variance,
        "random_intercept_status": (
            "estimated" if random_intercept_variance > 0.0 else "boundary_zero"
        ),
        "random_intercept_effect": effect,
        "variance_component_method": component_method,
        **extra,
    }
    return (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        random_intercept_variance,
        numpy.array(
            [[random_intercept_variance, 0.0], [0.0, 0.0]], dtype=float
        ),
    )


def shrink(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    coefficients: numpy.ndarray,
    components: tuple[float, float],
) -> tuple[list[int], dict[str, float]]:
    """Summarizes the shrunken random-intercept predictions (BLUPs).

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        coefficients: Fixed-effect coefficients.
        components: The residual and random-intercept variances.

    Returns:
        The number of rows in each cluster, and the mean and largest
        absolute BLUP.
    """
    residual_variance, random_intercept_variance = components
    fitted_residuals = values - design @ coefficients
    random_effects = []
    cluster_sizes = []
    for label in labels:
        group = clusters == label
        count = int(numpy.sum(group))
        cluster_sizes.append(count)
        group_sum = float(numpy.sum(fitted_residuals[group]))
        shrinkage = random_intercept_variance / (
            residual_variance + count * random_intercept_variance
        )
        random_effects.append(shrinkage * group_sum)
    effect = {
        "mean_absolute_blup": float(numpy.mean(numpy.abs(random_effects))),
        "max_absolute_blup": float(numpy.max(numpy.abs(random_effects))),
    }
    return cluster_sizes, effect
