"""The hierarchical linear model: moments, GLS and dispatch.

Calls written `parameter_estimation.<name>(...)` look the function up on
the package when they run, so a test that patches the package attribute
reaches them as it did when the package was one module.
"""

import math
from typing import Any

import numpy
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation import parameter_estimation
from scmle.evaluation.parameter_estimation.audit import crosscheck
from scmle.evaluation.parameter_estimation.audit import (
    reference_hierarchical_random_slope,
)
from scmle.evaluation.parameter_estimation.audit import (
    reference_hierarchical_random_slope_reml,
)
from scmle.evaluation.parameter_estimation.audit import (
    reference_hierarchical_reml,
)
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import MIN_CLUSTERS
from scmle.evaluation.parameter_estimation.declaration import (
    RANDOM_EFFECT_COLUMNS,
)
from scmle.evaluation.parameter_estimation.declaration import SCALE_FLOOR
from scmle.evaluation.parameter_estimation.mixed import columns
from scmle.evaluation.parameter_estimation.mixed import distinct
from scmle.evaluation.parameter_estimation.mixed import intercept
from scmle.evaluation.parameter_estimation.mixed import project
from scmle.evaluation.parameter_estimation.mixed import slope
from scmle.evaluation.parameter_estimation.reml import fit
from scmle.evaluation.parameter_estimation.reml import optimise
from scmle.evaluation.parameter_estimation.report import Fit


def recheck(
    design: numpy.ndarray,
    values: numpy.ndarray,
    cluster_values: numpy.ndarray,
    policy: Config,
    fitted: tuple[Any, ...],
) -> dict[str, Any]:
    """Runs the independent reference for the hierarchical model.

    Args:
        design: Design matrix with an intercept column.
        values: Outcome vector.
        cluster_values: Cluster labels of the kept rows.
        policy: Estimator policy, selecting method and random slope.
        fitted: The coefficients, diagnostics, residual variance, random
            intercept variance and random-effect covariance of the fit.

    Returns:
        The reference record matching the declared random-effects structure.
    """
    (
        coefficients,
        diagnostics,
        residual_variance,
        random_intercept_variance,
        random_effect_covariance,
    ) = fitted
    if policy.slope is not None:
        if policy.hierarchy == "reml":
            return reference_hierarchical_random_slope_reml(
                design,
                values,
                cluster_values,
                policy.slope,
                residual_variance,
                random_effect_covariance,
                coefficients,
                diagnostics["reml_objective"],
                policy,
            )
        return reference_hierarchical_random_slope(
            design,
            values,
            cluster_values,
            policy.slope,
            residual_variance,
            random_effect_covariance,
            coefficients,
            policy,
        )
    if policy.hierarchy == "reml":
        return reference_hierarchical_reml(
            design,
            values,
            cluster_values,
            residual_variance,
            random_intercept_variance,
            coefficients,
            diagnostics["reml_objective"],
            policy,
        )
    return crosscheck(
        design,
        values,
        cluster_values,
        residual_variance,
        random_intercept_variance,
        coefficients,
        policy,
    )


def nest(
    design: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    cluster_values: numpy.ndarray | None,
    names: tuple[str, ...],
) -> Fit:
    """Fits the hierarchical linear model and its reference calculation.

    Args:
        design: Design matrix with an intercept column.
        values: Outcome vector.
        policy: Estimator policy.
        cluster_values: Cluster labels of the kept rows.
        names: Parameter names.

    Returns:
        The fitted quantities.

    Raises:
        ValueError: If there are no cluster labels.
    """
    if cluster_values is None:
        raise errors.InputError("hierarchical_linear requires cluster labels")
    (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        random_intercept_variance,
        random_effect_covariance,
    ) = parameter_estimation.fit_hierarchical_linear(  # type: ignore[attr-defined]
        design, values, cluster_values, policy
    )
    reference = recheck(
        design,
        values,
        cluster_values,
        policy,
        (
            coefficients,
            diagnostics,
            residual_variance,
            random_intercept_variance,
            random_effect_covariance,
        ),
    )
    likelihood = "gaussian"
    degrees_of_freedom = max(len(dict.fromkeys(cluster_values.tolist())) - 1, 1)
    pivot = stats.t.ppf(0.5 + policy.confidence / 2.0, degrees_of_freedom)
    p_values = 2.0 * stats.t.sf(
        numpy.abs(
            coefficients
            / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
        ),
        degrees_of_freedom,
    )
    diagnostics = {
        **diagnostics,
        "likelihood": likelihood,
        "degrees_of_freedom": degrees_of_freedom,
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


def fit_hierarchical_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits a bounded two-stage hierarchical linear model.

    The default estimates variance components by moments from an initial
    fixed-effect fit.  The bounded REML policy uses a deterministic profile
    likelihood over the residual and random-intercept variances, followed by
    generalized least squares for the fixed effects.  When a random slope is
    declared, the same methods use one random-intercept/random-slope block.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        policy: Supplies the variance-component method, the random-slope
            position, the REML bounds, iteration limit and tolerance.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary, the residual variance, the random-intercept variance,
        and the 2x2 random-effect covariance matrix (intercept and slope; the
        slope entries are zero for a random intercept only).

    Raises:
        ValueError: If the labels are unhashable, there are fewer than two
            clusters, the design is rank deficient, the within-cluster
            variance is not positive, or the GLS system is not identifiable.
            Errors from the random-slope and REML fits also propagate.
    """
    if policy.slope is not None:
        return fit_hierarchical_random_slope(design, values, clusters, policy)
    labels, initial_residuals = start(design, values, clusters)
    residual_variance, random_intercept_variance = moments(
        clusters, labels, initial_residuals
    )
    if policy.hierarchy == "reml":
        return fit(
            design,
            values,
            clusters,
            labels,
            residual_variance,
            random_intercept_variance,
            policy,
        )
    return intercept(
        design,
        values,
        clusters,
        labels,
        (residual_variance, random_intercept_variance),
        (2, "two_stage_moment_gls", "moment_estimate_from_initial_ols"),
        {
            "uncertainty_method": "conditional_on_moment_variance_components",
            "variance_component_uncertainty": "not_propagated",
        },
    )


def fit_hierarchical_random_slope(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits one random slope with a random intercept by moments or REML.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        policy: Supplies the random-slope position and the variance-component
            method; the REML settings are used when the method is ``reml``.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary, the residual variance, the random-intercept variance,
        and the 2x2 random-effect covariance matrix.

    Raises:
        ValueError: If no random slope position is declared, the labels are
            unhashable, there are fewer than two clusters, the design is rank
            deficient, a cluster lacks variation in the slope feature, the
            within-cluster variance is not positive, or the GLS system is not
            identifiable.
    """
    random_slope_position = policy.slope
    if random_slope_position is None:
        raise errors.InputError("random slope position must be declared")
    labels, initial_residuals = start(design, values, clusters)
    residual_variance, random_effect_covariance, cluster_sizes = slopes(
        design, clusters, labels, initial_residuals, random_slope_position
    )
    if policy.hierarchy == "reml":
        return optimise(
            design,
            values,
            clusters,
            labels,
            random_slope_position,
            residual_variance,
            random_effect_covariance,
            cluster_sizes,
            policy,
        )
    return slope(
        design,
        values,
        clusters,
        (labels, random_slope_position, cluster_sizes),
        (residual_variance, random_effect_covariance),
        "two_stage_moment_gls",
        {},
    )


def start(
    design: numpy.ndarray, values: numpy.ndarray, clusters: numpy.ndarray
) -> tuple[tuple[Any, ...], numpy.ndarray]:
    """Checks a hierarchical design and fits the initial least squares.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.

    Returns:
        The distinct cluster labels and the residuals of the initial
        ordinary least-squares fit.

    Raises:
        ValueError: If the labels are unhashable, there are fewer than two
            clusters, or the design is rank deficient.
    """
    labels = distinct(clusters)
    if len(labels) < MIN_CLUSTERS:
        raise errors.InputError(
            "hierarchical_linear needs at least two clusters"
        )
    if numpy.linalg.matrix_rank(design) != design.shape[1]:
        raise errors.InputError("hierarchical design matrix is rank deficient")
    initial_coefficients = numpy.linalg.lstsq(design, values, rcond=None)[0]
    initial_residuals = values - design @ initial_coefficients
    return labels, initial_residuals


def moments(
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    initial_residuals: numpy.ndarray,
) -> tuple[float, float]:
    """Estimates the random-intercept variance components by moments.

    Args:
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        initial_residuals: Residuals of the initial least-squares fit.

    Returns:
        The within-cluster residual variance and the random-intercept
        variance, floored at zero.

    Raises:
        ValueError: If the within-cluster variance is not positive.
    """
    cluster_means = []
    cluster_sizes = []
    within_sum_of_squares = 0.0
    for label in labels:
        group = clusters == label
        group_residuals = initial_residuals[group]
        cluster_means.append(float(numpy.mean(group_residuals)))
        cluster_sizes.append(int(numpy.sum(group)))
        centered = group_residuals - numpy.mean(group_residuals)
        within_sum_of_squares += float(numpy.sum(centered**2))
    residual_variance = within_sum_of_squares / max(
        len(initial_residuals) - len(labels), 1
    )
    if not math.isfinite(residual_variance) or residual_variance <= SCALE_FLOOR:
        raise errors.InputError(
            "hierarchical_linear needs positive within-cluster variance"
        )
    between_variance = float(numpy.var(cluster_means, ddof=1))
    finite_sample_adjustment = residual_variance * float(
        numpy.mean(1.0 / numpy.asarray(cluster_sizes, dtype=float))
    )
    random_intercept_variance = max(
        0.0, between_variance - finite_sample_adjustment
    )
    return residual_variance, random_intercept_variance


def slopes(
    design: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    initial_residuals: numpy.ndarray,
    random_slope_position: int,
) -> tuple[float, numpy.ndarray, list[int]]:
    """Estimates the random intercept and slope covariance by moments.

    Args:
        design: Design matrix including the intercept column.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        initial_residuals: Residuals of the initial least-squares fit.
        random_slope_position: Feature index carrying the random slope.

    Returns:
        The within-cluster residual variance, the 2x2 random-effect
        covariance projected to the positive-semidefinite cone, and the
        number of rows in each cluster.

    Raises:
        ValueError: If a cluster lacks variation in the slope feature or the
            within-cluster variance is not positive.
    """
    random_effect_estimates = []
    random_effect_inverse_matrices = []
    cluster_sizes = []
    within_sum_of_squares = 0.0
    for label in labels:
        group = clusters == label
        random_effect_design = columns(design[group], random_slope_position)
        if (
            numpy.linalg.matrix_rank(random_effect_design)
            != RANDOM_EFFECT_COLUMNS
        ):
            raise errors.InputError(
                "each hierarchical cluster needs within-cluster variation "
                "for the declared random slope"
            )
        inverse_matrix = numpy.linalg.solve(
            random_effect_design.T @ random_effect_design,
            numpy.eye(2),
        )
        group_residuals = initial_residuals[group]
        random_effect_estimates.append(
            inverse_matrix @ random_effect_design.T @ group_residuals
        )
        random_effect_inverse_matrices.append(inverse_matrix)
        cluster_sizes.append(int(numpy.sum(group)))
        centered = group_residuals - numpy.mean(group_residuals)
        within_sum_of_squares += float(numpy.sum(centered**2))
    residual_variance = within_sum_of_squares / max(
        len(initial_residuals) - len(labels), 1
    )
    if not math.isfinite(residual_variance) or residual_variance <= SCALE_FLOOR:
        raise errors.InputError(
            "hierarchical_linear needs positive within-cluster variance"
        )
    effects = numpy.asarray(random_effect_estimates, dtype=float)
    centered_effects = effects - effects.mean(axis=0)
    effect_covariance = (
        centered_effects.T @ centered_effects / (len(labels) - 1)
    )
    effect_covariance -= residual_variance * numpy.mean(
        numpy.asarray(random_effect_inverse_matrices), axis=0
    )
    return residual_variance, project(effect_covariance), cluster_sizes
