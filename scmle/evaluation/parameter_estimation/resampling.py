"""Cluster and row bootstrap draws for sampling sensitivity.

Calls written `parameter_estimation.<name>(...)` look the function up on
the package when they run, so a test that patches the package attribute
reaches them as it did when the package was one module.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import functools
import hashlib
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.evaluation import parameter_estimation
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import INTERCEPT_LIMITS
from scmle.evaluation.parameter_estimation.declaration import MIN_CLUSTERS
from scmle.evaluation.parameter_estimation.declaration import MIN_REPLICATES

SLOPE_LIMITS = (
    "variance-component uncertainty is not propagated",
    "only one random slope is supported",
    "nested effects are unsupported",
    "crossed effects are unsupported",
    "fully Bayesian multilevel inference is unsupported",
)


def rerun(
    sampled_matrix: numpy.ndarray,
    sampled_values: numpy.ndarray,
    estimand: Estimand,
    bootstrap_policy: Config,
    clusters: Sequence[Any] | None,
    coefficients: numpy.ndarray,
) -> numpy.ndarray:
    """Estimates the parameters on one bootstrap draw and checks the result.

    Args:
        sampled_matrix: Feature rows of the draw.
        sampled_values: Outcomes of the draw.
        estimand: Scientific estimand and its explicit assumptions.
        bootstrap_policy: The estimator policy with bootstrapping turned off.
        clusters: Cluster labels of the draw, or None for row-level draws.
        coefficients: Point-estimate coefficient vector for shape checking.

    Returns:
        The parameter estimates of the draw.

    Raises:
        ValueError: If the estimate has an unexpected shape, is not finite,
            or fails its independent reference check.
    """
    result = parameter_estimation.estimate_parameters(
        sampled_matrix,
        sampled_values,
        estimand,
        bootstrap_policy,
        clusters=clusters,
    )
    estimate = numpy.asarray(
        [parameter.estimate for parameter in result.parameters],
        dtype=float,
    )
    if estimate.shape != numpy.asarray(coefficients).shape:
        raise ValueError(
            "bootstrap replicate returned an unexpected parameter shape"
        )
    if not numpy.isfinite(estimate).all():
        raise errors.InputError(
            "bootstrap replicate returned non-finite estimates"
        )
    if result.reference.get("passed") is not True:
        raise errors.InputError(
            "bootstrap replicate independent reference check failed"
        )
    return estimate


def hierarchical_bootstrap_sensitivity(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    estimand: Estimand,
    policy: Config,
    coefficients: numpy.ndarray,
    clusters: numpy.ndarray,
) -> dict[str, Any]:
    """Estimates grouped-parameter sensitivity by resampling whole clusters.

    Each bootstrap draw samples the source clusters with replacement and
    gives every selected copy a fresh identity. The declared moment-GLS or
    bounded-REML policy is then rerun on that draw. A replicate is accepted
    only when its independent reference calculation passes, so percentile
    intervals never combine estimates that failed the scientific check.

    Args:
        matrix: Complete feature rows used by the point estimate.
        values: Complete outcomes used by the point estimate.
        estimand: Scientific estimand and its explicit assumptions.
        policy: Hierarchical estimator policy with a positive replicate count.
        coefficients: Point-estimate coefficient vector for shape checking.
        clusters: Complete source cluster labels.

    Returns:
        Serializable bootstrap evidence with draw digests, replicate status,
        percentile intervals, independent reference counts, and limitations.

    Raises:
        ValueError: If fewer than two reference-checked replicates succeed,
            or the cluster labels are unhashable or form fewer than two
            clusters.
    """
    rng = numpy.random.default_rng(policy.seed)
    limitations = list(
        INTERCEPT_LIMITS
        if policy.random_slope_position is None
        else SLOPE_LIMITS
    )
    source_labels = tuple(dict.fromkeys(clusters.tolist()))
    source_unit_count = len(source_labels)
    estimates, draws, failures = replicate(
        functools.partial(clone, matrix, values, clusters, rng),
        {
            "resampling_unit": "cluster",
            "estimator_method": policy.hierarchical_method,
        },
        (estimand, policy, coefficients),
        "hierarchical",
    )
    return {
        "status": "partial" if failures else "available",
        "method": "nonparametric_percentile_bootstrap",
        "bootstrap_replicates": policy.bootstrap_replicates,
        "successful_replicates": len(estimates),
        "failed_replicates": len(failures),
        "confidence_level": policy.confidence_level,
        "seed": policy.seed,
        "resampling_unit": "cluster",
        "estimator_method": policy.hierarchical_method,
        "source_unit_count": source_unit_count,
        "source_row_count": len(values),
        "cluster_identity_policy": "fresh_identity_per_selected_cluster",
        "draws": draws,
        "failures": failures,
        "reference_checks": checks(estimates, failures),
        "limitations": limitations,
        "bootstrap_intervals": intervals(estimates, policy.confidence_level),
    }


def clone(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    rng: numpy.random.Generator,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Draws whole clusters and assigns fresh identities to selected copies.

    Args:
        matrix: Complete feature rows from the source estimate.
        values: Complete outcomes from the source estimate.
        clusters: Source cluster labels.
        rng: Seeded random generator owned by the sensitivity report.

    Returns:
        Resampled rows, outcomes, fresh cluster labels, and digest evidence.

    Raises:
        ValueError: If the cluster labels are not hashable or the source
            data contain fewer than two clusters.
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise errors.InputError(
            "hierarchical cluster labels must be hashable for bootstrap"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise errors.InputError(
            "hierarchical bootstrap needs at least two source clusters"
        )
    source_indices = rng.integers(0, len(labels), size=len(labels))
    row_groups = [
        numpy.flatnonzero(clusters == labels[index]) for index in source_indices
    ]
    indices = numpy.concatenate(row_groups)
    sampled_clusters = numpy.repeat(
        numpy.arange(len(row_groups), dtype=object),
        [len(group) for group in row_groups],
    )
    return (
        matrix[indices],
        values[indices],
        sampled_clusters,
        {
            "sampled_rows": len(indices),
            "sampled_units": len(row_groups),
            "source_unit_count": len(labels),
            "sample_digest": bootstrap_sample_digest(source_indices),
            "fresh_cluster_identity_digest": bootstrap_sample_digest(
                numpy.arange(len(row_groups), dtype=numpy.int64)
            ),
        },
    )


def causal_bootstrap_sensitivity(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    estimand: Estimand,
    policy: Config,
    coefficients: numpy.ndarray,
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Estimates causal sampling sensitivity with nonparametric bootstrap.

    Row-level causal policies resample observations independently. Clustered
    causal policies resample complete declared clusters, preserving each
    cluster as a sampling unit and assigning every selected copy a distinct
    bootstrap cluster label for the influence-function covariance.

    Args:
        matrix: Complete rows used by the point estimate.
        values: Complete outcomes used by the point estimate.
        estimand: Scientific estimand and its explicit causal assumptions.
        policy: Causal estimator policy with a positive replicate count.
        coefficients: Point-estimate coefficient vector for shape checking.
        clusters: Cluster labels for a clustered policy, otherwise ``None``.

    Returns:
        Serializable sampling-sensitivity evidence, including percentile
        intervals, resampling-unit evidence, and every failed replicate.

    Raises:
        ValueError: If the bootstrap cannot produce at least two valid,
            independently reference-checked replicates, or clustered labels
            are unhashable or form fewer than two clusters.
    """
    rng = numpy.random.default_rng(policy.seed)
    resampling_unit = "cluster" if clusters is not None else "row"
    source_unit_count = (
        len(tuple(dict.fromkeys(clusters.tolist())))
        if clusters is not None
        else len(values)
    )
    estimates, draws, failures = replicate(
        functools.partial(
            draw_causal_bootstrap_sample, matrix, values, clusters, rng
        ),
        {"resampling_unit": resampling_unit},
        (estimand, policy, coefficients),
        "causal",
    )
    return {
        "status": "partial" if failures else "available",
        "method": "nonparametric_percentile_bootstrap",
        "bootstrap_replicates": policy.bootstrap_replicates,
        "successful_replicates": len(estimates),
        "failed_replicates": len(failures),
        "confidence_level": policy.confidence_level,
        "seed": policy.seed,
        "resampling_unit": resampling_unit,
        "source_unit_count": source_unit_count,
        "source_row_count": len(values),
        "draws": draws,
        "failures": failures,
        "reference_checks": checks(estimates, failures),
        "bootstrap_intervals": intervals(estimates, policy.confidence_level),
    }


def draw_causal_bootstrap_sample(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None, dict[str, Any]]:
    """Draws one causal bootstrap sample and its auditable unit evidence.

    Args:
        matrix: Complete feature rows from the source estimate.
        values: Complete outcomes from the source estimate.
        clusters: Source cluster labels, or None to resample rows
            independently.
        rng: Seeded random generator owned by the sensitivity report.

    Returns:
        The resampled rows, the resampled outcomes, fresh cluster labels
        (None when `clusters` is None), and a dictionary of sampling
        evidence including a digest of the drawn indices.

    Raises:
        ValueError: If the cluster labels are not hashable or the source
            data contain fewer than two clusters.
    """
    if clusters is None:
        indices = rng.integers(0, len(values), size=len(values))
        return (
            matrix[indices],
            values[indices],
            None,
            {
                "sampled_rows": len(indices),
                "source_unit_count": len(values),
                "sample_digest": bootstrap_sample_digest(indices),
            },
        )

    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise errors.InputError(
            "cluster labels must be hashable for clustered bootstrap"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise errors.InputError(
            "clustered bootstrap needs at least two source clusters"
        )
    source_indices = rng.integers(0, len(labels), size=len(labels))
    row_groups = [
        numpy.flatnonzero(clusters == labels[index]) for index in source_indices
    ]
    indices = numpy.concatenate(row_groups)
    sampled_clusters = numpy.repeat(
        numpy.arange(len(row_groups), dtype=object),
        [len(group) for group in row_groups],
    )
    return (
        matrix[indices],
        values[indices],
        sampled_clusters,
        {
            "sampled_rows": len(indices),
            "sampled_units": len(row_groups),
            "source_unit_count": len(labels),
            "sample_digest": bootstrap_sample_digest(source_indices),
        },
    )


def bootstrap_sample_digest(indices: numpy.ndarray) -> str:
    """Returns a digest of bootstrap draw indices without exposing raw rows.

    Args:
        indices: Integer indices of the draw.

    Returns:
        The SHA-256 hex digest of the indices as little-endian 64-bit
        integers.
    """
    canonical = numpy.asarray(indices, dtype="<i8").tobytes()
    return hashlib.sha256(canonical).hexdigest()


def replicate(
    draw: Callable[
        [],
        tuple[
            numpy.ndarray,
            numpy.ndarray,
            numpy.ndarray | None,
            Mapping[str, Any],
        ],
    ],
    labels: Mapping[str, Any],
    point: tuple[Estimand, Config, numpy.ndarray],
    kind: str,
) -> tuple[
    list[numpy.ndarray], list[Mapping[str, Any]], list[Mapping[str, Any]]
]:
    """Reruns the estimator on bootstrap draws and keeps the checked ones.

    A replicate whose rerun raises a numerical or input error is recorded
    as a failure instead of an estimate.

    Args:
        draw: Draws one bootstrap sample: rows, outcomes, fresh cluster
            labels (or None) and the draw's evidence.
        labels: Resampling-unit fields recorded with every draw.
        point: The estimand, the policy with a positive replicate count,
            and the point-estimate coefficients for shape checking.
        kind: ``hierarchical`` or ``causal``, for the error message.

    Returns:
        The accepted estimates, one record per draw and one per failure.

    Raises:
        ValueError: If fewer than two replicates succeed; errors of the draw
            itself propagate.
    """
    estimand, policy, coefficients = point
    bootstrap_policy = dataclasses.replace(policy, bootstrap_replicates=0)
    estimates = []
    failures: list[Mapping[str, Any]] = []
    draws: list[Mapping[str, Any]] = []
    for index in range(policy.bootstrap_replicates):
        sampled_matrix, sampled_values, sampled_clusters, record = draw()
        draws.append({"replicate": index, **labels, **record})
        try:
            estimates.append(
                rerun(
                    sampled_matrix,
                    sampled_values,
                    estimand,
                    bootstrap_policy,
                    sampled_clusters.tolist()
                    if sampled_clusters is not None
                    else None,
                    coefficients,
                )
            )
        except (
            FloatingPointError,
            numpy.linalg.LinAlgError,
            OverflowError,
            ValueError,
        ) as error:
            # handled: returned as failed finding/problem/refusal with error
            # text
            failures.append(
                {
                    "replicate": index,
                    "resampling_unit": labels["resampling_unit"],
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )
    if len(estimates) < MIN_REPLICATES:
        raise errors.InputError(
            f"{kind} bootstrap needs at least two successful replicates; "
            f"received {len(estimates)} of {policy.bootstrap_replicates}"
        )
    return estimates, draws, failures


def checks(
    estimates: Sequence[numpy.ndarray], failures: Sequence[object]
) -> dict[str, int | str]:
    """Returns the reference-check record of a bootstrap.

    Args:
        estimates: The accepted estimates.
        failures: The failed replicates.

    Returns:
        The successful and failed counts and the acceptance requirement.
    """
    return {
        "successful": len(estimates),
        "failed": len(failures),
        "requirement": "every accepted replicate passes its independent "
        "point-estimate reference check",
    }


def intervals(
    estimates: Sequence[numpy.ndarray], confidence_level: float
) -> list[list[float]]:
    """Returns percentile bootstrap intervals for each parameter.

    Args:
        estimates: The accepted bootstrap estimates.
        confidence_level: Confidence level of the intervals.

    Returns:
        The lower bounds and the upper bounds, one per parameter.
    """
    bootstrap = numpy.asarray(estimates)
    alpha = (1.0 - confidence_level) / 2.0
    return [
        [float(item) for item in numpy.quantile(bootstrap, alpha, axis=0)],
        [
            float(item)
            for item in numpy.quantile(bootstrap, 1.0 - alpha, axis=0)
        ],
    ]
