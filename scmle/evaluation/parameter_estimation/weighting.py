"""Influence-function variance, Hajek weighting and outcome regressions.

These pieces are shared by the causal estimators and their independent
reference calculations.
"""

import math
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.declaration import CovarianceKind
from scmle.evaluation.parameter_estimation.declaration import MIN_CLUSTERS
from scmle.evaluation.parameter_estimation.declaration import MIN_INFLUENCE_ROWS
from scmle.evaluation.parameter_estimation.inputs import validate


def center(influence: numpy.ndarray) -> numpy.ndarray:
    """Validates observation-level influence scores and centers them.

    Args:
        influence: Observation-level influence scores.

    Returns:
        The scores minus their mean.

    Raises:
        ValueError: If the scores are not one-dimensional, are fewer than
            two, or are not finite.
    """
    centered_influence = numpy.asarray(influence, dtype=float)
    if centered_influence.ndim != 1:
        raise errors.InputError("influence scores must have shape (rows,)")
    if len(centered_influence) < MIN_INFLUENCE_ROWS:
        raise errors.InputError("influence uncertainty needs at least two rows")
    if not numpy.isfinite(centered_influence).all():
        raise errors.InputError(
            "influence scores must contain only finite values"
        )
    centered: numpy.ndarray = centered_influence - numpy.mean(
        centered_influence
    )
    return centered


def dispersion(
    influence: numpy.ndarray,
    covariance: CovarianceKind,
    clusters: numpy.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    """Computes IID or finite-sample clustered influence uncertainty.

    The clustered calculation treats each declared cluster as an independent
    sampling unit. It centers the observation-level influence scores before
    summing them within clusters and applies the finite-sample correction
    ``G / (G - 1) * (n - 1) / n``. This is intentionally separate from the
    HC3 and regression sandwich paths: causal ATE uncertainty is defined over
    an influence function, not over a model residual score.

    Args:
        influence: Observation-level influence scores.
        covariance: ``iid`` or ``clustered`` uncertainty.
        clusters: Cluster label of each row; needed for ``clustered``.

    Returns:
        The variance of the estimate and a dictionary describing the method
        (with cluster diagnostics for clustered uncertainty).

    Raises:
        ValueError: If the scores are not one-dimensional, are fewer than
            two, or are not finite; if `covariance` is neither ``iid`` nor
            ``clustered``; if clustered uncertainty has no labels, labels
            that are unhashable or of the wrong length, or fewer than two
            clusters; or if the clustered variance is not finite and
            non-negative.
    """
    centered_influence = center(influence)
    if covariance == "iid":
        variance = float(
            numpy.sum(centered_influence**2)
            / (len(centered_influence) - 1)
            / len(centered_influence)
        )
        return variance, {
            "method": "iid_centered_influence_function",
            "centered": True,
        }
    if covariance != "clustered":
        raise errors.InputError(
            "causal influence uncertainty supports only iid or clustered "
            "covariance"
        )
    variance, cluster_count, cluster_sizes = tally(centered_influence, clusters)
    if not math.isfinite(variance) or variance < 0.0:
        raise errors.InputError("clustered influence variance must be finite")
    return variance, {
        "method": "clustered_centered_influence_function",
        "centered": True,
        "cluster_diagnostics": {
            "cluster_count": cluster_count,
            "cluster_sizes": cluster_sizes,
            "uncertainty_method": "clustered_centered_influence_function",
        },
    }


def tally(
    centered_influence: numpy.ndarray, clusters: numpy.ndarray | None
) -> tuple[float, int, list[int]]:
    """Sums centered influence scores within clusters into a variance.

    Args:
        centered_influence: Centered observation-level influence scores.
        clusters: Cluster label of each row.

    Returns:
        The finite-sample clustered variance, the cluster count and the
        cluster sizes in first-appearance order.

    Raises:
        ValueError: If there are no labels, labels that are unhashable or
            of the wrong length, or fewer than two clusters.
    """
    if clusters is None:
        raise errors.InputError("clustered covariance needs cluster labels")
    cluster_values = validate(clusters.tolist(), len(centered_influence))
    if cluster_values is None:
        raise errors.InputError("clustered covariance needs cluster labels")
    try:
        labels = tuple(dict.fromkeys(cluster_values.tolist()))
    except TypeError as error:
        raise errors.InputError(
            "cluster labels must be hashable scalar values"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise errors.InputError(
            "clustered covariance needs at least two clusters"
        )
    cluster_sums = []
    cluster_sizes = []
    for label in labels:
        membership = cluster_values == label
        cluster_sums.append(float(numpy.sum(centered_influence[membership])))
        cluster_sizes.append(int(numpy.sum(membership)))
    cluster_sum_array = numpy.asarray(cluster_sums, dtype=float)
    row_count = len(centered_influence)
    correction = (len(labels) / (len(labels) - 1)) * (
        (row_count - 1) / row_count
    )
    variance = float(
        correction * numpy.sum(cluster_sum_array**2) / row_count**2
    )
    return variance, len(labels), cluster_sizes


def hajek(
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    propensity: numpy.ndarray,
    clusters: numpy.ndarray | None = None,
    covariance: CovarianceKind = "iid",
) -> tuple[float, float, dict[str, Any]]:
    """Computes normalized IPW means and an influence-function variance.

    Args:
        treatment: 0/1 treatment of every row.
        values: Outcome of every row.
        propensity: Clipped propensity score of every row.
        clusters: Cluster label of every row, for clustered uncertainty.
        covariance: ``iid`` or ``clustered`` uncertainty.

    Returns:
        The difference of the normalized treated and control means (the
        ATE), its variance, and a dictionary with the effective sample size
        and mean of each arm and the uncertainty description.

    Raises:
        ValueError: If either arm's weights sum to a non-positive value, or
            if the variance calculation rejects its inputs.
    """
    treated_weights = treatment / propensity
    control_weights = (1.0 - treatment) / (1.0 - propensity)
    treated_denominator = float(numpy.sum(treated_weights))
    control_denominator = float(numpy.sum(control_weights))
    if treated_denominator <= 0.0 or control_denominator <= 0.0:
        raise errors.InputError(
            "ipw_ate treatment weights must have positive support"
        )
    treated_mean = float(
        numpy.sum(treated_weights * values) / treated_denominator
    )
    control_mean = float(
        numpy.sum(control_weights * values) / control_denominator
    )
    influence = treated_weights * (values - treated_mean) - control_weights * (
        values - control_mean
    )
    variance, uncertainty = dispersion(influence, covariance, clusters)
    effective_sample_size = {
        "treated": float(
            treated_denominator**2 / numpy.sum(treated_weights**2)
        ),
        "control": float(
            control_denominator**2 / numpy.sum(control_weights**2)
        ),
    }
    return (
        treated_mean - control_mean,
        variance,
        {
            "effective_sample_size": effective_sample_size,
            "treatment_means": {
                "treated": treated_mean,
                "control": control_mean,
            },
            "uncertainty_method": uncertainty["method"],
            "uncertainty": uncertainty,
        },
    )


def outcomes(
    design: numpy.ndarray,
    values: numpy.ndarray,
    arms: tuple[numpy.ndarray, numpy.ndarray],
    evaluation: numpy.ndarray,
    message: str,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Fits separate least-squares outcome models for the two arms.

    Args:
        design: Design matrix of the training rows, with intercept.
        values: Outcomes of the training rows.
        arms: Masks of the treated and of the control training rows.
        evaluation: Design matrix of the rows to predict.
        message: Error message for an outcome design that is rank deficient.

    Returns:
        The predicted outcome under control and under treatment for the
        evaluation rows.

    Raises:
        ValueError: If either arm's outcome design is rank deficient.
    """
    treated, control = arms
    treated_design = design[treated]
    control_design = design[control]
    if (
        numpy.linalg.matrix_rank(treated_design) != treated_design.shape[1]
        or numpy.linalg.matrix_rank(control_design) != control_design.shape[1]
    ):
        raise errors.InputError(message)
    treated_coefficients = numpy.linalg.lstsq(
        treated_design, values[treated], rcond=None
    )[0]
    control_coefficients = numpy.linalg.lstsq(
        control_design, values[control], rcond=None
    )[0]
    return evaluation @ control_coefficients, evaluation @ treated_coefficients


def aipw(
    outcome_treated: numpy.ndarray,
    outcome_control: numpy.ndarray,
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    propensity: numpy.ndarray,
) -> numpy.ndarray:
    """Returns the augmented inverse-probability-weighted score of each row.

    Args:
        outcome_treated: Predicted outcome under treatment.
        outcome_control: Predicted outcome under control.
        treatment: 0/1 treatment.
        values: Observed outcome.
        propensity: Clipped propensity score.

    Returns:
        The outcome-model contrast plus the weighted residual corrections.
    """
    score: numpy.ndarray = (
        outcome_treated
        - outcome_control
        + treatment * (values - outcome_treated) / propensity
        - (1.0 - treatment) * (values - outcome_control) / (1.0 - propensity)
    )
    return score
