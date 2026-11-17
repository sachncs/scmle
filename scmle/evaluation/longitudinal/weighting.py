"""Propensity fits, weights, covariance and the independent recomputation."""

from collections.abc import Mapping
from collections.abc import Sequence
import datetime
import hashlib
import json
import math
from typing import Any

import numpy
import pandas
from scipy import special

from scmle.contracts import errors
from scmle.evaluation.longitudinal.declaration import Config
from scmle.evaluation.longitudinal.declaration import MATRIX_NDIM
from scmle.evaluation.longitudinal.declaration import PeriodATE


def regress(
    covariates: numpy.ndarray,
    treatment: numpy.ndarray,
    policy: Config,
) -> tuple[numpy.ndarray, int]:
    """Fits a deterministic ridge-logistic propensity model by Newton steps.

    The ridge penalty is fixed at 1e-8 and the starting coefficients are
    zero.

    Args:
        covariates: Two-dimensional matrix, rows by features, without an
            intercept column; one is added.
        treatment: Treatment indicator per row (0 or 1).
        policy: Supplies the iteration limit and convergence tolerance.

    Returns:
        A tuple of the fitted treatment probability for each row and the
        number of iterations used.

    Raises:
        ValueError: If `covariates` is not two-dimensional, the Hessian is
            singular, a step is not finite, or the fit does not converge
            within `policy.max_iterations`.
    """
    if covariates.ndim != MATRIX_NDIM:
        raise errors.InputError("covariates must be a two-dimensional matrix")
    design = numpy.column_stack((numpy.ones(len(treatment)), covariates))
    coefficients = numpy.zeros(design.shape[1], dtype=float)
    ridge = 1e-8
    ridge_matrix = ridge * numpy.eye(design.shape[1])
    for iteration in range(1, policy.max_iterations + 1):
        probabilities = special.expit(design @ coefficients)
        variance = numpy.maximum(probabilities * (1.0 - probabilities), 1e-12)
        gradient = design.T @ (probabilities - treatment) + ridge * coefficients
        hessian = design.T @ (variance[:, None] * design) + ridge_matrix
        try:
            step = numpy.linalg.solve(hessian, gradient)
        except numpy.linalg.LinAlgError as error:
            raise errors.InputError(
                "propensity model is not identifiable"
            ) from error
        if not numpy.isfinite(step).all():
            raise errors.InputError(
                "propensity model produced non-finite coefficients"
            )
        coefficients -= step
        if float(numpy.max(numpy.abs(step))) <= policy.tolerance:
            probabilities = special.expit(design @ coefficients)
            return probabilities, iteration
    raise errors.InputError(
        "propensity model did not converge within the policy bound"
    )


def effective_sample_size(weights: numpy.ndarray) -> float:
    """Returns the Kish effective sample size for positive arm weights.

    Args:
        weights: Non-negative weights of one arm.

    Returns:
        The squared sum of the weights divided by the sum of their squares,
        or 0.0 when the sum of squares is not positive.
    """
    total = float(numpy.sum(weights))
    squared = float(numpy.sum(weights**2))
    if squared <= 0.0:
        return 0.0
    return total**2 / squared


def weigh(
    propensity: numpy.ndarray,
    clipped: numpy.ndarray,
    treatment: numpy.ndarray,
    outcome: numpy.ndarray,
    policy: Config,
    period_value: Any,
) -> Mapping[str, Any]:
    """Computes one period's Hajek contrast and its influence values.

    Args:
        propensity: Fitted propensities of the period, before clipping.
        clipped: The propensities clipped to the policy bound.
        treatment: Treatment indicator per row (0 or 1).
        outcome: Outcome per row.
        policy: Supplies the effective-sample-size floor.
        period_value: The period's time value, used in error messages.

    Returns:
        The estimate, the common-support bounds, the effective sample size of
        each arm, the per-row influence values and the weighted mean of each
        arm (as `reference_treated_mean` and `reference_control_mean`).

    Raises:
        ValueError: If the treated and control propensity ranges do not
            overlap, an arm has no weighted support, or an arm's effective
            sample size is below the policy minimum.
    """
    treated_propensity = propensity[treatment == 1.0]
    control_propensity = propensity[treatment == 0.0]
    common_support_lower = float(
        max(numpy.min(treated_propensity), numpy.min(control_propensity))
    )
    common_support_upper = float(
        min(numpy.max(treated_propensity), numpy.max(control_propensity))
    )
    if common_support_lower >= common_support_upper:
        raise errors.InputError(
            f"period {period_value!r} has insufficient propensity overlap"
        )
    treated_weights = treatment / clipped
    control_weights = (1.0 - treatment) / (1.0 - clipped)
    treated_weight_sum = float(numpy.sum(treated_weights))
    control_weight_sum = float(numpy.sum(control_weights))
    if treated_weight_sum <= 0.0 or control_weight_sum <= 0.0:
        raise errors.InputError(
            f"period {period_value!r} has no weighted support"
        )
    treated_mean = float(
        numpy.sum(treated_weights * outcome) / treated_weight_sum
    )
    control_mean = float(
        numpy.sum(control_weights * outcome) / control_weight_sum
    )
    estimate = treated_mean - control_mean
    influence = treated_weights * (outcome - treated_mean) / (
        treated_weight_sum / len(outcome)
    ) - control_weights * (outcome - control_mean) / (
        control_weight_sum / len(outcome)
    )
    ess_treated = effective_sample_size(treated_weights)
    ess_control = effective_sample_size(control_weights)
    if (
        ess_treated < policy.minimum_effective_sample_size
        or ess_control < policy.minimum_effective_sample_size
    ):
        raise errors.InputError(
            f"period {period_value!r} has insufficient effective sample size"
        )
    return {
        "estimate": estimate,
        "common_support_lower": common_support_lower,
        "common_support_upper": common_support_upper,
        "effective_sample_size_treated": ess_treated,
        "effective_sample_size_control": ess_control,
        "influence": influence,
        "reference_treated_mean": treated_mean,
        "reference_control_mean": control_mean,
    }


def sandwich(fitted: Sequence[Mapping[str, Any]]) -> numpy.ndarray:
    """Estimates covariance from subject-aligned influence values.

    Args:
        fitted: One `wave` record per period, in period order.

    Returns:
        A symmetric matrix with one row and column per period.  Each entry
        sums, over subjects present in both periods, the product of their
        influence values, each divided by its period's row count and
        centred on the period mean; a diagonal entry with more than one
        subject is scaled by `n / (n - 1)`.  Entries are 0.0 when no subject
        is shared.
    """
    periods = len(fitted)
    covariance = numpy.zeros((periods, periods), dtype=float)
    influence_by_period: list[dict[Any, float]] = []
    for item in fitted:
        subjects = item["subjects"]
        influence = numpy.asarray(item["influence"], dtype=float)
        count = int(item["rows"])
        values = {
            subject: float(value / count)
            for subject, value in zip(subjects, influence, strict=True)
        }
        mean = float(numpy.mean(list(values.values())))
        influence_by_period.append(
            {subject: value - mean for subject, value in values.items()}
        )
    for first in range(periods):
        for second in range(first, periods):
            first_values = influence_by_period[first]
            second_values = influence_by_period[second]
            shared = set(first_values).intersection(second_values)
            if not shared:
                value = 0.0
            else:
                value = sum(
                    first_values[subject] * second_values[subject]
                    for subject in shared
                )
                if first == second and len(shared) > 1:
                    value *= len(shared) / (len(shared) - 1)
            covariance[first, second] = value
            covariance[second, first] = value
    return (covariance + covariance.T) / 2.0


def periodise(
    fitted: Mapping[str, Any], variance: float, critical_value: float
) -> PeriodATE:
    """Builds a serializable period result after covariance is available.

    Args:
        fitted: The `wave` record of the period.
        variance: Variance of the period estimate; negative values are
            treated as zero.
        critical_value: Normal quantile that scales the standard error into
            the interval half-width.

    Returns:
        The period estimate with its standard error, interval, and fit
        diagnostics, marked as converged.
    """
    standard_error = math.sqrt(max(float(variance), 0.0))
    estimate = float(fitted["estimate"])
    return PeriodATE(
        period=str(fitted["period"]),
        estimate=estimate,
        standard_error=standard_error,
        interval=(
            estimate - critical_value * standard_error,
            estimate + critical_value * standard_error,
        ),
        rows=int(fitted["rows"]),
        treated_rows=int(fitted["treated_rows"]),
        control_rows=int(fitted["control_rows"]),
        propensity_min=float(fitted["propensity_min"]),
        propensity_max=float(fitted["propensity_max"]),
        fraction_clipped=float(fitted["fraction_clipped"]),
        effective_sample_size_treated=float(
            fitted["effective_sample_size_treated"]
        ),
        effective_sample_size_control=float(
            fitted["effective_sample_size_control"]
        ),
        converged=True,
        iterations=int(fitted["iterations"]),
    )


def referee(
    fitted: Sequence[Mapping[str, Any]], tolerance: float
) -> dict[str, Any]:
    """Recalculates normalized weighted means through an independent path.

    Args:
        fitted: One `wave` record per period.
        tolerance: Largest absolute difference counted as agreement.

    Returns:
        A dictionary with the method name, the tolerance, an overall
        "passed" flag, and one record per period (period label, recomputed
        estimate, primary estimate, absolute difference, pass flag, arm row
        counts, propensity range, the method name, and the recomputed arm
        means).
    """
    records = []
    for item in fitted:
        propensity = numpy.asarray(item["propensity"], dtype=float)
        clipped = numpy.asarray(item["clipped_propensity"], dtype=float)
        treatment = numpy.asarray(item["treatment"], dtype=float)
        outcome = numpy.asarray(item["outcome"], dtype=float)
        treated_count = int(item["treated_rows"])
        control_count = int(item["control_rows"])
        treated_weights = numpy.where(treatment == 1.0, 1.0 / clipped, 0.0)
        control_weights = numpy.where(
            treatment == 0.0, 1.0 / (1.0 - clipped), 0.0
        )
        treated_mean = float(
            numpy.sum(treated_weights * outcome) / numpy.sum(treated_weights)
        )
        control_mean = float(
            numpy.sum(control_weights * outcome) / numpy.sum(control_weights)
        )
        primary = float(item["estimate"])
        reference_estimate = treated_mean - control_mean
        difference = abs(primary - reference_estimate)
        records.append(
            {
                "period": str(item["period"]),
                "estimate": reference_estimate,
                "primary_estimate": primary,
                "absolute_difference": difference,
                "passed": difference <= tolerance,
                "treated_rows": treated_count,
                "control_rows": control_count,
                "propensity_range": [
                    float(numpy.min(propensity)),
                    float(numpy.max(propensity)),
                ],
                "independent_method": "direct_normalized_weighted_means",
                "treated_mean": treated_mean,
                "control_mean": control_mean,
            }
        )
    return {
        "method": "independent_direct_hajek_recalculation",
        "tolerance": tolerance,
        "passed": all(bool(record["passed"]) for record in records),
        "periods": records,
    }


def periodize(value: Any) -> str:
    """Converts a pandas/numpy time value to a stable JSON label.

    Args:
        value: A time value from the time column.

    Returns:
        The ISO 8601 text for a timestamp, date, or datetime, otherwise
        `str` of the value with NumPy scalars unwrapped first.
    """
    if isinstance(value, pandas.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, numpy.generic):
        value = value.item()
    return str(value)


def imprint(frame: pandas.DataFrame) -> str:
    """Returns a deterministic digest of the supplied frame.

    Args:
        frame: Frame to fingerprint.

    Returns:
        The hexadecimal SHA-256 digest covering the column names, dtypes,
        shape, and hashed row values including the index.

    Raises:
        ValueError: If the frame values cannot be hashed.
    """
    try:
        values = (
            pandas.util.hash_pandas_object(frame, index=True)
            .to_numpy()
            .tobytes()
        )
    except (TypeError, ValueError) as error:
        raise errors.InputError(
            "frame values must be hashable for fingerprinting"
        ) from error
    descriptor = json.dumps(
        {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "shape": list(frame.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    digest.update(values)
    return digest.hexdigest()
