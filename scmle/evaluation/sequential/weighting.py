"""Propensity fits, regime means, covariance and the recomputation."""

import hashlib
import json
import math
from typing import Any

import numpy
import pandas
from scipy import special

from scmle.contracts import errors
from scmle.evaluation.sequential.declaration import Config
from scmle.evaluation.sequential.declaration import MATRIX_NDIM
from scmle.evaluation.sequential.report import RegimeEstimate


def regress(
    covariates: numpy.ndarray,
    treatment: numpy.ndarray,
    policy: Config,
) -> tuple[numpy.ndarray, int]:
    """Fits a deterministic ridge-logistic propensity by bounded Newton steps.

    Args:
        covariates: Two-dimensional matrix, subjects by features, without an
            intercept column; one is added.
        treatment: Treatment indicator per subject (0 or 1).
        policy: Supplies the iteration limit and convergence tolerance.

    Returns:
        A tuple of the fitted treatment probability for each subject and
        the number of iterations used.

    Raises:
        ValueError: If `covariates` is not two-dimensional, its rows do not
            match `treatment`, the Hessian is singular, a step is not
            finite, or the fit does not converge within
            `policy.max_iterations`.
    """
    if covariates.ndim != MATRIX_NDIM:
        raise errors.InputError("covariates must be a two-dimensional matrix")
    if len(covariates) != len(treatment):
        raise errors.InputError(
            "covariates and treatment must have matching rows"
        )
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
            return special.expit(design @ coefficients), iteration
    raise errors.InputError(
        "propensity model did not converge within policy bound"
    )


def overlap(propensity: numpy.ndarray, period: Any, minimum: float) -> None:
    """Rejects fitted propensities outside the declared positivity support.

    Args:
        propensity: Fitted probabilities for one period.
        period: Period value, used in the error message.
        minimum: Smallest allowed probability; the largest allowed is
            `1 - minimum`.

    Raises:
        ValueError: If any probability is below `minimum` or above
            `1 - minimum`.
    """
    lower = float(numpy.min(propensity))
    upper = float(numpy.max(propensity))
    if lower < minimum or upper > 1.0 - minimum:
        raise errors.InputError(
            f"period {period!r} violates declared positivity support "
            f"[{minimum}, {1.0 - minimum}]"
        )


def effective_sample_size(weights: numpy.ndarray) -> float:
    """Returns the Kish effective sample size for positive weights.

    Args:
        weights: Non-negative weights.

    Returns:
        The squared sum of the weights divided by the sum of their squares,
        or 0.0 when the sum of squares is not positive.
    """
    total = float(numpy.sum(weights))
    squared = float(numpy.sum(weights**2))
    if squared <= 0.0:
        return 0.0
    return total**2 / squared


def average(
    regime: str,
    treatment: int,
    mask: numpy.ndarray,
    weights: numpy.ndarray,
    outcome: numpy.ndarray,
    policy: Config,
    critical_value: float,
) -> RegimeEstimate:
    """Builds a normalized weighted regime mean and influence standard error.

    Args:
        regime: Name of the regime.
        treatment: Treatment value held in every period.
        mask: Boolean flag per subject for following the regime.
        weights: Capped cumulative weight per subject.
        outcome: Final-period outcome per subject.
        policy: Supplies the support and effective-sample-size minimums.
        critical_value: Normal quantile that scales the standard error into
            the interval half-width.

    Returns:
        The weighted mean over the selected subjects with its standard
        error, interval, row count, weight sum, and effective sample size.

    Raises:
        ValueError: If fewer than `policy.minimum_arm_rows` subjects follow
            the regime, their weights sum to zero or less, or their
            effective sample size is below the policy minimum.
    """
    selected_weights = weights[mask]
    selected_outcome = outcome[mask]
    if len(selected_weights) < policy.minimum_arm_rows:
        raise errors.InputError(
            f"regime {regime!r} has insufficient treatment-history support"
        )
    weight_sum = float(numpy.sum(selected_weights))
    if weight_sum <= 0.0:
        raise errors.InputError(f"regime {regime!r} has no weighted support")
    ess = effective_sample_size(selected_weights)
    if ess < policy.minimum_effective_sample_size:
        raise errors.InputError(
            f"regime {regime!r} has insufficient effective sample size"
        )
    estimate = float(
        numpy.sum(selected_weights * selected_outcome) / weight_sum
    )
    influence = selected_weights * (selected_outcome - estimate) / weight_sum
    standard_error = math.sqrt(max(float(numpy.sum(influence**2)), 0.0))
    return RegimeEstimate(
        regime=regime,
        treatment=treatment,
        estimate=estimate,
        standard_error=standard_error,
        interval=(
            estimate - critical_value * standard_error,
            estimate + critical_value * standard_error,
        ),
        rows=len(selected_weights),
        weight_sum=weight_sum,
        effective_sample_size=ess,
    )


def covary(
    target_mask: numpy.ndarray,
    comparator_mask: numpy.ndarray,
    weights: numpy.ndarray,
    outcome: numpy.ndarray,
    target: RegimeEstimate,
    comparator: RegimeEstimate,
) -> numpy.ndarray:
    """Estimates subject-level covariance of two normalized regime means.

    Args:
        target_mask: Boolean flag per subject for following the target
            regime.
        comparator_mask: Boolean flag per subject for following the
            comparator regime.
        weights: Capped cumulative weight per subject.
        outcome: Final-period outcome per subject.
        target: Target regime estimate.
        comparator: Comparator regime estimate.

    Returns:
        A symmetric 2 by 2 array ordered target then comparator, holding
        sums over subjects of the products of the influence values.
    """
    target_influence = numpy.zeros(len(outcome), dtype=float)
    comparator_influence = numpy.zeros(len(outcome), dtype=float)
    target_influence[target_mask] = (
        weights[target_mask]
        * (outcome[target_mask] - target.estimate)
        / target.weight_sum
    )
    comparator_influence[comparator_mask] = (
        weights[comparator_mask]
        * (outcome[comparator_mask] - comparator.estimate)
        / comparator.weight_sum
    )
    return numpy.asarray(
        [
            [
                numpy.sum(target_influence**2),
                numpy.sum(target_influence * comparator_influence),
            ],
            [
                numpy.sum(target_influence * comparator_influence),
                numpy.sum(comparator_influence**2),
            ],
        ],
        dtype=float,
    )


def referee(
    target_mask: numpy.ndarray,
    comparator_mask: numpy.ndarray,
    weights: numpy.ndarray,
    outcome: numpy.ndarray,
    target: RegimeEstimate,
    comparator: RegimeEstimate,
    tolerance: float,
) -> dict[str, Any]:
    """Recalculates normalized regime means through an independent path.

    Args:
        target_mask: Boolean flag per subject for following the target
            regime.
        comparator_mask: Boolean flag per subject for following the
            comparator regime.
        weights: Capped cumulative weight per subject.
        outcome: Final-period outcome per subject.
        target: Target regime estimate to compare against.
        comparator: Comparator regime estimate to compare against.
        tolerance: Largest absolute difference counted as agreement.

    Returns:
        A dictionary with the method name, the tolerance, an overall
        "passed" flag (true when every regime is within `tolerance`), one
        record per regime in target then comparator order (regime name,
        recomputed estimate, primary estimate, absolute difference, pass
        flag, rows, method name), and the recomputed contrast of target
        minus comparator.
    """
    records: list[dict[str, Any]] = []
    for name, mask, primary in (
        (target.regime, target_mask, target),
        (comparator.regime, comparator_mask, comparator),
    ):
        selected_weights = [float(value) for value in weights[mask]]
        selected_outcome = [float(value) for value in outcome[mask]]
        weighted_total = sum(
            weight * value
            for weight, value in zip(
                selected_weights, selected_outcome, strict=True
            )
        )
        reference_estimate = weighted_total / sum(selected_weights)
        difference = abs(primary.estimate - reference_estimate)
        records.append(
            {
                "regime": name,
                "estimate": reference_estimate,
                "primary_estimate": primary.estimate,
                "absolute_difference": difference,
                "passed": difference <= tolerance,
                "rows": len(selected_weights),
                "independent_method": "direct_normalized_history_weight_mean",
            }
        )
    contrast = float(records[0]["estimate"]) - float(records[1]["estimate"])
    return {
        "method": "independent_direct_normalized_history_weight_means",
        "tolerance": tolerance,
        "passed": all(bool(record["passed"]) for record in records),
        "regimes": records,
        "contrast": contrast,
    }


def periodize(value: Any) -> str:
    """Converts a period value into a stable report label.

    Args:
        value: A period value from the time column.

    Returns:
        The ISO 8601 text for a timestamp, otherwise `str` of the value
        with NumPy scalars unwrapped first.
    """
    if isinstance(value, pandas.Timestamp):
        return value.isoformat()
    if isinstance(value, numpy.generic):
        value = value.item()
    return str(value)


def imprint(frame: pandas.DataFrame) -> str:
    """Returns a deterministic SHA-256 digest of the supplied frame.

    Args:
        frame: Frame to fingerprint.

    Returns:
        The hexadecimal digest covering the column names, dtypes, shape,
        and hashed row values including the index.

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
