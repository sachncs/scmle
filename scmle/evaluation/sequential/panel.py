"""Validates and prepares the balanced long-format panel."""

from collections.abc import Sequence
from itertools import pairwise
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.evaluation.sequential.declaration import Config
from scmle.evaluation.sequential.declaration import Estimand
from scmle.evaluation.sequential.declaration import MIN_PERIODS


def conform(frame: pandas.DataFrame, estimand: Estimand) -> list[str]:
    """Lists the declared columns and requires all of them in the frame.

    Args:
        frame: Long-format panel.
        estimand: Declares the columns that must be present.

    Returns:
        The declared column names: subject, time, treatment, outcome and
        covariates.

    Raises:
        ValueError: If a declared column is missing from the frame.
    """
    required = [
        estimand.subject,
        estimand.time,
        estimand.treatment,
        estimand.outcome,
        *estimand.baseline_covariates,
        *estimand.time_varying_covariates,
    ]
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise errors.InputError(
            f"sequential frame is missing columns: {missing_columns}"
        )
    return required


def situate(frame: pandas.DataFrame, estimand: Estimand) -> None:
    """Checks the subject identities and time values of the panel.

    Args:
        frame: Long-format panel.
        estimand: Names the subject and time columns.

    Raises:
        ValueError: If subject identities or time values are missing or
            non-finite, subject-time pairs repeat, or the time values are
            invalid.
    """
    subject_name = estimand.subject
    time_name = estimand.time
    if frame[subject_name].isna().any() or frame[time_name].isna().any():
        raise errors.InputError(
            "subject identities and time values must not be missing"
        )
    for name in (subject_name, time_name):
        if pandas.api.types.is_numeric_dtype(frame[name]):
            values = frame[name].to_numpy(dtype=float)
            if not numpy.isfinite(values).all():
                raise errors.InputError(
                    f"column {name!r} contains non-finite values"
                )
    if frame.duplicated([subject_name, time_name]).any():
        raise errors.InputError("duplicate subject-time rows are not allowed")
    chronologize(frame, estimand)


def numericize(prepared: pandas.DataFrame, value_columns: list[str]) -> None:
    """Converts the value columns to numbers in place and checks them.

    Args:
        prepared: The prepared frame, changed in place.
        value_columns: Treatment, outcome and covariate columns.

    Raises:
        ValueError: If a column is non-numeric or holds non-finite values.
    """
    for name in value_columns:
        try:
            prepared[name] = pandas.to_numeric(prepared[name], errors="raise")
        except (TypeError, ValueError) as error:
            raise errors.InputError(
                f"column {name!r} must be numeric"
            ) from error
        values = prepared[name].to_numpy(dtype=float)
        if not numpy.isfinite(values).all():
            raise errors.InputError(
                f"column {name!r} contains non-finite values"
            )


def nonrepeating(values: list[Any]) -> None:
    """Rejects equal consecutive time values.

    Args:
        values: One subject's time values in row order.

    Raises:
        ValueError: If two consecutive values are equal.
    """
    if any(left == right for left, right in pairwise(values)):
        raise errors.InputError(
            "time values must be strictly increasing per subject"
        )


def prepare(
    frame: pandas.DataFrame,
    estimand: Estimand,
    policy: Config,
) -> tuple[pandas.DataFrame, int, int, int]:
    """Validates and prepares a balanced long-format panel.

    Args:
        frame: Long-format panel to check.
        estimand: Declares the columns that must be present.
        policy: Supplies the missingness rule and the minimum subject count.

    Returns:
        A tuple of the prepared frame (declared columns only, with the
        treatment, outcome and covariate columns converted to numeric and
        dropped subjects removed), the number of rows dropped, the number of
        distinct subjects in the input, and the number of subjects dropped.

    Raises:
        TypeError: If `frame` is not a pandas DataFrame.
        ValueError: If the frame is empty, declared columns are missing,
            subject or time values are missing or non-finite, subject-time
            pairs repeat, time values are invalid, values are missing under
            the "reject" policy, a column is non-numeric or non-finite,
            nothing remains after dropping, treatment is not binary,
            fewer than two periods remain, a subject lacks a row in some
            period, too few subjects remain, or a baseline covariate varies
            within a subject.
    """
    if not isinstance(frame, pandas.DataFrame):
        raise errors.KindError("frame must be a pandas DataFrame")
    if frame.empty:
        raise errors.InputError("sequential frame must not be empty")
    required = conform(frame, estimand)
    subject_name = estimand.subject
    time_name = estimand.time
    situate(frame, estimand)
    subjects_input = int(frame[subject_name].nunique())
    value_columns = [
        estimand.treatment,
        estimand.outcome,
        *estimand.baseline_covariates,
        *estimand.time_varying_covariates,
    ]
    missing_mask = frame[value_columns].isna().any(axis=1)
    subjects_dropped = 0
    rows_dropped = 0
    prepared = frame[required].copy()
    if missing_mask.any() and policy.missingness == "reject":
        raise errors.InputError(
            "missing treatment, outcome, or covariate values"
        )
    if missing_mask.any():
        dropped_subjects = set(frame.loc[missing_mask, subject_name].tolist())
        keep_mask = ~prepared[subject_name].isin(dropped_subjects)
        subjects_dropped = len(dropped_subjects)
        rows_dropped = int((~keep_mask).sum())
        prepared = prepared.loc[keep_mask].copy()
    numericize(prepared, value_columns)
    if prepared.empty:
        raise errors.InputError("missingness policy left no subjects")
    treatment_values = set(prepared[estimand.treatment].unique().tolist())
    if not treatment_values.issubset({0, 1}):
        raise errors.InputError("treatment must be binary with values 0 and 1")
    period_values = order(prepared, time_name)
    if len(period_values) < MIN_PERIODS:
        raise errors.InputError(
            "sequential estimation needs at least two periods"
        )
    counts = prepared.groupby(subject_name, sort=False).size()
    if not (counts == len(period_values)).all():
        raise errors.InputError(
            "each subject must have one row at every period"
        )
    if len(prepared[subject_name].unique()) < policy.minimum_subjects:
        raise errors.InputError("sequential frame has insufficient subjects")
    pretreat(prepared, estimand)
    return prepared, rows_dropped, subjects_input, subjects_dropped


def chronologize(frame: pandas.DataFrame, estimand: Estimand) -> None:
    """Rejects repeated consecutive time values within a subject.

    Only equal neighbouring values (in row order) are rejected; descending
    values are not.

    Args:
        frame: Long-format panel.
        estimand: Names the subject and time columns.

    Raises:
        ValueError: If a subject has two equal consecutive time values, or
            its time values cannot be examined in order.
        TypeError: Re-raised unchanged only if its message contains
            "strictly increasing", which the comparisons here never
            produce; every other `TypeError` becomes `ValueError`.
    """
    try:
        for item in frame.groupby(estimand.subject, sort=False, dropna=False):
            nonrepeating(item[1][estimand.time].tolist())
    except (TypeError, ValueError) as error:
        if "strictly increasing" in str(error):
            raise
        raise errors.InputError("time values have invalid ordering") from error


def order(frame: pandas.DataFrame, time_column: str) -> list[Any]:
    """Returns stable sorted period values.

    Args:
        frame: Long-format panel.
        time_column: Name of the time column.

    Returns:
        The distinct time values in ascending order.

    Raises:
        ValueError: If the time values cannot be sorted.
    """
    try:
        return frame[time_column].drop_duplicates().sort_values().tolist()
    except (TypeError, ValueError) as error:
        raise errors.InputError("time values have invalid ordering") from error


def pretreat(frame: pandas.DataFrame, estimand: Estimand) -> None:
    """Ensures declared baseline covariates are stable within subjects.

    Args:
        frame: Long-format panel.
        estimand: Names the subject column and the baseline covariates.

    Raises:
        ValueError: If a baseline covariate takes more than one value
            (counting missing as a value) within a subject.
    """
    grouped = frame.groupby(estimand.subject, sort=False)
    for name in estimand.baseline_covariates:
        if (grouped[name].nunique(dropna=False) > 1).any():  # noqa: PD101 -- per-subject distinct counts; Series.is_unique does not apply
            raise errors.InputError(
                f"baseline covariate {name!r} varies within a subject"
            )


def tabulate(
    frame: pandas.DataFrame,
    estimand: Estimand,
    period_values: Sequence[Any],
    subject_values: Sequence[Any],
) -> dict[str, numpy.ndarray]:
    """Builds deterministic subject-by-period arrays from the long frame.

    Args:
        frame: Prepared long-format panel.
        estimand: Names the treatment, outcome, and covariate columns.
        period_values: Periods in time order; fixes the period axis.
        subject_values: Subjects in order; fixes the subject axis.

    Returns:
        A dictionary with "treatment" and "outcome" (subjects by periods),
        "baseline" (subjects by baseline covariates, read at the first
        period), and "time_varying" (subjects by periods by time-varying
        covariates).

    Raises:
        ValueError: If a subject's first-period subject-time pair maps to
            more than one row.
    """
    indexed = frame.set_index([estimand.subject, estimand.time], drop=False)
    treatment = numpy.empty(
        (len(subject_values), len(period_values)), dtype=float
    )
    outcome = numpy.empty_like(treatment)
    baseline = numpy.empty(
        (len(subject_values), len(estimand.baseline_covariates))
    )
    time_varying = numpy.empty(
        (
            len(subject_values),
            len(period_values),
            len(estimand.time_varying_covariates),
        )
    )
    for subject_index, subject in enumerate(subject_values):
        first = indexed.loc[(subject, period_values[0])]
        if isinstance(first, pandas.DataFrame):
            raise ValueError("subject-time index is not unique")
        baseline[subject_index] = first[
            list(estimand.baseline_covariates)
        ].to_numpy(dtype=float)
        for period_index, period in enumerate(period_values):
            row = indexed.loc[(subject, period)]
            treatment[subject_index, period_index] = float(
                row[estimand.treatment]
            )
            outcome[subject_index, period_index] = float(row[estimand.outcome])
            time_varying[subject_index, period_index] = row[
                list(estimand.time_varying_covariates)
            ].to_numpy(dtype=float)
    return {
        "treatment": treatment,
        "outcome": outcome,
        "baseline": baseline,
        "time_varying": time_varying,
    }
