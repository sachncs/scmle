"""Validates and prepares the long-format panel before estimation."""

import itertools

import numpy
import pandas

from scmle.contracts import errors
from scmle.evaluation.longitudinal.declaration import Config
from scmle.evaluation.longitudinal.declaration import Estimand


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
        *estimand.covariates,
    ]
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise errors.InputError(
            f"longitudinal frame is missing columns: {missing_columns}"
        )
    return required


def situate(frame: pandas.DataFrame, estimand: Estimand) -> None:
    """Checks the subject identities and time values of the panel.

    Args:
        frame: Long-format panel.
        estimand: Names the subject and time columns.

    Raises:
        ValueError: If subject identities or time values are missing or
            non-finite, subject-time pairs repeat, or the time values are not
            strictly increasing within a subject.
    """
    if frame[estimand.subject].isna().any():
        raise errors.InputError("missing subject identities are not allowed")
    subject_labels = frame[estimand.subject].astype("string")
    if subject_labels.fillna("").str.strip().eq("").any():
        raise errors.InputError("missing subject identities are not allowed")
    if frame[estimand.time].isna().any():
        raise errors.InputError("missing time values are not allowed")
    for name in (estimand.subject, estimand.time):
        if pandas.api.types.is_numeric_dtype(frame[name]):
            values = frame[name].to_numpy(dtype=float)
            if not numpy.isfinite(values).all():
                raise errors.InputError(
                    f"column {name!r} contains non-finite values"
                )
    if frame.duplicated([estimand.subject, estimand.time]).any():
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


def prepare(
    frame: pandas.DataFrame,
    estimand: Estimand,
    policy: Config,
) -> tuple[pandas.DataFrame, int]:
    """Validates and numerically prepares a long-format input frame.

    Args:
        frame: Long-format panel to check.
        estimand: Declares the columns that must be present.
        policy: Supplies the missingness rule and the minimum row count.

    Returns:
        A tuple of the prepared frame (declared columns only; treatment,
        outcome and covariates converted to numbers; rows with a missing
        treatment, outcome or covariate removed under `complete_case`) and
        the number of rows removed.

    Raises:
        TypeError: If `frame` is not a pandas DataFrame.
        ValueError: If the frame is empty, declared columns are missing,
            subject identities or time values are missing or non-finite,
            subject-time pairs repeat, time values are invalid, values are
            missing under the "reject" policy, a column is non-numeric or
            non-finite, treatment is not binary, no rows remain, too few
            rows remain, or a covariate varies within a subject.
    """
    if not isinstance(frame, pandas.DataFrame):
        raise errors.KindError("frame must be a pandas DataFrame")
    if frame.empty:
        raise errors.InputError("longitudinal frame must not be empty")
    required = conform(frame, estimand)
    situate(frame, estimand)
    prepared = frame[required].copy()
    value_columns = [
        estimand.treatment,
        estimand.outcome,
        *estimand.covariates,
    ]
    missing_mask = prepared[value_columns].isna().any(axis=1)
    if missing_mask.any() and policy.missingness == "reject":
        raise errors.InputError(
            "missing treatment, outcome, or covariate values"
        )
    rows_dropped = int(missing_mask.sum())
    if rows_dropped:
        prepared = prepared.loc[~missing_mask].copy()
    numericize(prepared, value_columns)
    treatment_values = set(prepared[estimand.treatment].unique().tolist())
    if not treatment_values.issubset({0, 1}):
        raise errors.InputError("treatment must be binary with values 0 and 1")
    if prepared.empty:
        raise errors.InputError("missingness policy left no rows")
    if len(prepared) < policy.minimum_rows_per_period:
        raise errors.InputError("longitudinal frame has insufficient rows")
    pretreat(prepared, estimand)
    return prepared, rows_dropped


def monotone(times: pandas.Series) -> None:
    """Requires one subject's time values to be strictly increasing.

    Args:
        times: The subject's time values in row order.

    Raises:
        ValueError: If the values decrease or repeat.
    """
    if not times.is_monotonic_increasing:
        raise errors.InputError(
            "time values must be strictly increasing per subject"
        )
    values = times.tolist()
    if any(left == right for left, right in itertools.pairwise(values)):
        raise errors.InputError(
            "time values must be strictly increasing per subject"
        )


def chronologize(frame: pandas.DataFrame, estimand: Estimand) -> None:
    """Requires strictly increasing time values within each subject.

    Args:
        frame: Long-format panel.
        estimand: Names the subject and time columns.

    Raises:
        ValueError: If a subject's time values, in row order, are not
            strictly increasing or cannot be compared.
        TypeError: Re-raised only if its message contains "strictly
            increasing", which comparison errors do not; any other
            `TypeError` becomes `ValueError`.
    """
    subject_name = estimand.subject
    time_name = estimand.time
    try:
        for subject_group in frame.groupby(
            subject_name, sort=False, dropna=False
        ):
            monotone(subject_group[1][time_name])
    except (TypeError, ValueError) as error:
        if "strictly increasing" in str(error):
            raise
        raise errors.InputError("time values have invalid ordering") from error


def pretreat(frame: pandas.DataFrame, estimand: Estimand) -> None:
    """Checks the declared no-time-varying-covariates condition in the data.

    Args:
        frame: Long-format panel.
        estimand: Names the subject column and the covariates.

    Raises:
        ValueError: If a covariate takes more than one value (counting
            missing as a value) within a subject.
    """
    if not estimand.covariates:
        return
    grouped = frame.groupby(estimand.subject, sort=False, dropna=False)
    for name in estimand.covariates:
        if (grouped[name].nunique(dropna=False) > 1).any():  # noqa: PD101 -- per-subject distinct counts; Series.is_unique does not apply
            raise errors.InputError(
                f"covariate {name!r} varies within a subject despite the "
                "no_time_varying_covariates assumption"
            )
