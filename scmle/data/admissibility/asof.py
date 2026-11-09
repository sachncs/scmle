"""As-of times derived from event or change logs by point-in-time join.

Where a change log exists, the value a row could have used at its prediction
moment is the latest log entry written at or before that moment (an as-of or
point-in-time join; ties on the write time go to the later log entry). Rows
whose stored value differs from that reconstruction were overwritten or
backfilled after the moment, or hold a value written after it. The share of
such rows is reported with a Wilson interval.

This is a census of the rows given, not a test: the reconstruction is exact
arithmetic on the log. Without a log nothing here applies, and the as-of rule
must be an attestation (see `provenance`). No as-of time is inferred from the
data distribution alone.
"""

import dataclasses
import enum

import numpy
import pandas

from scmle.contracts import qualification
from scmle.data.admissibility import provenance

DETECTOR_ID = "f4-asof-reconstruction"
DEFECT_TYPE = "asof_overwrite"
EPOCH = pandas.Timestamp("1970-01-01")
SECOND = pandas.Timedelta(seconds=1)


class RowStatus(enum.StrEnum):
    """How a row's stored value relates to the log."""

    MATCH = "match"
    OVERWRITTEN = "overwritten"
    NOT_YET_WRITTEN = "not_yet_written"


def to_seconds(values: pandas.Series) -> numpy.ndarray:
    """Converts times to float seconds; datetimes count from the Unix epoch.

    Args:
        values: A datetime series (timezone is dropped) or a numeric,
            non-boolean series, which is passed through as seconds.

    Returns:
        A float64 array of seconds, in the order of `values`.

    Raises:
        ValueError: On values that are neither datetimes nor numbers, or on
            missing times.
    """
    if values.isna().any():
        raise ValueError("times must not be missing")
    if pandas.api.types.is_datetime64_any_dtype(values):
        naive = values.dt.tz_localize(None) if values.dt.tz else values
        return ((naive - EPOCH) / SECOND).to_numpy(dtype="float64")
    if pandas.api.types.is_numeric_dtype(values) and not (
        pandas.api.types.is_bool_dtype(values)
    ):
        return values.to_numpy(dtype="float64")
    raise ValueError("times must be datetimes or numbers")


def point_in_time(
    rows: pandas.DataFrame,
    log: pandas.DataFrame,
    key: str,
    moment: str,
    written_at: str,
    value: str,
) -> pandas.DataFrame:
    """Finds the latest logged value at or before each row's moment.

    Args:
        rows: One row per case, with `key` and `moment` columns.
        log: Change log with `key`, `written_at` and `value` columns, in
            the order the changes were applied.
        key: Join key present in both tables.
        moment: Column of `rows` holding the prediction moment (datetimes
            or numbers; see `to_seconds`).
        written_at: Column of `log` holding the write time.
        value: Column of `log` holding the written value.

    Returns:
        A frame indexed like `rows` with `reconstructed` (missing when no
        entry precedes the moment), `known_at` (write time in seconds, NaN
        when none) and `has_entry`. Among entries for one key with the same
        write time, the one later in `log` wins.
    """
    left = pandas.DataFrame(
        {
            "k": rows[key].to_numpy(),
            "t": to_seconds(rows[moment]),
            "i": numpy.arange(len(rows)),
        }
    )
    times = to_seconds(log[written_at])
    right = pandas.DataFrame(
        {
            "k": log[key].to_numpy(),
            "t": times,
            "w": times,
            "v": log[value].to_numpy(),
            "seq": numpy.arange(len(log)),
        }
    )
    right = right.sort_values(["t", "seq"], kind="stable").drop_duplicates(
        ["k", "t"], keep="last"
    )
    joined = pandas.merge_asof(
        left.sort_values("t", kind="stable"),
        right[["k", "t", "w", "v"]],
        on="t",
        by="k",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("i")
    return pandas.DataFrame(
        {
            "reconstructed": joined["v"].to_numpy(),
            "known_at": joined["w"].to_numpy(dtype="float64"),
            "has_entry": joined["w"].notna().to_numpy(),
        },
        index=rows.index,
    )


def agrees(
    stored: pandas.Series, reconstructed: pandas.Series
) -> pandas.Series:
    """Compares two series element-wise, treating two missing values as equal.

    Args:
        stored: The stored values.
        reconstructed: The values to compare with, same length as `stored`.

    Returns:
        A boolean series on the index of `stored`; True where the values
        are equal or both missing.
    """
    left = stored.to_numpy(dtype=object)
    right = reconstructed.to_numpy(dtype=object)
    both_missing = stored.isna().to_numpy() & reconstructed.isna().to_numpy()
    equal = numpy.array([a == b for a, b in zip(left, right, strict=True)])
    return pandas.Series(
        (equal | both_missing).astype(bool), index=stored.index, dtype=bool
    )


def classify_rows(
    stored: pandas.Series, joined: pandas.DataFrame
) -> pandas.Series:
    """Classifies every row against the reconstruction.

    Args:
        stored: The stored value of each row.
        joined: Output of `point_in_time`, aligned to `stored`.

    Returns:
        A series of `RowStatus` values (as strings) on the index of
        `stored`: `match` where the values agree, `not_yet_written` where
        no log entry precedes the moment but a value is stored, and
        `overwritten` otherwise.
    """
    match = agrees(stored, joined["reconstructed"])
    not_yet = (~joined["has_entry"]) & stored.notna()
    status = pandas.Series(RowStatus.OVERWRITTEN.value, index=stored.index)
    status[match] = RowStatus.MATCH.value
    status[not_yet] = RowStatus.NOT_YET_WRITTEN.value
    return status


def reconstruction_mismatch(
    rows: pandas.DataFrame,
    log: pandas.DataFrame,
    key: str,
    moment: str,
    stored: str,
    written_at: str,
    value: str,
) -> pandas.Series:
    """Flags rows whose stored value is not the log value at the moment.

    Args:
        rows: Cases with `key`, `moment` and the stored column.
        log: Change log (see `point_in_time`).
        key: Join key.
        moment: Prediction-moment column of `rows`.
        stored: Column of `rows` that holds the stored value.
        written_at: Write-time column of `log`.
        value: Value column of `log`.

    Returns:
        A boolean series aligned to `rows`; True for every row that is not
        a `match` (overwritten or not yet written).
    """
    joined = point_in_time(rows, log, key, moment, written_at, value)
    status = classify_rows(rows[stored], joined)
    return pandas.Series(
        (status != RowStatus.MATCH.value).to_numpy(),
        index=rows.index,
        dtype=bool,
    )


@dataclasses.dataclass(frozen=True, eq=False)
class AsOfAudit:
    """Outcome of reconstructing one field from its log.

    Attributes:
        field: Field name.
        status: Status of every row.
        log_check: Counts and tolerance, for the provenance record.
        as_of: Derived rule: the smallest lead over matching rows, or
            unknown when no row matches.
        interval: Wilson interval for the mismatch share.
        check: The ledger record.
    """

    field: str
    status: pandas.Series
    log_check: provenance.LogCheck
    as_of: provenance.AsOf
    interval: tuple[float, float]
    check: qualification.Check

    @property
    def counts(self) -> dict[str, int]:
        """Rows per status."""
        return {
            kind.value: int((self.status == kind.value).sum())
            for kind in RowStatus
        }
