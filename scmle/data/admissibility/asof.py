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

import numpy as np
import pandas as pd

from scmle.contracts import qualification
from scmle.data.admissibility import provenance
from scmle.data.detectors import certificates

DETECTOR_ID = "f4-asof-reconstruction"
DEFECT_TYPE = "asof_overwrite"
EPOCH = pd.Timestamp("1970-01-01")
SECOND = pd.Timedelta(seconds=1)


class RowStatus(enum.StrEnum):
    """How a row's stored value relates to the log."""

    MATCH = "match"
    OVERWRITTEN = "overwritten"
    NOT_YET_WRITTEN = "not_yet_written"


def to_seconds(values: pd.Series) -> np.ndarray:
    """Times as float seconds; datetimes count from the Unix epoch.

    Raises:
        ValueError: On values that are neither datetimes nor numbers, or on
            missing times.
    """
    if values.isna().any():
        raise ValueError("times must not be missing")
    if pd.api.types.is_datetime64_any_dtype(values):
        naive = values.dt.tz_localize(None) if values.dt.tz else values
        return ((naive - EPOCH) / SECOND).to_numpy(dtype="float64")
    if pd.api.types.is_numeric_dtype(values) and not (
        pd.api.types.is_bool_dtype(values)
    ):
        return values.to_numpy(dtype="float64")
    raise ValueError("times must be datetimes or numbers")


def point_in_time(
    rows: pd.DataFrame,
    log: pd.DataFrame,
    key: str,
    moment: str,
    written_at: str,
    value: str,
) -> pd.DataFrame:
    """Latest logged value at or before each row's moment.

    Args:
        rows: One row per case, with `key` and `moment` columns.
        log: Change log with `key`, `written_at` and `value` columns, in
            the order the changes were applied.
        key: Join key present in both tables.
        moment: Column of `rows` holding the prediction moment.
        written_at: Column of `log` holding the write time.
        value: Column of `log` holding the written value.

    Returns:
        A frame indexed like `rows` with `reconstructed` (missing when no
        entry precedes the moment), `known_at` (write time in seconds, NaN
        when none) and `has_entry`.
    """
    left = pd.DataFrame(
        {
            "k": rows[key].to_numpy(),
            "t": to_seconds(rows[moment]),
            "i": np.arange(len(rows)),
        }
    )
    times = to_seconds(log[written_at])
    right = pd.DataFrame(
        {
            "k": log[key].to_numpy(),
            "t": times,
            "w": times,
            "v": log[value].to_numpy(),
            "seq": np.arange(len(log)),
        }
    )
    right = right.sort_values(["t", "seq"], kind="stable").drop_duplicates(
        ["k", "t"], keep="last"
    )
    joined = pd.merge_asof(
        left.sort_values("t", kind="stable"),
        right[["k", "t", "w", "v"]],
        on="t",
        by="k",
        direction="backward",
        allow_exact_matches=True,
    ).sort_values("i")
    return pd.DataFrame(
        {
            "reconstructed": joined["v"].to_numpy(),
            "known_at": joined["w"].to_numpy(dtype="float64"),
            "has_entry": joined["w"].notna().to_numpy(),
        },
        index=rows.index,
    )


def agrees(stored: pd.Series, reconstructed: pd.Series) -> pd.Series:
    """Element-wise equality in which two missing values are equal."""
    left = stored.to_numpy(dtype=object)
    right = reconstructed.to_numpy(dtype=object)
    both_missing = stored.isna().to_numpy() & reconstructed.isna().to_numpy()
    equal = np.array([a == b for a, b in zip(left, right, strict=True)])
    return pd.Series(
        (equal | both_missing).astype(bool), index=stored.index, dtype=bool
    )


def classify_rows(stored: pd.Series, joined: pd.DataFrame) -> pd.Series:
    """Status of every row from its stored value and the reconstruction."""
    match = agrees(stored, joined["reconstructed"])
    not_yet = (~joined["has_entry"]) & stored.notna()
    status = pd.Series(RowStatus.OVERWRITTEN.value, index=stored.index)
    status[match] = RowStatus.MATCH.value
    status[not_yet] = RowStatus.NOT_YET_WRITTEN.value
    return status


def reconstruction_mismatch(
    rows: pd.DataFrame,
    log: pd.DataFrame,
    key: str,
    moment: str,
    stored: str,
    written_at: str,
    value: str,
) -> pd.Series:
    """True for rows whose stored value is not the log value at the moment.

    Args:
        rows: Cases with `key`, `moment` and the stored column.
        log: Change log (see `point_in_time`).
        key: Join key.
        moment: Prediction-moment column of `rows`.
        stored: Column of `rows` that holds the stored value.
        written_at: Write-time column of `log`.
        value: Value column of `log`.

    Returns:
        A boolean series aligned to `rows`.
    """
    joined = point_in_time(rows, log, key, moment, written_at, value)
    status = classify_rows(rows[stored], joined)
    return pd.Series(
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
    status: pd.Series
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


def audit_field(
    rows: pd.DataFrame,
    log: pd.DataFrame,
    key: str,
    moment: str,
    stored: str,
    written_at: str,
    value: str,
    tolerance: float = 0.0,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
    set_id: qualification.SetId = qualification.SetId.EXPLORATION,
    seed_path: qualification.SeedPath = (0, DETECTOR_ID, 0),
) -> AsOfAudit:
    """Reconstructs a field from its log and compares it with the stored one.

    The field fails when its mismatch share exceeds `tolerance`. The check
    quarantines only when `gate` is `GATING`; otherwise it warns.

    Args:
        rows: Cases with `key`, `moment` and the stored column.
        log: Change log (see `point_in_time`).
        key: Join key.
        moment: Prediction-moment column of `rows`.
        stored: Column of `rows` that holds the stored value.
        written_at: Write-time column of `log`.
        value: Value column of `log`.
        tolerance: Largest mismatch share allowed, in [0, 1].
        gate: Whether the detector has a current certificate.
        set_id: Set the rows belong to. The audit does not read the target.
        seed_path: Recorded on the check; the audit draws no randomness.

    Returns:
        The audit, including the `Check` for the ledger.

    Raises:
        ValueError: On an empty table or a bad tolerance.
    """
    if len(rows) == 0:
        raise ValueError("rows must not be empty")
    joined = point_in_time(rows, log, key, moment, written_at, value)
    status = classify_rows(rows[stored], joined)
    mismatched = int((status != RowStatus.MATCH.value).sum())
    log_check = provenance.LogCheck(len(rows), mismatched, tolerance)
    matched = (status == RowStatus.MATCH.value) & joined["has_entry"]
    leads = (
        to_seconds(rows[moment])[matched.to_numpy()]
        - joined["known_at"].to_numpy()[matched.to_numpy()]
    )
    as_of = (
        provenance.AsOf(provenance.AsOfKind.TIMESTAMP, float(leads.min()))
        if len(leads)
        else provenance.AsOf.unknown()
    )
    low, high = certificates.wilson_interval(mismatched, len(rows))
    decision = (
        qualification.Decision.PASS
        if log_check.passed
        else qualification.Decision.QUARANTINE
    )
    check = qualification.Check(
        check_id=f"asof.reconstruction.{stored}",
        null_hypothesis=(
            "the stored value equals the change-log value at the prediction "
            "moment"
        ),
        statistic=log_check.rate,
        p_value=None,
        interval=(low, high),
        effect=log_check.rate,
        n=len(rows),
        decision=certificates.demote(decision, gate),
        set_id=set_id,
        seed_path=seed_path,
        tier=qualification.Tier.UNVERIFIED,
    )
    return AsOfAudit(stored, status, log_check, as_of, (low, high), check)


def log_provenance(
    base: provenance.FieldProvenance, audit: AsOfAudit
) -> provenance.FieldProvenance:
    """Copy of `base` whose as-of rule comes from a log audit.

    Args:
        base: The record to update.
        audit: Audit of the same field.

    Returns:
        A record with source `log`; its tier is `confirmed` only if the
        audit passed.

    Raises:
        ValueError: If the audit named another field or derived no rule.
    """
    if audit.field != base.name:
        raise ValueError("audit belongs to another field")
    if not audit.as_of.is_known:
        raise ValueError("the log gives no as-of rule: no row matched")
    return dataclasses.replace(
        base,
        as_of=audit.as_of,
        as_of_source=provenance.AsOfSource.LOG,
        log_check=audit.log_check,
    )
