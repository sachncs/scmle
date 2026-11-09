"""The as-of view: the only read path to a non-target table.

Every feature read of another table goes through `AsOfView.visible`, which
returns only rows whose recorded `known_time` is at or before the target
row's cutoff (`t_i - lag`) and whose event time is also at or before it. The
synthesizer never receives the table, only the view.

Where `known_time` comes from, best first:

* `transaction`: the table has a column holding when the system recorded the
  row. Late arrivals and backfills are excluded correctly.
* `event_delay`: no such column; `known_time = event_time + delta_rec`, a
  declared maximum recording delay. A row recorded later than that delay is
  a leak this rule cannot see; the rule is at best `attested`.
* `non_versioned`: no history (a mutable dimension). There is no as-of rule;
  `visible` refuses unless the view was opened `permissive`, and then returns
  the current rows and the tier stays `unverified`.

This is a guarantee about the code, not the world: it holds only if the
recorded time is true. Python cannot make an attribute unreachable, so
"no raw read path" is held by this module being the only one that handles
tables, by the import test on `synthesize`, and by the metamorphic tests.
"""

from collections.abc import Callable
import dataclasses
import datetime
from typing import Any

import numpy
import pandas

from scmle.contracts import errors
from scmle.contracts import qualification
from scmle.data.admissibility import provenance
from scmle.data.ingest import profile
from scmle.data.readers import connectors

TRANSACTION = "transaction"
EVENT_DELAY = "event_delay"
NON_VERSIONED = "non_versioned"
TARGET_COLUMN = "target_row"
CUTOFF_COLUMN = "cutoff_days"
EVENT_COLUMN = "event_days"
KNOWN_COLUMN = "known_days"
RESERVED = (TARGET_COLUMN, CUTOFF_COLUMN, EVENT_COLUMN, KNOWN_COLUMN)
SECONDS_PER_DAY = 86_400.0
EPOCH = pandas.Timestamp("1970-01-01")
TIER_ORDER = {
    qualification.Tier.UNVERIFIED: 0,
    qualification.Tier.ATTESTED: 1,
    qualification.Tier.CONFIRMED: 2,
}


class NoAsOfRuleError(ValueError):
    """A table without history was read where an as-of rule is required."""


def to_days(values: pandas.Series) -> numpy.ndarray:
    """Converts times to float days.

    Datetimes are measured from the Unix epoch, numbers are taken as given,
    and strings are parsed as timestamps (timezone-aware values are converted
    to UTC). Missing values become NaN.

    Args:
        values: The column of times.

    Returns:
        A float64 array of days, one per value, in the order of `values`.
    """
    if pandas.api.types.is_datetime64_any_dtype(values):
        stamps = values
    elif pandas.api.types.is_numeric_dtype(values) and not (
        pandas.api.types.is_bool_dtype(values)
    ):
        return values.to_numpy(dtype="float64")
    else:
        stamps = pandas.to_datetime(values, errors="raise", utc=True)
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    seconds = (stamps - EPOCH).dt.total_seconds()
    return seconds.to_numpy(dtype="float64") / SECONDS_PER_DAY


@dataclasses.dataclass(frozen=True, slots=True)
class KnownTime:
    """How a table's known time is obtained, and what vouches for it.

    Attributes:
        mode: `transaction`, `event_delay` or `non_versioned`.
        known_column: Column holding the recorded time (`transaction`).
        delta_rec_days: Declared maximum recording delay, in days. Also the
            fallback for a missing recorded time within a transaction table.
        attestation: A named person's statement about the rule, if any.
        log_check: Result of reconstructing a field from a change log.
    """

    mode: str
    known_column: str | None = None
    delta_rec_days: float | None = None
    attestation: provenance.Attestation | None = None
    log_check: provenance.LogCheck | None = None

    def __post_init__(self) -> None:
        """Validates the combination of fields.

        Raises:
            ValueError: If `mode` is `transaction` without `known_column`,
                is `event_delay` without `delta_rec_days`, is not one of
                the three modes, or if `delta_rec_days` is negative.
        """
        if self.mode == TRANSACTION and not self.known_column:
            raise ValueError("transaction mode needs known_column")
        if self.mode == EVENT_DELAY and self.delta_rec_days is None:
            raise ValueError("event_delay mode needs delta_rec_days")
        if self.mode not in (TRANSACTION, EVENT_DELAY, NON_VERSIONED):
            raise ValueError(f"unknown mode: {self.mode}")
        if self.delta_rec_days is not None and self.delta_rec_days < 0:
            raise ValueError("delta_rec_days must be non-negative")

    def tier(self, today: datetime.date | None = None) -> qualification.Tier:
        """Gives the evidence tier of the rule.

        `confirmed` needs a passing log check and a recorded-time column; a
        delay rule is `attested` at best; a table without history, or a rule
        nobody vouches for, is `unverified`.

        Args:
            today: Date on which the attestation must be valid. When None,
                the attestation's validity dates are not checked.

        Returns:
            The tier of the rule.
        """
        if self.mode == NON_VERSIONED:
            return qualification.Tier.UNVERIFIED
        attested = self.attestation is not None and (
            today is None or self.attestation.valid_on(today)
        )
        if (
            self.mode == TRANSACTION
            and self.log_check is not None
            and self.log_check.passed
        ):
            return qualification.Tier.CONFIRMED
        if attested:
            return qualification.Tier.ATTESTED
        return qualification.Tier.UNVERIFIED


def weakest(tiers: list[qualification.Tier]) -> qualification.Tier:
    """Finds the lowest tier in a list.

    Args:
        tiers: The tiers to compare.

    Returns:
        The weakest tier by evidence order (`unverified` < `attested` <
        `confirmed`), or `unverified` for an empty list.
    """
    if not tiers:
        return qualification.Tier.UNVERIFIED
    return min(tiers, key=TIER_ORDER.__getitem__)


class AsOfView:
    """Rows of one table, readable only as of each target row's cutoff."""

    def __init__(
        self,
        frame: pandas.DataFrame,
        key: str,
        rule: KnownTime,
        event_time: str | None = None,
        name: str = "",
        permissive: bool = False,
    ) -> None:
        """Opens a view on a table.

        Args:
            frame: The table.
            key: Column joining the table to its target rows.
            rule: How known time is obtained.
            event_time: Column of when the fact happened; required unless
                the table is `non_versioned`.
            name: Table name, for provenance and messages.
            permissive: Allow a `non_versioned` read (unverified).

        Raises:
            ValueError: If the table uses a reserved column name, lacks a
                needed column, or is versioned but has no `event_time`.
        """
        clash = [c for c in RESERVED if c in frame.columns]
        if clash:
            raise ValueError(f"table uses reserved column(s): {clash}")
        needed = [key] + ([event_time] if event_time else [])
        if rule.known_column:
            needed.append(rule.known_column)
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            raise ValueError(f"table lacks column(s): {missing}")
        if rule.mode != NON_VERSIONED and event_time is None:
            raise ValueError("a versioned table needs event_time")
        self.name = name or key
        self.key = key
        self.rule = rule
        self.event_time = event_time
        self.permissive = permissive
        self.columns = tuple(str(c) for c in frame.columns)
        self.reader = self.freeze(frame)

    def freeze(self, frame: pandas.DataFrame) -> Callable[[], pandas.DataFrame]:
        """Prepares the rows once and returns a closure over them.

        Rows with a missing key are dropped, and the event and known times
        are added as `EVENT_COLUMN` and `KNOWN_COLUMN`, both in days. A
        non-versioned table is known from minus infinity. In a transaction
        table a missing recorded time falls back to the event time plus
        `delta_rec_days`, which is infinite when that delay is None.

        Args:
            frame: The table.

        Returns:
            A function that returns the prepared rows.

        Raises:
            ScmleError: If the rule is `transaction` without a known column.
        """
        rows = frame.loc[frame[self.key].notna()].reset_index(drop=True)
        rows = rows.copy()
        if self.event_time is not None:
            event = to_days(rows[self.event_time])
        else:
            event = numpy.full(len(rows), numpy.nan)
        rows[EVENT_COLUMN] = event
        if self.rule.mode == NON_VERSIONED:
            known = numpy.full(len(rows), -numpy.inf)
        else:
            delay = self.rule.delta_rec_days
            fallback = event + (numpy.inf if delay is None else delay)
            if self.rule.mode == TRANSACTION:
                if self.rule.known_column is None:
                    raise errors.ScmleError(
                        "transaction rule has no known column"
                    )
                recorded = to_days(rows[self.rule.known_column])
                known = numpy.where(numpy.isnan(recorded), fallback, recorded)
            else:
                known = fallback
        rows[KNOWN_COLUMN] = known
        return lambda: rows

    @property
    def tier(self) -> qualification.Tier:
        """The evidence tier of this table's as-of rule."""
        return self.rule.tier()

    def visible(self, targets: pandas.DataFrame) -> pandas.DataFrame:
        """Selects the rows readable by each target row.

        Args:
            targets: Columns `key` (the join value), `TARGET_COLUMN` (an
                integer label of the target row) and `CUTOFF_COLUMN` (the
                cutoff in days, `moment - lag`).

        Returns:
            One row per (target row, matching table row) with
            `known_days <= cutoff` and `event_days <= cutoff`, with every
            table column plus `TARGET_COLUMN`, `CUTOFF_COLUMN`, and the
            derived `EVENT_COLUMN` and `KNOWN_COLUMN`.

        Raises:
            NoAsOfRuleError: For a `non_versioned` table unless `permissive`.
        """
        if self.rule.mode == NON_VERSIONED and not self.permissive:
            raise NoAsOfRuleError(
                f"{self.name} has no history, so no as-of rule exists"
            )
        left = targets[[self.key, TARGET_COLUMN, CUTOFF_COLUMN]]
        left = left.loc[left[self.key].notna()]
        joined = left.merge(self.reader(), on=self.key, how="inner")
        if self.rule.mode == NON_VERSIONED:
            return joined.reset_index(drop=True)
        keep = (joined[KNOWN_COLUMN] <= joined[CUTOFF_COLUMN]) & (
            joined[EVENT_COLUMN] <= joined[CUTOFF_COLUMN]
        )
        return joined.loc[keep].reset_index(drop=True)

    def evidence(self, today: datetime.date | None = None) -> dict[str, Any]:
        """Describes the rule and its tier, for the feature manifest.

        Args:
            today: Date on which the attestation must be valid, or None to
                skip the date check.

        Returns:
            A mapping with the table name, rule mode, known column,
            recording delay in days and the tier value.
        """
        return {
            "table": self.name,
            "mode": self.rule.mode,
            "known_column": self.rule.known_column,
            "delta_rec_days": self.rule.delta_rec_days,
            "tier": self.rule.tier(today).value,
        }

    @classmethod
    def from_connector(
        cls,
        connector: connectors.SourceConnector,
        key: str,
        rule: KnownTime,
        event_time: str | None = None,
        name: str = "",
        permissive: bool = False,
        batch_rows: int = 1_000,
    ) -> "AsOfView":
        """Reads a table through `consistent_read` and opens a view on it.

        A source that changes during the read is rejected by
        `consistent_read` with `TornReadError`, which propagates.

        Args:
            connector: The source to read completely.
            key: Column joining the table to its target rows.
            rule: How known time is obtained.
            event_time: Column of when the fact happened; required unless
                the table is `non_versioned`.
            name: Table name, for provenance and messages.
            permissive: Allow a `non_versioned` read (unverified).
            batch_rows: Rows per read batch.

        Returns:
            A view over the rows that were read.
        """
        read = profile.consistent_read(connector, batch_rows)
        return cls(read.frame, key, rule, event_time, name, permissive)
