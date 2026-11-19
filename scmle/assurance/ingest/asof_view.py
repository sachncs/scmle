"""The as-of view: the only read path to a non-target table.

Every feature read of another table goes through `View.visible`, which
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

from scmle.assurance.admissibility import provenance
from scmle.contracts import errors
from scmle.contracts import qualification

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


class HistoryError(ValueError):
    """A table without history was read where an as-of rule is required."""


def convert(values: pandas.Series) -> numpy.ndarray:
    """Converts times to float days.

    Datetimes are measured from the Unix epoch, numbers are taken as given,
    and strings are parsed as timestamps by `stamp` (timezone-aware values
    are converted to UTC). Missing values become NaN.

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
        stamps = stamp(values)
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_convert("UTC").dt.tz_localize(None)
    seconds = (stamps - EPOCH).dt.total_seconds()
    return seconds.to_numpy(dtype="float64") / SECONDS_PER_DAY


def stamp(values: pandas.Series) -> pandas.Series:
    """Parses text times as UTC timestamps.

    ISO 8601 is read per value, so one column may mix precisions ("2016-07-04"
    beside "2023-08-29 09:23:46", as real databases hold); any other text
    must share one format, which pandas infers.

    Args:
        values: The column of text times.

    Returns:
        Timezone-aware UTC timestamps; missing values stay missing.

    Raises:
        ValueError: If the text is not ISO 8601 and has no single format.
    """
    iso = pandas.to_datetime(
        values, format="ISO8601", utc=True, errors="coerce"
    )
    if bool((iso.isna() & values.notna()).any()):
        return pandas.to_datetime(values, errors="raise", utc=True)
    return iso


@dataclasses.dataclass(frozen=True, slots=True)
class Timing:
    """How a table's known time is obtained, and what vouches for it.

    Attributes:
        mode: `transaction`, `event_delay` or `non_versioned`.
        recorded: Column holding the recorded time (`transaction`).
        delay: Declared maximum recording delay, in days. Also the
            fallback for a missing recorded time within a transaction table.
        attestation: A named person's statement about the rule, if any.
        check: Result of reconstructing a field from a change log.
    """

    mode: str
    recorded: str | None = None
    delay: float | None = None
    attestation: provenance.Attestation | None = None
    check: provenance.Reconstruction | None = None

    def __post_init__(self) -> None:
        """Validates the combination of fields.

        Raises:
            ValueError: If `mode` is `transaction` without `recorded`,
                is `event_delay` without `delay`, is not one of
                the three modes, or if `delay` is negative.
        """
        if self.mode == TRANSACTION and not self.recorded:
            raise errors.InputError("transaction mode needs known_column")
        if self.mode == EVENT_DELAY and self.delay is None:
            raise errors.InputError("event_delay mode needs delta_rec_days")
        if self.mode not in (TRANSACTION, EVENT_DELAY, NON_VERSIONED):
            raise errors.InputError(f"unknown mode: {self.mode}")
        if self.delay is not None and self.delay < 0:
            raise errors.InputError("delta_rec_days must be non-negative")

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
            today is None or self.attestation.effective(today)
        )
        if (
            self.mode == TRANSACTION
            and self.check is not None
            and self.check.passed
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


class View:
    """Rows of one table, readable only as of each target row's cutoff."""

    def __init__(
        self,
        frame: pandas.DataFrame,
        key: str,
        rule: Timing,
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
                needed column, or is versioned but has no `time`.
        """
        clash = [c for c in RESERVED if c in frame.columns]
        if clash:
            raise errors.InputError(f"table uses reserved column(s): {clash}")
        needed = [key] + ([event_time] if event_time else [])
        if rule.recorded:
            needed.append(rule.recorded)
        missing = [c for c in needed if c not in frame.columns]
        if missing:
            raise errors.InputError(f"table lacks column(s): {missing}")
        if rule.mode != NON_VERSIONED and event_time is None:
            raise errors.InputError("a versioned table needs event_time")
        self.name = name or key
        self.key = key
        self.rule = rule
        self.time = event_time
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
        if self.time is not None:
            event = convert(rows[self.time])
        else:
            event = numpy.full(len(rows), numpy.nan)
        rows[EVENT_COLUMN] = event
        if self.rule.mode == NON_VERSIONED:
            known = numpy.full(len(rows), -numpy.inf)
        else:
            delay = self.rule.delay
            fallback = event + (numpy.inf if delay is None else delay)
            if self.rule.mode == TRANSACTION:
                if self.rule.recorded is None:
                    raise errors.ScmleError(
                        "transaction rule has no known column"
                    )
                recorded = convert(rows[self.rule.recorded])
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
            HistoryError: For a `non_versioned` table unless `permissive`.
        """
        if self.rule.mode == NON_VERSIONED and not self.permissive:
            raise HistoryError(
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
            "known_column": self.rule.recorded,
            "delta_rec_days": self.rule.delay,
            "tier": self.rule.tier(today).value,
        }
