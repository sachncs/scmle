"""Per-period drift: each time period of `X` compared with the first.

The exploration rows are ordered by the time column and cut into contiguous
periods of equal size. Every later period is compared with the first, column
by column, with the tests of `shift.compare` (Kolmogorov-Smirnov for numeric
columns, chi-square for categorical ones). The Benjamini-Hochberg adjustment
runs once over every period and column, so the false discovery rate is held
over the whole family, not per period.

The number of periods follows from the rows: each period holds at least
`shift.MIN_ROWS` rows, the same floor the two-sample tests of `shift` need, and
there are at most `MAX_PERIODS`. With 50 rows a side the two-sided
Kolmogorov-Smirnov test at 0.05 detects a distance of about
`1.36 * sqrt(2 / 50) = 0.27`; more, smaller periods would only detect
larger shifts, and ten periods (deciles of time) keep the family at ten
times the columns. Rows without a time are left out.

Like the deployment-sample tests, these detect covariate drift, not label or
concept drift, and a pass is a failure to detect, not evidence of none.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy
import pandas
from scipy import stats

from scmle.assurance.robustness import shift

MAX_PERIODS = 10
LEGACY = "not_run"


@dataclasses.dataclass(frozen=True, slots=True)
class Period:
    """One contiguous stretch of time and its tests against the first.

    Attributes:
        index: Position in time order, 0 for the first (the baseline).
        start: Earliest time in the period, as text.
        end: Latest time in the period, as text.
        rows: Rows in the period.
        features: Tests against the first period, with the adjustment over
            every period; empty for the first period.
    """

    index: int
    start: str
    end: str
    rows: int
    features: tuple[shift.Drift, ...]

    @property
    def rejected(self) -> tuple[str, ...]:
        """Columns whose adjusted p-value is at most the level."""
        return tuple(d.column for d in self.features if d.rejected)


@dataclasses.dataclass(frozen=True, slots=True)
class Periods:
    """The per-period drift verdict.

    Attributes:
        status: `pass` (no period differs detectably from the first),
            `fail` (at least one does) or `unverifiable`.
        reason: One sentence saying why.
        column: The time column, or None.
        periods: The periods in time order; empty when unverifiable.
    """

    status: shift.Status
    reason: str
    column: str | None = None
    periods: tuple[Period, ...] = ()

    def to_dict(self) -> Mapping[str, Any]:
        """Converts the record to a JSON-ready dictionary.

        Returns:
            `status` as its string value, `reason`, `column` and `periods`,
            one entry per period with its fields, its tests as dictionaries
            and `rejected`.
        """
        return {
            "status": self.status.value,
            "reason": self.reason,
            "column": self.column,
            "periods": [
                {
                    "index": p.index,
                    "start": p.start,
                    "end": p.end,
                    "rows": p.rows,
                    "features": [d.to_dict() for d in p.features],
                    "rejected": list(p.rejected),
                }
                for p in self.periods
            ],
        }


def count(rows: int) -> int:
    """The number of periods `rows` timed rows are cut into.

    Args:
        rows: Rows with a time.

    Returns:
        `rows // shift.MIN_ROWS`, at most `MAX_PERIODS`.
    """
    return min(rows // shift.MIN_ROWS, MAX_PERIODS)


def order(values: pandas.Series) -> pandas.Series:
    """Turns time values into something that sorts.

    Args:
        values: The time column.

    Returns:
        Numbers stay numbers; anything else is read as a date, and what
        cannot be read becomes missing.
    """
    if pandas.api.types.is_numeric_dtype(
        values
    ) or pandas.api.types.is_datetime64_any_dtype(values):
        return values
    return pandas.to_datetime(values, errors="coerce", format="mixed")


def unverifiable(reason: str, column: str | None) -> Periods:
    """Builds a verdict that says nothing could be checked, and why.

    Args:
        reason: One sentence saying why.
        column: The time column, or None.

    Returns:
        An `unverifiable` verdict with no periods.
    """
    return Periods(shift.Status.UNVERIFIABLE, reason, column)


UNTIMED = unverifiable("the data have no time column", None)


def adjust(
    tested: list[tuple[shift.Drift, ...]], alpha: float
) -> list[tuple[shift.Drift, ...]]:
    """Re-runs Benjamini-Hochberg over every period's tests at once.

    Args:
        tested: Each later period's tests, adjusted within the period.
        alpha: Level applied to the adjusted p-values.

    Returns:
        The same tests with the adjusted p-value and the decision taken
        over the whole family.
    """
    raw = [d.p_value for part in tested for d in part]
    if not raw:
        return tested
    adjusted = iter(stats.false_discovery_control(raw, method="bh"))
    return [
        tuple(
            dataclasses.replace(
                d, adjusted_p_value=float(q), rejected=bool(q <= alpha)
            )
            for d, q in zip(part, adjusted, strict=False)
        )
        for part in tested
    ]


def check(
    features: pandas.DataFrame, time: str | None, candidates: Sequence[str]
) -> str:
    """Says why the periods cannot be compared, if they cannot.

    Args:
        features: Exploration rows.
        time: The time column, or None.
        candidates: Feature columns to compare.

    Returns:
        The reason, or an empty string when the comparison can run as far
        as the columns go (the row count is checked by `assess`).
    """
    if time is None:
        return UNTIMED.reason
    if time not in features.columns:
        return f"the time column {time!r} is not among the exploration columns"
    if not [c for c in candidates if c in features.columns and c != time]:
        return "no feature column to compare across periods"
    if order(features[time]).notna().sum() == 0:
        return f"the time column {time!r} holds no orderable values"
    return ""


def assess(
    features: pandas.DataFrame,
    time: str | None,
    candidates: Sequence[str],
    alpha: float = shift.ALPHA,
) -> Periods:
    """Compares each time period of the exploration rows with the first.

    Args:
        features: Exploration rows, holding the time column.
        time: The time column, or None when the data have none.
        candidates: Feature columns to compare; those not in `features`
            are skipped.
        alpha: Level of the adjusted tests.

    Returns:
        `fail` when any later period has a column whose adjusted p-value is
        at most `alpha`, `pass` when none has, and `unverifiable` without a
        time column, without a feature column, without orderable times or
        with too few timed rows for two periods.
    """
    reason = check(features, time, candidates)
    if reason or time is None:
        return unverifiable(reason, time)
    columns = [c for c in candidates if c in features.columns and c != time]
    keys = order(features[time])
    timed = keys.notna().to_numpy()
    rows = features.loc[timed, columns].reset_index(drop=True)
    stamps = keys[timed].reset_index(drop=True)
    total = count(len(rows))
    if total < 2:  # noqa: PLR2004 -- a comparison needs two periods
        return unverifiable(
            f"{len(rows)} timed rows make fewer than 2 periods of "
            f"{shift.MIN_ROWS} rows",
            time,
        )
    chunks = numpy.array_split(
        numpy.argsort(stamps.to_numpy(), kind="stable"), total
    )
    first = rows.iloc[chunks[0]]
    tested = adjust(
        [shift.compare(rows.iloc[c], first, alpha) for c in chunks[1:]], alpha
    )
    found = tuple(
        Period(
            index,
            str(stamps.iloc[chunk].min()),
            str(stamps.iloc[chunk].max()),
            len(chunk),
            () if index == 0 else tested[index - 1],
        )
        for index, chunk in enumerate(chunks)
    )
    drifted = sum(1 for p in found if p.rejected)
    tests = sum(len(p.features) for p in found)
    family = f"(Benjamini-Hochberg over {tests} tests at {alpha})"
    if drifted:
        return Periods(
            shift.Status.FAIL,
            f"{drifted} of {total - 1} later periods differ from the first "
            f"{family}",
            time,
            found,
        )
    return Periods(
        shift.Status.PASS,
        f"no later period differs detectably from the first {family}",
        time,
        found,
    )
