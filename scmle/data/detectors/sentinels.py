"""Sentinels: codes that stand in for "missing" and hide in every statistic.

A numeric value is a sentinel candidate when its point mass is far larger
than a smooth density estimated from the rest of the column can explain, and
the count must also be at least `MIN_RATIO` times the expected count (an
effect floor, because the local density is itself estimated). For
a candidate value v with count c among n values, the other values m = n - c
give a local mass: the share of them inside a window of half-width h around
v (h reaches the k nearest others) times the ratio of the column's grid
spacing to the window width on the support. A one-sided exact binomial test
compares c with that mass; the family is Bonferroni-corrected over every
candidate tested. Text cells are matched exactly (case and surrounding
spaces ignored) against a dictionary the charter declares. Flagged cells can
be recoded to missing, and the recode is logged.

This is a proposed method, not a published procedure; its operating
characteristics come from its certificate. A real zero-inflated or heaped
(rounded) column also shows a point mass and is flagged as a candidate; a
flag is a candidate, not proof.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy
import pandas
from scipy import stats

MIN_COUNT = 5
MIN_WINDOW = 30
WINDOW_SHARE = 0.05
MIN_RATIO = 3.0
ALPHA = 0.01
MIN_VALUES = 30
TEXT_CODES = (
    "n/a",
    "na",
    "null",
    "none",
    "nan",
    "?",
    "-",
    "--",
    "missing",
    "unknown",
    "#n/a",
)


@dataclasses.dataclass(frozen=True)
class SentinelFinding:
    """One suspected sentinel value in one column.

    Attributes:
        column: Column name.
        value: The suspected code (number or text).
        count: Cells holding it.
        expected: Count expected under the smooth density (numeric), or
            None for a dictionary match.
        p_value: One-sided binomial p-value (numeric), or None.
        kind: `spike` (numeric point mass), `declared_numeric` or
            `dictionary` (text).
    """

    column: str
    value: float | str
    count: int
    expected: float | None
    p_value: float | None
    kind: str


@dataclasses.dataclass(frozen=True)
class Sentinels:
    """Sentinel findings and the cells they cover.

    Attributes:
        findings: One per suspected code, ordered by column then value.
        rows: Rows examined.
        tier: Always `unverified`: no provenance confirms a code.
    """

    findings: tuple[SentinelFinding, ...]
    rows: int
    tier: str = "unverified"

    def to_dict(self) -> dict[str, Any]:
        """Returns the report as a JSON-ready record.

        Returns:
            A mapping of every field to its value.
        """
        return dataclasses.asdict(self)


def grid_spacing(unique_values: numpy.ndarray) -> float:
    """Finds the grid spacing of a column.

    Args:
        unique_values: Distinct values in ascending order.

    Returns:
        The median positive gap between neighbouring values, or 0.0 when
        there is none.
    """
    gaps = numpy.diff(unique_values)
    gaps = gaps[gaps > 0]
    return float(numpy.median(gaps)) if gaps.size else 0.0


def expected_share(
    others: numpy.ndarray, value: float, spacing: float
) -> float:
    """Share of all draws a smooth density would put on `value`.

    Args:
        others: The column's values excluding every copy of `value`.
        value: Candidate value.
        spacing: Grid spacing of the column.

    Returns:
        Local density times spacing, in [0, 1].
    """
    distance = numpy.sort(numpy.abs(others - value))
    k = min(len(others), max(MIN_WINDOW, math.ceil(WINDOW_SHARE * len(others))))
    half = float(distance[k - 1])
    inside = int((distance <= half).sum())
    low, high = float(others.min()), float(others.max())
    if value < low or value > high:
        width = half
    else:
        width = min(half, value - low) + min(half, high - value)
    if width <= 0.0:
        return 1.0
    return min(1.0, inside / len(others) * spacing / width)


def numeric_spikes(
    column: str, series: pandas.Series, min_count: int = MIN_COUNT
) -> list[SentinelFinding]:
    """Finds candidate spikes of one numeric column.

    Args:
        column: Column name, recorded in the findings.
        series: The column's values.
        min_count: Fewest copies of a value that make it a candidate.

    Returns:
        Findings with their uncorrected p-values; the caller applies the
        multiplicity correction. The list is empty when the column has fewer
        than `MIN_VALUES` finite values.
    """
    values = series.to_numpy(dtype=float)
    values = values[numpy.isfinite(values)]
    if len(values) < MIN_VALUES:
        return []
    unique, counts = numpy.unique(values, return_counts=True)
    found = []
    spacing = grid_spacing(unique[counts < min_count])
    if spacing == 0.0:
        spacing = grid_spacing(unique)
    for value, count in zip(unique, counts, strict=True):
        if count < min_count:
            continue
        others = values[values != value]
        if len(others) < 2 * MIN_WINDOW:
            continue
        share = expected_share(others, float(value), spacing)
        if count < MIN_RATIO * share * len(values):
            continue
        p_value = float(
            stats.binomtest(int(count), len(values), share, "greater").pvalue
        )
        found.append(
            SentinelFinding(
                column,
                float(value),
                int(count),
                share * len(values),
                p_value,
                "spike",
            )
        )
    return found


def text_findings(
    column: str, series: pandas.Series, codes: Sequence[str]
) -> list[SentinelFinding]:
    """Finds dictionary matches in a text column.

    Args:
        column: Column name, recorded in the findings.
        series: The column's values.
        codes: Dictionary of codes; case and surrounding spaces are ignored.

    Returns:
        One finding per code present, in sorted order, holding the first
        spelling found.
    """
    lowered = {c.strip().lower() for c in codes}
    present = series.dropna()
    cleaned = present.astype(str).str.strip().str.lower()
    out = []
    for token in sorted(lowered):
        hits = int((cleaned == token).sum())
        if hits:
            original = str(present[cleaned == token].iloc[0])
            out.append(
                SentinelFinding(
                    column, original, hits, None, None, "dictionary"
                )
            )
    return out


def find_sentinels(
    frame: pandas.DataFrame,
    columns: Sequence[str] | None = None,
    text_codes: Sequence[str] = TEXT_CODES,
    numeric_codes: Sequence[float] = (),
    alpha: float = ALPHA,
) -> Sentinels:
    """Finds suspected sentinel codes.

    Args:
        frame: Exploration rows.
        columns: Columns to examine; defaults to all.
        text_codes: Declared text dictionary.
        numeric_codes: Declared numeric codes (always reported if present).
        alpha: Family level for the spike tests, in (0, 1).

    Returns:
        The report; spikes are those with Bonferroni-adjusted p < alpha.

    Raises:
        ValueError: If `alpha` is not strictly between 0 and 1.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be within (0, 1)")
    names = list(frame.columns) if columns is None else list(columns)
    candidates: list[SentinelFinding] = []
    declared = [float(c) for c in numeric_codes]
    findings: list[SentinelFinding] = []
    for name in names:
        series = frame[name]
        if pandas.api.types.is_bool_dtype(series):
            continue
        if pandas.api.types.is_numeric_dtype(series):
            candidates += numeric_spikes(str(name), series)
            numbers = series.to_numpy(dtype=float)
            for code in declared:
                hits = int((numbers == code).sum())
                if hits:
                    findings.append(
                        SentinelFinding(
                            str(name),
                            code,
                            hits,
                            None,
                            None,
                            "declared_numeric",
                        )
                    )
        elif pandas.api.types.is_string_dtype(series) or series.dtype == object:
            findings += text_findings(str(name), series, text_codes)
    threshold = alpha / max(len(candidates), 1)
    seen = {(f.column, f.value) for f in findings}
    for finding in candidates:
        key = (finding.column, finding.value)
        significant = finding.p_value is not None and (
            finding.p_value < threshold
        )
        if significant and key not in seen:
            findings.append(finding)
    findings.sort(key=lambda f: (f.column, str(f.value)))
    return Sentinels(tuple(findings), len(frame))


def sentinel_mask(
    frame: pandas.DataFrame, report: Sentinels
) -> pandas.DataFrame:
    """Marks the cells that hold a reported code.

    Args:
        frame: The table the report was made on.
        report: Findings from `find_sentinels`.

    Returns:
        A boolean frame of the shape of `frame`, True where a cell holds a
        code reported for its column.
    """
    mask = pandas.DataFrame(
        data=False, index=frame.index, columns=frame.columns
    )
    for finding in report.findings:
        series = frame[finding.column]
        if finding.kind == "dictionary":
            token = str(finding.value).strip().lower()
            hit = series.astype(str).str.strip().str.lower() == token
            hit &= series.notna()
        else:
            hit = series == finding.value
        mask[finding.column] |= hit.fillna(value=False).to_numpy(dtype=bool)
    return mask


def recode_missing(
    frame: pandas.DataFrame, report: Sentinels
) -> tuple[pandas.DataFrame, list[dict[str, Any]]]:
    """Replaces reported codes by missing values and logs each recode.

    Args:
        frame: The table; it is not modified.
        report: Findings from `find_sentinels`.

    Returns:
        The recoded copy and one log record per finding.
    """
    mask = sentinel_mask(frame, report)
    out = frame.copy()
    log = []
    for finding in report.findings:
        cells = mask[finding.column]
        column = out[finding.column]
        if pandas.api.types.is_integer_dtype(column):
            column = column.astype("float64")
        column = column.copy()
        column[cells.to_numpy()] = numpy.nan
        out[finding.column] = column
        log.append(
            {
                "column": finding.column,
                "value": finding.value,
                "recoded": int(cells.sum()),
                "kind": finding.kind,
            }
        )
    return out, log


def sentinel_rows(frame: pandas.DataFrame) -> pandas.Series:
    """Flags the rows that hold a sentinel code.

    Args:
        frame: The table to search.

    Returns:
        A boolean per row, True when any cell holds a code found by
        `find_sentinels` with its default settings.
    """
    report = find_sentinels(frame)
    return pandas.Series(
        sentinel_mask(frame, report).any(axis=1).to_numpy(dtype=bool),
        index=frame.index,
    )
