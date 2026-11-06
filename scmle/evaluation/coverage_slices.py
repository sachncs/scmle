"""Coverage and interval width or set size per declared slice (F12).

The slice columns are declared in the charter's `uncertainty` block, so they
are part of the registered coverage claim and fixed before the final rows are
read. For each level of a slice the report gives the number of rows, the
share covered with its Clopper-Pearson interval and the mean width (regression)
or set size (classification).

This is description, not a test. The Holm-corrected under-coverage flag of
`coverage.slice_coverage` stays out of this report: its familywise false-flag
rate missed its pre-registered bound (G5, `docs/` and the F12 plan), so no
flag may be read from a slice. A slice with fewer than
`coverage.MIN_SLICE_ROWS` rows is counted and not reported. A numeric column is
cut into `BANDS` bands at quantiles of the development rows, never of the
final rows; a column with more than `MAX_LEVELS` levels cannot be sliced.
"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd

from scmle.evaluation import coverage

MAX_LEVELS = 12
BANDS = 4
NOTE = (
    "descriptive: per-slice coverage with its exact interval; not a test, no "
    "slice is flagged (the flag missed its pre-registered bound, G5)"
)


def labels_for(
    values: pd.Series, reference: pd.Series | None = None
) -> pd.Series | None:
    """Slice label of each row, or None when the column cannot be sliced.

    Args:
        values: The column on the rows being scored.
        reference: The same column on the development rows, where the band
            edges of a numeric column are taken.
    """
    seen = reference if reference is not None else values
    if pd.api.types.is_numeric_dtype(seen) and not pd.api.types.is_bool_dtype(
        seen
    ):
        clean = pd.to_numeric(seen, errors="coerce").dropna()
        if clean.nunique() > MAX_LEVELS:
            edges = np.unique(
                np.quantile(clean, np.linspace(0, 1, BANDS + 1)[1:-1])
            )
            numbers = pd.to_numeric(values, errors="coerce").to_numpy(float)
            bands = np.searchsorted(edges, numbers, side="right") + 1
            labels = [
                f"band {band}" if not np.isnan(n) else "missing"
                for band, n in zip(bands, numbers, strict=True)
            ]
            return pd.Series(labels, index=values.index)
    if seen.nunique() > MAX_LEVELS or values.nunique() > MAX_LEVELS:
        return None
    return values.astype(str)


def summary(
    inside: np.ndarray,
    sizes: np.ndarray,
    labels: Sequence[Any],
    nominal: float,
    level: float = 0.95,
) -> dict[str, Any]:
    """Coverage and mean size of each slice level, with no flag.

    Args:
        inside: Whether the interval or set held, per row.
        sizes: Interval width or set size, per row.
        labels: Slice label of each row.
        nominal: The requested coverage.
        level: Confidence level of each reported interval.
    """
    flags = np.asarray(inside, dtype=bool)
    widths = np.asarray(sizes, dtype=float)
    names = np.asarray([str(v) for v in labels])
    found: list[dict[str, Any]] = []
    skipped = 0
    for name in sorted(set(names)):
        mask = names == name
        rows = int(mask.sum())
        if rows < coverage.MIN_SLICE_ROWS:
            skipped += 1
            continue
        hits = int(flags[mask].sum())
        lower, upper = coverage.clopper_pearson(hits, rows, level)
        chosen = widths[mask]
        found.append(
            {
                "slice": name,
                "rows": rows,
                "covered": hits,
                "coverage": hits / rows,
                "lower": lower,
                "upper": upper,
                "mean_size": None
                if np.isinf(chosen).any()
                else float(chosen.mean()),
            }
        )
    return {
        "descriptive": True,
        "nominal": nominal,
        "level": level,
        "slices": found,
        "skipped_for_support": skipped,
        "note": NOTE,
    }
