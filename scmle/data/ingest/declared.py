"""Reconciliation of the extract to totals the charter declared (F9).

The charter's `reconcile` block states, before any data is read, what the
source system says each extracted view contains: a row count, column sums and
a checksum of a key column. The stage compares them with the frames it
actually read; one mismatch stops the run and names the quantity, because
features built on a short or duplicated extract are wrong everywhere.

Counts and checksums are exact. A sum may differ by the charter's relative
`delta_rec`; a declared sum of zero is compared as an absolute gap. This is
agreement of aggregates: offsetting row errors pass, and the declared numbers
are only as good as whoever owns the source system (a person's statement, tier
`attested` at best). `scmle.data.ingest.reconcile` queries the source instead
and is a different check.
"""

from collections.abc import Mapping
import hashlib
from typing import Any

import pandas as pd

from scmle.contracts import errors
from scmle.data.charter import ingest as charter_ingest

FILE = "extract_reconciliation.json"


def key_checksum(values: pd.Series) -> str:
    """SHA-256 of the sorted values as text, joined by newlines.

    The order of the rows does not matter; a changed, missing or repeated
    value does.
    """
    text = "\n".join(sorted(str(v) for v in values.tolist()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check(
    name: str, extract: Any, source: Any, gap: float, limit: float
) -> dict:
    """One compared quantity."""
    return {
        "name": name,
        "extract": extract,
        "source": source,
        "gap": gap,
        "limit": limit,
        "passed": gap <= limit,
    }


def column_of(frame: pd.DataFrame, view: str, column: str) -> pd.Series:
    """A column of a view, or a `ConfigError` naming it."""
    if column not in frame.columns:
        raise errors.ConfigError(
            f"the reconcile block names column {column!r} that view "
            f"{view!r} lacks"
        )
    return frame[column]


def reconcile(
    frames: Mapping[str, pd.DataFrame],
    spec: charter_ingest.ReconcileSpec,
    tolerance: float,
) -> dict[str, Any]:
    """Compares every declared total with the extracted view.

    Args:
        frames: View name to the frame read from the source.
        spec: The charter's declared totals.
        tolerance: Relative gap a sum may have.

    Returns:
        The `extract_reconciliation.json` document.

    Raises:
        ConfigError: If a declared view or column is not in the extract.
    """
    checks = []
    for table in spec.tables:
        frame = frames.get(table.view)
        if frame is None:
            raise errors.ConfigError(
                f"the reconcile block names view {table.view!r}, which the "
                f"ingest plan does not read; views: {sorted(frames)}"
            )
        if table.rows is not None:
            checks.append(
                check(
                    f"{table.view}.row_count",
                    len(frame),
                    table.rows,
                    float(abs(len(frame) - table.rows)),
                    0.0,
                )
            )
        for total in table.sums:
            found = float(
                pd.to_numeric(
                    column_of(frame, table.view, total.column), errors="coerce"
                ).sum()
            )
            denominator = abs(total.total) or 1.0
            checks.append(
                check(
                    f"{table.view}.sum.{total.column}",
                    found,
                    total.total,
                    abs(found - total.total) / denominator,
                    tolerance,
                )
            )
        if table.checksum is not None and table.checksum_column is not None:
            found_sum = key_checksum(
                column_of(frame, table.view, table.checksum_column)
            )
            checks.append(
                check(
                    f"{table.view}.checksum.{table.checksum_column}",
                    found_sum,
                    table.checksum,
                    0.0 if found_sum == table.checksum else 1.0,
                    0.0,
                )
            )
    return {
        "source": "totals declared in the charter",
        "tolerance": tolerance,
        "checks": checks,
        "passed": all(c["passed"] for c in checks),
    }


def require(report: Mapping[str, Any]) -> None:
    """Stops the run when any declared total is not reproduced.

    Raises:
        BlockedError: Naming each quantity that differs.
    """
    failed = [c for c in report["checks"] if not c["passed"]]
    if failed:
        raise errors.BlockedError(
            "the extract does not reproduce the totals the charter declared: "
            + "; ".join(
                f"{c['name']} extract {c['extract']} vs declared {c['source']}"
                for c in failed
            ),
            {"reconciliation": dict(report)},
        )
