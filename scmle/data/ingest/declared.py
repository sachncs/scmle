"""Reconciliation of the extract to totals the charter declared.

The charter's `reconcile` block states, before any data is read, what the
source system says each extracted view contains: a row count, column sums and
a checksum of a key column. The stage compares them with the frames it
actually read; one mismatch stops the run and names the quantity, because
features built on a short or duplicated extract are wrong everywhere.

Counts and checksums are exact. A sum may differ by the charter's relative
`delta_rec`; a declared sum of zero is compared as an absolute gap. This is
agreement of aggregates: offsetting row errors pass, and the declared numbers
are only as good as whoever owns the source system (a person's statement, tier
`attested` at best).

`reconcile_source` is the second, independent check: for every declared view
it queries the source database itself (`reconcile.reconcile_extract`: row
count and column sums; plus the key checksum) and compares the answers with
the frame that was extracted, so a short, duplicated or altered extract is
caught even when the declared numbers are right or absent. The source's
answer is as good as the connection to it; the check is about aggregates too.
"""

from collections.abc import Callable
from collections.abc import Mapping
import hashlib
from typing import Any

import pandas

from scmle.contracts import errors
from scmle.data.charter import ingest
from scmle.data.ingest import profile
from scmle.data.ingest import reconcile as source_reconcile

FILE = "extract_reconciliation.json"


def key_checksum(values: pandas.Series) -> str:
    """SHA-256 of the sorted values as text, joined by newlines.

    The order of the rows does not matter; a changed, missing or repeated
    value does.

    Args:
        values: The key column.

    Returns:
        The hex digest.
    """
    text = "\n".join(sorted(str(v) for v in values.tolist()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def check(
    name: str, extract: Any, source: Any, gap: float, limit: float
) -> dict:
    """Builds the record of one compared quantity.

    Args:
        name: Name of the quantity, e.g. `view.row_count`.
        extract: Value found in the extract.
        source: Value declared or queried from the source.
        gap: Size of the difference (absolute or relative, as the caller
            defines it).
        limit: Largest gap that still passes.

    Returns:
        A mapping of the arguments plus `passed`, True when `gap <= limit`.
    """
    return {
        "name": name,
        "extract": extract,
        "source": source,
        "gap": gap,
        "limit": limit,
        "passed": gap <= limit,
    }


def column_of(frame: pandas.DataFrame, view: str, column: str) -> pandas.Series:
    """Takes a column of a view.

    Args:
        frame: The extracted view.
        view: Name of the view, for the error message.
        column: Column to take.

    Returns:
        The column.

    Raises:
        ConfigError: If the view lacks the column.
    """
    if column not in frame.columns:
        raise errors.ConfigError(
            f"the reconcile block names column {column!r} that view "
            f"{view!r} lacks"
        )
    return frame[column]


def reconcile(
    frames: Mapping[str, pandas.DataFrame],
    spec: ingest.ReconcileSpec,
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
                pandas.to_numeric(
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


def source_check(
    name: str, extract: Any, source: Any, tolerance: float
) -> dict:
    """Compares one quantity with the source's own answer.

    Counts and checksums are exact (`tolerance` 0); a source total of zero is
    compared as an absolute gap.

    Args:
        name: Name of the quantity.
        extract: Value found in the extract.
        source: Value the source database gave.
        tolerance: Largest relative gap that passes.

    Returns:
        The comparison record, as `check` forms it, with the gap relative to
        the source value.
    """
    gap = abs(float(extract) - float(source))
    denominator = abs(float(source)) or 1.0
    return check(name, extract, source, gap / denominator, tolerance)


def source_totals(
    view: str,
    frame: pandas.DataFrame,
    table: str,
    totals: ingest.TableTotals,
    source: profile.SqlSource,
    tolerance: float,
) -> list[dict]:
    """Compares one extracted view with queries of its source table.

    The row count and the declared sum columns go through
    `reconcile.reconcile_extract`; a quantity whose source total is zero (so
    that function forms no ratio) is compared as an absolute gap instead of
    being skipped. The checksum is computed over the source's own column.
    A quantity that `reconcile.reconcile_extract` does not return (including
    when it raises `ValueError`) is compared against a source value of zero.
    A view that lacks a declared column makes `column_of` raise
    `ConfigError`, which propagates.

    Args:
        view: Name of the extracted view.
        frame: The extracted view.
        table: Source table the view comes from.
        totals: The declared sum columns and checksum column for the view.
        source: The source to query.
        tolerance: Relative gap a sum may have.

    Returns:
        One comparison record per quantity: row count, each sum and, when
        declared, the key checksum.
    """
    measures = [total.column for total in totals.sums]
    numeric = pandas.DataFrame(
        {
            m: pandas.to_numeric(column_of(frame, view, m), errors="coerce")
            for m in measures
        },
        index=frame.index,
    )
    found: dict[str, tuple[float, float]] = {}
    try:
        outcome = source_reconcile.reconcile_extract(
            source, numeric, table, measures, tolerance=tolerance
        )
    except ValueError:
        outcome = None
    prefix = f"{table}."
    if outcome is not None:
        for result in outcome.result.results:
            found[result.item.name.removeprefix(prefix)] = (
                result.item.extract,
                result.item.source,
            )
    own = {"row_count": float(len(frame))}
    own.update({f"sum.{m}": float(numeric[m].sum()) for m in measures})
    checks = []
    for name, extract in own.items():
        theirs = found.get(name, (extract, 0.0))[1]
        limit = 0.0 if name == "row_count" else tolerance
        checks.append(source_check(f"{view}.{name}", extract, theirs, limit))
    if totals.checksum_column is not None:
        column = column_of(frame, view, totals.checksum_column)
        values = source.fetch(
            profile.statement(
                "SELECT {c} FROM {t}",
                c=profile.quote(totals.checksum_column),
                t=profile.quote(table),
            )
        )
        their_sum = key_checksum(
            pandas.Series([row[0] for row in values or []])
        )
        my_sum = key_checksum(column)
        checks.append(
            {
                "name": f"{view}.checksum.{totals.checksum_column}",
                "extract": my_sum,
                "source": their_sum,
                "gap": 0.0 if my_sum == their_sum else 1.0,
                "limit": 0.0,
                "passed": my_sum == their_sum,
            }
        )
    return checks


def reconcile_source(
    frames: Mapping[str, pandas.DataFrame],
    tables: Mapping[str, str],
    spec: ingest.ReconcileSpec,
    factory: Callable[[], Any],
    tolerance: float,
) -> list[dict]:
    """Queries the source database for each declared view and compares.

    Args:
        frames: View name to the frame read from the source.
        tables: View name to its source table.
        spec: The charter's reconcile block (which views and columns).
        factory: DB-API connection factory of the source.
        tolerance: Relative gap a sum may have.

    Returns:
        The compared quantities, as `reconcile` forms them.

    Raises:
        ConfigError: If a declared view or column is not in the extract.
    """
    source = profile.SqlSource(factory)
    checks: list[dict] = []
    for totals in spec.tables:
        frame = frames.get(totals.view)
        if frame is None:
            raise errors.ConfigError(
                f"the reconcile block names view {totals.view!r}, which the "
                f"ingest plan does not read; views: {sorted(frames)}"
            )
        checks.extend(
            source_totals(
                totals.view,
                frame,
                tables[totals.view],
                totals,
                source,
                tolerance,
            )
        )
    return checks


def require(report: Mapping[str, Any]) -> None:
    """Stops the run when any declared or source-queried total differs.

    Args:
        report: The reconciliation document, with `checks` and optionally
            `source_checks`.

    Raises:
        BlockedError: If any check failed; names each quantity that differs.
    """
    failed = [c for c in report["checks"] if not c["passed"]]
    queried = [c for c in report.get("source_checks", []) if not c["passed"]]
    if failed or queried:
        parts = [
            f"{c['name']} extract {c['extract']} vs declared {c['source']}"
            for c in failed
        ]
        parts += [
            f"{c['name']} extract {c['extract']} vs source query {c['source']}"
            for c in queried
        ]
        raise errors.BlockedError(
            "the extract does not reproduce the totals of the source "
            "(declared in the charter or queried from the database): "
            + "; ".join(parts),
            {"reconciliation": dict(report)},
        )
