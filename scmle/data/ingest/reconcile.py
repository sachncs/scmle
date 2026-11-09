"""Reconciliation of an assembled table to source totals.

Extends `admissibility.reconciliation`: the source side of every `Item` comes
from a query pushed down to the source instead of a number typed by a person.
Four kinds of quantity are compared:

* row count of the assembled table against `COUNT(*)` of its base table;
* conservation of each measure (`SUM`) against the base table;
* fan-out of each join, output rows over left rows, which must be 1 for a
  many-to-one join;
* end-of-data feature totals (count and sum features) against the source
  aggregate.

Agreement is about aggregates, not rows: offsetting row-level errors pass. A
source whose total is zero cannot form a gap ratio and is listed as
`not_reconciled` rather than passed.
"""

from collections.abc import Sequence
import dataclasses
import sys
from typing import Any

import pandas

from scmle.contracts import qualification
from scmle.data.admissibility import reconciliation
from scmle.data.detectors import certificates
from scmle.data.ingest import profile

DETECTOR_ID = "ingest-reconcile"
DEFECT_FANOUT = "fanout"
DEFECT_DROPPED = "dropped_rows"
DEFECT_TORN = "torn_read"
DEFECT_TYPES = (DEFECT_FANOUT, DEFECT_DROPPED, DEFECT_TORN)
SOURCE_NAME = "source query at read time"


@dataclasses.dataclass(frozen=True, slots=True)
class JoinCheck:
    """Row counts around one join.

    Attributes:
        name: Join label.
        left_rows: Rows entering the join.
        output_rows: Rows leaving it.
    """

    name: str
    left_rows: int
    output_rows: int


@dataclasses.dataclass(frozen=True, slots=True)
class FeatureTotal:
    """An end-of-data total of a feature against a source query.

    Attributes:
        name: Feature name.
        extract: Total of the feature over all target entities with the
            moment set after the last record.
        sql: Query returning the source aggregate in one cell.
    """

    name: str
    extract: float
    sql: str


@dataclasses.dataclass(frozen=True, slots=True)
class ExtractReconciliation:
    """Reconciliation plus the quantities it could not form.

    Attributes:
        result: The ordinary reconciliation.
        not_reconciled: Names of quantities whose source total was zero or
            null, so no gap ratio exists; they are `unverified`.
    """

    result: reconciliation.Reconciliation
    not_reconciled: tuple[str, ...]

    @property
    def passed(self) -> bool:
        """Whether there is no unexplained gap and nothing unreconciled."""
        return self.result.passed and not self.not_reconciled

    def to_dict(self) -> dict[str, Any]:
        """Serializes the reconciliation, for `reconciliation.json`.

        Returns:
            The ordinary reconciliation mapping plus `not_reconciled` and
            the overall `passed`.
        """
        return {
            **self.result.to_dict(),
            "not_reconciled": list(self.not_reconciled),
            "passed": self.passed,
        }


def reconcile_extract(
    source: profile.SqlSource,
    assembled: pandas.DataFrame,
    base_table: str,
    measures: Sequence[str] = (),
    joins: Sequence[JoinCheck] = (),
    feature_totals: Sequence[FeatureTotal] = (),
    tolerance: float = 0.0,
    gate: certificates.GateStatus = certificates.GateStatus.DESCRIPTIVE,
) -> ExtractReconciliation:
    """Compares an assembled table with the source it came from.

    Args:
        source: Read-only source; the totals are queried now.
        assembled: The table after extraction, joins and aggregation.
        base_table: Source table the rows are one-to-one with.
        measures: Columns that exist in both tables; their sums must be
            conserved.
        joins: Row counts around each join.
        feature_totals: End-of-data totals of count and sum features.
        tolerance: Relative gap that needs no explanation.
        gate: Whether this detector has a current certificate.

    Returns:
        The reconciliation and the quantities that could not be formed.

    Raises:
        ValueError: If no quantity could be formed because every source
            total was zero or null.
    """
    items = []
    skipped = []

    def add(name: str, extract: float, expected: Any) -> None:
        if expected is None or float(expected) == 0.0:
            skipped.append(name)
            return
        items.append(
            reconciliation.Item(
                name, float(extract), float(expected), SOURCE_NAME
            )
        )

    quoted = profile.quote(base_table)
    add(
        f"{base_table}.row_count",
        len(assembled),
        source.scalar(profile.statement("SELECT COUNT(*) FROM {t}", t=quoted)),
    )
    for measure in measures:
        total = source.scalar(
            profile.statement(
                "SELECT SUM({m}) FROM {t}",
                m=profile.quote(measure),
                t=quoted,
            )
        )
        add(
            f"{base_table}.sum.{measure}",
            float(assembled[measure].sum()),
            total,
        )
    for join in joins:
        add(f"fanout.{join.name}", join.output_rows, join.left_rows)
    for total_check in feature_totals:
        add(
            f"feature_total.{total_check.name}",
            total_check.extract,
            source.scalar(total_check.sql),
        )
    if not items:
        raise ValueError("nothing could be reconciled: all source totals zero")
    result = reconciliation.reconcile(items, tolerance, gate)
    return ExtractReconciliation(result, tuple(skipped))


def code_digest() -> str:
    """Hashes this module's source, the key of its certificate.

    Returns:
        The code hash computed by `certificates.code_hash`.
    """
    return certificates.code_hash(sys.modules[__name__])


def blocks(report: ExtractReconciliation) -> bool:
    """Checks whether the reconciliation should stop the run.

    Args:
        report: The reconciliation to inspect.

    Returns:
        True if any check carries a blocking decision, which happens only
        when the gate is decisive.
    """
    return any(
        check.decision is qualification.Decision.BLOCK
        for check in report.result.checks
    )
