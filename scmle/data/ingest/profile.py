"""Push-down profile and consistent reads (F9 section 1).

Uniqueness and inclusion are universal statements: a sample can refute them
but cannot confirm them. So every statistic here is an aggregate query run in
the source, and rows are pulled only by `consistent_read`, which needs a total
order and compares a snapshot token before and after the read. A read whose
token or row count changed is rejected as torn.

Honest limits. The token is only as good as the source: for a `DbApiConnector`
without a `snapshot_query` it is a digest of the whole result (exact, but a
full scan); a source that offers no token is read as `verified=False`.
Read-only enforcement is a textual allow-list plus a query-only session, not a
privilege boundary.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import re
from typing import Any

import pandas as pd

from scmle.data.ingest import budget as budgets
from scmle.data.readers import connectors

ORDER_BY = re.compile(r"\border\s+by\b", re.IGNORECASE)


def quote(name: str) -> str:
    """Quotes one SQL identifier.

    Raises:
        ValueError: On an empty name or one with a NUL or double quote.
    """
    if not name or "\x00" in name or '"' in name:
        raise ValueError(f"unsafe identifier: {name!r}")
    return f'"{name}"'


def quote_all(names: Sequence[str]) -> str:
    """Comma-separated quoted identifiers."""
    return ", ".join(quote(name) for name in names)


class SqlSource:
    """Read-only SQL access that counts queries and charges a budget.

    Every statement passes the connector allow-list (`refuse_writes`) and
    runs in a query-only session where the driver supports it.
    """

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        meter: budgets.BudgetMeter | None = None,
        uri: str = "sqlite://ingest",
        read_only_statement: str | None = None,
    ) -> None:
        """Initializes the source around an injected DB-API factory."""
        self.connector = connectors.DbApiConnector(
            uri,
            "SELECT 1",
            connection_factory,
            read_only_statement=read_only_statement,
        )
        self.meter = meter
        self.queries = 0

    def fetch(
        self,
        sql: str,
        stage: str = "",
        unit: str = "",
        cost_rows: int = 0,
    ) -> list[tuple[Any, ...]] | None:
        """Runs one read-only query.

        Returns:
            The rows, or None when the budget refused the step (the boundary
            is on the meter).

        Raises:
            ConnectorError: If the statement is not a read-only query.
        """
        connectors.refuse_writes(sql, self.connector.source_uri)
        if (
            self.meter is not None
            and stage
            and not self.meter.admit(stage, unit, cost_rows)
        ):
            return None
        self.queries += 1
        with self.connector.session(sql) as cursor:
            return [tuple(row) for row in cursor.fetchall()]

    def scalar(self, sql: str, stage: str = "", unit: str = "") -> Any | None:
        """The first cell of the first row, or None if refused or empty."""
        rows = self.fetch(sql, stage, unit)
        return rows[0][0] if rows else None

    def columns(self, table: str, stage: str = "") -> tuple[str, ...]:
        """Column names of a table, in order (one metadata query)."""
        sql = f"SELECT * FROM {quote(table)} WHERE 1 = 0"
        connectors.refuse_writes(sql, self.connector.source_uri)
        if (
            self.meter is not None
            and stage
            and not self.meter.admit(stage, table, 0)
        ):
            return ()
        self.queries += 1
        with self.connector.session(sql) as cursor:
            return tuple(str(item[0]) for item in cursor.description)


def value_kind(value: Any) -> str:
    """`number`, `text`, `other` or `empty` for a column's minimum value."""
    if value is None:
        return "empty"
    if isinstance(value, bool):
        return "other"
    if isinstance(value, int | float):
        return "number"
    if isinstance(value, str):
        return "text"
    return "other"


@dataclasses.dataclass(frozen=True, slots=True)
class ColumnProfile:
    """Exact aggregates of one column at the snapshot read.

    Attributes:
        name: Column name.
        kind: `number`, `text`, `other` or `empty`.
        non_null: Count of non-null values.
        distinct: Count of distinct non-null values.
        minimum: Smallest value, as text, or None.
        maximum: Largest value, as text, or None.
    """

    name: str
    kind: str
    non_null: int
    distinct: int
    minimum: str | None
    maximum: str | None

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class TableProfile:
    """Row count and column profiles of one table."""

    name: str
    rows: int
    columns: tuple[ColumnProfile, ...]

    def column(self, name: str) -> ColumnProfile:
        """The profile of one column."""
        for item in self.columns:
            if item.name == name:
                return item
        raise KeyError(f"{self.name} has no column {name!r}")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return {
            "name": self.name,
            "rows": self.rows,
            "columns": [c.to_dict() for c in self.columns],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Catalog:
    """Profiles of the profiled tables, as `ingest_catalog.json`.

    Attributes:
        tables: One profile per table that was profiled.
        partial: True when the budget stopped profiling before every table.
        fingerprint: Digest of the aggregates. It detects a change in any
            profiled count, distinct count or extreme; it is not a content
            hash, so a change that preserves them is not seen.
    """

    tables: tuple[TableProfile, ...]
    partial: bool = False

    @property
    def fingerprint(self) -> str:
        """Digest of the aggregates in this catalog."""
        document = json.dumps(
            [t.to_dict() for t in self.tables], sort_keys=True
        )
        return hashlib.sha256(document.encode("utf-8")).hexdigest()

    def table(self, name: str) -> TableProfile:
        """The profile of one table."""
        for item in self.tables:
            if item.name == name:
                return item
        raise KeyError(f"table {name!r} was not profiled")

    def to_dict(self) -> dict[str, Any]:
        """`ingest_catalog.json`."""
        return {
            "partial": self.partial,
            "fingerprint": self.fingerprint,
            "tables": [t.to_dict() for t in self.tables],
        }


def profile_table(
    source: SqlSource, table: str, stage: str = "profile"
) -> TableProfile | None:
    """Profiles one table by aggregate queries; None if the budget stops it."""
    quoted = quote(table)
    count = source.scalar(f"SELECT COUNT(*) FROM {quoted}", stage, table)
    if count is None:
        return None
    rows = int(count)
    profiles = []
    names = source.columns(table, stage)
    if not names:
        return None
    for name in names:
        column = quote(name)
        found = source.fetch(
            f"SELECT COUNT({column}), COUNT(DISTINCT {column}), "
            f"MIN({column}), MAX({column}) FROM {quoted}",
            stage,
            f"{table}.{name}",
            rows,
        )
        if found is None:
            return None
        non_null, distinct, low, high = found[0]
        profiles.append(
            ColumnProfile(
                name,
                value_kind(low),
                int(non_null),
                int(distinct),
                None if low is None else str(low),
                None if high is None else str(high),
            )
        )
    return TableProfile(table, rows, tuple(profiles))


def profile_catalog(
    source: SqlSource, tables: Sequence[str], stage: str = "profile"
) -> Catalog:
    """Profiles tables in the order given, stopping at a table boundary.

    A table whose profile the budget cannot finish is left out whole, so every
    profile in the catalog is complete; `partial` says some were not done.
    """
    done = []
    partial = False
    for table in tables:
        profile = profile_table(source, table, stage)
        if profile is None:
            partial = True
            break
        done.append(profile)
    return Catalog(tuple(done), partial)


class TornReadError(connectors.ConnectorError):
    """The source changed while it was being read."""

    def __init__(self, source_uri: str, attempts: int, reason: str) -> None:
        """Records how many attempts were made and why the last failed."""
        super().__init__(
            "torn_read",
            f"source changed during the read ({reason}) after "
            f"{attempts} attempt(s)",
            source_uri,
            {"attempts": attempts, "reason": reason},
        )
        self.attempts = attempts
        self.reason = reason


@dataclasses.dataclass(frozen=True, slots=True)
class ConsistentRead:
    """Rows read under a snapshot check.

    Attributes:
        frame: The rows.
        token: Snapshot token before and after the read, or None.
        verified: True only when a token was compared before and after.
        attempts: Reads made, including torn ones.
    """

    frame: pd.DataFrame
    token: str | None
    verified: bool
    attempts: int


def has_total_order(query: str) -> bool:
    """Whether the query text has an `ORDER BY` outside quoted text."""
    cleaned = connectors.LITERALS_AND_COMMENTS.sub(" ", query)
    return ORDER_BY.search(cleaned) is not None


def read_once(
    connector: connectors.SourceConnector, batch_rows: int
) -> pd.DataFrame:
    """Reads every batch of a connector into one frame."""
    rows: list[tuple[Any, ...]] = []
    checkpoint = None
    names: list[str] = []
    while True:
        batch = connector.read_rows(checkpoint, batch_rows)
        names = [c.name for c in batch.schema.columns]
        rows.extend(batch.rows)
        checkpoint = batch.checkpoint
        if not batch.has_more:
            break
    return pd.DataFrame(rows, columns=names)


def consistent_read(
    connector: connectors.SourceConnector,
    batch_rows: int = 1_000,
    token: Callable[[], str] | None = None,
    attempts: int = 2,
    require_order: bool = True,
) -> ConsistentRead:
    """Reads a source completely, rejecting a torn read.

    Args:
        connector: Any connector; `DbApiConnector` also checks the rows before
            each batch offset (`source_changed`).
        batch_rows: Rows per batch.
        token: Snapshot-token callback. Defaults to the connector's
            `snapshot_token` when it has one; with none the read is returned
            with `verified=False` and must be treated as `unverified`.
        attempts: Reads to try before giving up.
        require_order: Refuse a query with no `ORDER BY`, because batches of
            an unordered query are not a function of the offset.

    Raises:
        ConnectorError: `unordered_query` when the order is missing.
        TornReadError: When every attempt saw a change.
    """
    query = getattr(connector, "query", None)
    if require_order and isinstance(query, str) and not has_total_order(query):
        raise connectors.ConnectorError(
            "unordered_query",
            "batched reads need an ORDER BY on a unique key",
            connector.source_uri,
            {},
        )
    probe = token or getattr(connector, "snapshot_token", None)
    reason = "no attempt"
    for attempt in range(1, max(attempts, 1) + 1):
        before = probe() if probe else None
        try:
            frame = read_once(connector, batch_rows)
        except connectors.CheckpointError as error:
            if error.code != "source_changed":
                raise
            reason = "rows before a batch offset changed"
            continue
        after = probe() if probe else None
        if before != after:
            reason = "snapshot token changed"
            continue
        return ConsistentRead(frame, before, probe is not None, attempt)
    raise TornReadError(connector.source_uri, attempts, reason)
