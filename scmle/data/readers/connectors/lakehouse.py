"""Connector for caller-owned warehouse and lakehouse query engines."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import hashlib
from urllib.parse import urlsplit

from scmle.contracts import errors
from scmle.data.readers.connectors import adapter
from scmle.data.readers.connectors import failure
from scmle.data.readers.connectors import layout
from scmle.data.readers.connectors import locator


class Lakehouse(adapter.Adapter):
    """Connector for caller-owned warehouse or lakehouse query engines.

    The connector injects the query engine boundary instead of importing a
    warehouse SDK. A DuckDB, Spark, BigQuery, Snowflake, or lakehouse
    integration can provide the two callbacks while this contract owns
    bounded reads, retries, checkpoints, schema validation, and failure
    evidence.

    Attributes:
        query: Versioned query or table expression used by the engine.
        schema_reader: Callback returning the schema for `query`.
        reader: Callback returning one batch for a query, offset and
            limit.
        snapshot_reader: Callback returning a source-state token, or None.
    """

    schemes = ("warehouse", "lakehouse")

    def __init__(
        self,
        source_uri: str,
        query: str,
        schema_reader: Callable[[str], layout.Columns],
        batch_reader: Callable[
            [str, int, int], tuple[layout.Columns, Sequence[layout.Row], bool]
        ],
        retry_budget: failure.Retries | None = None,
        snapshot_reader: Callable[[str], str] | None = None,
    ) -> None:
        """Initializes a warehouse connector around injected query callbacks.

        Args:
            source_uri: Logical ``warehouse://`` or ``lakehouse://`` URI.
            query: Versioned query or table expression used by the engine.
            schema_reader: Callback returning the source schema for ``query``.
            batch_reader: Callback accepting query, row offset, and row
                limit and returning the batch schema, its rows, and whether
                more rows remain.
            retry_budget: Per-operation retry policy.
            snapshot_reader: Optional callback returning a token for the
                state of ``query`` (a table version, ``AS OF`` timestamp or
                content hash). When given, a resumed read whose token
                differs from the one the checkpoint was bound to is refused
                (``source_changed``). Without it the connector cannot tell
                that the source changed between batches.

        Raises:
            SchemeError: If the scheme is neither
                ``warehouse`` nor ``lakehouse``.
            ValueError: If `query` is blank.
            TypeError: If `schema_reader` or `reader` is not
                callable.
        """
        value = source_uri.strip()
        scheme = urlsplit(value).scheme.lower()
        if scheme not in self.schemes:
            raise failure.SchemeError(value, scheme, self.schemes)
        if not query.strip():
            raise errors.InputError("query must not be empty")
        schema_candidate: object = schema_reader
        batch_candidate: object = batch_reader
        if not callable(schema_candidate) or not callable(batch_candidate):
            raise errors.KindError(
                "schema_reader and batch_reader must be callable"
            )
        self.query = query
        self.schema_reader = schema_reader
        self.reader = batch_reader
        self.snapshot_reader = snapshot_reader
        super().__init__(value, retry_budget)
        self.identity = locator.identify(f"{self.uri}|query={self.query}")

    def schema(self) -> layout.Columns:
        """Reads and validates the schema without consuming a checkpoint.

        The read is retried under the connector's retry budget.

        Returns:
            The schema returned by the schema callback, once validated.
        """
        return failure.persevere(
            lambda: self.vet(self.schema_reader(self.query)),
            "schema",
            self.uri,
            self.budget,
        )

    def read(
        self, checkpoint: locator.Bookmark | None = None, limit: int = 1_000
    ) -> layout.Page:
        """Reads one bounded batch from the injected warehouse engine.

        A non-positive `limit` raises ``ValueError``; a checkpoint bound to
        a different source raises ``CheckpointError``
        (``checkpoint_mismatch``).

        Args:
            checkpoint: Row-offset checkpoint to resume from; None starts
                at row 0.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The validated batch with a checkpoint advanced by the rows
            returned.
        """
        self.validate(limit)
        value = self.rebind(checkpoint)
        return failure.persevere(
            lambda: self.accept(value, limit),
            "read_rows",
            self.uri,
            self.budget,
        )

    def accept(self, checkpoint: locator.Bookmark, limit: int) -> layout.Page:
        """Validates one callback result and binds its next checkpoint.

        When a snapshot reader is configured, its token (hashed with
        SHA-256) is stored in the next checkpoint and compared with the one
        in `checkpoint`.

        Args:
            checkpoint: Bound checkpoint to resume from.
            limit: Maximum number of rows the callback may return.

        Returns:
            The batch with the next checkpoint.

        Raises:
            CheckpointError: ``source_changed`` if the snapshot token
                differs from the one recorded in the checkpoint.
            ConnectorError: If the callback's ``has_more`` is not a bool,
                it returns more than `limit` rows, or a row's width differs
                from the schema's.
        """
        token = (
            hashlib.sha256(
                self.snapshot_reader(self.query).encode()
            ).hexdigest()
            if self.snapshot_reader
            else None
        )
        if (
            token is not None
            and checkpoint.digest is not None
            and checkpoint.digest != token
        ):
            raise failure.CheckpointError(
                self.uri,
                "source_changed",
                "warehouse snapshot token changed since the last batch",
                {"offset": checkpoint.offset},
            )
        schema, raw_rows, has_more = self.reader(
            self.query, checkpoint.offset, limit
        )
        schema = self.vet(schema)
        if not isinstance(has_more, bool):
            raise failure.ConnectorError(
                "invalid_batch_more_flag",
                "batch reader has_more must be bool",
                self.uri,
                {"value_type": type(has_more).__name__},
            )
        rows = tuple(tuple(row) for row in raw_rows)
        if len(rows) > limit:
            raise failure.ConnectorError(
                "batch_limit_exceeded",
                "batch reader returned more rows than requested",
                self.uri,
                {"limit": limit, "received": len(rows)},
            )
        width = len(schema.columns)
        if any(len(row) != width for row in rows):
            raise failure.ConnectorError(
                "row_width_mismatch",
                "warehouse row width does not match its schema",
                self.uri,
                {"expected": width},
            )
        after = checkpoint.advance(len(rows))
        if token is not None:
            after = dataclasses.replace(after, digest=token)
        return layout.Page(
            schema=schema,
            rows=rows,
            checkpoint=after,
            more=has_more,
        )

    def vet(self, schema: layout.Columns) -> layout.Columns:
        """Validates a callback schema at the connector boundary.

        Args:
            schema: The value returned by a callback.

        Returns:
            `schema`, unchanged.

        Raises:
            ConnectorError: ``invalid_schema`` if it is not a
                `Columns`.
        """
        if not isinstance(schema, layout.Columns):
            raise failure.ConnectorError(
                "invalid_schema",
                "schema reader must return Columns",
                self.uri,
                {"schema_type": type(schema).__name__},
            )
        return schema
