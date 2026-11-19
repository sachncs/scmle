"""Connector for read-only queries through an injected DB-API factory."""

from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Sequence
import contextlib
import dataclasses
import hashlib
import re
import sqlite3
from typing import Any
from urllib.parse import urlsplit

from scmle.contracts import errors
from scmle.data.readers.connectors import adapter
from scmle.data.readers.connectors import failure
from scmle.data.readers.connectors import layout
from scmle.data.readers.connectors import locator

WRITE_KEYWORDS = frozenset(
    {
        "alter",
        "attach",
        "call",
        "copy",
        "create",
        "delete",
        "detach",
        "drop",
        "exec",
        "execute",
        "grant",
        "insert",
        "into",
        "merge",
        "pragma",
        "reindex",
        "revoke",
        "truncate",
        "update",
        "vacuum",
    }
)
READ_STARTS = frozenset({"select", "with", "values"})
LITERALS_AND_COMMENTS = re.compile(
    r"'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"|--[^\n]*|/\*.*?\*/", re.DOTALL
)
WORD = re.compile(r"[A-Za-z]+")


def readonly(query: str, source_uri: str) -> None:
    """Rejects a statement that is not a single read-only query.

    The allow-list is textual and deliberately strict: after comments and
    quoted text are removed the statement must be one ``SELECT``, ``WITH``
    or ``VALUES`` statement that contains no data-definition, data-change,
    ``INTO`` or session-control keyword. It refuses some harmless text (a
    column named ``update`` must be quoted) and is a guard against mistakes,
    not a sandbox: a database role without write privilege is the real
    control and cannot be verified from here.

    Args:
        query: SQL text to check.
        source_uri: Source URI recorded in the failure.

    Raises:
        ConnectorError: ``write_statement_refused`` with the reason.
    """
    cleaned = LITERALS_AND_COMMENTS.sub(" ", query).strip().rstrip(";").strip()
    words = [word.lower() for word in WORD.findall(cleaned)]
    reason = ""
    if ";" in cleaned:
        reason = "more than one statement"
    elif not words or words[0] not in READ_STARTS:
        reason = "statement does not start with SELECT, WITH or VALUES"
    else:
        found = sorted(WRITE_KEYWORDS.intersection(words))
        if found:
            reason = f"contains write keyword(s) {found}"
    if reason:
        raise failure.ConnectorError(
            "write_statement_refused",
            f"query refused by the read-only allow-list: {reason}",
            source_uri,
            {"reason": reason},
        )


def feed(hasher: Any, rows: Sequence[Sequence[Any]]) -> None:
    """Feeds rows into a hash in a canonical text form.

    Args:
        hasher: A hash object with an ``update(bytes)`` method, updated in
            place.
        rows: Rows to digest; each is hashed as the UTF-8 ``repr`` of its
            tuple followed by a newline.
    """
    for row in rows:
        hasher.update(repr(tuple(row)).encode("utf-8"))
        hasher.update(b"\n")


def release(cursor: Any, connection: Any) -> None:
    """Closes resources without masking the primary operation error.

    Args:
        cursor: Cursor to close, or None if none was opened.
        connection: Connection to close.
    """
    if cursor is not None:
        with contextlib.suppress(Exception):
            cursor.close()
    with contextlib.suppress(Exception):
        connection.close()


class DbApiConnector(adapter.Adapter):
    """Connector for a query executed through an injected DB-API factory.

    The factory is called for each operation and must return a DB-API 2.0
    connection. This keeps the connector dependency-free and makes lifecycle,
    retry, and failure behavior explicit to the caller.

    The query is checked against a read-only allow-list at construction
    (``readonly``) and, for ``sqlite3`` connections, the session is put
    in ``query_only`` mode while it runs. Other drivers can pass
    ``preamble`` (for example ``SET TRANSACTION READ ONLY``),
    which runs before the query. A database role without write privilege is
    still the only complete control.

    Batches re-execute the query. Every batch carries a digest of the rows
    before its offset in ``Bookmark.digest`` and the next
    batch refuses to continue (``source_changed``) when the rows it skips no
    longer hash to that digest. Use an ``ORDER BY`` so the order is total.

    Attributes:
        query: The read-only SQL query whose rows form the source.
        factory: Callable returning a fresh DB-API connection.
        guarded: Whether the allow-list and read-only session are
            enforced.
        preamble: Statement run before the query, or None.
        probe: Query whose result identifies the source state, or
            None.
    """

    schemes = ("dbapi", "sqlite")

    def __init__(
        self,
        source_uri: str,
        query: str,
        factory: Callable[[], Any],
        retry_budget: failure.Retries | None = None,
        guarded: bool = True,
        preamble: str | None = None,
        probe: str | None = None,
    ) -> None:
        """Initializes a DB-API connector.

        Args:
            source_uri: Logical ``dbapi://`` or ``sqlite://`` source URI.
            query: Read-only SQL query whose rows form the source.
            factory: Callable returning a fresh DB-API connection.
            retry_budget: Per-operation retry policy.
            guarded: Enforce the read-only allow-list (a statement that
                fails it is refused with ``ConnectorError``) and the
                ``sqlite3`` ``query_only`` session mode. When False,
                neither is applied.
            preamble: Optional statement run before the query to
                start a read-only transaction on drivers that support it.
            probe: Optional cheap query whose result identifies the
                state of the source (a version counter, a maximum id). Without
                it the token is a digest of the whole query result.

        Raises:
            SchemeError: If the URI scheme is not ``dbapi``
                or ``sqlite``.
            ValueError: If `query` is blank.
            TypeError: If `factory` is not callable.
        """
        value = source_uri.strip()
        parsed = urlsplit(value)
        scheme = parsed.scheme.lower()
        if scheme not in self.schemes:
            raise failure.SchemeError(value, scheme, self.schemes)
        if not query.strip():
            raise errors.InputError("query must not be empty")
        if not callable(factory):
            raise errors.KindError("factory must be callable")
        if guarded:
            readonly(query, value)
            if probe is not None:
                readonly(probe, value)
        self.query = query
        self.factory = factory
        self.guarded = guarded
        self.preamble = preamble
        self.probe = probe
        super().__init__(value, retry_budget)
        self.identity = locator.identify(f"{self.uri}|query={self.query}")

    @contextlib.contextmanager
    def session(self, sql: str) -> Generator[Any, None, None]:
        """Opens a connection and executes ``sql`` on a cursor.

        The cursor and connection are closed on exit, ignoring errors from
        closing. The session is read-only as configured by `guarded`.

        Args:
            sql: Statement to execute before yielding.

        Yields:
            The DB-API cursor positioned on the result of ``sql``.
        """
        connection = self.factory()
        cursor: Any = None
        try:
            if self.guarded and isinstance(connection, sqlite3.Connection):
                connection.execute("PRAGMA query_only = ON")
            cursor = connection.cursor()
            if self.preamble:
                cursor.execute(self.preamble)
            cursor.execute(sql)
            yield cursor
        finally:
            release(cursor, connection)

    def schema(self) -> layout.Columns:
        """Executes the query and describes the result columns.

        Returns:
            The schema built from the cursor description; a missing type
            code becomes ``unknown``.
        """
        return failure.persevere(
            self.introspect,
            "schema",
            self.uri,
            self.budget,
        )

    def introspect(self) -> layout.Columns:
        """Performs one non-retried DB-API schema operation.

        Returns:
            The schema built from the cursor description.
        """
        with self.session(self.query) as cursor:
            return self.translate(cursor.description)

    def snapshot(self) -> str:
        """Returns a token that changes when the source content changes.

        With ``probe`` the token digests that query's rows; without
        it, every row of the main query (a full scan).

        Returns:
            The SHA-256 hex digest over the rows of the snapshot query, or
            of the main query when none was given.
        """
        return failure.persevere(
            self.fingerprint,
            "snapshot_token",
            self.uri,
            self.budget,
        )

    def fingerprint(self) -> str:
        """Performs one non-retried token computation.

        Returns:
            The SHA-256 hex digest over every row the token query yields.
        """
        hasher = hashlib.sha256()
        with self.session(self.probe or self.query) as cursor:
            while rows := cursor.fetchmany(10_000):
                feed(hasher, rows)
        return hasher.hexdigest()

    def read(
        self, checkpoint: locator.Bookmark | None = None, limit: int = 1_000
    ) -> layout.Page:
        """Reads DB-API rows using portable offset/cursor checkpoints.

        A non-positive `limit` raises ``ValueError``; a checkpoint bound to
        a different source raises ``CheckpointError``
        (``checkpoint_mismatch``).

        Args:
            checkpoint: Row-offset checkpoint to resume from; None starts
                at row 0.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The rows after the checkpoint offset, the schema, a checkpoint
            advanced by the rows returned and carrying the digest of all
            rows up to it, and whether more rows remain.
        """
        self.validate(limit)
        value = self.rebind(checkpoint)
        return failure.persevere(
            lambda: self.fetch(value, limit),
            "read_rows",
            self.uri,
            self.budget,
        )

    def fetch(self, checkpoint: locator.Bookmark, limit: int) -> layout.Page:
        """Performs one non-retried DB-API row operation.

        Args:
            checkpoint: Bound checkpoint to resume from.
            limit: Maximum number of rows to return.

        Returns:
            The batch after the checkpoint offset. ``has_more`` is true when
            one more row than `limit` could be fetched.

        Raises:
            CheckpointError: ``source_changed`` when the rows before the
                checkpoint offset no longer match the checkpoint digest.
        """
        with self.session(self.query) as cursor:
            schema = self.translate(cursor.description)
            hasher = hashlib.sha256()
            self.skip(cursor, checkpoint.offset, limit, hasher)
            expected = checkpoint.digest
            if expected is not None and hasher.hexdigest() != expected:
                raise failure.CheckpointError(
                    self.uri,
                    "source_changed",
                    "rows before the checkpoint changed since the last batch",
                    {"offset": checkpoint.offset},
                )
            fetched = cursor.fetchmany(limit + 1)
            rows = tuple(tuple(row) for row in fetched[:limit])
            feed(hasher, rows)
            next_checkpoint = dataclasses.replace(
                checkpoint.advance(len(rows)),
                digest=hasher.hexdigest(),
            )
            return layout.Page(
                schema=schema,
                rows=rows,
                checkpoint=next_checkpoint,
                more=len(fetched) > limit,
            )

    def translate(
        self, description: Sequence[Sequence[Any]] | None
    ) -> layout.Columns:
        """Converts DB-API cursor metadata into a validated schema.

        Args:
            description: The cursor's ``description``: one sequence per
                column whose first item is the name and second the type code.

        Returns:
            The schema; a missing or empty type code is reported as
            ``unknown``.

        Raises:
            ValueError: If the description is None or empty.
        """
        if not description:
            raise errors.InputError("query did not provide a result schema")
        return layout.Columns(
            tuple(
                layout.Field(
                    str(column[0]),
                    str(column[1])
                    if len(column) > 1 and column[1]
                    else "unknown",
                )
                for column in description
            )
        )

    def skip(
        self, cursor: Any, offset: int, batch_size: int, hasher: Any
    ) -> None:
        """Consumes and hashes rows before a checkpoint.

        Args:
            cursor: Cursor positioned at the first row.
            offset: Number of rows to skip.
            batch_size: Maximum rows fetched per call.
            hasher: Hash object updated with the skipped rows.

        Raises:
            CheckpointError: ``source_changed`` when fewer than ``offset``
                rows remain, which means rows were deleted.
        """
        remaining = offset
        while remaining:
            skipped = cursor.fetchmany(min(remaining, batch_size))
            if not skipped:
                raise failure.CheckpointError(
                    self.uri,
                    "source_changed",
                    "source has fewer rows than the checkpoint offset",
                    {"offset": offset, "missing": remaining},
                )
            feed(hasher, skipped)
            remaining -= len(skipped)
