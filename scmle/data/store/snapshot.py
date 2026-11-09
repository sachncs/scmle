"""Snapshot identity, schema version, freshness and lineage for every read.

A read through `SnapshotReader` returns the frame and a `SnapshotRecord`:
which object was read (URI, content hash, size, modification time, backend
version), under which declared schema version and whether the data matched
it, how old the data was against the contract's freshness limit, and which
earlier snapshots it was derived from. The record's `snapshot_id` is a hash
of the URI, the content hash and the schema digest, so the same bytes under
the same contract always give the same id, and a changed object gives a new
one. Every read is appended to a `ReadLog` (and to a JSON-lines file when one
is given), which is what a later training run is traced back through.

`connector_reader` adapts a store to the `read_bytes` callable of
`ObjectStoreConnector`, and decoding uses the same `read_tabular_bytes` as the
source connectors.
"""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import hashlib
import json
import pathlib
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from scmle import durability
from scmle.contracts import errors
from scmle.data.readers import connectors
from scmle.data.store import objectstore

FRESH = "fresh"
STALE = "stale"
UNKNOWN = "unknown"
KINDS = ("int", "float", "bool", "datetime", "string")


class SnapshotError(errors.DataError):
    """A snapshot could not be read or recorded."""

    category = "snapshot"


def kind_of(series: pd.Series) -> str:
    """The storage-independent kind of a column.

    `int`, `float`, `bool`, `datetime` or `string`, so a contract does not
    depend on how a particular reader or pandas version names a dtype.

    Args:
        series: The column.

    Returns:
        One of `KINDS`: `bool`, `int`, `float` or `datetime` by the
        series' dtype, tested in that order, and `string` for anything else.
    """
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    if pd.api.types.is_integer_dtype(series):
        return "int"
    if pd.api.types.is_float_dtype(series):
        return "float"
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    return "string"


def observed_schema(frame: pd.DataFrame) -> dict[str, str]:
    """Reads the schema a frame actually has.

    Args:
        frame: The data.

    Returns:
        Column name (as a string) to kind, in column order.
    """
    return {str(name): kind_of(frame[name]) for name in frame.columns}


def schema_digest(schema: Mapping[str, str]) -> str:
    """Hashes a schema, independent of column order.

    Args:
        schema: Column name to kind.

    Returns:
        `sha256:` followed by the hex SHA-256 of the sorted name and kind
        pairs.
    """
    text = json.dumps(sorted(schema.items()), separators=(",", ":"))
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


@dataclasses.dataclass(frozen=True, slots=True)
class DataContract:
    """What a consumer expects of a dataset.

    Attributes:
        name: Dataset name.
        schema_version: Version label of the declared schema.
        columns: Expected column name to kind (see `KINDS`).
        max_age_seconds: Oldest acceptable data, or None for no limit.
        key_columns: Columns that identify a row (for deduplication).
    """

    name: str
    schema_version: str
    columns: Mapping[str, str]
    max_age_seconds: float | None = None
    key_columns: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the contract.

        Raises:
            ConfigError: If it is incomplete or names an unknown kind.
        """
        if not self.name or not self.schema_version or not self.columns:
            raise errors.ConfigError(
                "a contract needs a name, a schema version and columns"
            )
        for column, kind in self.columns.items():
            if kind not in KINDS:
                raise errors.ConfigError(
                    f"column {column!r}: kind must be one of {KINDS}"
                )
        missing = [c for c in self.key_columns if c not in self.columns]
        if missing:
            raise errors.ConfigError(f"key columns not in columns: {missing}")
        if self.max_age_seconds is not None and self.max_age_seconds <= 0:
            raise errors.ConfigError("max_age_seconds must be positive")

    @property
    def digest(self) -> str:
        """Hash of the declared schema."""
        return schema_digest(self.columns)

    def mismatches(self, frame: pd.DataFrame) -> list[str]:
        """Lists how the frame differs from the declared schema.

        Args:
            frame: The data.

        Returns:
            One message per missing column, unexpected column and column of
            the wrong kind, in that order; empty if the schemas are equal.
        """
        seen = observed_schema(frame)
        problems = [
            f"missing column {c}" for c in self.columns if c not in seen
        ]
        problems += [
            f"unexpected column {c}" for c in seen if c not in self.columns
        ]
        problems += [
            f"column {c} is {seen[c]}, expected {kind}"
            for c, kind in self.columns.items()
            if c in seen and seen[c] != kind
        ]
        return problems


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotRecord:
    """Everything recorded about one read.

    Attributes:
        snapshot_id: Hash of URI, content hash and schema digest.
        uri: Where the object was read.
        sha256: Content hash.
        size: Bytes.
        modified: The object's modification time (ISO 8601 UTC).
        version: The backend's version id, if any.
        dataset: Contract name.
        schema_version: Declared schema version.
        schema_digest: Hash of the observed schema.
        schema_matches: Whether the observed schema equals the declared one.
        rows: Rows read.
        read_at: When the read happened (ISO 8601 UTC).
        age_seconds: `read_at` minus `modified`.
        max_age_seconds: The contract's limit, if any.
        freshness: `fresh`, `stale` or `unknown` (no limit declared).
        lineage: Snapshot ids this one was derived from.
    """

    snapshot_id: str
    uri: str
    sha256: str
    size: int
    modified: str
    version: str | None
    dataset: str
    schema_version: str
    schema_digest: str
    schema_matches: bool
    rows: int
    read_at: str
    age_seconds: float
    max_age_seconds: float | None
    freshness: str
    lineage: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Converts the record to plain data.

        Returns:
            A dict of the fields, with `lineage` as a list.
        """
        record = dataclasses.asdict(self)
        record["lineage"] = list(self.lineage)
        return record


class ReadLog:
    """Every snapshot read, in order, in memory and optionally on disk.

    Attributes:
        path: The JSON-lines file appended to, or None for memory only.
        records: The records appended so far, in order.
    """

    def __init__(self, path: str | pathlib.Path | None = None) -> None:
        """Initializes the log.

        Args:
            path: JSON-lines file appended to on every read.
        """
        self.path = None if path is None else pathlib.Path(path)
        self.records: list[SnapshotRecord] = []

    def append(self, record: SnapshotRecord) -> None:
        """Adds a record, and appends it to the file if there is one.

        Args:
            record: The snapshot record to log.
        """
        self.records.append(record)
        if self.path is not None:
            line = json.dumps(record.to_dict(), sort_keys=True) + "\n"
            durability.append_text(self.path, line)

    def trace(self, snapshot_id: str) -> list[SnapshotRecord]:
        """Follows a snapshot back through the snapshots it derives from.

        Args:
            snapshot_id: The snapshot to start from.

        Returns:
            The snapshot and then each ancestor once, breadth first, so the
            nearest come first.

        Raises:
            SnapshotError: If the id or one of its ancestors was never read.
        """
        by_id = {record.snapshot_id: record for record in self.records}
        order: list[SnapshotRecord] = []
        pending = [snapshot_id]
        while pending:
            current = pending.pop(0)
            if current in {record.snapshot_id for record in order}:
                continue
            found = by_id.get(current)
            if found is None:
                raise SnapshotError(f"snapshot {current} was never recorded")
            order.append(found)
            pending.extend(found.lineage)
        return order


def parse_time(text: str) -> datetime.datetime:
    """Parses an ISO 8601 time.

    Text that is not ISO 8601 raises ValueError, from
    `datetime.datetime.fromisoformat`.

    Args:
        text: The time; a value without an offset is taken to be local time.

    Returns:
        The same moment as an aware datetime in UTC.
    """
    return datetime.datetime.fromisoformat(text).astimezone(datetime.UTC)


class SnapshotReader:
    """Reads tabular objects from a store and records each read.

    Attributes:
        store: Where objects are read from.
        log: Where reads are recorded.
        clock: Source of the current time.
    """

    def __init__(
        self,
        store: objectstore.ObjectStore,
        log: ReadLog | None = None,
        clock: Callable[[], datetime.datetime] | None = None,
    ) -> None:
        """Initializes the reader.

        Args:
            store: Where objects are read from.
            log: Where reads are recorded (in memory by default).
            clock: Source of the current time (UTC `datetime.now` by default).
        """
        self.store = store
        self.log = log or ReadLog()
        self.clock = clock or (lambda: datetime.datetime.now(datetime.UTC))

    def read(
        self,
        key: str,
        contract: DataContract,
        parents: Sequence[SnapshotRecord | str] = (),
    ) -> tuple[pd.DataFrame, SnapshotRecord]:
        """Reads one object and records the snapshot.

        A schema mismatch or a stale object does not stop the read; it is
        recorded, and the quality gate decides what happens (fail,
        quarantine or warn). Errors from the store, such as `NotFoundError`
        for a missing object, propagate unchanged.

        Args:
            key: Object key (CSV, JSON lines or Parquet by extension).
            contract: What the consumer expects.
            parents: Snapshots (or their ids) this read derives from.

        Returns:
            The frame and its record.

        Raises:
            SnapshotError: If the object cannot be decoded.
        """
        payload, meta = self.store.get(key)
        uri = self.store.uri(key)
        try:
            frame = connectors.read_tabular_bytes(uri, payload)
        except (ValueError, errors.ScmleError) as error:
            raise SnapshotError(f"cannot decode {uri}: {error}") from error
        observed = observed_schema(frame)
        digest = schema_digest(observed)
        now = self.clock().astimezone(datetime.UTC)
        age = max(0.0, (now - parse_time(meta.modified)).total_seconds())
        limit = contract.max_age_seconds
        if limit is None:
            freshness = UNKNOWN
        else:
            freshness = FRESH if age <= limit else STALE
        identity = hashlib.sha256(
            f"{uri}|{meta.sha256}|{digest}".encode()
        ).hexdigest()
        record = SnapshotRecord(
            snapshot_id="snap-" + identity[:32],
            uri=uri,
            sha256=meta.sha256,
            size=meta.size,
            modified=meta.modified,
            version=meta.version,
            dataset=contract.name,
            schema_version=contract.schema_version,
            schema_digest=digest,
            schema_matches=not contract.mismatches(frame),
            rows=len(frame),
            read_at=objectstore.iso(now),
            age_seconds=round(age, 3),
            max_age_seconds=limit,
            freshness=freshness,
            lineage=tuple(
                p if isinstance(p, str) else p.snapshot_id for p in parents
            ),
        )
        self.log.append(record)
        return frame, record


def connector_reader(store: objectstore.ObjectStore) -> Callable[[str], bytes]:
    """Adapts a store to the `read_bytes` callable of `ObjectStoreConnector`.

    The URI's path (after the store's own prefix) is the object key, so a
    connector constructed with `store.uri(key)` reads through the store.

    Args:
        store: The store to read from.

    Returns:
        A function from a source URI to the bytes of the object whose key is
        the URI's path without the store's own prefix. It raises whatever
        `store.get` raises, such as `NotFoundError`.
    """
    root = urlsplit(store.location).path.strip("/")

    def read_bytes(source_uri: str) -> bytes:
        """Reads the object that a URI under the store names.

        Args:
            source_uri: A URI whose path ends in an object key.

        Returns:
            The object's content.
        """
        path = urlsplit(source_uri).path.strip("/")
        key = path.removeprefix(root + "/") if root else path
        return store.get(key)[0]

    return read_bytes
