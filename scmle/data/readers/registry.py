"""Source snapshot registry and schema-evolution decisions."""

from collections.abc import Mapping
import dataclasses
import json
import os
import pathlib
import tempfile
from typing import Any
from typing import Self

from scmle import durability
from scmle.contracts import data
from scmle.data.readers import connectors


@dataclasses.dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Immutable identity and source schema for one observed snapshot.

    Attributes:
        source_id: Identity of the logical source.
        source_uri: Canonical URI of the source.
        snapshot_id: Identity of this snapshot of the source.
        schema_hash: Hash of the snapshot's column names and dtypes.
        data_hash: Hash of the snapshot's row contents.
        raw_source_hash: Hash of the raw source bytes, or an empty string
            when they were not available.
        row_count: Number of rows in the snapshot.
        schema: Mapping from column name to source type name.
        captured_at: Capture time as supplied by the caller.
        watermark: Source watermark, or an empty string if none.
    """

    source_id: str
    source_uri: str
    snapshot_id: str
    schema_hash: str
    data_hash: str
    raw_source_hash: str
    row_count: int
    schema: Mapping[str, str]
    captured_at: str
    watermark: str = ""

    def __post_init__(self) -> None:
        """Validates identity, shape, and schema fields.

        Raises:
            ValueError: If a required identity or time field is blank, the
                row count is negative, the schema is empty, or a schema
                column name or type name is blank.
        """
        for name in (
            "source_id",
            "source_uri",
            "snapshot_id",
            "schema_hash",
            "data_hash",
            "captured_at",
        ):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be empty")
        if self.row_count < 0:
            raise ValueError("row_count must not be negative")
        if not self.schema:
            raise ValueError("schema must not be empty")
        if any(
            not name.strip() or not type_name.strip()
            for name, type_name in self.schema.items()
        ):
            raise ValueError("schema names and types must not be empty")

    @classmethod
    def from_manifest(
        cls,
        manifest: data.DatasetManifest,
        schema: Mapping[str, str],
        captured_at: str,
    ) -> Self:
        """Builds a source snapshot from the immutable dataset manifest.

        Args:
            manifest: Manifest of the ingested dataset. An empty URI is
                treated as ``memory://``.
            schema: Mapping from column name to source type name.
            captured_at: Capture time to record.

        Returns:
            A snapshot whose identity, hashes and row count come from the
            manifest; the raw source hash is taken from the manifest's
            ``raw-source`` lineage hash, or is empty when absent.
        """
        source_uri = connectors.canonical_source_uri(
            manifest.uri or "memory://"
        )
        return cls(
            source_id=connectors.source_identity(source_uri),
            source_uri=source_uri,
            snapshot_id=manifest.snapshot_id,
            schema_hash=manifest.schema_hash,
            data_hash=manifest.data_hash,
            raw_source_hash=manifest.lineage.object_hashes.get(
                "raw-source", ""
            ),
            row_count=manifest.row_count,
            schema=dict(schema),
            captured_at=captured_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the snapshot.

        Returns:
            A JSON-compatible mapping of every field, with the schema
            sorted by column name.
        """
        return {
            "source_id": self.source_id,
            "source_uri": self.source_uri,
            "snapshot_id": self.snapshot_id,
            "schema_hash": self.schema_hash,
            "data_hash": self.data_hash,
            "raw_source_hash": self.raw_source_hash,
            "row_count": self.row_count,
            "schema": dict(sorted(self.schema.items())),
            "captured_at": self.captured_at,
            "watermark": self.watermark,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a source snapshot from serialized evidence.

        Args:
            record: Mapping as produced by `to_dict`. The keys other than
                ``raw_source_hash`` and ``watermark`` are required; those
                two default to an empty string.

        Returns:
            The reconstructed snapshot.
        """
        return cls(
            source_id=str(record["source_id"]),
            source_uri=str(record["source_uri"]),
            snapshot_id=str(record["snapshot_id"]),
            schema_hash=str(record["schema_hash"]),
            data_hash=str(record["data_hash"]),
            raw_source_hash=str(record.get("raw_source_hash", "")),
            row_count=int(record["row_count"]),
            schema={
                str(name): str(type_name)
                for name, type_name in dict(record["schema"]).items()
            },
            captured_at=str(record["captured_at"]),
            watermark=str(record.get("watermark", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class SchemaCompatibility:
    """Decision describing how one source schema changed from another.

    Attributes:
        previous_snapshot_id: Snapshot compared against, or None for the
            first snapshot of a source.
        current_snapshot_id: Snapshot being judged.
        added: Sorted names of columns present only in the current schema.
        removed: Sorted names of columns present only in the previous
            schema.
        changed: Sorted names of columns in both schemas whose type name
            differs.
    """

    previous_snapshot_id: str | None
    current_snapshot_id: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        """Whether no column was removed or changed (additions allowed)."""
        return not self.removed and not self.changed

    @property
    def status(self) -> str:
        """The stable schema-evolution status.

        One of ``initial`` (no previous snapshot), ``breaking`` (a column
        was removed or changed), ``additive`` (columns only added) or
        ``unchanged``.
        """
        if self.previous_snapshot_id is None:
            return "initial"
        if self.removed or self.changed:
            return "breaking"
        return "additive" if self.added else "unchanged"

    def to_dict(self) -> dict[str, Any]:
        """Serializes the compatibility decision.

        Returns:
            A JSON-compatible mapping with both snapshot ids, ``status``,
            ``compatible`` and the added, removed and changed column names
            as lists.
        """
        return {
            "previous_snapshot_id": self.previous_snapshot_id,
            "current_snapshot_id": self.current_snapshot_id,
            "status": self.status,
            "compatible": self.compatible,
            "added": list(self.added),
            "removed": list(self.removed),
            "changed": list(self.changed),
        }


def compare_schemas(
    previous: SourceSnapshot | None, current: SourceSnapshot
) -> SchemaCompatibility:
    """Compares two snapshots without silently accepting breaking changes.

    Args:
        previous: The earlier snapshot of the source, or None if there is
            none.
        current: The snapshot being judged.

    Returns:
        The decision. With no previous snapshot all column lists are empty
        and the status is ``initial``; otherwise the lists hold the sorted
        names of added, removed and type-changed columns.
    """
    if previous is None:
        return SchemaCompatibility(None, current.snapshot_id, (), (), ())
    previous_names = set(previous.schema)
    current_names = set(current.schema)
    added = tuple(sorted(current_names - previous_names))
    removed = tuple(sorted(previous_names - current_names))
    changed = tuple(
        sorted(
            name
            for name in previous_names & current_names
            if previous.schema[name] != current.schema[name]
        )
    )
    return SchemaCompatibility(
        previous.snapshot_id,
        current.snapshot_id,
        added,
        removed,
        changed,
    )


@dataclasses.dataclass
class SourceRegistry:
    """Append-only source snapshot history for a run or source project.

    Attributes:
        snapshots: Mapping from source id to that source's snapshots, in
            registration order.
    """

    snapshots: dict[str, tuple[SourceSnapshot, ...]] = dataclasses.field(
        default_factory=dict
    )

    def register(self, snapshot: SourceSnapshot) -> None:
        """Registers a snapshot idempotently and rejects conflicting repeats.

        Registering a snapshot whose id is already present with identical
        evidence is a no-op.

        Args:
            snapshot: The snapshot to append to its source's history.

        Raises:
            ValueError: If the snapshot id is already registered for the
                source with different evidence.
        """
        history = self.snapshots.get(snapshot.source_id, ())
        for existing in history:
            if existing.snapshot_id == snapshot.snapshot_id:
                same_content = (
                    existing.source_id == snapshot.source_id
                    and existing.source_uri == snapshot.source_uri
                    and existing.schema_hash == snapshot.schema_hash
                    and existing.data_hash == snapshot.data_hash
                    and existing.raw_source_hash == snapshot.raw_source_hash
                    and existing.row_count == snapshot.row_count
                    and dict(existing.schema) == dict(snapshot.schema)
                    and existing.watermark == snapshot.watermark
                )
                if not same_content:
                    raise ValueError(
                        "snapshot id is already registered with different "
                        "evidence"
                    )
                return
        self.snapshots[snapshot.source_id] = (*history, snapshot)

    def latest(self, source_id: str) -> SourceSnapshot | None:
        """Finds the most recently registered snapshot for a source.

        Args:
            source_id: Identity of the source.

        Returns:
            The latest snapshot, or None if the source has none.
        """
        history = self.snapshots.get(source_id, ())
        return history[-1] if history else None

    def compatibility(self, snapshot: SourceSnapshot) -> SchemaCompatibility:
        """Compares a candidate snapshot with the source's latest snapshot.

        Args:
            snapshot: The candidate snapshot (not registered by this call).

        Returns:
            The schema decision against the latest registered snapshot of
            the same source; ``initial`` if there is none.
        """
        return compare_schemas(self.latest(snapshot.source_id), snapshot)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the registry.

        Returns:
            A JSON-compatible mapping with ``schema_version`` 1 and the
            sources sorted by id, each with its snapshots in registration
            order.
        """
        return {
            "schema_version": 1,
            "sources": [
                {
                    "source_id": source_id,
                    "snapshots": [snapshot.to_dict() for snapshot in snapshots],
                }
                for source_id, snapshots in sorted(self.snapshots.items())
            ],
        }

    def to_json(self) -> str:
        """Serializes the registry with canonical JSON ordering.

        Returns:
            Compact, ASCII-only JSON of `to_dict` with sorted keys.
        """
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a source registry from serialized evidence.

        Snapshots are re-registered, so conflicting duplicate snapshot ids
        are rejected as by `register`.

        Args:
            record: Mapping as produced by `to_dict`.

        Returns:
            The reconstructed registry.

        Raises:
            ValueError: If the schema version is not 1, ``sources`` or a
                source's ``snapshots`` is not a list, or a snapshot's source
                id differs from the source it is listed under.
            TypeError: If an entry of ``sources`` is not a mapping.
        """
        if record.get("schema_version") != 1:
            raise ValueError("unsupported source registry schema version")
        sources = record.get("sources")
        if not isinstance(sources, list):
            raise ValueError("sources must be a list")
        registry = cls()
        for source in sources:
            if not isinstance(source, Mapping):
                raise TypeError("each source must be a mapping")
            source_id = str(source["source_id"])
            snapshots = source.get("snapshots")
            if not isinstance(snapshots, list):
                raise ValueError("source snapshots must be a list")
            for item in snapshots:
                snapshot = SourceSnapshot.from_dict(item)
                if snapshot.source_id != source_id:
                    raise ValueError(
                        "snapshot source id does not match registry"
                    )
                registry.register(snapshot)
        return registry

    @classmethod
    def from_json(cls, serialized: str) -> Self:
        """Rebuilds a source registry from canonical JSON.

        Args:
            serialized: JSON text as produced by `to_json`.

        Returns:
            The reconstructed registry.

        Raises:
            TypeError: If the JSON document is not an object.
        """
        record = json.loads(serialized)
        if not isinstance(record, Mapping):
            raise TypeError("serialized registry must contain an object")
        return cls.from_dict(record)


@dataclasses.dataclass(frozen=True, slots=True)
class SourceRegistryStore:
    """Durable, atomic file store for source history across runs.

    Attributes:
        path: Location of the JSON file holding the registry.
    """

    path: pathlib.Path

    def load(self) -> SourceRegistry:
        """Loads the registry or returns an empty registry for a new store.

        Returns:
            The stored registry, or an empty one if the file does not exist.
        """
        if not self.path.exists():
            return SourceRegistry()
        return SourceRegistry.from_json(self.path.read_text(encoding="utf-8"))

    def save(self, registry: SourceRegistry) -> None:
        """Atomically publishes a complete registry snapshot.

        A failed serialization or write leaves the previous registry intact.
        The caller owns coordination between multiple writers; this store
        guarantees that readers never observe a partial JSON document.

        Args:
            registry: The complete registry to publish.

        Raises:
            TypeError: If `registry` is not a `SourceRegistry`.
        """
        if not isinstance(registry, SourceRegistry):
            raise TypeError("registry must be SourceRegistry")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", dir=self.path.parent
        )
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                handle.write(registry.to_json())
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            pathlib.Path(temporary_name).replace(self.path)
            durability.sync_directory(self.path.parent)
        except BaseException:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
            raise

    def register(self, snapshot: SourceSnapshot) -> SchemaCompatibility:
        """Compares, registers, and durably publishes one source snapshot.

        Args:
            snapshot: The snapshot to record.

        Returns:
            The schema decision against the latest stored snapshot of the
            same source, computed before the snapshot is added.
        """
        registry = self.load()
        compatibility = registry.compatibility(snapshot)
        registry.register(snapshot)
        self.save(registry)
        return compatibility
