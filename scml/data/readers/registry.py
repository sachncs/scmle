"""Source snapshot registry and schema-evolution decisions."""

from collections.abc import Mapping
import dataclasses
import json
from typing import Any
from typing import Self

from scml.contracts import data as data_contracts
from scml.data.readers import connectors


@dataclasses.dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Immutable identity and source schema for one observed snapshot."""

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
        """Validate identity, shape, and schema fields."""
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
        manifest: data_contracts.DatasetManifest,
        schema: Mapping[str, str],
        captured_at: str,
    ) -> Self:
        """Build a source snapshot from the immutable dataset manifest."""
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
        """Return a deterministic JSON-compatible snapshot record."""
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
        """Rebuild a source snapshot from serialized evidence."""
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
    """Decision describing how one source schema changed from another."""

    previous_snapshot_id: str | None
    current_snapshot_id: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    changed: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        """Whether the change is unchanged or additive-only."""
        return not self.removed and not self.changed

    @property
    def status(self) -> str:
        """Return the stable schema-evolution status."""
        if self.previous_snapshot_id is None:
            return "initial"
        if self.removed or self.changed:
            return "breaking"
        return "additive" if self.added else "unchanged"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible compatibility decision."""
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
    """Compare two snapshots without silently accepting breaking changes."""
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
    """Append-only source snapshot history for a run or source project."""

    snapshots: dict[str, tuple[SourceSnapshot, ...]] = dataclasses.field(
        default_factory=dict
    )

    def register(self, snapshot: SourceSnapshot) -> None:
        """Register a snapshot idempotently and reject conflicting repeats."""
        history = self.snapshots.get(snapshot.source_id, ())
        for existing in history:
            if existing.snapshot_id == snapshot.snapshot_id:
                if existing != snapshot:
                    raise ValueError(
                        "snapshot id is already registered with different "
                        "evidence"
                    )
                return
        self.snapshots[snapshot.source_id] = (*history, snapshot)

    def latest(self, source_id: str) -> SourceSnapshot | None:
        """Return the most recently registered snapshot for a source."""
        history = self.snapshots.get(source_id, ())
        return history[-1] if history else None

    def compatibility(self, snapshot: SourceSnapshot) -> SchemaCompatibility:
        """Compare a candidate snapshot with the source's latest snapshot."""
        return compare_schemas(self.latest(snapshot.source_id), snapshot)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic source registry evidence."""
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
        """Serialize the registry with canonical JSON ordering."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a source registry from serialized evidence."""
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
        """Rebuild a source registry from canonical JSON."""
        record = json.loads(serialized)
        if not isinstance(record, Mapping):
            raise TypeError("serialized registry must contain an object")
        return cls.from_dict(record)
