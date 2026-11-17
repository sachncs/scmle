"""Source snapshot registry and schema-evolution decisions."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import json
import os
import pathlib
import tempfile
from typing import Any
from typing import ClassVar
from typing import Self

from scmle import durability
from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.readers import connectors

DIGESTS = {
    "schema_hash": "schema",
    "data_hash": "data",
    "raw_source_hash": "raw",
}


def nest(layout: records.Layout) -> records.Layout:
    """Moves a renamed version-1 snapshot's hashes under ``digests``.

    Args:
        layout: The snapshot after its other keys were renamed.

    Returns:
        The snapshot with ``schema_hash``, ``data_hash`` and
        ``raw_source_hash`` nested as ``digests.schema``, ``digests.data``
        and ``digests.raw``.
    """
    rest = {key: value for key, value in layout.items() if key not in DIGESTS}
    rest["digests"] = {
        new: layout[old] for old, new in DIGESTS.items() if old in layout
    }
    return rest


def flatten(layout: records.Layout) -> records.Layout:
    """Undoes `nest`.

    Args:
        layout: A version-2 snapshot.

    Returns:
        The snapshot with its digests as top-level version-1 keys.
    """
    rest = {key: value for key, value in layout.items() if key != "digests"}
    digests = layout.get("digests", {})
    rest.update(
        {old: digests[new] for old, new in DIGESTS.items() if new in digests}
    )
    return rest


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """Immutable identity and source schema for one observed snapshot.

    A snapshot carries no version of its own: the `Sources` registry that
    holds it is versioned and applies `migrations` to it.

    Attributes:
        source: Identity of the logical source.
        uri: Canonical URI of the source.
        snapshot: Identity of this snapshot of the source.
        digests: Hashes keyed by the hashed thing: ``schema`` (the
            snapshot's column names and dtypes), ``data`` (its row
            contents) and ``raw`` (the raw source bytes, or an empty string
            when they were not available).
        rows: Number of rows in the snapshot.
        layout: Mapping from column name to source type name.
        captured: Capture time as supplied by the caller.
        watermark: Source watermark, or an empty string if none.
        migrations: The snapshot's layout steps, applied by `Sources`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "source_id": "source",
                "source_uri": "uri",
                "snapshot_id": "snapshot",
                "row_count": "rows",
                "schema": "layout",
                "captured_at": "captured",
            },
            change=nest,
            revert=flatten,
        ),
    )

    source: str
    uri: str
    snapshot: str
    digests: Mapping[str, str]
    rows: int
    layout: Mapping[str, str]
    captured: str
    watermark: str = ""

    def __post_init__(self) -> None:
        """Validates identity, shape, and schema fields.

        Raises:
            ValueError: If a required identity or time field is blank, the
                digests are not exactly ``schema``, ``data`` and ``raw`` or
                the schema or data digest is blank, the row count is
                negative, the layout is empty, or a column name or type name
                is blank.
        """
        for name in ("source", "uri", "snapshot", "captured"):
            if not getattr(self, name).strip():
                raise errors.InputError(f"{name} must not be empty")
        if set(self.digests) != {"schema", "data", "raw"}:
            raise errors.InputError("digests must hold schema, data and raw")
        for name in ("schema", "data"):
            if not self.digests[name].strip():
                raise errors.InputError(f"{name} digest must not be empty")
        object.__setattr__(self, "digests", dict(self.digests))
        if self.rows < 0:
            raise errors.InputError("rows must not be negative")
        if not self.layout:
            raise errors.InputError("layout must not be empty")
        if any(
            not name.strip() or not type_name.strip()
            for name, type_name in self.layout.items()
        ):
            raise errors.InputError("layout names and types must not be empty")

    @classmethod
    def restore(
        cls,
        manifest: data.DatasetManifest,
        layout: Mapping[str, str],
        captured: str,
    ) -> Self:
        """Builds a source snapshot from the immutable dataset manifest.

        Args:
            manifest: Manifest of the ingested dataset. An empty URI is
                treated as ``memory://``.
            layout: Mapping from column name to source type name.
            captured: Capture time to record.

        Returns:
            A snapshot whose identity, hashes and row count come from the
            manifest; the raw source hash is taken from the manifest's
            ``raw-source`` lineage hash, or is empty when absent.
        """
        uri = connectors.canonicalize(manifest.uri or "memory://")
        return cls(
            source=connectors.source_identity(uri),
            uri=uri,
            snapshot=manifest.snapshot,
            digests={
                "schema": manifest.digests["schema"],
                "data": manifest.digests["data"],
                "raw": manifest.lineage.digests.get("raw-source", ""),
            },
            rows=manifest.rows,
            layout=dict(layout),
            captured=captured,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the snapshot.

        Returns:
            A JSON-compatible mapping of every field, with the layout
            sorted by column name.
        """
        return {
            "source": self.source,
            "uri": self.uri,
            "snapshot": self.snapshot,
            "digests": dict(self.digests),
            "rows": self.rows,
            "layout": dict(sorted(self.layout.items())),
            "captured": self.captured,
            "watermark": self.watermark,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a source snapshot from serialized evidence.

        Args:
            record: Mapping as produced by `to_dict`. The keys other than
                ``digests.raw`` and ``watermark`` are required; those two
                default to an empty string.

        Returns:
            The reconstructed snapshot.
        """
        digests = dict(record["digests"])
        return cls(
            source=str(record["source"]),
            uri=str(record["uri"]),
            snapshot=str(record["snapshot"]),
            digests={
                "schema": str(digests["schema"]),
                "data": str(digests["data"]),
                "raw": str(digests.get("raw", "")),
            },
            rows=int(record["rows"]),
            layout={
                str(name): str(type_name)
                for name, type_name in dict(record["layout"]).items()
            },
            captured=str(record["captured"]),
            watermark=str(record.get("watermark", "")),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Compatibility:
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


def diff(previous: Observation | None, current: Observation) -> Compatibility:
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
        return Compatibility(None, current.snapshot, (), (), ())
    previous_names = set(previous.layout)
    current_names = set(current.layout)
    added = tuple(sorted(current_names - previous_names))
    removed = tuple(sorted(previous_names - current_names))
    changed = tuple(
        sorted(
            name
            for name in previous_names & current_names
            if previous.layout[name] != current.layout[name]
        )
    )
    return Compatibility(
        previous.snapshot,
        current.snapshot,
        added,
        removed,
        changed,
    )


def lift(layout: records.Layout) -> records.Layout:
    """Moves every snapshot of a version-1 registry to version 2.

    Args:
        layout: A version-1 registry without its version key.

    Returns:
        The registry with each snapshot's keys renamed and its hashes
        nested.
    """
    return shift(layout, Observation.migrations[0].upgrade)


def lower(layout: records.Layout) -> records.Layout:
    """Undoes `lift`.

    Args:
        layout: A version-2 registry without its version key.

    Returns:
        The registry with every snapshot in version-1 form.
    """
    return shift(layout, Observation.migrations[0].downgrade)


def shift(
    layout: records.Layout,
    step: Callable[[records.Layout], records.Layout],
) -> records.Layout:
    """Applies a step to every snapshot of a stored registry.

    Args:
        layout: The stored registry.
        step: Moves one stored snapshot.

    Returns:
        The registry with every snapshot moved; entries that are not the
        expected lists or mappings are left for loading to reject.
    """
    sources = layout.get("sources")
    if not isinstance(sources, list):
        return layout
    return {
        **layout,
        "sources": [
            {
                **source,
                "snapshots": [
                    step(dict(item)) if isinstance(item, Mapping) else item
                    for item in source["snapshots"]
                ],
            }
            if isinstance(source, Mapping)
            and isinstance(source.get("snapshots"), list)
            else source
            for source in sources
        ],
    }


@dataclasses.dataclass
class Sources:
    """Append-only source snapshot history for a run or source project.

    Attributes:
        snapshots: Mapping from source id to that source's snapshots, in
            registration order.
        migrations: The registry's version steps, oldest first; version 2
            renamed the keys of every snapshot.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(change=lift, revert=lower),
    )

    snapshots: dict[str, tuple[Observation, ...]] = dataclasses.field(
        default_factory=dict
    )

    def register(self, snapshot: Observation) -> None:
        """Registers a snapshot idempotently and rejects conflicting repeats.

        Registering a snapshot whose id is already present with identical
        evidence is a no-op.

        Args:
            snapshot: The snapshot to append to its source's history.

        Raises:
            ValueError: If the snapshot id is already registered for the
                source with different evidence.
        """
        history = self.snapshots.get(snapshot.source, ())
        for existing in history:
            if existing.snapshot == snapshot.snapshot:
                same_content = (
                    existing.source == snapshot.source
                    and existing.uri == snapshot.uri
                    and dict(existing.digests) == dict(snapshot.digests)
                    and existing.rows == snapshot.rows
                    and dict(existing.layout) == dict(snapshot.layout)
                    and existing.watermark == snapshot.watermark
                )
                if not same_content:
                    raise errors.InputError(
                        "snapshot id is already registered with different "
                        "evidence"
                    )
                return
        self.snapshots[snapshot.source] = (*history, snapshot)

    def latest(self, source_id: str) -> Observation | None:
        """Finds the most recently registered snapshot for a source.

        Args:
            source_id: Identity of the source.

        Returns:
            The latest snapshot, or None if the source has none.
        """
        history = self.snapshots.get(source_id, ())
        return history[-1] if history else None

    def compatibility(self, snapshot: Observation) -> Compatibility:
        """Compares a candidate snapshot with the source's latest snapshot.

        Args:
            snapshot: The candidate snapshot (not registered by this call).

        Returns:
            The schema decision against the latest registered snapshot of
            the same source; ``initial`` if there is none.
        """
        return diff(self.latest(snapshot.source), snapshot)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the registry.

        Returns:
            A JSON-compatible mapping with the current ``schema_version``
            and the sources sorted by id, each with its snapshots in
            registration order.
        """
        return {
            "schema_version": records.ORIGIN + len(self.migrations),
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
        are rejected as by `register`. A registry stored at an older schema
        version is migrated first.

        Args:
            record: Mapping as produced by `to_dict`.

        Returns:
            The reconstructed registry.

        Raises:
            ValueError: If the schema version is missing or newer than this
                build, ``sources`` or a source's ``snapshots`` is not a list,
                or a snapshot's source id differs from the source it is
                listed under.
            TypeError: If an entry of ``sources`` is not a mapping.
        """
        if not isinstance(record.get("schema_version"), int):
            raise errors.InputError(
                "unsupported source registry schema version"
            )
        layout = records.load("Sources", cls.migrations, record)
        sources = layout.get("sources")
        if not isinstance(sources, list):
            raise errors.InputError("sources must be a list")
        registry = cls()
        for source in sources:
            if not isinstance(source, Mapping):
                raise errors.KindError("each source must be a mapping")
            source_id = str(source["source_id"])
            snapshots = source.get("snapshots")
            if not isinstance(snapshots, list):
                raise errors.InputError("source snapshots must be a list")
            for item in snapshots:
                snapshot = Observation.from_dict(item)
                if snapshot.source != source_id:
                    raise errors.InputError(
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
            raise errors.KindError("serialized registry must contain an object")
        return cls.from_dict(record)


@dataclasses.dataclass(frozen=True, slots=True)
class Registry:
    """Durable, atomic file store for source history across runs.

    Attributes:
        path: Location of the JSON file holding the registry.
    """

    path: pathlib.Path

    def load(self) -> Sources:
        """Loads the registry or returns an empty registry for a new store.

        Returns:
            The stored registry, or an empty one if the file does not exist.
        """
        if not self.path.exists():
            return Sources()
        return Sources.from_json(self.path.read_text(encoding="utf-8"))

    def save(self, registry: Sources) -> None:
        """Atomically publishes a complete registry snapshot.

        A failed serialization or write leaves the previous registry intact.
        The caller owns coordination between multiple writers; this store
        guarantees that readers never observe a partial JSON document.

        Args:
            registry: The complete registry to publish.

        Raises:
            TypeError: If `registry` is not a `Sources`.
        """
        if not isinstance(registry, Sources):
            raise errors.KindError("registry must be Sources")
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

    def register(self, snapshot: Observation) -> Compatibility:
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
