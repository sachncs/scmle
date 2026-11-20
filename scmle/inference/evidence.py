"""Content-linked inference evidence for completed model runs."""

from collections.abc import Mapping
import dataclasses
import datetime
import hashlib
import json
import os
import pathlib
import tempfile
from typing import Any
from typing import ClassVar

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import lineage
from scmle.contracts import records


@dataclasses.dataclass(frozen=True, slots=True)
class Link:
    """Identity of one inference event linked to an exact model bundle.

    Attributes:
        event: Unique identifier of the inference event.
        input_fingerprint: Fingerprint of the request input.
        output_fingerprint: Fingerprint of the produced output.
        bundle_digest: Digest of the model bundle that served the event.
        timestamp: ISO 8601 time of the event, including a timezone.
        graph_digest: Digest of the evidence graph after the event was added.
        KEYS: Version-1 keys to the written ones (`event`, `digests.payload`,
            `digests.prediction`, `digests.bundle`, `digests.graph`);
            `downgrade` restores version 1. The node digest is anchored at
            version 1: it hashes the version-1 keys, so renaming a written
            key never breaks a stored chain.
    """

    KEYS: ClassVar[records.Migration] = records.nest(
        {"event_id": "event"},
        groups={
            "digests": {
                "input_fingerprint": "payload",
                "output_fingerprint": "prediction",
                "bundle_digest": "bundle",
                "graph_digest": "graph",
            }
        },
    )

    event: str
    input_fingerprint: str
    output_fingerprint: str
    bundle_digest: str
    timestamp: str
    graph_digest: str

    def __post_init__(self) -> None:
        """Validates the identity fields and the timestamp.

        Raises:
            ValueError: If a field is not a non-blank string, the timestamp
                is not ISO 8601, or the timestamp has no timezone.
        """
        for name in (
            "event",
            "input_fingerprint",
            "output_fingerprint",
            "bundle_digest",
            "timestamp",
            "graph_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{name} must not be empty")
        parsed = datetime.datetime.fromisoformat(self.timestamp)
        if parsed.tzinfo is None:
            raise errors.InputError("timestamp must include a timezone")

    def to_dict(self) -> dict[str, str]:
        """Converts the link to a JSON-native record.

        Returns:
            The event and timestamp, and the four digests under `digests`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))

    def node(self) -> lineage.Node:
        """Builds the inference node of the event.

        Returns:
            A node `inference:<event_id>` whose attributes are the identity
            fields (without the graph digest) under their version-1 keys and
            whose digest is the SHA-256 of their canonical JSON; the keys stay
            the version-1 ones so the digest of a stored event never changes.
        """
        payload = {
            "event_id": self.event,
            "input_fingerprint": self.input_fingerprint,
            "output_fingerprint": self.output_fingerprint,
            "bundle_digest": self.bundle_digest,
            "timestamp": self.timestamp,
        }
        digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        return lineage.Node(
            identifier=f"inference:{self.event}",
            kind=lineage.Kind.INFERENCE,
            digest=digest,
            attributes=dict(payload),
        )


class Linker:
    """Appends exact-bundle inference nodes to a durable evidence graph."""

    def link(
        self,
        graph_path: str | pathlib.Path,
        *,
        event_id: str,
        input_fingerprint: str,
        output_fingerprint: str,
        bundle_digest: str,
        timestamp: str,
    ) -> Link:
        """Links an inference event and atomically persists the graph.

        The caller supplies fingerprints from the serving boundary. The
        linker never infers identity from prediction values or mutable logs.
        Repeating the exact event leaves the graph content unchanged (the
        file is still rewritten); reusing an event ID with any changed field
        is rejected.

        Args:
            graph_path: Path of the evidence graph JSON file.
            event_id: Unique identifier of the inference event.
            input_fingerprint: Fingerprint of the request input.
            output_fingerprint: Fingerprint of the produced output.
            bundle_digest: Digest of the model bundle; it must equal the
                digest of the graph's `bundle` node.
            timestamp: ISO 8601 time of the event, including a timezone.

        Returns:
            The link, carrying the digest of the graph after the event was
            added.

        Raises:
            ValueError: If an identity field is blank, the timestamp is
                invalid or lacks a timezone, the graph has no bundle node,
                `bundle_digest` does not match the graph's bundle, the event
                ID is already used with a different identity, or the graph
                file fails the checks of `load`.
        """
        candidate = Link(
            event=event_id,
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            bundle_digest=bundle_digest,
            timestamp=timestamp,
            graph_digest="pending",
        )
        path = pathlib.Path(graph_path)
        graph = self.load(path)
        bundle = graph.nodes.get("bundle")
        if bundle is None or bundle.kind is not lineage.Kind.BUNDLE:
            raise errors.InputError("evidence graph has no bundle node")
        if bundle.digest != bundle_digest:
            raise errors.InputError(
                "inference bundle digest does not match graph"
            )
        node = candidate.node()
        graph.add_node(node)
        graph.connect(lineage.Edge("bundle", node.identifier, "serves"))
        graph_record = graph.to_dict()
        graph_record["graph_digest"] = graph.digest()
        self.write(path, graph_record)
        return dataclasses.replace(
            candidate, graph_digest=str(graph_record["graph_digest"])
        )

    @staticmethod
    def load(path: str | pathlib.Path) -> lineage.Graph:
        """Loads and validates a persisted evidence graph.

        Args:
            path: Path of the evidence graph JSON file.

        Returns:
            The graph rebuilt from the file.

        Raises:
            TypeError: If the file does not hold a JSON object.
            ValueError: If the stored `graph_digest` does not match the
                content. A file without `graph_digest` is accepted.
        """
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise errors.KindError("evidence graph must be a JSON object")
        graph = lineage.Graph.from_dict(record)
        expected_digest = record.get("graph_digest")
        if expected_digest is not None and expected_digest != graph.digest():
            raise errors.InputError(
                "evidence graph digest does not match content"
            )
        return graph

    @staticmethod
    def write(path: pathlib.Path, record: dict[str, Any]) -> None:
        """Persists graph JSON with flushed contents and directory metadata.

        The record is written to a temporary file in the same directory,
        synced, and moved over `path`; the temporary file is removed on any
        failure.

        Args:
            path: Destination file; missing parent directories are created.
            record: JSON-serializable graph record.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline="\n"
            ) as handle:
                json.dump(record, handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            pathlib.Path(temporary_name).replace(path)
            durability.sync_directory(path.parent)
        except BaseException:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
            raise
