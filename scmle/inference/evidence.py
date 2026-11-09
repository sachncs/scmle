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

from scmle import durability
from scmle.contracts import lineage


@dataclasses.dataclass(frozen=True, slots=True)
class InferenceEvidenceLink:
    """Identity of one inference event linked to an exact model bundle.

    Attributes:
        event_id: Unique identifier of the inference event.
        input_fingerprint: Fingerprint of the request input.
        output_fingerprint: Fingerprint of the produced output.
        bundle_digest: Digest of the model bundle that served the event.
        timestamp: ISO 8601 time of the event, including a timezone.
        graph_digest: Digest of the evidence graph after the event was added.
    """

    event_id: str
    input_fingerprint: str
    output_fingerprint: str
    bundle_digest: str
    timestamp: str
    graph_digest: str

    def __post_init__(self) -> None:
        """Validate the identity fields and the timestamp.

        Raises:
            ValueError: If a field is not a non-blank string, the timestamp
                is not ISO 8601, or the timestamp has no timezone.
        """
        for name in (
            "event_id",
            "input_fingerprint",
            "output_fingerprint",
            "bundle_digest",
            "timestamp",
            "graph_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        parsed = datetime.datetime.fromisoformat(
            self.timestamp.replace("Z", "+00:00")
        )
        if parsed.tzinfo is None:
            raise ValueError("timestamp must include a timezone")

    def to_dict(self) -> dict[str, str]:
        """Convert the link to a JSON-native record.

        Returns:
            A dictionary with one string entry per field.
        """
        return dataclasses.asdict(self)


class InferenceEvidenceLinker:
    """Append exact-bundle inference nodes to a durable evidence graph."""

    def link(
        self,
        graph_path: str | pathlib.Path,
        *,
        event_id: str,
        input_fingerprint: str,
        output_fingerprint: str,
        bundle_digest: str,
        timestamp: str,
    ) -> InferenceEvidenceLink:
        """Link an inference event and atomically persist the graph.

        The caller supplies fingerprints from the serving boundary. The
        linker never infers identity from prediction values or mutable logs.
        Repeating the exact event is idempotent; reusing an event ID with any
        changed identity is rejected.

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
                or `bundle_digest` does not match the graph's bundle.
        """
        candidate = InferenceEvidenceLink(
            event_id=event_id,
            input_fingerprint=input_fingerprint,
            output_fingerprint=output_fingerprint,
            bundle_digest=bundle_digest,
            timestamp=timestamp,
            graph_digest="pending",
        )
        path = pathlib.Path(graph_path)
        graph = self.load(path)
        bundle = graph.nodes.get("bundle")
        if bundle is None or bundle.kind is not lineage.EvidenceNodeKind.BUNDLE:
            raise ValueError("evidence graph has no bundle node")
        if bundle.digest != bundle_digest:
            raise ValueError("inference bundle digest does not match graph")
        node_id = f"inference:{event_id}"
        payload = {
            "event_id": candidate.event_id,
            "input_fingerprint": candidate.input_fingerprint,
            "output_fingerprint": candidate.output_fingerprint,
            "bundle_digest": candidate.bundle_digest,
            "timestamp": candidate.timestamp,
        }
        node_digest = hashlib.sha256(
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        node = lineage.EvidenceNode(
            node_id=node_id,
            kind=lineage.EvidenceNodeKind.INFERENCE,
            digest=node_digest,
            attributes={
                "event_id": candidate.event_id,
                "input_fingerprint": candidate.input_fingerprint,
                "output_fingerprint": candidate.output_fingerprint,
                "bundle_digest": candidate.bundle_digest,
                "timestamp": candidate.timestamp,
            },
        )
        graph.add_node(node)
        graph.add_edge(lineage.EvidenceEdge("bundle", node_id, "serves"))
        graph_record = graph.to_dict()
        graph_record["graph_digest"] = graph.digest()
        self.write_atomically(path, graph_record)
        return InferenceEvidenceLink(
            event_id=candidate.event_id,
            input_fingerprint=candidate.input_fingerprint,
            output_fingerprint=candidate.output_fingerprint,
            bundle_digest=candidate.bundle_digest,
            timestamp=candidate.timestamp,
            graph_digest=str(graph_record["graph_digest"]),
        )

    @staticmethod
    def load(path: str | pathlib.Path) -> lineage.EvidenceGraph:
        """Load and validate a persisted evidence graph.

        Args:
            path: Path of the evidence graph JSON file.

        Returns:
            The graph rebuilt from the file.

        Raises:
            TypeError: If the file does not hold a JSON object.
            ValueError: If the stored `graph_digest` does not match the
                content.
        """
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        if not isinstance(record, Mapping):
            raise TypeError("evidence graph must be a JSON object")
        graph = lineage.EvidenceGraph.from_dict(record)
        expected_digest = record.get("graph_digest")
        if expected_digest is not None and expected_digest != graph.digest():
            raise ValueError("evidence graph digest does not match content")
        return graph

    @staticmethod
    def write_atomically(path: pathlib.Path, record: dict[str, Any]) -> None:
        """Persist graph JSON with flushed contents and directory metadata.

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
            os.replace(temporary_name, path)
            durability.sync_directory(path.parent)
        except BaseException:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
            raise
