"""Content-linked evidence graph contracts for an ML run.

The graph is deliberately small and modality-neutral. It records identities
and relationships, not mutable payloads, so a packaged run can answer which
source, curation decision, split, model, and evaluation produced an artifact
without replaying free-form logs.
"""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import json
from typing import Any
from typing import Self


class EvidenceNodeKind(enum.StrEnum):
    """Evidence object categories in the lifecycle graph."""

    RAW_SOURCE = "raw_source"
    SNAPSHOT = "snapshot"
    CURATION = "curation"
    FEATURE_GRAPH = "feature_graph"
    SPLIT = "split"
    MODEL = "model"
    EVALUATION = "evaluation"
    EXPLANATION = "explanation"
    BUNDLE = "bundle"
    INFERENCE = "inference"


def require_text(value: str, name: str) -> str:
    """Return a trimmed non-empty contract string."""
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceNode:
    """One immutable content identity in an evidence graph."""

    node_id: str
    kind: EvidenceNodeKind
    digest: str
    attributes: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize node identity metadata."""
        require_text(self.node_id, "node_id")
        require_text(self.digest, "digest")
        if not isinstance(self.kind, EvidenceNodeKind):
            object.__setattr__(self, "kind", EvidenceNodeKind(self.kind))
        normalized = dict(self.attributes)
        if any(not key.strip() for key in normalized):
            raise ValueError("attribute names must not be empty")
        if any(not isinstance(value, str) for value in normalized.values()):
            raise TypeError("attribute values must be strings")
        object.__setattr__(self, "attributes", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible node record."""
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "digest": self.digest,
            "attributes": dict(sorted(self.attributes.items())),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a node from its serialized representation."""
        return cls(
            node_id=str(record["node_id"]),
            kind=EvidenceNodeKind(record["kind"]),
            digest=str(record["digest"]),
            attributes=dict(record.get("attributes", {})),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EvidenceEdge:
    """A directed lifecycle relationship between two evidence nodes."""

    source_id: str
    target_id: str
    relationship: str

    def __post_init__(self) -> None:
        """Validate edge endpoints and relationship semantics."""
        require_text(self.source_id, "source_id")
        require_text(self.target_id, "target_id")
        require_text(self.relationship, "relationship")
        if self.source_id == self.target_id:
            raise ValueError("evidence edges must not be self-loops")

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-compatible edge record."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild an edge from its serialized representation."""
        return cls(
            source_id=str(record["source_id"]),
            target_id=str(record["target_id"]),
            relationship=str(record["relationship"]),
        )


class EvidenceGraph:
    """Deterministic directed acyclic graph of run evidence.

    Node registration is idempotent only when the complete node identity is
    identical. Conflicting re-registration and dangling or cyclic edges fail
    immediately, so a partial graph cannot be mistaken for valid lineage.
    """

    schema_version = 1

    def __init__(self, nodes: Iterable[EvidenceNode] = ()) -> None:
        """Create a graph and register initial nodes."""
        self.nodes: dict[str, EvidenceNode] = {}
        self.edges: list[EvidenceEdge] = []
        for node in nodes:
            self.add_node(node)

    def add_node(self, node: EvidenceNode) -> None:
        """Register one node without silently replacing an identity."""
        if not isinstance(node, EvidenceNode):
            raise TypeError("node must be EvidenceNode")
        existing = self.nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting evidence node: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: EvidenceEdge) -> None:
        """Register one edge and reject dangling or cyclic lineage."""
        if not isinstance(edge, EvidenceEdge):
            raise TypeError("edge must be EvidenceEdge")
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("evidence edge refers to an unknown node")
        if edge in self.edges:
            return
        if self.path_exists(edge.target_id, edge.source_id):
            raise ValueError("evidence graph must remain acyclic")
        self.edges.append(edge)

    def path_exists(self, source_id: str, target_id: str) -> bool:
        """Return whether a directed path connects two registered nodes."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return False
        pending = [source_id]
        visited: set[str] = set()
        outgoing = self.outgoing()
        while pending:
            current = pending.pop()
            if current == target_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            pending.extend(outgoing.get(current, ()))
        return False

    def outgoing(self) -> dict[str, tuple[str, ...]]:
        """Return deterministic outgoing adjacency lists."""
        result: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            result[edge.source_id].append(edge.target_id)
        return {
            node_id: tuple(sorted(targets))
            for node_id, targets in sorted(result.items())
        }

    def topological_order(self) -> tuple[str, ...]:
        """Return a deterministic topological node order."""
        self.validate()
        incoming = dict.fromkeys(self.nodes, 0)
        outgoing = self.outgoing()
        for edge in self.edges:
            incoming[edge.target_id] += 1
        ready = sorted(
            node_id for node_id, count in incoming.items() if count == 0
        )
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for target in outgoing[current]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("evidence graph contains a cycle")
        return tuple(order)

    def validate(self) -> None:
        """Validate endpoints, duplicates, and acyclicity."""
        seen: set[EvidenceEdge] = set()
        for edge in self.edges:
            if (
                edge.source_id not in self.nodes
                or edge.target_id not in self.nodes
            ):
                raise ValueError("evidence graph contains a dangling edge")
            if edge in seen:
                raise ValueError("evidence graph contains duplicate edges")
            seen.add(edge)
        self.topological_order_without_validation()

    def topological_order_without_validation(self) -> tuple[str, ...]:
        """Compute order without recursively invoking ``validate``."""
        incoming = dict.fromkeys(self.nodes, 0)
        outgoing = self.outgoing()
        for edge in self.edges:
            incoming[edge.target_id] += 1
        ready = sorted(
            node_id for node_id, count in incoming.items() if count == 0
        )
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for target in outgoing[current]:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("evidence graph contains a cycle")
        return tuple(order)

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic graph content without an external digest."""
        self.validate()
        return {
            "schema_version": self.schema_version,
            "nodes": [
                self.nodes[node_id].to_dict() for node_id in sorted(self.nodes)
            ],
            "edges": [
                edge.to_dict()
                for edge in sorted(
                    self.edges,
                    key=lambda item: (
                        item.source_id,
                        item.target_id,
                        item.relationship,
                    ),
                )
            ],
            "topological_order": list(
                self.topological_order_without_validation()
            ),
        }

    def canonical_json(self) -> str:
        """Return canonical graph content suitable for hashing and signing."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def digest(self) -> str:
        """Return the SHA-256 digest of canonical graph content."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild and validate a graph from serialized evidence."""
        if int(record.get("schema_version", 0)) != cls.schema_version:
            raise ValueError("unsupported evidence graph schema version")
        graph = cls(
            EvidenceNode.from_dict(item) for item in record.get("nodes", ())
        )
        for item in record.get("edges", ()):
            graph.add_edge(EvidenceEdge.from_dict(item))
        graph.validate()
        expected_order = tuple(record.get("topological_order", ()))
        if expected_order and expected_order != graph.topological_order():
            raise ValueError("evidence graph topological order is invalid")
        return graph


__all__ = [
    "EvidenceEdge",
    "EvidenceGraph",
    "EvidenceNode",
    "EvidenceNodeKind",
    "require_text",
]
