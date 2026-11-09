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


class Kind(enum.StrEnum):
    """Evidence object categories in the lifecycle graph."""

    RAW_SOURCE = "raw_source"
    SNAPSHOT = "snapshot"
    CURATION = "curation"
    FEATURE_GRAPH = "feature_graph"
    SPLIT = "split"
    MODEL = "model"
    EVALUATION = "evaluation"
    PARAMETER_ESTIMATION = "parameter_estimation"
    EXPLANATION = "explanation"
    BUNDLE = "bundle"
    INFERENCE = "inference"


def require_text(value: str, name: str) -> str:
    """Returns a trimmed non-empty contract string.

    Args:
        value: The text to check.
        name: The field name, for the error message.

    Returns:
        `value` with surrounding whitespace removed.

    Raises:
        ValueError: If the trimmed text is empty.
    """
    result = value.strip()
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class Node:
    """One immutable content identity in an evidence graph."""

    node_id: str
    kind: Kind
    digest: str
    attributes: Mapping[str, str] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validates and normalizes node identity metadata.

        Raises:
            TypeError: If an attribute value is not a string.
            ValueError: If the id or digest is blank, the kind is not a valid
                node kind, or an attribute name is empty.
        """
        require_text(self.node_id, "node_id")
        require_text(self.digest, "digest")
        kind: Any = self.kind  # may be a raw value at runtime
        if not isinstance(kind, Kind):
            object.__setattr__(self, "kind", Kind(kind))
        normalized = dict(self.attributes)
        if any(not key.strip() for key in normalized):
            raise ValueError("attribute names must not be empty")
        if any(not isinstance(value, str) for value in normalized.values()):
            raise TypeError("attribute values must be strings")
        object.__setattr__(self, "attributes", normalized)

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible node record."""
        return {
            "node_id": self.node_id,
            "kind": self.kind.value,
            "digest": self.digest,
            "attributes": dict(sorted(self.attributes.items())),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a node from its serialized representation.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The node.
        """
        return cls(
            node_id=str(record["node_id"]),
            kind=Kind(record["kind"]),
            digest=str(record["digest"]),
            attributes=dict(record.get("attributes", {})),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """A directed lifecycle relationship between two evidence nodes."""

    source_id: str
    target_id: str
    relationship: str

    def __post_init__(self) -> None:
        """Validates edge endpoints and relationship semantics.

        Raises:
            ValueError: If an endpoint or the relationship is blank, or the edge
                connects a node to itself.
        """
        require_text(self.source_id, "source_id")
        require_text(self.target_id, "target_id")
        require_text(self.relationship, "relationship")
        if self.source_id == self.target_id:
            raise ValueError("evidence edges must not be self-loops")

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-compatible edge record."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relationship": self.relationship,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds an edge from its serialized representation.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The edge.
        """
        return cls(
            source_id=str(record["source_id"]),
            target_id=str(record["target_id"]),
            relationship=str(record["relationship"]),
        )


class Graph:
    """Deterministic directed acyclic graph of run evidence.

    Node registration is idempotent only when the complete node identity is
    identical. Conflicting re-registration and dangling or cyclic edges fail
    immediately, so a partial graph cannot be mistaken for valid lineage.
    """

    schema_version = 1

    def __init__(self, nodes: Iterable[Node] = ()) -> None:
        """Creates a graph and registers initial nodes.

        Args:
            nodes: The nodes to register.
        """
        self.nodes: dict[str, Node] = {}
        self.edges: list[Edge] = []
        for node in nodes:
            self.add_node(node)

    def add_node(self, node: Node) -> None:
        """Registers one node without silently replacing an identity.

        Args:
            node: The node to register.

        Raises:
            TypeError: If `node` is not an `Node`.
            ValueError: If a different node already has the same id.
        """
        if not isinstance(node, Node):
            raise TypeError("node must be Node")
        existing = self.nodes.get(node.node_id)
        if existing is not None and existing != node:
            raise ValueError(f"conflicting evidence node: {node.node_id}")
        self.nodes[node.node_id] = node

    def add_edge(self, edge: Edge) -> None:
        """Registers one edge and rejects dangling or cyclic lineage.

        An edge that is already registered is ignored.

        Args:
            edge: The edge to register.

        Raises:
            TypeError: If `edge` is not an `Edge`.
            ValueError: If an endpoint is not a registered node, or the edge
                would create a cycle.
        """
        if not isinstance(edge, Edge):
            raise TypeError("edge must be Edge")
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            raise ValueError("evidence edge refers to an unknown node")
        if edge in self.edges:
            return
        if self.path_exists(edge.target_id, edge.source_id):
            raise ValueError("evidence graph must remain acyclic")
        self.edges.append(edge)

    def path_exists(self, source_id: str, target_id: str) -> bool:
        """Returns whether a directed path connects two registered nodes.

        Args:
            source_id: Id of the start node.
            target_id: Id of the end node.

        Returns:
            True if both nodes are registered and a path leads from the first to
            the second (a node reaches itself); False otherwise.
        """
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
        """Returns deterministic outgoing adjacency lists."""
        result: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            result[edge.source_id].append(edge.target_id)
        return {
            node_id: tuple(sorted(targets))
            for node_id, targets in sorted(result.items())
        }

    def topological_order(self) -> tuple[str, ...]:
        """Returns a deterministic topological node order.

        Returns:
            The node ids ordered so every edge goes forward; ties are broken by
            sorted id.

        Raises:
            ValueError: If the graph has a dangling or duplicate edge or
                contains a cycle.
        """
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
        """Validates endpoints, duplicates, and acyclicity.

        Raises:
            ValueError: If an edge has an unknown endpoint, an edge is
                duplicated, or the graph contains a cycle.
        """
        seen: set[Edge] = set()
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
        """Computes order without recursively invoking ``validate``.

        The caller must already have checked that every edge joins registered
        nodes; an unknown endpoint raises `KeyError` instead.

        Returns:
            The node ids ordered so every edge goes forward; ties are broken by
            sorted id.

        Raises:
            ValueError: If the graph contains a cycle.
        """
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
        """Returns deterministic graph content without an external digest.

        The graph is validated first, so a dangling, duplicate or cyclic
        edge raises `ValueError`.
        """
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
        """Returns canonical graph content suitable for hashing and signing."""
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    def digest(self) -> str:
        """Returns the hex SHA-256 digest of the canonical graph content."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds and validates a graph from serialized evidence.

        Nodes and edges are registered in order, so a conflicting node or an
        edge with an unknown endpoint or that closes a cycle also raises
        `ValueError`.

        Args:
            record: A mapping with the schema version, nodes, edges and, if
                present, the topological order.

        Returns:
            The graph.

        Raises:
            ValueError: If the schema version is unsupported or a recorded
                topological order differs from the graph's.
        """
        if int(record.get("schema_version", 0)) != cls.schema_version:
            raise ValueError("unsupported evidence graph schema version")
        graph = cls(Node.from_dict(item) for item in record.get("nodes", ()))
        for item in record.get("edges", ()):
            graph.add_edge(Edge.from_dict(item))
        graph.validate()
        expected_order = tuple(record.get("topological_order", ()))
        if expected_order and expected_order != graph.topological_order():
            raise ValueError("evidence graph topological order is invalid")
        return graph


__all__ = [
    "Edge",
    "Graph",
    "Kind",
    "Node",
    "require_text",
]
