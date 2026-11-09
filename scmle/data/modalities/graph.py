"""Dependency-free graph data contracts, ingestion, and baseline preparation.

This module intentionally stops at a data foundation. It represents a graph,
validates its structural invariants, profiles its contents, creates leakage-
safe inductive node splits, and prepares bounded numeric features. Graph
models, graph explainers, and graph serving are not part of this module.
"""

from collections import defaultdict
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
from datetime import datetime
from datetime import UTC
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy
import pandas

from scmle.contracts import data
from scmle.data.modalities.curation import build_finding
from scmle.data.modalities.curation import Curation
from scmle.data.modalities.curation import label_counts


class GraphError(ValueError):
    """Base error for invalid graph data or graph input."""


class GraphInputError(GraphError):
    """Raised when a graph input file cannot be parsed."""


class GraphTemporalError(GraphError):
    """Raised when temporal graph evidence is incomplete or invalid."""


class GraphValidationError(GraphError):
    """Raised when a graph fails structural validation.

    Attributes:
        report: The complete validation report that caused the error.
    """

    def __init__(self, report: "Validation") -> None:
        """Builds an error containing the complete validation report.

        Args:
            report: The validation report; its messages, joined by ``; ``,
                form the error message.
        """
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.issues))


@dataclasses.dataclass(frozen=True, slots=True)
class Node:
    """Immutable node with an optional numeric feature vector and label.

    Attributes:
        node_id: Non-empty node identifier.
        features: Finite numeric feature vector, stored as floats; may be
            empty.
        label: Optional node label.
    """

    node_id: str
    features: tuple[float, ...] = ()
    label: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        """Validates the node identifier and numeric feature vector.

        Raises:
            ValueError: If node_id is not a non-empty string or a feature is
                not finite.
            TypeError: If a feature is not an int or float.
        """
        node_id: object = self.node_id
        if not isinstance(node_id, str) or not node_id.strip():
            raise ValueError("node_id must be a non-empty string")
        if any(not isinstance(value, (int, float)) for value in self.features):
            raise TypeError("node features must be numeric")
        if any(not math.isfinite(float(value)) for value in self.features):
            raise ValueError("node features must be finite")
        object.__setattr__(
            self,
            "features",
            tuple(float(value) for value in self.features),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """Immutable weighted edge between two node identifiers.

    Attributes:
        source: Non-empty identifier of the source node.
        target: Non-empty identifier of the target node.
        weight: Finite edge weight, stored as a float.
        relation: Non-empty relation name.
        event_time: Optional timezone-aware time, stored in UTC.
    """

    source: str
    target: str
    weight: float = 1.0
    relation: str = "default"
    event_time: datetime | None = None

    def __post_init__(self) -> None:
        """Validates endpoint, relation, and weight values.

        Raises:
            ValueError: If an endpoint or the relation is not a non-empty
                string, the weight is not finite, or the event time has no
                timezone.
            TypeError: If the weight is not numeric or the event time is not
                a datetime.
        """
        source: object = self.source
        target: object = self.target
        if not isinstance(source, str) or not source.strip():
            raise ValueError("edge source must be a non-empty string")
        if not isinstance(target, str) or not target.strip():
            raise ValueError("edge target must be a non-empty string")
        if not isinstance(self.weight, (int, float)):
            raise TypeError("edge weight must be numeric")
        if not math.isfinite(float(self.weight)):
            raise ValueError("edge weight must be finite")
        relation: object = self.relation
        if not isinstance(relation, str) or not relation.strip():
            raise ValueError("edge relation must be a non-empty string")
        if self.event_time is not None:
            if not isinstance(self.event_time, datetime):
                raise TypeError("event_time must be a datetime")
            if (
                self.event_time.tzinfo is None
                or self.event_time.utcoffset() is None
            ):
                raise ValueError("event_time must be timezone-aware")
            object.__setattr__(
                self,
                "event_time",
                self.event_time.astimezone(UTC),
            )
        object.__setattr__(self, "weight", float(self.weight))


@dataclasses.dataclass(frozen=True, slots=True)
class TemporalSplit:
    """Time-ordered graph edges separated at an explicit UTC cutoff.

    Attributes:
        cutoff: Timezone-aware cutoff, converted to UTC.
        train_edges: Edges with event time at or before the cutoff.
        future_edges: Edges with event time after the cutoff.
    """

    cutoff: datetime
    train_edges: tuple[Edge, ...]
    future_edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        """Validates the cutoff and timestamp completeness of both partitions.

        Raises:
            ValueError: If the cutoff is not timezone-aware or an edge in
                either partition has no event time.
        """
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise ValueError("cutoff must be timezone-aware")
        object.__setattr__(
            self,
            "cutoff",
            self.cutoff.astimezone(UTC),
        )
        if any(edge.event_time is None for edge in self.train_edges):
            raise ValueError("train_edges must contain event times")
        if any(edge.event_time is None for edge in self.future_edges):
            raise ValueError("future_edges must contain event times")

    @property
    def has_future_leakage(self) -> bool:
        """Whether a training edge occurs after the cutoff."""
        cutoff = self.cutoff.astimezone(UTC)
        return any(
            edge.event_time is not None and edge.event_time > cutoff
            for edge in self.train_edges
        )


def canonical_event_time(value: datetime | str) -> datetime:
    """Parses and normalizes a timezone-aware graph event timestamp to UTC.

    Args:
        value: A datetime, or ISO 8601 text; a trailing ``Z`` means UTC.

    Returns:
        The same instant as a UTC datetime.

    Raises:
        GraphTemporalError: If the text is not ISO 8601, the value is
            neither datetime nor text, or the time has no timezone.
    """
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith(("Z", "z")):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise GraphTemporalError(
                f"invalid graph event time: {value!r}"
            ) from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise GraphTemporalError("graph event time must be datetime or text")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise GraphTemporalError("graph event time must be timezone-aware")
    return parsed.astimezone(UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class Issue:
    """One deterministic graph validation finding.

    Attributes:
        code: Machine-readable issue code.
        message: Human-readable description.
        affected: Identifiers of the nodes or edge parts concerned.
    """

    code: str
    message: str
    affected: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible issue mapping.

        Returns:
            A dict with `code`, `message` and `affected` (as a list).
        """
        return {
            "code": self.code,
            "message": self.message,
            "affected": list(self.affected),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Validation:
    """Complete graph validation result.

    Attributes:
        issues: Every issue found; empty when the graph is valid.
    """

    issues: tuple[Issue, ...]

    @property
    def is_valid(self) -> bool:
        """Whether no validation issues were found."""
        return not self.issues

    @property
    def error_codes(self) -> tuple[str, ...]:
        """The unique issue codes, sorted."""
        return tuple(sorted({issue.code for issue in self.issues}))

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible report mapping.

        Returns:
            A dict with `is_valid`, `error_codes` and `issues` (each issue as
            a dict).
        """
        return {
            "is_valid": self.is_valid,
            "error_codes": list(self.error_codes),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class NodeProfile:
    """Summary statistics for graph nodes and their degrees.

    Attributes:
        node_count: Number of nodes.
        isolated_node_count: Number of nodes with degree zero.
        labeled_node_count: Number of nodes with a label.
        feature_dimension: Longest feature vector among the nodes.
        degree_min: Smallest degree; 0 for an empty graph.
        degree_max: Largest degree; 0 for an empty graph.
        degree_mean: Mean degree; 0.0 for an empty graph.
    """

    node_count: int
    isolated_node_count: int
    labeled_node_count: int
    feature_dimension: int
    degree_min: int
    degree_max: int
    degree_mean: float

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible node profile.

        Returns:
            A dict of the profile fields.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class EdgeProfile:
    """Summary statistics for graph edges and weights.

    Attributes:
        edge_count: Number of edges.
        duplicate_edge_count: Number of edges beyond the first for each
            duplicate-detection key.
        self_loop_count: Number of edges whose source equals the target.
        weight_min: Smallest weight, or None without edges.
        weight_max: Largest weight, or None without edges.
        weight_mean: Mean weight, or None without edges.
    """

    edge_count: int
    duplicate_edge_count: int
    self_loop_count: int
    weight_min: float | None
    weight_max: float | None
    weight_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible edge profile.

        Returns:
            A dict of the profile fields.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """Combined graph, node, and edge profile.

    Attributes:
        graph_identity: SHA-256 identity of the graph content.
        directed: Whether the graph is directed.
        component_count: Number of weakly connected components.
        node: Node profile.
        edge: Edge profile.
    """

    graph_identity: str
    directed: bool
    component_count: int
    node: NodeProfile
    edge: EdgeProfile

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible graph profile.

        Returns:
            A dict with the identity, direction, component count and the
            node and edge profiles as nested dicts.
        """
        return {
            "graph_identity": self.graph_identity,
            "directed": self.directed,
            "component_count": self.component_count,
            "node": self.node.to_dict(),
            "edge": self.edge.to_dict(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class InductiveNodeSplit:
    """Leakage-safe node partitions and within-partition edge partitions.

    Attributes:
        train_nodes: Sorted identifiers of the training nodes.
        validation_nodes: Sorted identifiers of the validation nodes.
        test_nodes: Sorted identifiers of the test nodes.
        train_edges: Edges with both endpoints in the training nodes.
        validation_edges: Edges with both endpoints in the validation nodes.
        test_edges: Edges with both endpoints in the test nodes.
        withheld_edges: Edges whose endpoints fall in different partitions.
        seed: Seed used for the shuffle.
    """

    train_nodes: tuple[str, ...]
    validation_nodes: tuple[str, ...]
    test_nodes: tuple[str, ...]
    train_edges: tuple[Edge, ...]
    validation_edges: tuple[Edge, ...]
    test_edges: tuple[Edge, ...]
    withheld_edges: tuple[Edge, ...]
    seed: int

    @property
    def node_partitions(self) -> tuple[tuple[str, ...], ...]:
        """The train, validation and test node partitions, in that order."""
        return (self.train_nodes, self.validation_nodes, self.test_nodes)


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedFeatures:
    """Bounded numeric features for a graph baseline.

    Attributes:
        node_ids: Identifier of each row.
        feature_names: Name of each column.
        values: One tuple of feature values per node.
        standardized: Whether columns were standardised to zero mean and
            unit variance.
    """

    node_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    standardized: bool

    @property
    def shape(self) -> tuple[int, int]:
        """The number of nodes and the number of features, as a pair."""
        return (len(self.node_ids), len(self.feature_names))

    def to_numpy(self) -> numpy.ndarray:
        """Returns a writable copy of the prepared feature matrix.

        Returns:
            A float64 array of shape ``(nodes, features)``.
        """
        return numpy.asarray(self.values, dtype=numpy.float64).copy()


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset:
    """Immutable graph dataset with canonical node and edge ordering.

    Attributes:
        nodes: Nodes, sorted by identifier; identifiers are unique.
        edges: Edges, sorted by source, target, relation, weight and event
            time.
        directed: Whether edges are directed.
        feature_names: Names of the node feature columns; when given, they
            match the populated feature length.
        source_uri: URI the graph was loaded from.
    """

    nodes: tuple[Node, ...] = ()
    edges: tuple[Edge, ...] = ()
    directed: bool = True
    feature_names: tuple[str, ...] = ()
    source_uri: str = "memory://"

    def __post_init__(self) -> None:
        """Normalizes collections and checks basic shape invariants.

        Raises:
            ValueError: If node identifiers repeat, populated feature
                vectors differ in length, feature_names does not match that
                length, or a feature name is blank.
        """
        nodes = tuple(sorted(self.nodes, key=lambda node: node.node_id))
        edges = tuple(
            sorted(
                self.edges,
                key=lambda edge: (
                    edge.source,
                    edge.target,
                    edge.relation,
                    edge.weight,
                    edge.event_time.isoformat() if edge.event_time else "",
                ),
            )
        )
        if len({node.node_id for node in nodes}) != len(nodes):
            raise ValueError("node identifiers must be unique")
        dimensions = {len(node.features) for node in nodes if node.features}
        if len(dimensions) > 1:
            raise ValueError(
                "all populated node feature vectors must have equal length"
            )
        if (
            self.feature_names
            and dimensions
            and len(self.feature_names) != next(iter(dimensions))
        ):
            raise ValueError(
                "feature_names must match the node feature dimension"
            )
        if any(not name.strip() for name in self.feature_names):
            raise ValueError("feature_names must not contain empty names")
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @classmethod
    def from_edges(
        cls,
        edges: Iterable[Edge],
        nodes: Iterable[Node | str] | None = None,
        directed: bool = True,
        feature_names: Iterable[str] = (),
        source_uri: str = "memory://",
    ) -> "Dataset":
        """Builds a graph and derives nodes when no node collection is given.

        Args:
            edges: The edges.
            nodes: Nodes or node identifiers. When None, one featureless
                node is created for every edge endpoint.
            directed: Whether edges are directed.
            feature_names: Names of the node feature columns.
            source_uri: URI recorded as the graph source.

        Returns:
            The new graph. It is not validated beyond the invariants checked
            on construction; call `require_valid` for endpoint checks.
        """
        edge_values = tuple(edges)
        if nodes is None:
            node_values = tuple(
                Node(node_id)
                for node_id in sorted(
                    {
                        endpoint
                        for edge in edge_values
                        for endpoint in (edge.source, edge.target)
                    }
                )
            )
        else:
            node_values = tuple(
                value if isinstance(value, Node) else Node(value)
                for value in nodes
            )
        return cls(
            nodes=node_values,
            edges=edge_values,
            directed=directed,
            feature_names=tuple(feature_names),
            source_uri=source_uri,
        )

    @classmethod
    def from_edge_file(
        cls,
        path: str | Path,
        nodes: Iterable[Node | str] | None = None,
        directed: bool = True,
        source_column: str = "source",
        target_column: str = "target",
        weight_column: str = "weight",
        relation_column: str = "relation",
        time_column: str | None = None,
    ) -> "Dataset":
        """Loads CSV or JSONL edges into a validated graph dataset.

        Args:
            path: Edge file (``.csv``, ``.jsonl`` or ``.ndjson``).
            nodes: Nodes or node identifiers. When None, nodes are derived
                from the edge endpoints.
            directed: Whether edges are directed.
            source_column: Name of the source column or key.
            target_column: Name of the target column or key.
            weight_column: Name of the weight column or key.
            relation_column: Name of the relation column or key.
            time_column: Name of the event-time column or key, or None for
                untimed edges.

        Returns:
            The graph, with its source URI set to the file's URI. Errors
            from `read_edges` propagate; a graph that fails `validate`
            raises GraphValidationError.
        """
        edges = read_edges(
            path,
            source_column=source_column,
            target_column=target_column,
            weight_column=weight_column,
            relation_column=relation_column,
            time_column=time_column,
        )
        graph = cls.from_edges(
            edges,
            nodes=nodes,
            directed=directed,
            source_uri=Path(path).expanduser().resolve(strict=False).as_uri(),
        )
        graph.require_valid()
        return graph

    @property
    def node_ids(self) -> tuple[str, ...]:
        """The node identifiers, in canonical (sorted) order."""
        return tuple(node.node_id for node in self.nodes)

    @property
    def identity(self) -> str:
        """The full SHA-256 hex identity of the graph content."""
        return graph_identity(
            self.nodes,
            self.edges,
            directed=self.directed,
            feature_names=self.feature_names,
        )

    def validate(self) -> Validation:
        """Finds missing endpoints, duplicates, self-loops, and bad weights.

        Returns:
            A report with the issues found, in edge order followed by
            duplicates. Codes are `missing_endpoint`, `self_loop`,
            `invalid_weight` and `duplicate_edge`.
        """
        node_ids = set(self.node_ids)
        issues: list[Issue] = []
        duplicate_counts: defaultdict[tuple[str, str, str, str], int] = (
            defaultdict(int)
        )
        for edge in self.edges:
            if edge.source not in node_ids:
                issues.append(
                    Issue(
                        "missing_endpoint",
                        f"edge source '{edge.source}' is not a declared node",
                        (edge.source,),
                    )
                )
            if edge.target not in node_ids:
                issues.append(
                    Issue(
                        "missing_endpoint",
                        f"edge target '{edge.target}' is not a declared node",
                        (edge.target,),
                    )
                )
            if edge.source == edge.target:
                issues.append(
                    Issue(
                        "self_loop",
                        f"self-loop found on node '{edge.source}'",
                        (edge.source,),
                    )
                )
            if not math.isfinite(edge.weight):
                issues.append(
                    Issue(
                        "invalid_weight",
                        f"edge '{edge.source}->{edge.target}' has a non-finite "
                        "weight",
                        (edge.source, edge.target),
                    )
                )
            key = self.edge_key(edge)
            duplicate_counts[key] += 1
        for key, count in sorted(duplicate_counts.items()):
            if count > 1:
                issues.append(
                    Issue(
                        "duplicate_edge",
                        f"edge {key[0]}->{key[1]} ({key[2]}) occurs "
                        f"{count} times",
                        key,
                    )
                )
        return Validation(tuple(issues))

    def require_valid(self) -> None:
        """Raises `GraphValidationError` unless the graph is valid.

        Raises:
            GraphValidationError: If `validate` finds any issue.
        """
        report = self.validate()
        if not report.is_valid:
            raise GraphValidationError(report)

    def edge_key(self, edge: Edge) -> tuple[str, str, str, str]:
        """Returns the canonical duplicate-detection key for an edge.

        Args:
            edge: The edge.

        Returns:
            The tuple (first endpoint, second endpoint, relation, ISO event
            time or empty text). The endpoints are sorted when the graph is
            undirected. The weight is not part of the key.
        """
        endpoints = (edge.source, edge.target)
        if not self.directed:
            endpoints = tuple(sorted(endpoints))  # type: ignore[assignment]
        event_time = edge.event_time.isoformat() if edge.event_time else ""
        return (endpoints[0], endpoints[1], edge.relation, event_time)

    def connected_components(self) -> tuple[frozenset[str], ...]:
        """Returns deterministic weakly connected components.

        Edges are treated as undirected and edges to undeclared nodes are
        ignored.

        Returns:
            The components as frozensets of node identifiers, ordered by
            their smallest identifier.
        """
        adjacency: dict[str, set[str]] = {
            node_id: set() for node_id in self.node_ids
        }
        for edge in self.edges:
            if edge.source in adjacency and edge.target in adjacency:
                adjacency[edge.source].add(edge.target)
                adjacency[edge.target].add(edge.source)
        remaining = set(adjacency)
        components: list[frozenset[str]] = []
        while remaining:
            root = min(remaining)
            pending = [root]
            component: set[str] = set()
            while pending:
                node_id = pending.pop()
                if node_id not in remaining:
                    continue
                remaining.remove(node_id)
                component.add(node_id)
                pending.extend(sorted(adjacency[node_id] & remaining))
            components.append(frozenset(component))
        return tuple(sorted(components, key=min))

    def profile_nodes(self) -> NodeProfile:
        """Profiles node counts, labels, features, and undirected degree.

        A self-loop adds one to its node's degree.

        Returns:
            The node profile.
        """
        degrees = dict.fromkeys(self.node_ids, 0)
        for edge in self.edges:
            if edge.source in degrees:
                degrees[edge.source] += 1
            if edge.target in degrees and edge.target != edge.source:
                degrees[edge.target] += 1
        degree_values = tuple(degrees.values())
        return NodeProfile(
            node_count=len(self.nodes),
            isolated_node_count=sum(value == 0 for value in degree_values),
            labeled_node_count=sum(
                node.label is not None for node in self.nodes
            ),
            feature_dimension=max(
                (len(node.features) for node in self.nodes), default=0
            ),
            degree_min=min(degree_values, default=0),
            degree_max=max(degree_values, default=0),
            degree_mean=float(numpy.mean(degree_values))
            if degree_values
            else 0.0,
        )

    def profile_edges(self) -> EdgeProfile:
        """Profiles edge counts, duplicate edges, loops, and weights.

        Returns:
            The edge profile.
        """
        weights = tuple(edge.weight for edge in self.edges)
        duplicate_count = sum(
            max(0, count - 1) for count in self.edge_key_counts().values()
        )
        return EdgeProfile(
            edge_count=len(self.edges),
            duplicate_edge_count=duplicate_count,
            self_loop_count=sum(
                edge.source == edge.target for edge in self.edges
            ),
            weight_min=min(weights) if weights else None,
            weight_max=max(weights) if weights else None,
            weight_mean=float(numpy.mean(weights)) if weights else None,
        )

    def edge_key_counts(self) -> dict[tuple[str, str, str, str], int]:
        """Returns canonical edge multiplicities.

        Returns:
            A dict from `edge_key` to the number of edges with that key.
        """
        counts: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
        for edge in self.edges:
            counts[self.edge_key(edge)] += 1
        return dict(counts)

    def profile(self) -> Profile:
        """Returns the combined graph profile.

        Returns:
            The profile with identity, direction, component count, and node
            and edge profiles.
        """
        return Profile(
            graph_identity=self.identity,
            directed=self.directed,
            component_count=len(self.connected_components()),
            node=self.profile_nodes(),
            edge=self.profile_edges(),
        )

    def curate(self) -> Curation:
        """Builds graph-structure evidence before choosing a graph model.

        Returns:
            A curation report for modality ``graph`` whose sample count is
            the node count. The profile holds the graph profile, node label
            counts and temporal status (`complete`, `mixed` or `absent`).
            Findings cover partly timestamped edges, duplicate edges,
            isolated nodes, missing declared features and mixed labelled and
            unlabelled nodes.
        """
        profile = self.profile()
        labels = label_counts(node.label for node in self.nodes)
        profile_data = profile.to_dict()
        profile_data["node"]["label_counts"] = labels
        temporal_edges = [
            edge for edge in self.edges if edge.event_time is not None
        ]
        event_times = tuple(
            edge.event_time
            for edge in temporal_edges
            if edge.event_time is not None
        )
        if event_times and len(event_times) == len(self.edges):
            profile_data["temporal"] = {
                "status": "complete",
                "event_time_min": min(event_times).isoformat(),
                "event_time_max": max(event_times).isoformat(),
            }
        elif temporal_edges:
            profile_data["temporal"] = {
                "status": "mixed",
                "timed_edge_count": len(temporal_edges),
                "untimed_edge_count": len(self.edges) - len(temporal_edges),
            }
        else:
            profile_data["temporal"] = {"status": "absent"}
        quality_findings = []
        risk_findings = []
        if temporal_edges and len(temporal_edges) != len(self.edges):
            quality_findings.append(
                build_finding(
                    "quality.graph-mixed-temporal-state",
                    data.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {"temporal": profile_data["temporal"]},
                    "Supply event times for every edge or explicitly use a "
                    "non-temporal graph capability.",
                    blocking=False,
                )
            )
        if profile.edge.duplicate_edge_count:
            quality_findings.append(
                build_finding(
                    "quality.graph-duplicate-edges",
                    data.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {
                        "duplicate_edge_count": (
                            profile.edge.duplicate_edge_count
                        )
                    },
                    "Declare whether parallel edges are valid or "
                    "deduplicate them.",
                    blocking=False,
                )
            )
        if profile.node.isolated_node_count:
            quality_findings.append(
                build_finding(
                    "quality.graph-isolated-nodes",
                    data.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {"isolated_node_count": profile.node.isolated_node_count},
                    "Confirm isolated nodes are valid for the intended task.",
                    blocking=False,
                )
            )
        if not self.feature_names:
            quality_findings.append(
                build_finding(
                    "quality.graph-no-declared-features",
                    data.FindingSeverity.INFO,
                    {"source": self.source_uri},
                    {"node_count": profile.node.node_count},
                    "Declare node features or select a topology-only model "
                    "explicitly.",
                    blocking=False,
                )
            )
        if "unlabeled" in labels and len(labels) > 1:
            risk_findings.append(
                build_finding(
                    "risk.graph-mixed-label-state",
                    data.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {"label_counts": labels},
                    "Declare whether unlabeled nodes are valid for the "
                    "intended graph task before training.",
                    blocking=False,
                )
            )
        return Curation(
            modality="graph",
            dataset_hash=self.identity,
            sample_count=len(self.nodes),
            profile=profile_data,
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
        )


def graph_identity(
    nodes: Iterable[Node],
    edges: Iterable[Edge],
    directed: bool = True,
    feature_names: Iterable[str] = (),
) -> str:
    """Returns a canonical SHA-256 identity independent of input ordering.

    For an undirected graph the endpoints of each edge are sorted before
    hashing.

    Args:
        nodes: The nodes.
        edges: The edges.
        directed: Whether edges are directed.
        feature_names: Names of the node feature columns; hashed in sorted
            order.

    Returns:
        The SHA-256 hex digest of the canonical JSON of the content.
    """
    node_records: list[dict[str, Any]] = [
        {
            "node_id": node.node_id,
            "features": list(node.features),
            "label": node.label,
        }
        for node in nodes
    ]
    edge_records: list[dict[str, Any]] = [
        {
            "source": edge.source,
            "target": edge.target,
            "weight": edge.weight,
            "relation": edge.relation,
            "event_time": (
                edge.event_time.isoformat() if edge.event_time else None
            ),
        }
        for edge in edges
    ]
    if not directed:
        for record in edge_records:
            source = str(record["source"])
            target = str(record["target"])
            record["source"], record["target"] = sorted((source, target))
    payload = {
        "directed": directed,
        "feature_names": sorted(str(name) for name in feature_names),
        "nodes": sorted(
            node_records, key=lambda record: str(record["node_id"])
        ),
        "edges": sorted(
            edge_records,
            key=lambda record: (
                str(record["source"]),
                str(record["target"]),
                str(record["relation"]),
                float(record["weight"]),
                str(record["event_time"]),
            ),
        ),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_edges(
    path: str | Path,
    source_column: str = "source",
    target_column: str = "target",
    weight_column: str = "weight",
    relation_column: str = "relation",
    time_column: str | None = None,
) -> tuple[Edge, ...]:
    """Reads CSV or JSONL edge records with bounded schema handling.

    The format is chosen from the file name: ``.csv``, ``.jsonl`` or
    ``.ndjson`` (case-insensitive). Blank JSONL lines are skipped.

    Args:
        path: Edge file.
        source_column: Name of the source column or key.
        target_column: Name of the target column or key.
        weight_column: Name of the weight column or key; a missing weight
            defaults to 1.0.
        relation_column: Name of the relation column or key; a missing
            relation defaults to ``default``.
        time_column: Name of the event-time column or key, or None for
            untimed edges.

    Returns:
        The edges in file order.

    Raises:
        GraphInputError: If the file does not exist, has another extension,
            cannot be read or parsed, a JSONL line is not an object, or a
            record is invalid (see `edges_from_records`).
    """
    input_path = Path(path)
    if not input_path.exists():
        raise GraphInputError(f"graph edge file not found: {input_path}")
    suffix = input_path.name.lower()
    if suffix.endswith(".csv"):
        try:
            frame = pandas.read_csv(input_path)
        except (
            OSError,
            pandas.errors.ParserError,
            UnicodeDecodeError,
        ) as error:
            raise GraphInputError(
                f"failed to read CSV edges: {error}"
            ) from error
        csv_records = [
            {str(key): value for key, value in record.items()}
            for record in frame.to_dict(orient="records")
        ]
        return edges_from_records(
            csv_records,
            source_column=source_column,
            target_column=target_column,
            weight_column=weight_column,
            relation_column=relation_column,
            time_column=time_column,
        )
    if suffix.endswith((".jsonl", ".ndjson")):
        records: list[dict[str, Any]] = []
        try:
            with input_path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    content = line.strip()
                    if not content:
                        continue
                    try:
                        record = json.loads(content)
                    except json.JSONDecodeError as error:
                        raise GraphInputError(
                            f"malformed JSONL at line {line_number}: "
                            f"{error.msg}"
                        ) from error
                    if not isinstance(record, dict):
                        raise GraphInputError(
                            f"JSONL line {line_number} must be an object"
                        )
                    records.append(record)
        except OSError as error:
            raise GraphInputError(
                f"failed to read JSONL edges: {error}"
            ) from error
        return edges_from_records(
            records,
            source_column=source_column,
            target_column=target_column,
            weight_column=weight_column,
            relation_column=relation_column,
            time_column=time_column,
        )
    raise GraphInputError(
        "graph edges must use a .csv, .jsonl, or .ndjson file"
    )


def record_weight(
    record: Mapping[str, Any], weight_column: str, index: int
) -> float:
    """Reads the weight of an edge record.

    Args:
        record: The edge record.
        weight_column: Key of the weight; a missing key defaults to 1.0.
        index: 1-based record number, for the message.

    Returns:
        The weight as a float.

    Raises:
        GraphInputError: If the weight is null or non-numeric.
    """
    weight_value = record.get(weight_column, 1.0)
    if pandas.isna(weight_value):
        raise GraphInputError(f"edge record {index} has an invalid weight")
    try:
        return float(weight_value)
    except (TypeError, ValueError) as error:
        raise GraphInputError(
            f"edge record {index} has a non-numeric weight"
        ) from error


def record_event_time(
    record: Mapping[str, Any], time_column: str | None, index: int
) -> datetime | None:
    """Reads the event time of an edge record.

    Args:
        record: The edge record.
        time_column: Key of the event time, or None for untimed edges.
        index: 1-based record number, for the message.

    Returns:
        The event time normalised to UTC, or None when there is no time
        column or the value is missing or null.

    Raises:
        GraphInputError: If the event time is invalid.
    """
    if time_column is None:
        return None
    time_value = record.get(time_column)
    if time_value is None or pandas.isna(time_value):
        return None
    try:
        return canonical_event_time(time_value)
    except GraphTemporalError as error:
        raise GraphInputError(
            f"edge record {index} has an invalid event time"
        ) from error


def edges_from_records(
    records: Iterable[Mapping[str, Any]],
    source_column: str = "source",
    target_column: str = "target",
    weight_column: str = "weight",
    relation_column: str = "relation",
    time_column: str | None = None,
) -> tuple[Edge, ...]:
    """Converts mappings into immutable edges with deterministic errors.

    Args:
        records: Mappings holding one edge each.
        source_column: Key of the source identifier.
        target_column: Key of the target identifier.
        weight_column: Key of the weight; a missing key defaults to 1.0.
        relation_column: Key of the relation; a missing or null value
            becomes ``default``.
        time_column: Key of the event time, or None for untimed edges. A
            missing or null value leaves the edge untimed.

    Returns:
        The edges in record order; identifiers are converted to text.

    Raises:
        GraphInputError: If a record lacks the source or target key or has a
            null endpoint, a null or non-numeric weight, an invalid event
            time, or values that `Edge` rejects. The message gives the
            1-based record number.
    """
    edges: list[Edge] = []
    for index, record in enumerate(records, start=1):
        if source_column not in record or target_column not in record:
            raise GraphInputError(
                f"edge record {index} must contain '{source_column}' and "
                f"'{target_column}'"
            )
        source = record[source_column]
        target = record[target_column]
        if pandas.isna(source) or pandas.isna(target):
            raise GraphInputError(f"edge record {index} has a missing endpoint")
        weight = record_weight(record, weight_column, index)
        relation_value = record.get(relation_column, "default")
        if pandas.isna(relation_value):
            relation_value = "default"
        event_time = record_event_time(record, time_column, index)
        try:
            edges.append(
                Edge(
                    source=str(source),
                    target=str(target),
                    weight=weight,
                    relation=str(relation_value),
                    event_time=event_time,
                )
            )
        except (TypeError, ValueError) as error:
            raise GraphInputError(
                f"invalid edge record {index}: {error}"
            ) from error
    return tuple(edges)


def inductive_node_split(
    graph: Dataset,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> InductiveNodeSplit:
    """Creates deterministic node partitions with no retained edge leakage.

    Nodes are shuffled with the seed and divided by the fractions; every
    partition with a positive fraction receives at least one node when there
    are enough nodes. Edges inside a partition are retained for it and edges
    between partitions are withheld.

    Args:
        graph: The graph; it must be valid.
        train_fraction: Share of nodes for training.
        validation_fraction: Share of nodes for validation.
        test_fraction: Share of nodes for testing.
        seed: Seed of the node shuffle.

    Returns:
        The node and edge partitions with the seed.

    Raises:
        ValueError: If a fraction is negative or not finite, or the
            fractions do not sum to one. An invalid graph raises
            GraphValidationError from `Dataset.require_valid`.
    """
    graph.require_valid()
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value < 0 or not math.isfinite(value) for value in fractions):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")
    node_ids = numpy.asarray(graph.node_ids, dtype=object)
    generator = numpy.random.default_rng(seed)
    shuffled = node_ids[generator.permutation(len(node_ids))]
    positive_indices = [index for index, value in enumerate(fractions) if value]
    counts = [0, 0, 0]
    if len(node_ids) >= len(positive_indices):
        for index in positive_indices:
            counts[index] = 1
        remaining = len(node_ids) - len(positive_indices)
        raw_counts = [remaining * value for value in fractions]
        floor_counts = [int(value) for value in raw_counts]
        for index, count in enumerate(floor_counts):
            counts[index] += count
        remainder = remaining - sum(floor_counts)
        order = sorted(
            range(len(fractions)),
            key=lambda index: (
                raw_counts[index] - floor_counts[index],
                fractions[index],
                -index,
            ),
            reverse=True,
        )
        for index in order[:remainder]:
            counts[index] += 1
    else:
        order = sorted(
            range(len(fractions)),
            key=lambda index: (fractions[index], -index),
            reverse=True,
        )
        for index in order[: len(node_ids)]:
            counts[index] = 1
    train_end = counts[0]
    validation_end = train_end + counts[1]
    train_nodes = tuple(sorted(str(value) for value in shuffled[:train_end]))
    validation_nodes = tuple(
        sorted(str(value) for value in shuffled[train_end:validation_end])
    )
    test_nodes = tuple(
        sorted(str(value) for value in shuffled[validation_end:])
    )
    partitions = {
        node_id: index
        for index, nodes in enumerate(
            (train_nodes, validation_nodes, test_nodes)
        )
        for node_id in nodes
    }
    retained: list[list[Edge]] = [[], [], []]
    withheld: list[Edge] = []
    for edge in graph.edges:
        source_partition = partitions[edge.source]
        target_partition = partitions[edge.target]
        if source_partition == target_partition:
            retained[source_partition].append(edge)
        else:
            withheld.append(edge)
    return InductiveNodeSplit(
        train_nodes=train_nodes,
        validation_nodes=validation_nodes,
        test_nodes=test_nodes,
        train_edges=tuple(retained[0]),
        validation_edges=tuple(retained[1]),
        test_edges=tuple(retained[2]),
        withheld_edges=tuple(withheld),
        seed=seed,
    )


def structural_degrees(graph: Dataset) -> dict[str, list[float]]:
    """Computes the out-degree, in-degree and weighted degree of each node.

    Args:
        graph: The graph.

    Returns:
        For each node id, the three degrees in that order; the weighted
        degree is the summed weight of incident edges. Edges with an
        endpoint outside the node set are not counted at that endpoint.
    """
    degrees = {node_id: [0.0, 0.0, 0.0] for node_id in graph.node_ids}
    for edge in graph.edges:
        if edge.source in degrees:
            degrees[edge.source][0] += 1.0
            degrees[edge.source][2] += edge.weight
        if edge.target in degrees:
            degrees[edge.target][1] += 1.0
            degrees[edge.target][2] += edge.weight
    return degrees


def feature_values(
    rows: list[list[float]], width: int, *, standardize: bool
) -> tuple[tuple[float, ...], ...]:
    """Turns feature rows into the immutable values of a feature matrix.

    Args:
        rows: One list of features per node, already cut to ``width``.
        width: Number of columns kept.
        standardize: Whether to scale each column to zero mean and unit
            variance (constant columns are only centred).

    Returns:
        The values as nested tuples; empty rows when there is no row or no
        column.

    Raises:
        ValueError: If a feature value is not finite.
    """
    if not (rows and width):
        return tuple(() for row in rows)
    matrix = numpy.asarray(rows, dtype=numpy.float64)
    if not numpy.isfinite(matrix).all():
        raise ValueError("prepared features must be finite")
    if standardize:
        means = matrix.mean(axis=0)
        scales = matrix.std(axis=0)
        scales[scales == 0] = 1.0
        matrix = (matrix - means) / scales
    return tuple(tuple(float(value) for value in row) for row in matrix)


def prepare_features(
    graph: Dataset,
    max_nodes: int = 100_000,
    max_features: int = 512,
    standardize: bool = False,
    include_structural: bool = True,
) -> PreparedFeatures:
    """Prepares bounded numeric node features for a baseline model.

    User-provided features are followed by deterministic degree features when
    ``include_structural`` is true. Nodes and features are truncated in
    canonical order, and no graph-neural-network dependency is introduced.
    Structural features are out-degree, in-degree and weighted degree (the
    summed weight of incident edges).

    Args:
        graph: The graph; it must be valid.
        max_nodes: Largest number of nodes kept, in canonical order.
        max_features: Largest number of columns kept.
        standardize: Whether to scale each column to zero mean and unit
            variance (constant columns are only centred).
        include_structural: Whether to append the degree features.

    Returns:
        The prepared features. Without any column the rows are empty.

    Raises:
        ValueError: If ``max_nodes`` or ``max_features`` is below one, or a
            feature value is not finite. An invalid graph raises
            GraphValidationError from `Dataset.require_valid`.
    """
    graph.require_valid()
    if max_nodes < 1 or max_features < 1:
        raise ValueError("max_nodes and max_features must be at least one")
    nodes = graph.nodes[:max_nodes]
    feature_dimension = max((len(node.features) for node in nodes), default=0)
    names = list(graph.feature_names)
    if feature_dimension and not names:
        names = [f"feature_{index}" for index in range(feature_dimension)]
    rows = [list(node.features) for node in nodes]
    if include_structural:
        degrees = structural_degrees(graph)
        for index, node in enumerate(nodes):
            rows[index].extend(degrees[node.node_id])
        names.extend(("out_degree", "in_degree", "weighted_degree"))
    width = min(max_features, max((len(row) for row in rows), default=0))
    rows = [row[:width] for row in rows]
    names = names[:width]
    return PreparedFeatures(
        node_ids=tuple(node.node_id for node in nodes),
        feature_names=tuple(names),
        values=feature_values(rows, width, standardize=standardize),
        standardized=standardize,
    )


__all__ = [
    "Dataset",
    "Edge",
    "EdgeProfile",
    "GraphError",
    "GraphTemporalError",
    "GraphValidationError",
    "InductiveNodeSplit",
    "Issue",
    "Node",
    "NodeProfile",
    "PreparedFeatures",
    "Profile",
    "TemporalSplit",
    "Validation",
    "canonical_event_time",
    "edges_from_records",
    "graph_identity",
    "inductive_node_split",
    "prepare_features",
    "read_edges",
]
