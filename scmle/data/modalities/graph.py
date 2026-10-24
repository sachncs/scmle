"""Dependency-free graph data contracts, ingestion, and baseline preparation.

This module intentionally stops at a data foundation. It represents a graph,
validates its structural invariants, profiles its contents, creates leakage-
safe inductive node splits, and prepares bounded numeric features. Graph
models, graph explainers, and graph serving are separate capability slices.
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

import numpy as np
import pandas as pd

from scmle.contracts import data as data_contracts
from scmle.data.modalities.curation import build_finding
from scmle.data.modalities.curation import label_counts
from scmle.data.modalities.curation import ModalityCurationReport


class GraphError(ValueError):
    """Base error for invalid graph data or graph input."""


class GraphInputError(GraphError):
    """Raised when a graph input file cannot be parsed."""


class GraphTemporalError(GraphError):
    """Raised when temporal graph evidence is incomplete or invalid."""


class GraphValidationError(GraphError):
    """Raised when a graph fails structural validation."""

    def __init__(self, report: "GraphValidationReport") -> None:
        """Builds an error containing the complete validation report."""
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.issues))


@dataclasses.dataclass(frozen=True, slots=True)
class GraphNode:
    """Immutable node with an optional numeric feature vector and label."""

    node_id: str
    features: tuple[float, ...] = ()
    label: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        """Validates the node identifier and numeric feature vector."""
        if not isinstance(self.node_id, str) or not self.node_id.strip():
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
class GraphEdge:
    """Immutable weighted edge between two node identifiers."""

    source: str
    target: str
    weight: float = 1.0
    relation: str = "default"
    event_time: datetime | None = None

    def __post_init__(self) -> None:
        """Validates endpoint, relation, and weight values."""
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("edge source must be a non-empty string")
        if not isinstance(self.target, str) or not self.target.strip():
            raise ValueError("edge target must be a non-empty string")
        if not isinstance(self.weight, (int, float)):
            raise TypeError("edge weight must be numeric")
        if not math.isfinite(float(self.weight)):
            raise ValueError("edge weight must be finite")
        if not isinstance(self.relation, str) or not self.relation.strip():
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
class TemporalGraphSplit:
    """Time-ordered graph edges separated at an explicit UTC cutoff."""

    cutoff: datetime
    train_edges: tuple[GraphEdge, ...]
    future_edges: tuple[GraphEdge, ...]

    def __post_init__(self) -> None:
        """Validate the cutoff and timestamp completeness of both partitions."""
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
        """Return whether a training edge occurs after the cutoff."""
        cutoff = self.cutoff.astimezone(UTC)
        return any(
            edge.event_time is not None and edge.event_time > cutoff
            for edge in self.train_edges
        )


def canonical_event_time(value: datetime | str) -> datetime:
    """Parse and normalize a timezone-aware graph event timestamp to UTC."""
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
class GraphValidationIssue:
    """One deterministic graph validation finding."""

    code: str
    message: str
    affected: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible issue mapping."""
        return {
            "code": self.code,
            "message": self.message,
            "affected": list(self.affected),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class GraphValidationReport:
    """Complete graph validation result."""

    issues: tuple[GraphValidationIssue, ...]

    @property
    def is_valid(self) -> bool:
        """Returns whether no validation issues were found."""
        return not self.issues

    @property
    def error_codes(self) -> tuple[str, ...]:
        """Returns unique issue codes in deterministic order."""
        return tuple(sorted({issue.code for issue in self.issues}))

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible report mapping."""
        return {
            "is_valid": self.is_valid,
            "error_codes": list(self.error_codes),
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class NodeProfile:
    """Summary statistics for graph nodes and their degrees."""

    node_count: int
    isolated_node_count: int
    labeled_node_count: int
    feature_dimension: int
    degree_min: int
    degree_max: int
    degree_mean: float

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible node profile."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class EdgeProfile:
    """Summary statistics for graph edges and weights."""

    edge_count: int
    duplicate_edge_count: int
    self_loop_count: int
    weight_min: float | None
    weight_max: float | None
    weight_mean: float | None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible edge profile."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class GraphProfile:
    """Combined graph, node, and edge profile."""

    graph_identity: str
    directed: bool
    component_count: int
    node: NodeProfile
    edge: EdgeProfile

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible graph profile."""
        return {
            "graph_identity": self.graph_identity,
            "directed": self.directed,
            "component_count": self.component_count,
            "node": self.node.to_dict(),
            "edge": self.edge.to_dict(),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class InductiveNodeSplit:
    """Leakage-safe node partitions and within-partition edge partitions."""

    train_nodes: tuple[str, ...]
    validation_nodes: tuple[str, ...]
    test_nodes: tuple[str, ...]
    train_edges: tuple[GraphEdge, ...]
    validation_edges: tuple[GraphEdge, ...]
    test_edges: tuple[GraphEdge, ...]
    withheld_edges: tuple[GraphEdge, ...]
    seed: int

    @property
    def node_partitions(self) -> tuple[tuple[str, ...], ...]:
        """Returns train, validation, and test node partitions."""
        return (self.train_nodes, self.validation_nodes, self.test_nodes)

    def has_edge_leakage(self) -> bool:
        """Returns whether any retained edge crosses node partitions."""
        partitions = {
            node_id: index
            for index, nodes in enumerate(self.node_partitions)
            for node_id in nodes
        }
        retained = self.train_edges + self.validation_edges + self.test_edges
        return any(
            partitions.get(edge.source) != partitions.get(edge.target)
            for edge in retained
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PreparedFeatures:
    """Bounded numeric features for a graph baseline."""

    node_ids: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: tuple[tuple[float, ...], ...]
    standardized: bool

    @property
    def shape(self) -> tuple[int, int]:
        """Returns the prepared matrix shape."""
        return (len(self.node_ids), len(self.feature_names))

    def to_numpy(self) -> np.ndarray:
        """Returns a writable copy of the prepared feature matrix."""
        return np.asarray(self.values, dtype=np.float64).copy()


@dataclasses.dataclass(frozen=True, slots=True)
class GraphData:
    """Immutable graph dataset with canonical node and edge ordering."""

    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    directed: bool = True
    feature_names: tuple[str, ...] = ()
    source_uri: str = "memory://"

    def __post_init__(self) -> None:
        """Normalizes collections and checks basic shape invariants."""
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
        edges: Iterable[GraphEdge],
        nodes: Iterable[GraphNode | str] | None = None,
        directed: bool = True,
        feature_names: Iterable[str] = (),
        source_uri: str = "memory://",
    ) -> "GraphData":
        """Builds a graph and derives nodes when no node collection is given."""
        edge_values = tuple(edges)
        if nodes is None:
            node_values = tuple(
                GraphNode(node_id)
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
                value if isinstance(value, GraphNode) else GraphNode(value)
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
        nodes: Iterable[GraphNode | str] | None = None,
        directed: bool = True,
        source_column: str = "source",
        target_column: str = "target",
        weight_column: str = "weight",
        relation_column: str = "relation",
        time_column: str | None = None,
    ) -> "GraphData":
        """Loads CSV or JSONL edges into a validated graph dataset."""
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
        """Returns canonical node identifiers."""
        return tuple(node.node_id for node in self.nodes)

    @property
    def identity(self) -> str:
        """Returns the full SHA-256 identity of graph content."""
        return graph_identity(
            self.nodes,
            self.edges,
            directed=self.directed,
            feature_names=self.feature_names,
        )

    def validate(self) -> GraphValidationReport:
        """Finds missing endpoints, duplicates, self-loops, and bad weights."""
        node_ids = set(self.node_ids)
        issues: list[GraphValidationIssue] = []
        duplicate_counts: defaultdict[tuple[str, str, str, str], int] = (
            defaultdict(int)
        )
        for edge in self.edges:
            if edge.source not in node_ids:
                issues.append(
                    GraphValidationIssue(
                        "missing_endpoint",
                        f"edge source '{edge.source}' is not a declared node",
                        (edge.source,),
                    )
                )
            if edge.target not in node_ids:
                issues.append(
                    GraphValidationIssue(
                        "missing_endpoint",
                        f"edge target '{edge.target}' is not a declared node",
                        (edge.target,),
                    )
                )
            if edge.source == edge.target:
                issues.append(
                    GraphValidationIssue(
                        "self_loop",
                        f"self-loop found on node '{edge.source}'",
                        (edge.source,),
                    )
                )
            if not math.isfinite(edge.weight):
                issues.append(
                    GraphValidationIssue(
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
                    GraphValidationIssue(
                        "duplicate_edge",
                        f"edge {key[0]}->{key[1]} ({key[2]}) occurs "
                        f"{count} times",
                        key,
                    )
                )
        return GraphValidationReport(tuple(issues))

    def require_valid(self) -> None:
        """Raises `GraphValidationError` unless the graph is valid."""
        report = self.validate()
        if not report.is_valid:
            raise GraphValidationError(report)

    def edge_key(self, edge: GraphEdge) -> tuple[str, str, str, str]:
        """Returns the canonical duplicate-detection key for an edge."""
        endpoints = (edge.source, edge.target)
        if not self.directed:
            endpoints = tuple(sorted(endpoints))  # type: ignore[assignment]
        event_time = edge.event_time.isoformat() if edge.event_time else ""
        return (endpoints[0], endpoints[1], edge.relation, event_time)

    def connected_components(self) -> tuple[frozenset[str], ...]:
        """Returns deterministic weakly connected components."""
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
        return tuple(sorted(components, key=lambda component: min(component)))

    def profile_nodes(self) -> NodeProfile:
        """Profiles node counts, labels, features, and undirected degree."""
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
            degree_mean=float(np.mean(degree_values)) if degree_values else 0.0,
        )

    def profile_edges(self) -> EdgeProfile:
        """Profiles edge counts, duplicate edges, loops, and weights."""
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
            weight_mean=float(np.mean(weights)) if weights else None,
        )

    def edge_key_counts(self) -> dict[tuple[str, str, str, str], int]:
        """Returns canonical edge multiplicities."""
        counts: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
        for edge in self.edges:
            counts[self.edge_key(edge)] += 1
        return dict(counts)

    def profile(self) -> GraphProfile:
        """Returns the combined graph profile."""
        return GraphProfile(
            graph_identity=self.identity,
            directed=self.directed,
            component_count=len(self.connected_components()),
            node=self.profile_nodes(),
            edge=self.profile_edges(),
        )

    def temporal_split(self, cutoff: datetime | str) -> TemporalGraphSplit:
        """Split fully timestamped edges into past and future partitions.

        Edges at the cutoff belong to training; edges after the cutoff belong
        to the future evaluation partition. Untimestamped edges fail closed
        because they cannot be assigned without inventing temporal order.
        """
        cutoff_time = canonical_event_time(cutoff)
        if any(edge.event_time is None for edge in self.edges):
            raise GraphTemporalError(
                "temporal splitting requires an event time on every edge"
            )
        train_edges = tuple(
            edge
            for edge in self.edges
            if edge.event_time is not None and edge.event_time <= cutoff_time
        )
        future_edges = tuple(
            edge
            for edge in self.edges
            if edge.event_time is not None and edge.event_time > cutoff_time
        )
        return TemporalGraphSplit(
            cutoff=cutoff_time,
            train_edges=train_edges,
            future_edges=future_edges,
        )

    def curate(self) -> ModalityCurationReport:
        """Build graph-structure evidence before choosing a graph model."""
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
                    data_contracts.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {"temporal": profile_data["temporal"]},
                    "Supply event times for every edge or explicitly use a "
                    "non-temporal graph capability.",
                    False,
                )
            )
        if profile.edge.duplicate_edge_count:
            quality_findings.append(
                build_finding(
                    "quality.graph-duplicate-edges",
                    data_contracts.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {
                        "duplicate_edge_count": (
                            profile.edge.duplicate_edge_count
                        )
                    },
                    "Declare whether parallel edges are valid or "
                    "deduplicate them.",
                    False,
                )
            )
        if profile.node.isolated_node_count:
            quality_findings.append(
                build_finding(
                    "quality.graph-isolated-nodes",
                    data_contracts.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {"isolated_node_count": profile.node.isolated_node_count},
                    "Confirm isolated nodes are valid for the intended task.",
                    False,
                )
            )
        if not self.feature_names:
            quality_findings.append(
                build_finding(
                    "quality.graph-no-declared-features",
                    data_contracts.FindingSeverity.INFO,
                    {"source": self.source_uri},
                    {"node_count": profile.node.node_count},
                    "Declare node features or select a topology-only model "
                    "explicitly.",
                    False,
                )
            )
        if "unlabeled" in labels and len(labels) > 1:
            risk_findings.append(
                build_finding(
                    "risk.graph-mixed-label-state",
                    data_contracts.FindingSeverity.WARNING,
                    {"source": self.source_uri},
                    {"label_counts": labels},
                    "Declare whether unlabeled nodes are valid for the "
                    "intended graph task before training.",
                    False,
                )
            )
        return ModalityCurationReport(
            modality="graph",
            dataset_hash=self.identity,
            sample_count=len(self.nodes),
            profile=profile_data,
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
        )


def graph_identity(
    nodes: Iterable[GraphNode],
    edges: Iterable[GraphEdge],
    directed: bool = True,
    feature_names: Iterable[str] = (),
) -> str:
    """Returns a canonical SHA-256 identity independent of input ordering."""
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
) -> tuple[GraphEdge, ...]:
    """Reads CSV or JSONL edge records with bounded schema handling."""
    input_path = Path(path)
    if not input_path.exists():
        raise GraphInputError(f"graph edge file not found: {input_path}")
    suffix = input_path.name.lower()
    if suffix.endswith(".csv"):
        try:
            frame = pd.read_csv(input_path)
        except (OSError, pd.errors.ParserError, UnicodeDecodeError) as error:
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
    if suffix.endswith(".jsonl") or suffix.endswith(".ndjson"):
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


def edges_from_records(
    records: Iterable[Mapping[str, Any]],
    source_column: str = "source",
    target_column: str = "target",
    weight_column: str = "weight",
    relation_column: str = "relation",
    time_column: str | None = None,
) -> tuple[GraphEdge, ...]:
    """Converts mappings into immutable edges with deterministic errors."""
    edges: list[GraphEdge] = []
    for index, record in enumerate(records, start=1):
        if source_column not in record or target_column not in record:
            raise GraphInputError(
                f"edge record {index} must contain '{source_column}' and "
                f"'{target_column}'"
            )
        source = record[source_column]
        target = record[target_column]
        if pd.isna(source) or pd.isna(target):
            raise GraphInputError(f"edge record {index} has a missing endpoint")
        weight_value = record.get(weight_column, 1.0)
        if pd.isna(weight_value):
            raise GraphInputError(f"edge record {index} has an invalid weight")
        try:
            weight = float(weight_value)
        except (TypeError, ValueError) as error:
            raise GraphInputError(
                f"edge record {index} has a non-numeric weight"
            ) from error
        relation_value = record.get(relation_column, "default")
        if pd.isna(relation_value):
            relation_value = "default"
        event_time = None
        if time_column is not None:
            time_value = record.get(time_column)
            if time_value is not None and not pd.isna(time_value):
                try:
                    event_time = canonical_event_time(time_value)
                except GraphTemporalError as error:
                    raise GraphInputError(
                        f"edge record {index} has an invalid event time"
                    ) from error
        try:
            edges.append(
                GraphEdge(
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
    graph: GraphData,
    train_fraction: float = 0.8,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.1,
    seed: int = 0,
) -> InductiveNodeSplit:
    """Creates deterministic node partitions with no retained edge leakage."""
    graph.require_valid()
    fractions = (train_fraction, validation_fraction, test_fraction)
    if any(value < 0 or not math.isfinite(value) for value in fractions):
        raise ValueError("split fractions must be finite and non-negative")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to one")
    node_ids = np.asarray(graph.node_ids, dtype=object)
    generator = np.random.default_rng(seed)
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
    retained: list[list[GraphEdge]] = [[], [], []]
    withheld: list[GraphEdge] = []
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


def prepare_features(
    graph: GraphData,
    max_nodes: int = 100_000,
    max_features: int = 512,
    standardize: bool = False,
    include_structural: bool = True,
) -> PreparedFeatures:
    """Prepares bounded numeric node features for a baseline model.

    User-provided features are followed by deterministic degree features when
    ``include_structural`` is true. Nodes and features are truncated in
    canonical order, and no graph-neural-network dependency is introduced.
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
        degrees = {node_id: [0.0, 0.0, 0.0] for node_id in graph.node_ids}
        for edge in graph.edges:
            if edge.source in degrees:
                degrees[edge.source][0] += 1.0
                degrees[edge.source][2] += edge.weight
            if edge.target in degrees:
                degrees[edge.target][1] += 1.0
                degrees[edge.target][2] += edge.weight
        for index, node in enumerate(nodes):
            rows[index].extend(degrees[node.node_id])
        names.extend(("out_degree", "in_degree", "weighted_degree"))
    width = min(max_features, max((len(row) for row in rows), default=0))
    rows = [row[:width] for row in rows]
    names = names[:width]
    if rows and width:
        matrix = np.asarray(rows, dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("prepared features must be finite")
        if standardize:
            means = matrix.mean(axis=0)
            scales = matrix.std(axis=0)
            scales[scales == 0] = 1.0
            matrix = (matrix - means) / scales
        values = tuple(tuple(float(value) for value in row) for row in matrix)
    else:
        values = tuple(() for row in rows)
    return PreparedFeatures(
        node_ids=tuple(node.node_id for node in nodes),
        feature_names=tuple(names),
        values=values,
        standardized=standardize,
    )


def load_graph(
    path: str | Path,
    nodes: Iterable[GraphNode | str] | None = None,
    directed: bool = True,
    time_column: str | None = None,
) -> GraphData:
    """Loads a CSV or JSONL graph using the default edge column names."""
    return GraphData.from_edge_file(
        path,
        nodes=nodes,
        directed=directed,
        time_column=time_column,
    )


__all__ = [
    "EdgeProfile",
    "GraphData",
    "GraphEdge",
    "GraphError",
    "GraphNode",
    "GraphProfile",
    "GraphTemporalError",
    "GraphValidationError",
    "GraphValidationIssue",
    "GraphValidationReport",
    "InductiveNodeSplit",
    "NodeProfile",
    "PreparedFeatures",
    "TemporalGraphSplit",
    "canonical_event_time",
    "edges_from_records",
    "graph_identity",
    "inductive_node_split",
    "load_graph",
    "prepare_features",
    "read_edges",
]
