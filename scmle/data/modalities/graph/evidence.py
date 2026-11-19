"""Profiles, partitions and feature matrices produced from a graph."""

import dataclasses
from typing import Any
from typing import ClassVar

import numpy

from scmle.contracts import records
from scmle.data.modalities.graph.elements import Edge


@dataclasses.dataclass(frozen=True, slots=True)
class Degrees:
    """Summary statistics for graph nodes and their degrees.

    Attributes:
        nodes: Number of nodes.
        isolated: Number of nodes with degree zero.
        labeled: Number of nodes with a label.
        dimension: Longest feature vector among the nodes.
        degree_min: Smallest degree; 0 for an empty graph.
        degree_max: Largest degree; 0 for an empty graph.
        degree_mean: Mean degree; 0.0 for an empty graph.
    """

    nodes: int
    isolated: int
    labeled: int
    dimension: int
    degree_min: int
    degree_max: int
    degree_mean: float

    KEYS: ClassVar[records.Migration] = records.nest(
        {
            "node_count": "nodes",
            "isolated_node_count": "isolated",
            "labeled_node_count": "labeled",
            "feature_dimension": "dimension",
        },
        groups={
            "degree": {
                "degree_min": "low",
                "degree_max": "high",
                "degree_mean": "mean",
            }
        },
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible node profile.

        Returns:
            A dict of the profile fields in single-word keys, with the
            degree extremes and mean under `degree.{low, high, mean}`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Connectivity:
    """Summary statistics for graph edges and weights.

    Attributes:
        edges: Number of edges.
        duplicates: Number of edges beyond the first for each
            duplicate-detection key.
        loops: Number of edges whose source equals the target.
        weight_min: Smallest weight, or None without edges.
        weight_max: Largest weight, or None without edges.
        weight_mean: Mean weight, or None without edges.
    """

    edges: int
    duplicates: int
    loops: int
    weight_min: float | None
    weight_max: float | None
    weight_mean: float | None

    KEYS: ClassVar[records.Migration] = records.nest(
        {
            "edge_count": "edges",
            "duplicate_edge_count": "duplicates",
            "self_loop_count": "loops",
        },
        groups={
            "weight": {
                "weight_min": "low",
                "weight_max": "high",
                "weight_mean": "mean",
            }
        },
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible edge profile.

        Returns:
            A dict of the profile fields in single-word keys, with the
            weight extremes and mean under `weight.{low, high, mean}`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
    """Combined graph, node, and edge profile.

    Attributes:
        identity: SHA-256 identity of the graph content.
        directed: Whether the graph is directed.
        components: Number of weakly connected components.
        node: Node profile.
        edge: Edge profile.
    """

    identity: str
    directed: bool
    components: int
    node: Degrees
    edge: Connectivity

    KEYS: ClassVar[records.Migration] = records.nest(
        {"component_count": "components"},
        groups={"digests": {"graph_identity": "graph"}},
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible graph profile.

        Returns:
            A dict with the identity under `digests.graph`, direction,
            component count and the node and edge profiles as nested dicts.
        """
        return self.KEYS.upgrade(
            {
                "graph_identity": self.identity,
                "directed": self.directed,
                "component_count": self.components,
                "node": self.node.to_dict(),
                "edge": self.edge.to_dict(),
            }
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Partition:
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
    def divisions(self) -> tuple[tuple[str, ...], ...]:
        """The train, validation and test node partitions, in that order."""
        return (self.train_nodes, self.validation_nodes, self.test_nodes)


@dataclasses.dataclass(frozen=True, slots=True)
class Features:
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

    def materialize(self) -> numpy.ndarray:
        """Returns a writable copy of the prepared feature matrix.

        Returns:
            A float64 array of shape ``(nodes, features)``.
        """
        return numpy.asarray(self.values, dtype=numpy.float64).copy()
