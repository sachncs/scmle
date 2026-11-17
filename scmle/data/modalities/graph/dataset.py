"""The immutable graph dataset."""

from collections.abc import Iterable
import dataclasses
from pathlib import Path

from scmle.contracts import errors
from scmle.data.modalities.graph.elements import Edge
from scmle.data.modalities.graph.elements import Node
from scmle.data.modalities.graph.files import edgelist
from scmle.data.modalities.graph.survey import Survey


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset(Survey):
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
            raise errors.InputError("node identifiers must be unique")
        dimensions = {len(node.features) for node in nodes if node.features}
        if len(dimensions) > 1:
            raise errors.InputError(
                "all populated node feature vectors must have equal length"
            )
        if (
            self.feature_names
            and dimensions
            and len(self.feature_names) != next(iter(dimensions))
        ):
            raise errors.InputError(
                "feature_names must match the node feature dimension"
            )
        if any(not name.strip() for name in self.feature_names):
            raise errors.InputError(
                "feature_names must not contain empty names"
            )
        object.__setattr__(self, "nodes", nodes)
        object.__setattr__(self, "edges", edges)
        object.__setattr__(self, "feature_names", tuple(self.feature_names))

    @classmethod
    def connect(
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
            on construction; call `ensure` for endpoint checks.
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
            from `edgelist` propagate; a graph that fails `validate`
            raises ValidationError.
        """
        edges = edgelist(
            path,
            source_column=source_column,
            target_column=target_column,
            weight_column=weight_column,
            relation_column=relation_column,
            time_column=time_column,
        )
        graph = cls.connect(
            edges,
            nodes=nodes,
            directed=directed,
            source_uri=Path(path).expanduser().resolve(strict=False).as_uri(),
        )
        graph.ensure()
        return graph
