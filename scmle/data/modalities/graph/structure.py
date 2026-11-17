"""Identity, validation and components of a graph."""

from collections import defaultdict
import math

from scmle.data.modalities.graph.elements import Edge
from scmle.data.modalities.graph.elements import Issue
from scmle.data.modalities.graph.elements import Node
from scmle.data.modalities.graph.elements import Validation
from scmle.data.modalities.graph.elements import ValidationError
from scmle.data.modalities.graph.identity import graph_identity


class Structure:
    """Identity, validation and components of a graph.

    Attributes:
        nodes: Nodes, sorted by identifier.
        edges: Edges, in canonical order.
        directed: Whether edges are directed.
        feature_names: Names of the node feature columns.
    """

    __slots__ = ()

    nodes: tuple[Node, ...]
    edges: tuple[Edge, ...]
    directed: bool
    feature_names: tuple[str, ...]

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
            key = self.canonicalize(edge)
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

    def ensure(self) -> None:
        """Raises `ValidationError` unless the graph is valid.

        Raises:
            ValidationError: If `validate` finds any issue.
        """
        report = self.validate()
        if not report.is_valid:
            raise ValidationError(report)

    def canonicalize(self, edge: Edge) -> tuple[str, str, str, str]:
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
