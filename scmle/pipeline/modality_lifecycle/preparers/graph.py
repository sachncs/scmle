"""The graph node preparer."""

from __future__ import annotations

from typing import Any
from typing import cast
from typing import TYPE_CHECKING

import numpy

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import curation
from scmle.data.modalities import graph
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import scoring

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.linear_model import LogisticRegression


REMARKS = (
    (
        "The baseline is inductive node classification on declared "
        "node features."
    ),
    (
        "It excludes topology-derived features across partitions and "
        "does not claim graph-neural-network support."
    ),
)
OCCLUSION_LIMITATIONS = (
    (
        "The perturbation masks declared node features, not "
        "graph topology or neighbor identity."
    ),
    (
        "Topology remains review context because the baseline "
        "does not consume edges."
    ),
    "Score changes are associational and not causal.",
)
LIMITATIONS = (
    "The baseline uses declared node features, not graph topology.",
    "One-hop context is evidence for review, not GNN attribution.",
    "No causal or counterfactual graph claim is made.",
)


class GraphPreparer(curators.Preparer):
    """Adapter for inductive node classification on graph node features.

    Attributes:
        dataset: Validated graph with declared node features.
    """

    modality = modalities.Modality.GRAPH

    def __init__(self, dataset: graph.Dataset) -> None:
        """Binds a validated graph with declared node features.

        Args:
            dataset: Graph the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def digest(self) -> str:
        """The full graph identity."""
        return self.dataset.identity

    @property
    def samples(self) -> tuple[graph.Node, ...]:
        """The canonical graph nodes."""
        return self.dataset.nodes

    def curate(self) -> curation.Curation:
        """Builds topology, feature, and label evidence.

        Returns:
            The curation report for the graph.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> containers.Partitions:
        """Splits nodes inductively and retains partition-local edge evidence.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions of nodes, with the node IDs of
            each under `groups`.
        """
        result = graph.induct(self.dataset, seed=seed)
        by_id = {node.identifier: node for node in self.dataset.nodes}
        return containers.Partitions(
            tuple(by_id[node_id] for node_id in result.train_nodes),
            tuple(by_id[node_id] for node_id in result.validation_nodes),
            tuple(by_id[node_id] for node_id in result.test_nodes),
            {
                "train": result.train_nodes,
                "validation": result.validation_nodes,
                "test": result.test_nodes,
            },
        )

    def features(self, samples: Sequence[graph.Node]) -> containers.Batch:
        """Prepares declared node features without cross-partition topology.

        Args:
            samples: Nodes to featurize.

        Returns:
            Batch with one row per node, identified by its node ID, and one
            column per declared node feature.

        Raises:
            ValueError: If the graph declares no node features.
        """
        if not self.dataset.features:
            raise errors.InputError(
                "graph baseline requires declared node features"
            )
        rows = numpy.asarray(
            [sample.features for sample in samples], dtype=float
        )
        return containers.Batch(
            tuple(sample.identifier for sample in samples),
            self.dataset.features,
            rows,
        )

    def sample_id(self, sample: graph.Node) -> str:
        """Gives a node identity.

        Args:
            sample: Graph node.

        Returns:
            The node ID.
        """
        return sample.identifier

    def label(self, sample: graph.Node) -> object | None:
        """Gives the node label.

        Args:
            sample: Graph node.

        Returns:
            The node's label, or None if it has none.
        """
        return sample.label

    def neighbourhood(
        self,
        node_id: str,
        max_nodes: int = constants.MAX_GRAPH_CONTEXT_NODES,
        max_edges: int = constants.MAX_GRAPH_CONTEXT_EDGES,
    ) -> dict[str, object]:
        """Builds bounded, deterministic one-hop topology evidence.

        The result is review context for a node-feature baseline. It does not
        claim that the model used topology or that edge removal is a faithful
        graph-neural-network attribution method.

        Args:
            node_id: Node at the center of the subgraph.
            max_nodes: Maximum number of nodes returned, including the center.
            max_edges: Maximum number of edges returned.

        Returns:
            Mapping with the center node, the selected node IDs (center first,
            then neighbors in sorted order), the edges among them, the limits,
            flags for whether nodes or edges were truncated, and the observed
            counts of incident edges and neighbors.

        Raises:
            ValueError: If `node_id` is blank or unknown, or a limit is below 1.
        """
        if not node_id.strip():
            raise errors.InputError("node_id must not be empty")
        if max_nodes < 1:
            raise errors.InputError("max_nodes must be positive")
        if max_edges < 1:
            raise errors.InputError("max_edges must be positive")
        if node_id not in self.dataset.identifiers:
            raise errors.InputError(f"unknown graph node: {node_id}")

        incident_edges = tuple(
            edge
            for edge in self.dataset.edges
            if node_id in (edge.source, edge.target)
        )
        neighbor_ids = tuple(
            sorted(
                {
                    edge.target if edge.source == node_id else edge.source
                    for edge in incident_edges
                }
            )
        )
        selected_neighbors = neighbor_ids[: max_nodes - 1]
        selected_node_ids = (node_id, *selected_neighbors)
        selected_node_set = set(selected_node_ids)
        candidate_edges = tuple(
            edge
            for edge in incident_edges
            if edge.source in selected_node_set
            and edge.target in selected_node_set
        )
        selected_edges = candidate_edges[:max_edges]
        return {
            "center_node_id": node_id,
            "node_ids": list(selected_node_ids),
            "edges": [
                {
                    "source": edge.source,
                    "target": edge.target,
                    "weight": edge.weight,
                    "relation": edge.relation,
                }
                for edge in selected_edges
            ],
            "limits": {"max_nodes": max_nodes, "max_edges": max_edges},
            "truncated": {
                "nodes": len(neighbor_ids) > len(selected_neighbors),
                "edges": len(candidate_edges) > len(selected_edges),
            },
            "observed_counts": {
                "incident_edges": len(incident_edges),
                "neighbors": len(neighbor_ids),
            },
        }

    def elucidate(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],
        coefficients: numpy.ndarray,
    ) -> dict[str, object]:
        """Builds one-hop graph context and node-feature evidence.

        Args:
            sample: Graph node to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features.
            coefficients: Linear coefficients for the model's features, one per
                feature name.

        Returns:
            Mapping with the one-hop subgraph, the ten largest feature
            contributions (value times coefficient), the decision score and the
            change in score when each of up to 64 features is zeroed.

        Raises:
            TypeError: If `sample` is not a `Node`.
        """
        if not isinstance(sample, graph.Node):
            raise errors.KindError("sample must be a Node")
        subgraph = self.neighbourhood(sample.identifier)
        subgraph_node_ids = cast("list[str]", subgraph["node_ids"])
        neighbors = subgraph_node_ids[1:]
        edges = cast("list[dict[str, object]]", subgraph["edges"])
        values = numpy.asarray(sample.features, dtype=numpy.float64)
        decision = numpy.asarray(model.decision_function(values[None]))
        anchored = scoring.anchor(model, values[None], decision)
        feature_terms = scoring.rank(
            (name, float(values[index] * coefficients[index]))
            for index, name in enumerate(feature_names)
        )
        raw_features = knockouts(self, sample, model, feature_names, anchored)
        return {
            "status": "available",
            "method": "one_hop_graph_context_and_node_feature_contribution",
            "node_id": sample.identifier,
            "neighbors": neighbors,
            "edges": edges,
            "subgraph": subgraph,
            "feature_terms": [[name, value] for name, value in feature_terms],
            "decision_score": float(decision.reshape(-1)[0]),
            "model_uses_topology": False,
            "raw_input_occlusion": {
                "status": "available",
                "method": "raw_graph_node_feature_occlusion",
                "features": raw_features,
                "baseline": "zero replacement in source node features",
                "feature_space_only": False,
                "bounded_by": constants.MAX_RAW_GRAPH_FEATURES,
                "limitations": list(OCCLUSION_LIMITATIONS),
            },
            "limitations": list(LIMITATIONS),
        }

    def monitor(self, observed: curators.Preparer) -> dict[str, object]:
        """Builds graph identity and profile drift evidence.

        Args:
            observed: Adapter over the newly observed graph.

        Returns:
            Mapping with both graph identities and, per profile key, whether its
            value differs.

        Raises:
            TypeError: If `observed` is not a `GraphPreparer`.
        """
        if not isinstance(observed, GraphPreparer):
            raise errors.KindError("observed adapter must be a GraphPreparer")
        reference = self.dataset.profile().to_dict()
        current = observed.dataset.profile().to_dict()
        return {
            "reference_hash": self.dataset.identity,
            "observed_hash": observed.dataset.identity,
            "profile_delta": {
                key: current.get(key) != reference.get(key)
                for key in sorted(set(current) | set(reference))
            },
        }

    def remarks(self) -> tuple[str, ...]:
        """Describes graph lifecycle limits.

        Returns:
            Plain statements of what the baseline does and does not claim.
        """
        return REMARKS


def knockouts(
    preparer: GraphPreparer,
    sample: graph.Node,
    model: LogisticRegression,
    feature_names: Sequence[str],
    anchored: tuple[int, float],
) -> list[dict[str, object]]:
    """Scores the node with each of its first features zeroed in turn.

    Args:
        preparer: Preparer that featurizes the masked node.
        sample: Graph node to explain.
        model: Fitted classifier.
        feature_names: Names of the model's features.
        anchored: The tracked class and the unmasked node's score for it.

    Returns:
        For each of up to `MAX_RAW_GRAPH_FEATURES` features, its name and
        index and the score change (unmasked minus masked).
    """
    selected_class, original_score = anchored
    source_values = tuple(sample.features)
    raw_features: list[dict[str, object]] = []
    feature_limit = min(len(feature_names), constants.MAX_RAW_GRAPH_FEATURES)
    for index in range(feature_limit):
        masked_values = list(source_values)
        masked_values[index] = 0.0
        masked_sample = graph.Node(
            identifier=sample.identifier,
            features=tuple(masked_values),
            label=sample.label,
        )
        masked_features = preparer.features((masked_sample,)).values
        masked_score = scoring.score(model, masked_features, selected_class)
        raw_features.append(
            {
                "feature": feature_names[index],
                "feature_index": index,
                "score_delta_from_full": original_score - masked_score,
            }
        )
    return raw_features
