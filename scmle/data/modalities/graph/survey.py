"""Profiling and curation of a graph."""

from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence

import numpy

from scmle.contracts import data
from scmle.data.modalities.curation import Curation
from scmle.data.modalities.curation import flaw
from scmle.data.modalities.curation import tally
from scmle.data.modalities.graph.elements import Edge
from scmle.data.modalities.graph.evidence import Connectivity
from scmle.data.modalities.graph.evidence import Degrees
from scmle.data.modalities.graph.evidence import Profile
from scmle.data.modalities.graph.structure import Structure


def timeline(edges: Sequence[Edge]) -> dict[str, str | int]:
    """Summarises how completely the edges carry event times.

    Args:
        edges: The edges of a graph.

    Returns:
        Status ``"complete"`` with the earliest and latest event time when
        every edge is timed, ``"mixed"`` with the timed and untimed counts
        when only some are, and ``"absent"`` when none is.
    """
    temporal_edges = [edge for edge in edges if edge.time is not None]
    event_times = tuple(
        edge.time for edge in temporal_edges if edge.time is not None
    )
    if event_times and len(event_times) == len(edges):
        return {
            "status": "complete",
            "event_time_min": min(event_times).isoformat(),
            "event_time_max": max(event_times).isoformat(),
        }
    if temporal_edges:
        return {
            "status": "mixed",
            "timed_edge_count": len(temporal_edges),
            "untimed_edge_count": len(edges) - len(temporal_edges),
        }
    return {"status": "absent"}


def findings(
    graph: "Survey",
    profile: Profile,
    temporal: Mapping[str, object],
    labels: dict[str, int],
) -> tuple[tuple[data.Finding, ...], tuple[data.Finding, ...]]:
    """Builds the quality and risk findings of a graph curation.

    Args:
        graph: The graph being curated.
        profile: Its profile.
        temporal: Its temporal summary (see `timeline`).
        labels: Its node label counts.

    Returns:
        The quality findings (mixed temporal state, duplicate edges,
        isolated nodes, no declared features) and the risk findings (mixed
        label state).
    """
    quality_findings = []
    risk_findings = []
    if temporal["status"] == "mixed":
        quality_findings.append(
            flaw(
                "quality.graph-mixed-temporal-state",
                data.Severity.WARNING,
                {"source": graph.uri},
                {"temporal": temporal},
                "Supply event times for every edge or explicitly use a "
                "non-temporal graph capability.",
                blocking=False,
            )
        )
    if profile.edge.duplicates:
        quality_findings.append(
            flaw(
                "quality.graph-duplicate-edges",
                data.Severity.WARNING,
                {"source": graph.uri},
                {"duplicate_edge_count": (profile.edge.duplicates)},
                "Declare whether parallel edges are valid or deduplicate them.",
                blocking=False,
            )
        )
    if profile.node.isolated:
        quality_findings.append(
            flaw(
                "quality.graph-isolated-nodes",
                data.Severity.WARNING,
                {"source": graph.uri},
                {"isolated_node_count": profile.node.isolated},
                "Confirm isolated nodes are valid for the intended task.",
                blocking=False,
            )
        )
    if not graph.features:
        quality_findings.append(
            flaw(
                "quality.graph-no-declared-features",
                data.Severity.INFO,
                {"source": graph.uri},
                {"node_count": profile.node.nodes},
                "Declare node features or select a topology-only model "
                "explicitly.",
                blocking=False,
            )
        )
    if "unlabeled" in labels and len(labels) > 1:
        risk_findings.append(
            flaw(
                "risk.graph-mixed-label-state",
                data.Severity.WARNING,
                {"source": graph.uri},
                {"label_counts": labels},
                "Declare whether unlabeled nodes are valid for the "
                "intended graph task before training.",
                blocking=False,
            )
        )
    return tuple(quality_findings), tuple(risk_findings)


class Survey(Structure):
    """Profiling and curation of a graph.

    Attributes:
        source_uri: URI the graph was loaded from.
    """

    __slots__ = ()

    uri: str

    def census(self) -> Degrees:
        """Profiles node counts, labels, features, and undirected degree.

        A self-loop adds one to its node's degree.

        Returns:
            The node profile.
        """
        degrees = dict.fromkeys(self.identifiers, 0)
        for edge in self.edges:
            if edge.source in degrees:
                degrees[edge.source] += 1
            if edge.target in degrees and edge.target != edge.source:
                degrees[edge.target] += 1
        degree_values = tuple(degrees.values())
        return Degrees(
            nodes=len(self.nodes),
            isolated=sum(value == 0 for value in degree_values),
            labeled=sum(node.label is not None for node in self.nodes),
            dimension=max(
                (len(node.features) for node in self.nodes), default=0
            ),
            degree_min=min(degree_values, default=0),
            degree_max=max(degree_values, default=0),
            degree_mean=float(numpy.mean(degree_values))
            if degree_values
            else 0.0,
        )

    def inspect(self) -> Connectivity:
        """Profiles edge counts, duplicate edges, loops, and weights.

        Returns:
            The edge profile.
        """
        weights = tuple(edge.weight for edge in self.edges)
        duplicate_count = sum(
            max(0, count - 1) for count in self.tally().values()
        )
        return Connectivity(
            edges=len(self.edges),
            duplicates=duplicate_count,
            loops=sum(edge.source == edge.target for edge in self.edges),
            weight_min=min(weights) if weights else None,
            weight_max=max(weights) if weights else None,
            weight_mean=float(numpy.mean(weights)) if weights else None,
        )

    def tally(self) -> dict[tuple[str, str, str, str], int]:
        """Returns canonical edge multiplicities.

        Returns:
            A dict from `canonicalize` to the number of edges with that key.
        """
        counts: defaultdict[tuple[str, str, str, str], int] = defaultdict(int)
        for edge in self.edges:
            counts[self.canonicalize(edge)] += 1
        return dict(counts)

    def profile(self) -> Profile:
        """Returns the combined graph profile.

        Returns:
            The profile with identity, direction, component count, and node
            and edge profiles.
        """
        return Profile(
            identity=self.identity,
            directed=self.directed,
            components=len(self.components()),
            node=self.census(),
            edge=self.inspect(),
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
        labels = tally(node.label for node in self.nodes)
        profile_data = profile.to_dict()
        profile_data["node"]["label_counts"] = labels
        profile_data["temporal"] = timeline(self.edges)
        quality_findings, risk_findings = findings(
            self, profile, profile_data["temporal"], labels
        )
        return Curation(
            modality="graph",
            dataset_hash=self.identity,
            samples=len(self.nodes),
            profile=profile_data,
            findings=quality_findings,
            risk_findings=risk_findings,
        )
