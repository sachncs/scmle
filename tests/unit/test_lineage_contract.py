"""Tests for content-linked evidence graph contracts."""

import pytest

from scmle.contracts import lineage


def node(node_id: str, kind: lineage.EvidenceNodeKind) -> lineage.EvidenceNode:
    """Build a deterministic evidence node fixture."""
    return lineage.EvidenceNode(
        node_id=node_id,
        kind=kind,
        digest=f"sha256:{node_id}",
        attributes={"path": f"{node_id}.json"},
    )


def graph() -> lineage.EvidenceGraph:
    """Build a small source-to-bundle graph."""
    result = lineage.EvidenceGraph(
        (
            node("source", lineage.EvidenceNodeKind.RAW_SOURCE),
            node("snapshot", lineage.EvidenceNodeKind.SNAPSHOT),
            node("curation", lineage.EvidenceNodeKind.CURATION),
            node("bundle", lineage.EvidenceNodeKind.BUNDLE),
        )
    )
    result.add_edge(lineage.EvidenceEdge("source", "snapshot", "materializes"))
    result.add_edge(lineage.EvidenceEdge("snapshot", "curation", "curated"))
    result.add_edge(lineage.EvidenceEdge("curation", "bundle", "packages"))
    return result


def test_graph_is_deterministic_and_content_addressed():
    """Equivalent registration orders produce the same graph digest."""
    first = graph()
    second = lineage.EvidenceGraph(
        (
            node("bundle", lineage.EvidenceNodeKind.BUNDLE),
            node("curation", lineage.EvidenceNodeKind.CURATION),
            node("snapshot", lineage.EvidenceNodeKind.SNAPSHOT),
            node("source", lineage.EvidenceNodeKind.RAW_SOURCE),
        )
    )
    for edge in (
        lineage.EvidenceEdge("curation", "bundle", "packages"),
        lineage.EvidenceEdge("source", "snapshot", "materializes"),
        lineage.EvidenceEdge("snapshot", "curation", "curated"),
    ):
        second.add_edge(edge)

    assert first.path_exists("source", "bundle")
    assert first.topological_order() == (
        "source",
        "snapshot",
        "curation",
        "bundle",
    )
    assert first.digest() == second.digest()


def test_graph_round_trip_preserves_digest_and_order():
    """Serialized evidence can be validated and restored independently."""
    original = graph()

    restored = lineage.EvidenceGraph.from_dict(original.to_dict())

    assert restored.digest() == original.digest()
    assert restored.to_dict() == original.to_dict()


def test_graph_rejects_conflicts_dangling_edges_and_cycles():
    """Evidence cannot become ambiguous or cyclic through registration."""
    result = graph()
    with pytest.raises(ValueError, match="conflicting evidence node"):
        result.add_node(node("source", lineage.EvidenceNodeKind.SNAPSHOT))
    with pytest.raises(ValueError, match="unknown node"):
        result.add_edge(lineage.EvidenceEdge("bundle", "missing", "points"))
    with pytest.raises(ValueError, match="acyclic"):
        result.add_edge(lineage.EvidenceEdge("bundle", "source", "loops"))


def test_graph_rejects_invalid_node_and_edge_metadata():
    """Empty identities and self-loops fail at the contract boundary."""
    with pytest.raises(ValueError, match="digest"):
        lineage.EvidenceNode("source", lineage.EvidenceNodeKind.RAW_SOURCE, "")
    with pytest.raises(ValueError, match="self-loops"):
        lineage.EvidenceEdge("source", "source", "invalid")
