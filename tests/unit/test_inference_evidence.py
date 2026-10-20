"""Tests for content-linked inference evidence."""

import json

import pytest

from scml.contracts import lineage
from scml.inference import evidence


def build_graph() -> lineage.EvidenceGraph:
    """Build the smallest valid source-to-bundle graph."""
    result = lineage.EvidenceGraph(
        (
            lineage.EvidenceNode(
                "source",
                lineage.EvidenceNodeKind.RAW_SOURCE,
                "sha256:source",
            ),
            lineage.EvidenceNode(
                "bundle",
                lineage.EvidenceNodeKind.BUNDLE,
                "sha256:bundle",
            ),
        )
    )
    result.add_edge(lineage.EvidenceEdge("source", "bundle", "packages"))
    return result


def write_graph(path, graph: lineage.EvidenceGraph) -> None:
    """Write a graph in the same shape produced by the linker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = graph.to_dict()
    record["graph_digest"] = graph.digest()
    path.write_text(json.dumps(record), encoding="utf-8")


def test_linker_persists_exact_bundle_link_and_is_idempotent(tmp_path):
    """An inference event is linked to the exact serving bundle once."""
    path = tmp_path / "evidence" / "graph.json"
    write_graph(path, build_graph())
    linker = evidence.InferenceEvidenceLinker()

    first = linker.link(
        path,
        event_id="event-1",
        input_fingerprint="sha256:input",
        output_fingerprint="sha256:output",
        bundle_digest="sha256:bundle",
        timestamp="2026-10-20T10:00:00+00:00",
    )
    second = linker.link(
        path,
        event_id="event-1",
        input_fingerprint="sha256:input",
        output_fingerprint="sha256:output",
        bundle_digest="sha256:bundle",
        timestamp="2026-10-20T10:00:00+00:00",
    )

    assert first == second
    restored = linker.load(path)
    assert (
        restored.nodes["inference:event-1"].kind
        is lineage.EvidenceNodeKind.INFERENCE
    )
    assert restored.path_exists("bundle", "inference:event-1")
    assert first.graph_digest == restored.digest()


def test_linker_rejects_conflicting_event_and_bundle_identity(tmp_path):
    """An event ID cannot be reused for different content or a new bundle."""
    path = tmp_path / "graph.json"
    write_graph(path, build_graph())
    linker = evidence.InferenceEvidenceLinker()
    arguments = {
        "event_id": "event-1",
        "input_fingerprint": "sha256:input",
        "output_fingerprint": "sha256:output",
        "bundle_digest": "sha256:bundle",
        "timestamp": "2026-10-20T10:00:00+00:00",
    }
    linker.link(path, **arguments)

    with pytest.raises(ValueError, match="conflicting evidence node"):
        linker.link(path, **{**arguments, "output_fingerprint": "sha256:other"})
    with pytest.raises(ValueError, match="does not match graph"):
        linker.link(path, **{**arguments, "bundle_digest": "sha256:new"})


def test_linker_validates_graph_integrity_and_event_timestamp(tmp_path):
    """Tampered graph content and timezone-free events fail closed."""
    path = tmp_path / "graph.json"
    write_graph(path, build_graph())
    record = json.loads(path.read_text(encoding="utf-8"))
    record["graph_digest"] = "sha256:tampered"
    path.write_text(json.dumps(record), encoding="utf-8")
    linker = evidence.InferenceEvidenceLinker()
    with pytest.raises(ValueError, match="digest does not match"):
        linker.load(path)

    write_graph(path, build_graph())
    with pytest.raises(ValueError, match="timezone"):
        linker.link(
            path,
            event_id="event-1",
            input_fingerprint="sha256:input",
            output_fingerprint="sha256:output",
            bundle_digest="sha256:bundle",
            timestamp="2026-10-20T10:00:00",
        )


def test_linker_rejects_missing_bundle(tmp_path):
    """An inference graph without a bundle cannot claim serving evidence."""
    path = tmp_path / "graph.json"
    graph = lineage.EvidenceGraph(
        (
            lineage.EvidenceNode(
                "source", lineage.EvidenceNodeKind.RAW_SOURCE, "sha256:source"
            ),
        )
    )
    write_graph(path, graph)

    with pytest.raises(ValueError, match="no bundle node"):
        evidence.InferenceEvidenceLinker().link(
            path,
            event_id="event-1",
            input_fingerprint="sha256:input",
            output_fingerprint="sha256:output",
            bundle_digest="sha256:bundle",
            timestamp="2026-10-20T10:00:00+00:00",
        )
