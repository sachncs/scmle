"""Tests for the dependency-free graph data foundation."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from scml.data.modalities.graph import edges_from_records
from scml.data.modalities.graph import graph_identity
from scml.data.modalities.graph import GraphData
from scml.data.modalities.graph import GraphEdge
from scml.data.modalities.graph import GraphInputError
from scml.data.modalities.graph import GraphNode
from scml.data.modalities.graph import GraphValidationError
from scml.data.modalities.graph import inductive_node_split
from scml.data.modalities.graph import prepare_features
from scml.data.modalities.graph import read_edges


@pytest.fixture
def valid_csv(tmp_path: Path) -> Path:
    """Returns a valid weighted CSV edge fixture."""
    path = tmp_path / "valid.csv"
    path.write_text(
        "source,target,weight\na,b,1.5\nb,c,2\nd,e,1\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def valid_jsonl(tmp_path: Path) -> Path:
    """Returns a valid JSONL edge fixture."""
    path = tmp_path / "valid.jsonl"
    records = [
        {"source": "a", "target": "b", "weight": 1.5},
        {"source": "b", "target": "c"},
    ]
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def empty_csv(tmp_path: Path) -> Path:
    """Returns an empty-but-well-formed edge fixture."""
    path = tmp_path / "empty.csv"
    path.write_text("source,target,weight\n", encoding="utf-8")
    return path


@pytest.fixture
def malformed_jsonl(tmp_path: Path) -> Path:
    """Returns a malformed JSONL fixture."""
    path = tmp_path / "malformed.jsonl"
    path.write_text('{"source": "a"\n', encoding="utf-8")
    return path


@pytest.fixture
def shifted_graph() -> GraphData:
    """Returns a feature-shifted graph fixture."""
    return GraphData.from_edges(
        [GraphEdge("a", "b"), GraphEdge("b", "c")],
        nodes=[
            GraphNode("a", (10.0, 20.0)),
            GraphNode("b", (11.0, 21.0)),
            GraphNode("c", (12.0, 22.0)),
        ],
        feature_names=("x", "y"),
    )


def test_reads_csv_and_jsonl_edges(valid_csv: Path, valid_jsonl: Path) -> None:
    """CSV and JSONL ingestion create immutable weighted edges."""
    csv_edges = read_edges(valid_csv)
    jsonl_edges = read_edges(valid_jsonl)

    assert csv_edges[0] == GraphEdge("a", "b", 1.5)
    assert jsonl_edges[1] == GraphEdge("b", "c")
    assert isinstance(csv_edges, tuple)


def test_empty_fixture_is_a_valid_empty_graph(empty_csv: Path) -> None:
    """An empty edge file remains valid and preserves explicit nodes."""
    graph = GraphData.from_edge_file(empty_csv, nodes=["orphan"])

    assert graph.edges == ()
    assert graph.node_ids == ("orphan",)
    assert graph.validate().is_valid


def test_malformed_fixture_fails_closed(malformed_jsonl: Path) -> None:
    """Malformed JSONL is reported as a bounded input error."""
    with pytest.raises(GraphInputError, match="malformed JSONL"):
        read_edges(malformed_jsonl)


def test_invalid_weight_fails_closed() -> None:
    """Non-numeric and non-finite weights cannot enter the graph contract."""
    with pytest.raises(GraphInputError, match="non-numeric"):
        edges_from_records([{"source": "a", "target": "b", "weight": "bad"}])
    with pytest.raises(GraphInputError, match="invalid weight"):
        edges_from_records([{"source": "a", "target": "b", "weight": np.nan}])
    with pytest.raises(ValueError, match="finite"):
        GraphEdge("a", "b", float("inf"))


def test_duplicate_self_loop_and_missing_node_validation() -> None:
    """Validation reports every requested structural error class."""
    duplicate = GraphData.from_edges(
        [GraphEdge("a", "b"), GraphEdge("a", "b")], nodes=["a", "b"]
    )
    self_loop = GraphData.from_edges([GraphEdge("a", "a")])
    missing_node = GraphData.from_edges(
        [GraphEdge("a", "missing")], nodes=["a"]
    )

    assert "duplicate_edge" in duplicate.validate().error_codes
    assert "self_loop" in self_loop.validate().error_codes
    assert "missing_endpoint" in missing_node.validate().error_codes
    with pytest.raises(GraphValidationError):
        missing_node.require_valid()


def test_undirected_duplicate_detection_and_identity_are_canonical() -> None:
    """Undirected reversals are duplicates and ordering does not affect identity."""
    first = GraphData.from_edges(
        [GraphEdge("a", "b"), GraphEdge("b", "a")],
        nodes=["b", "a"],
        directed=False,
    )
    second = GraphData.from_edges(
        [GraphEdge("b", "a"), GraphEdge("a", "b")],
        nodes=["a", "b"],
        directed=False,
    )

    assert first.identity == second.identity
    assert first.validate().error_codes == ("duplicate_edge",)
    assert (
        graph_identity(first.nodes, first.edges, directed=False)
        == first.identity
    )


def test_components_and_profiles_are_deterministic(
    shifted_graph: GraphData,
) -> None:
    """Components and node/edge profiles summarize valid graph evidence."""
    profile = shifted_graph.profile()

    assert shifted_graph.connected_components() == (frozenset({"a", "b", "c"}),)
    assert profile.component_count == 1
    assert profile.node.node_count == 3
    assert profile.node.feature_dimension == 2
    assert profile.edge.edge_count == 2
    assert profile.edge.weight_mean == 1.0


def test_graph_curation_records_structure_and_feature_policy(
    shifted_graph: GraphData,
) -> None:
    """Graph curation explains structure before model selection."""
    report = shifted_graph.curate()

    assert report.modality == "graph"
    assert report.sample_count == 3
    assert report.profile["node"]["feature_dimension"] == 2
    assert report.profile["node"]["label_counts"] == {"unlabeled": 3}
    assert report.quality_findings == ()
    assert report.release_ready


def test_graph_curation_reports_mixed_label_state():
    """Graph curation exposes mixed node labels before graph modelling."""
    graph = GraphData.from_edges(
        [GraphEdge("a", "b")],
        nodes=[GraphNode("a", label="source"), GraphNode("b")],
    )

    report = graph.curate()

    assert report.profile["node"]["label_counts"] == {
        "str:source": 1,
        "unlabeled": 1,
    }
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.graph-mixed-label-state",
    ]


def test_inductive_split_is_deterministic_and_withholds_cross_partition_edges(
    shifted_graph: GraphData,
) -> None:
    """Repeated seeded splits match and retain no crossing edges."""
    first = inductive_node_split(shifted_graph, seed=41)
    second = inductive_node_split(shifted_graph, seed=41)

    assert first == second
    assert set(first.train_nodes).isdisjoint(first.validation_nodes)
    assert set(first.train_nodes).isdisjoint(first.test_nodes)
    assert set(first.validation_nodes).isdisjoint(first.test_nodes)
    assert not first.has_edge_leakage()
    assert len(first.withheld_edges) == 2


def test_feature_preparation_is_bounded_and_shift_fixture_is_observable(
    shifted_graph: GraphData,
) -> None:
    """Baseline preparation is finite, bounded, and exposes feature shift."""
    prepared = prepare_features(
        shifted_graph,
        max_nodes=2,
        max_features=3,
        standardize=True,
    )
    unshifted = GraphData.from_edges(
        shifted_graph.edges,
        nodes=[
            GraphNode("a", (0.0, 0.0)),
            GraphNode("b", (1.0, 1.0)),
            GraphNode("c", (2.0, 2.0)),
        ],
        feature_names=("x", "y"),
    )
    baseline = prepare_features(unshifted, max_features=3)

    assert prepared.shape == (2, 3)
    assert prepared.feature_names == ("x", "y", "out_degree")
    assert np.isfinite(prepared.to_numpy()).all()
    assert shifted_graph.identity != unshifted.identity
    assert baseline.to_numpy().shape == (3, 3)


def test_immutable_contracts_reject_mutation() -> None:
    """Node and edge contracts are frozen value objects."""
    node = GraphNode("a")
    edge = GraphEdge("a", "b")

    with pytest.raises(dataclasses.FrozenInstanceError):
        node.node_id = "b"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        edge.weight = 2.0  # type: ignore[misc]
