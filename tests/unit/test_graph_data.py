"""Tests for the dependency-free graph data foundation."""

import dataclasses
from datetime import datetime
from datetime import UTC
import json
from pathlib import Path

import numpy
import pytest

from scmle.data.modalities.graph import Dataset
from scmle.data.modalities.graph import Edge
from scmle.data.modalities.graph import edgelist
from scmle.data.modalities.graph import identify
from scmle.data.modalities.graph import induct
from scmle.data.modalities.graph import link
from scmle.data.modalities.graph import Node
from scmle.data.modalities.graph import ParseError
from scmle.data.modalities.graph import prepare
from scmle.data.modalities.graph import ValidationError


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
def shifted_graph() -> Dataset:
    """Returns a feature-shifted graph fixture."""
    return Dataset.connect(
        [Edge("a", "b"), Edge("b", "c")],
        nodes=[
            Node("a", (10.0, 20.0)),
            Node("b", (11.0, 21.0)),
            Node("c", (12.0, 22.0)),
        ],
        feature_names=("x", "y"),
    )


def test_reads_csv_and_jsonl_edges(valid_csv: Path, valid_jsonl: Path) -> None:
    """CSV and JSONL ingestion create immutable weighted edges."""
    csv_edges = edgelist(valid_csv)
    jsonl_edges = edgelist(valid_jsonl)

    assert csv_edges[0] == Edge("a", "b", 1.5)
    assert jsonl_edges[1] == Edge("b", "c")
    assert isinstance(csv_edges, tuple)


def test_empty_fixture_is_a_valid_empty_graph(empty_csv: Path) -> None:
    """An empty edge file remains valid and preserves explicit nodes."""
    graph = Dataset.load(empty_csv, nodes=["orphan"])

    assert graph.edges == ()
    assert graph.identifiers == ("orphan",)
    assert graph.validate().valid


def test_malformed_fixture_fails_closed(malformed_jsonl: Path) -> None:
    """Malformed JSONL is reported as a bounded input error."""
    with pytest.raises(ParseError, match="malformed JSONL"):
        edgelist(malformed_jsonl)


def test_invalid_weight_fails_closed() -> None:
    """Non-numeric and non-finite weights cannot enter the graph contract."""
    with pytest.raises(ParseError, match="non-numeric"):
        link([{"source": "a", "target": "b", "weight": "bad"}])
    with pytest.raises(ParseError, match="invalid weight"):
        link([{"source": "a", "target": "b", "weight": numpy.nan}])
    with pytest.raises(ValueError, match="finite"):
        Edge("a", "b", float("inf"))


def test_duplicate_self_loop_and_missing_node_validation() -> None:
    """Validation reports every requested structural error class."""
    duplicate = Dataset.connect(
        [Edge("a", "b"), Edge("a", "b")], nodes=["a", "b"]
    )
    self_loop = Dataset.connect([Edge("a", "a")])
    missing_node = Dataset.connect([Edge("a", "missing")], nodes=["a"])

    assert "duplicate_edge" in duplicate.validate().codes
    assert "self_loop" in self_loop.validate().codes
    assert "missing_endpoint" in missing_node.validate().codes
    with pytest.raises(ValidationError):
        missing_node.ensure()


def test_undirected_duplicate_detection_and_identity_are_canonical() -> None:
    """Undirected reversals are duplicates and ordering does not affect identity."""
    first = Dataset.connect(
        [Edge("a", "b"), Edge("b", "a")],
        nodes=["b", "a"],
        directed=False,
    )
    second = Dataset.connect(
        [Edge("b", "a"), Edge("a", "b")],
        nodes=["a", "b"],
        directed=False,
    )

    assert first.identity == second.identity
    assert first.validate().codes == ("duplicate_edge",)
    assert identify(first.nodes, first.edges, directed=False) == first.identity


def test_components_and_profiles_are_deterministic(
    shifted_graph: Dataset,
) -> None:
    """Components and node/edge profiles summarize valid graph evidence."""
    profile = shifted_graph.profile()

    assert shifted_graph.components() == (frozenset({"a", "b", "c"}),)
    assert profile.components == 1
    assert profile.node.nodes == 3
    assert profile.node.dimension == 2
    assert profile.edge.edges == 2
    assert profile.edge.weight_mean == 1.0


def test_graph_curation_records_structure_and_feature_policy(
    shifted_graph: Dataset,
) -> None:
    """Graph curation explains structure before model selection."""
    report = shifted_graph.curate()

    assert report.modality == "graph"
    assert report.samples == 3
    assert report.profile["node"]["dimension"] == 2
    assert report.profile["node"]["label_counts"] == {"unlabeled": 3}
    assert report.findings == ()
    assert report.release


def test_graph_curation_reports_mixed_label_state():
    """Graph curation exposes mixed node labels before graph modelling."""
    graph = Dataset.connect(
        [Edge("a", "b")],
        nodes=[Node("a", label="source"), Node("b")],
    )

    report = graph.curate()

    assert report.profile["node"]["label_counts"] == {
        "str:source": 1,
        "unlabeled": 1,
    }
    assert [finding.identifier for finding in report.risk_findings] == [
        "risk.graph-mixed-label-state",
    ]


def test_temporal_edges_are_normalized_and_curated() -> None:
    """Timestamped edges are normalized to UTC and curated."""
    edges = link(
        [
            {
                "source": "a",
                "target": "b",
                "event_time": "2026-01-01T01:00:00+01:00",
            },
            {
                "source": "b",
                "target": "c",
                "event_time": "2026-01-02T00:00:00Z",
            },
            {
                "source": "c",
                "target": "a",
                "event_time": "2026-01-03T00:00:00+00:00",
            },
        ],
        time_column="event_time",
    )
    graph = Dataset.connect(edges, nodes=["a", "b", "c"])

    assert edges[0].time == datetime(2026, 1, 1, tzinfo=UTC)
    assert graph.curate().profile["temporal"]["status"] == "complete"


def test_temporal_graphs_fail_closed_on_mixed_or_invalid_time_state() -> None:
    """Missing and malformed event times remain visible before splitting."""
    mixed_graph = Dataset.connect(
        [
            Edge(
                "a",
                "b",
                time=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            Edge("b", "c"),
        ],
        nodes=["a", "b", "c"],
    )

    report = mixed_graph.curate()

    assert report.profile["temporal"] == {
        "status": "mixed",
        "timed_edge_count": 1,
        "untimed_edge_count": 1,
    }
    quality_ids = {finding.identifier for finding in report.findings}
    assert "quality.graph-mixed-temporal-state" in quality_ids
    with pytest.raises(ParseError, match="invalid event time"):
        link(
            [{"source": "a", "target": "b", "event_time": "no-zone"}],
            time_column="event_time",
        )


def test_temporal_edge_identity_preserves_parallel_events() -> None:
    """Events with distinct timestamps are not collapsed as duplicates."""
    graph = Dataset.connect(
        [
            Edge(
                "a",
                "b",
                time=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            Edge(
                "a",
                "b",
                time=datetime(2026, 1, 2, tzinfo=UTC),
            ),
        ],
        nodes=["a", "b"],
    )

    assert graph.validate().valid
    assert graph.profile().edge.duplicates == 0


def test_inductive_split_is_deterministic_and_withholds_cross_partition_edges(
    shifted_graph: Dataset,
) -> None:
    """Repeated seeded splits match and retain no crossing edges."""
    first = induct(shifted_graph, seed=41)
    second = induct(shifted_graph, seed=41)

    assert first == second
    assert set(first.train_nodes).isdisjoint(first.validation_nodes)
    assert set(first.train_nodes).isdisjoint(first.test_nodes)
    assert set(first.validation_nodes).isdisjoint(first.test_nodes)
    partitions = {
        node_id: index
        for index, nodes in enumerate(first.divisions)
        for node_id in nodes
    }
    retained = first.train_edges + first.validation_edges + first.test_edges
    assert all(
        partitions.get(edge.source) == partitions.get(edge.target)
        for edge in retained
    )
    assert len(first.withheld_edges) == 2


def test_feature_preparation_is_bounded_and_shift_fixture_is_observable(
    shifted_graph: Dataset,
) -> None:
    """Baseline preparation is finite, bounded, and exposes feature shift."""
    prepared = prepare(
        shifted_graph,
        max_nodes=2,
        max_features=3,
        standardize=True,
    )
    unshifted = Dataset.connect(
        shifted_graph.edges,
        nodes=[
            Node("a", (0.0, 0.0)),
            Node("b", (1.0, 1.0)),
            Node("c", (2.0, 2.0)),
        ],
        feature_names=("x", "y"),
    )
    baseline = prepare(unshifted, max_features=3)

    assert prepared.shape == (2, 3)
    assert prepared.feature_names == ("x", "y", "out_degree")
    assert numpy.isfinite(prepared.materialize()).all()
    assert shifted_graph.identity != unshifted.identity
    assert baseline.materialize().shape == (3, 3)


def test_immutable_contracts_reject_mutation() -> None:
    """Node and edge contracts are frozen value objects."""
    node = Node("a")
    edge = Edge("a", "b")

    with pytest.raises(dataclasses.FrozenInstanceError):
        node.identifier = "b"  # type: ignore[misc]
    with pytest.raises(dataclasses.FrozenInstanceError):
        edge.weight = 2.0  # type: ignore[misc]
