"""Behavioural tests for the graph modality."""

from datetime import datetime
from datetime import timedelta
from datetime import timezone
from datetime import UTC

import numpy
import pytest

from scmle.contracts import errors
from scmle.data.modalities.graph import dataset
from scmle.data.modalities.graph import elements
from scmle.data.modalities.graph import evidence
from scmle.data.modalities.graph import files
from scmle.data.modalities.graph import identity
from scmle.data.modalities.graph import preparation
from scmle.data.modalities.graph import structure
from scmle.data.modalities.graph import survey


def edge(source, target, **kwargs):
    return elements.Edge(source, target, **kwargs)


def at(day):
    return datetime(2020, 1, day, tzinfo=UTC)


def chain(*pairs, **kwargs):
    return dataset.Dataset.connect([edge(a, b) for a, b in pairs], **kwargs)


def test_a_node_stores_float_features():
    node = elements.Node("a", (1, 2.5), label="x")
    assert node.features == (1.0, 2.5)
    assert all(type(value) is float for value in node.features)
    with pytest.raises(errors.InputError, match="non-empty string"):
        elements.Node(" ")
    with pytest.raises(errors.InputError, match="non-empty string"):
        elements.Node(1)
    with pytest.raises(errors.KindError, match="numeric"):
        elements.Node("a", ("1",))
    with pytest.raises(errors.InputError, match="finite"):
        elements.Node("a", (float("nan"),))


def test_an_edge_normalises_weight_and_time():
    shifted = datetime(2020, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    built = edge("a", "b", weight=2, event_time=shifted)
    assert built.weight == 2.0
    assert type(built.weight) is float
    assert built.event_time == at(1)
    assert built.event_time.utcoffset() == timedelta(0)


@pytest.mark.parametrize(
    ("kwargs", "kind", "message"),
    [
        ({"source": ""}, errors.InputError, "source"),
        ({"target": None}, errors.InputError, "target"),
        ({"weight": "1"}, errors.KindError, "weight must be numeric"),
        ({"weight": float("inf")}, errors.InputError, "finite"),
        ({"relation": " "}, errors.InputError, "relation"),
        ({"event_time": "2020"}, errors.KindError, "datetime"),
        (
            {"event_time": datetime.fromisoformat("2020-01-01")},
            errors.InputError,
            "aware",
        ),
    ],
)
def test_an_edge_rejects_bad_parts(kwargs, kind, message):
    arguments = {"source": "a", "target": "b"} | kwargs
    with pytest.raises(kind, match=message):
        elements.Edge(**arguments)


def test_a_chronology_detects_lookahead():
    early, late = (
        edge("a", "b", event_time=at(1)),
        edge("b", "c", event_time=at(3)),
    )
    fine = elements.Chronology(at(2), (early,), (late,))
    assert not fine.lookahead
    assert elements.Chronology(at(2), (early, late), ()).lookahead
    assert not elements.Chronology(at(3), (late,), ()).lookahead
    with pytest.raises(errors.InputError, match="cutoff"):
        elements.Chronology(datetime.fromisoformat("2020-01-01"), (), ())
    with pytest.raises(errors.InputError, match="train_edges"):
        elements.Chronology(at(1), (edge("a", "b"),), ())
    with pytest.raises(errors.InputError, match="future_edges"):
        elements.Chronology(at(1), (), (edge("a", "b"),))


def test_normalize_parses_event_times():
    assert elements.normalize("2020-01-01T00:00:00Z") == at(1)
    assert elements.normalize("2020-01-01T01:00:00+01:00") == at(1)
    assert elements.normalize(at(1)) is not None
    with pytest.raises(elements.TemporalError, match="invalid graph event"):
        elements.normalize("x")
    with pytest.raises(elements.TemporalError, match="datetime or text"):
        elements.normalize(5)
    with pytest.raises(elements.TemporalError, match="timezone-aware"):
        elements.normalize("2020-01-01T00:00:00")


def test_a_validation_report_summarises_issues():
    report = elements.Validation(
        (
            elements.Issue("b", "one", ("x",)),
            elements.Issue("a", "two"),
            elements.Issue("b", "three"),
        )
    )
    assert not report.is_valid
    assert report.error_codes == ("a", "b")
    assert report.to_dict() == {
        "valid": False,
        "codes": ["a", "b"],
        "issues": [
            {"code": "b", "message": "one", "affected": ["x"]},
            {"code": "a", "message": "two", "affected": []},
            {"code": "b", "message": "three", "affected": []},
        ],
    }
    error = elements.ValidationError(report)
    assert str(error) == "one; two; three"
    assert error.report is report
    assert elements.Validation(()).is_valid


def test_graph_identity_ignores_order_and_direction_when_undirected():
    nodes = [elements.Node("a"), elements.Node("b")]
    forward = identity.graph_identity(nodes, [edge("a", "b")], directed=False)
    backward = identity.graph_identity(
        nodes[::-1], [edge("b", "a")], directed=False
    )
    assert forward == backward
    assert len(forward) == 64
    directed = identity.graph_identity(nodes, [edge("a", "b")])
    assert directed != identity.graph_identity(nodes, [edge("b", "a")])
    assert identity.graph_identity(nodes, [], feature_names=["y", "x"]) == (
        identity.graph_identity(nodes, [], feature_names=["x", "y"])
    )
    timed = identity.graph_identity(nodes, [edge("a", "b", event_time=at(1))])
    assert timed != directed
    weighted = identity.graph_identity(nodes, [edge("a", "b", weight=2)])
    assert weighted != directed


def test_a_dataset_sorts_and_checks_shapes():
    graph = dataset.Dataset(
        nodes=(elements.Node("b", (1.0,)), elements.Node("a", (2.0,))),
        edges=(edge("b", "a"), edge("a", "b")),
        feature_names=("f",),
    )
    assert graph.node_ids == ("a", "b")
    assert [(e.source, e.target) for e in graph.edges] == [
        ("a", "b"),
        ("b", "a"),
    ]
    with pytest.raises(errors.InputError, match="unique"):
        dataset.Dataset(nodes=(elements.Node("a"), elements.Node("a")))
    with pytest.raises(errors.InputError, match="equal length"):
        dataset.Dataset(
            nodes=(elements.Node("a", (1.0,)), elements.Node("b", (1.0, 2.0)))
        )
    with pytest.raises(errors.InputError, match="feature dimension"):
        dataset.Dataset(
            nodes=(elements.Node("a", (1.0,)),), feature_names=("x", "y")
        )
    with pytest.raises(errors.InputError, match="empty names"):
        dataset.Dataset(feature_names=(" ",))


def test_connect_derives_nodes_from_endpoints():
    graph = chain(("b", "c"), ("a", "b"))
    assert graph.node_ids == ("a", "b", "c")
    given = dataset.Dataset.connect(
        [edge("a", "b")], nodes=["a", elements.Node("b", label=1)]
    )
    assert given.nodes[1].label == 1
    assert dataset.Dataset.connect([], directed=False).directed is False


def test_validate_finds_every_structural_issue():
    graph = dataset.Dataset.connect(
        [
            edge("a", "a"),
            edge("a", "z"),
            edge("a", "b"),
            edge("a", "b", weight=2),
        ],
        nodes=["a", "b"],
    )
    report = graph.validate()
    assert [issue.code for issue in report.issues] == [
        "self_loop",
        "missing_endpoint",
        "duplicate_edge",
    ]
    assert report.issues[1].affected == ("z",)
    assert report.issues[2].message == "edge a->b (default) occurs 2 times"
    with pytest.raises(elements.ValidationError):
        graph.ensure()
    missing_source = dataset.Dataset.connect([edge("q", "a")], nodes=["a"])
    assert "source 'q'" in missing_source.validate().issues[0].message


def test_canonicalize_sorts_endpoints_only_when_undirected():
    loose = chain(("b", "a"), directed=False)
    assert loose.canonicalize(edge("b", "a")) == ("a", "b", "default", "")
    strict = chain(("b", "a"))
    assert strict.canonicalize(edge("b", "a", event_time=at(1))) == (
        "b",
        "a",
        "default",
        "2020-01-01T00:00:00+00:00",
    )
    undirected = chain(("a", "b"), ("b", "a"), directed=False)
    assert [i.code for i in undirected.validate().issues] == ["duplicate_edge"]


def test_connected_components_are_weak_and_ordered():
    graph = dataset.Dataset.connect(
        [edge("c", "d"), edge("b", "a"), edge("a", "z")],
        nodes=["a", "b", "c", "d", "e"],
    )
    assert graph.connected_components() == (
        frozenset({"a", "b"}),
        frozenset({"c", "d"}),
        frozenset({"e"}),
    )
    assert isinstance(graph, structure.Structure)


def test_lines_reads_jsonl_objects(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text(
        '{"source": "a", "target": "b"}\n\n{"source": "b", "target": "c"}\n'
    )
    assert files.lines(path) == [
        {"source": "a", "target": "b"},
        {"source": "b", "target": "c"},
    ]
    path.write_text('{"a": 1}\n[1]\n')
    with pytest.raises(elements.ParseError, match="line 2 must be an object"):
        files.lines(path)
    path.write_text("{bad\n")
    with pytest.raises(elements.ParseError, match="malformed JSONL at line 1"):
        files.lines(path)
    with pytest.raises(elements.ParseError, match="failed to read"):
        files.lines(tmp_path)


def test_link_converts_records_with_defaults():
    edges = files.link(
        [
            {"source": 1, "target": 2},
            {"source": "a", "target": "b", "weight": "2.5", "relation": None},
            {"source": "a", "target": "c", "t": "2020-01-01T00:00:00Z"},
        ],
        time_column="t",
    )
    assert [(e.source, e.target, e.weight, e.relation) for e in edges] == [
        ("1", "2", 1.0, "default"),
        ("a", "b", 2.5, "default"),
        ("a", "c", 1.0, "default"),
    ]
    assert edges[2].event_time == at(1)
    assert edges[0].event_time is None


@pytest.mark.parametrize(
    ("record", "message"),
    [
        ({"source": "a"}, "record 1 must contain 'source' and 'target'"),
        ({"source": None, "target": "b"}, "missing endpoint"),
        ({"source": "a", "target": "b", "weight": None}, "invalid weight"),
        ({"source": "a", "target": "b", "weight": "x"}, "non-numeric weight"),
        ({"source": "a", "target": "b", "t": "bad"}, "invalid event time"),
        (
            {"source": "a", "target": "b", "weight": "inf"},
            "invalid edge record 1",
        ),
    ],
)
def test_link_rejects_bad_records(record, message):
    with pytest.raises(elements.ParseError, match=message):
        files.link([record], time_column="t")


def test_edgelist_reads_csv_and_jsonl(tmp_path):
    csv = tmp_path / "e.CSV"
    csv.write_text("from,to,w\na,b,2\nb,c,3\n")
    edges = files.edgelist(csv, "from", "to", "w")
    assert [(e.source, e.target, e.weight) for e in edges] == [
        ("a", "b", 2.0),
        ("b", "c", 3.0),
    ]
    ndjson = tmp_path / "e.ndjson"
    ndjson.write_text('{"source": "a", "target": "b"}\n')
    assert len(files.edgelist(ndjson)) == 1
    with pytest.raises(elements.ParseError, match="not found"):
        files.edgelist(tmp_path / "missing.csv")
    other = tmp_path / "e.txt"
    other.write_text("x")
    with pytest.raises(
        elements.ParseError, match=r"\.csv, \.jsonl, or \.ndjson"
    ):
        files.edgelist(other)
    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"source,target\n\xff\xfe,b\n")
    with pytest.raises(elements.ParseError, match="failed to read CSV"):
        files.edgelist(bad)


def test_from_edge_file_validates_the_graph(tmp_path):
    path = tmp_path / "e.jsonl"
    path.write_text('{"source": "a", "target": "b"}\n')
    graph = dataset.Dataset.from_edge_file(path)
    assert graph.source_uri == path.resolve().as_uri()
    assert graph.node_ids == ("a", "b")
    path.write_text('{"source": "a", "target": "a"}\n')
    with pytest.raises(elements.ValidationError, match="self-loop"):
        dataset.Dataset.from_edge_file(path)


def test_census_counts_degrees_and_isolated_nodes():
    graph = dataset.Dataset.connect(
        [edge("a", "b"), edge("a", "a")],
        nodes=[elements.Node("a", (1.0, 2.0), label=1), "b", "c"],
    )
    degrees = graph.census()
    assert degrees == evidence.Degrees(
        node_count=3,
        isolated_node_count=1,
        labeled_node_count=1,
        feature_dimension=2,
        degree_min=0,
        degree_max=2,
        degree_mean=1.0,
    )
    empty = dataset.Dataset().census()
    assert (empty.degree_min, empty.degree_max, empty.degree_mean) == (
        0,
        0,
        0.0,
    )


def test_profile_edges_counts_duplicates_loops_and_weights():
    graph = dataset.Dataset.connect(
        [
            edge("a", "b", weight=1),
            edge("a", "b", weight=3),
            edge("b", "b", weight=5),
        ]
    )
    profile = graph.profile_edges()
    assert profile == evidence.Connectivity(3, 1, 1, 1.0, 5.0, 3.0)
    assert graph.tally() == {
        ("a", "b", "default", ""): 2,
        ("b", "b", "default", ""): 1,
    }
    none = dataset.Dataset().profile_edges()
    assert (none.weight_min, none.weight_max, none.weight_mean) == (
        None,
        None,
        None,
    )


def test_profile_combines_identity_and_components():
    graph = chain(("a", "b"), ("c", "d"))
    profile = graph.profile()
    assert profile.graph_identity == graph.identity
    assert profile.component_count == 2
    record = profile.to_dict()
    assert record["digests"] == {"graph": graph.identity}
    assert record["components"] == 2
    assert record["node"]["degree"] == {"low": 1, "high": 1, "mean": 1.0}
    assert record["edge"]["weight"] == {"low": 1.0, "high": 1.0, "mean": 1.0}


def test_timeline_reports_temporal_completeness():
    assert survey.timeline(
        [edge("a", "b", event_time=at(2)), edge("b", "c", event_time=at(1))]
    ) == {
        "status": "complete",
        "event_time_min": "2020-01-01T00:00:00+00:00",
        "event_time_max": "2020-01-02T00:00:00+00:00",
    }
    assert survey.timeline(
        [edge("a", "b", event_time=at(1)), edge("b", "c")]
    ) == {
        "status": "mixed",
        "timed_edge_count": 1,
        "untimed_edge_count": 1,
    }
    assert survey.timeline([edge("a", "b")]) == {"status": "absent"}
    assert survey.timeline([]) == {"status": "absent"}


def test_curate_reports_findings_in_order():
    graph = dataset.Dataset.connect(
        [edge("a", "b", event_time=at(1)), edge("a", "b")],
        nodes=[elements.Node("a", label=1), "b", "c"],
    )
    report = graph.curate()
    assert report.modality == "graph"
    assert report.sample_count == 3
    assert report.dataset_hash == graph.identity
    assert report.profile["node"]["label_counts"] == {
        "int:1": 1,
        "unlabeled": 2,
    }
    assert report.profile["temporal"]["status"] == "mixed"
    assert [f.identifier for f in report.quality_findings] == [
        "quality.graph-mixed-temporal-state",
        "quality.graph-isolated-nodes",
        "quality.graph-no-declared-features",
    ]
    assert [f.identifier for f in report.risk_findings] == [
        "risk.graph-mixed-label-state"
    ]


def test_curate_flags_duplicate_edges_and_stays_quiet_when_clean():
    duplicated = chain(("a", "b"), ("a", "b"), feature_names=())
    names = [f.identifier for f in duplicated.curate().quality_findings]
    assert names[0] == "quality.graph-duplicate-edges"
    clean = dataset.Dataset(
        nodes=(elements.Node("a", (1.0,)), elements.Node("b", (2.0,))),
        edges=(edge("a", "b"),),
        feature_names=("f",),
    )
    report = clean.curate()
    assert report.quality_findings == ()
    assert report.risk_findings == ()


@pytest.mark.parametrize(
    ("fractions", "count", "expected"),
    [
        ((0.8, 0.1, 0.1), 10, [8, 1, 1]),
        ((0.8, 0.1, 0.1), 100, [80, 10, 10]),
        ((0.8, 0.1, 0.1), 3, [1, 1, 1]),
        ((0.8, 0.1, 0.1), 2, [1, 1, 0]),
        ((0.8, 0.1, 0.1), 0, [0, 0, 0]),
        ((1.0, 0.0, 0.0), 4, [4, 0, 0]),
        ((0.5, 0.5, 0.0), 5, [3, 2, 0]),
        ((0.2, 0.4, 0.4), 1, [0, 1, 0]),
        ((0.6, 0.2, 0.2), 7, [4, 2, 1]),
        ((0.6, 0.2, 0.2), 4, [2, 1, 1]),
        ((0.98, 0.01, 0.01), 5, [3, 1, 1]),
        ((0.6, 0.2, 0.2), 8, [5, 2, 1]),
    ],
)
def test_apportion_divides_by_largest_remainder(fractions, count, expected):
    assert preparation.apportion(fractions, count) == expected


@pytest.mark.parametrize(
    ("count", "expected"), [(10, [8, 1, 1]), (100, [80, 10, 10])]
)
def test_apportion_keeps_shares_proportional(count, expected):
    assert preparation.apportion((0.8, 0.1, 0.1), count) == expected


def test_induct_withholds_edges_between_partitions():
    graph = chain(*[(f"n{i}", f"n{i + 1}") for i in range(9)])
    partition = preparation.induct(graph, 0.8, 0.1, 0.1, seed=4)
    sizes = [len(nodes) for nodes in partition.divisions]
    assert sizes == [8, 1, 1]
    assert partition.seed == 4
    kept = (
        len(partition.train_edges)
        + len(partition.validation_edges)
        + len(partition.test_edges)
    )
    assert kept + len(partition.withheld_edges) == 9
    for retained in partition.train_edges:
        assert retained.source in partition.train_nodes
        assert retained.target in partition.train_nodes
    assert preparation.induct(graph, 0.8, 0.1, 0.1, seed=4) == partition
    assert list(partition.train_nodes) == sorted(partition.train_nodes)


def test_induct_rejects_bad_fractions_and_invalid_graphs():
    graph = chain(("a", "b"))
    with pytest.raises(errors.InputError, match="non-negative"):
        preparation.induct(graph, 1.1, -0.1, 0.0)
    with pytest.raises(errors.InputError, match="non-negative"):
        preparation.induct(graph, float("nan"), 0.0, 0.0)
    with pytest.raises(errors.InputError, match="sum to one"):
        preparation.induct(graph, 0.5, 0.1, 0.1)
    with pytest.raises(elements.ValidationError):
        preparation.induct(chain(("a", "a")))


def test_valence_counts_degrees_and_weights():
    graph = dataset.Dataset.connect(
        [edge("a", "b", weight=2), edge("b", "c", weight=3), edge("a", "z")],
        nodes=["a", "b", "c"],
    )
    assert preparation.valence(graph) == {
        "a": [2.0, 0.0, 3.0],
        "b": [1.0, 1.0, 5.0],
        "c": [0.0, 1.0, 3.0],
    }


def test_featurize_standardizes_columns():
    values = preparation.featurize(
        [[1.0, 5.0], [3.0, 5.0]], 2, standardize=True
    )
    assert values == ((-1.0, 0.0), (1.0, 0.0))
    raw = preparation.featurize([[1.0], [3.0]], 1, standardize=False)
    assert raw == ((1.0,), (3.0,))
    assert preparation.featurize([[], []], 0, standardize=True) == ((), ())
    assert preparation.featurize([], 3, standardize=True) == ()
    with pytest.raises(errors.InputError, match="finite"):
        preparation.featurize([[numpy.inf]], 1, standardize=False)


def test_prepare_features_appends_degrees_and_truncates():
    graph = dataset.Dataset(
        nodes=(elements.Node("a", (1.0,)), elements.Node("b", (2.0,))),
        edges=(edge("a", "b", weight=4),),
    )
    prepared = preparation.prepare_features(graph)
    assert prepared.feature_names == (
        "feature_0",
        "out_degree",
        "in_degree",
        "weighted_degree",
    )
    assert prepared.values == ((1.0, 1.0, 0.0, 4.0), (2.0, 0.0, 1.0, 4.0))
    assert prepared.shape == (2, 4)
    assert not prepared.standardized
    small = preparation.prepare_features(graph, max_nodes=1, max_features=2)
    assert small.node_ids == ("a",)
    assert small.values == ((1.0, 1.0),)
    assert small.feature_names == ("feature_0", "out_degree")
    plain = preparation.prepare_features(graph, include_structural=False)
    assert plain.values == ((1.0,), (2.0,))
    matrix = plain.to_numpy()
    matrix[0, 0] = 9.0
    assert plain.values[0] == (1.0,)
    bare = preparation.prepare_features(
        chain(("a", "b")), include_structural=False
    )
    assert bare.values == ((), ())
    assert bare.feature_names == ()
    with pytest.raises(errors.InputError, match="at least one"):
        preparation.prepare_features(graph, max_nodes=0)
    with pytest.raises(errors.InputError, match="at least one"):
        preparation.prepare_features(graph, max_features=0)
