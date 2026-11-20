"""The graph kind: labelled nodes through the shared stages.

The kind reads a directory with a node table and an edge file into one row
per node with as-of structural features, keys a timed graph's rows by when
each node is first seen so the latest nodes are held out (MM-11), and hands
the rows to the shared stages. The end-to-end runs register the kind and
its capabilities for the test, as `kinds.KINDS` and `roster.enrol` will
once the lead wires them.
"""

from datetime import datetime
from datetime import timedelta
from datetime import UTC
import json

import numpy
import pandas
import pytest

from scmle.api import project
from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.readers import tabular
from scmle.pipeline import kinds
from scmle.pipeline import profiler
from scmle.pipeline.kinds import graph
from scmle.pipeline.kinds import rows
from tests import support

START = datetime(2026, 1, 1, tzinfo=UTC)


def network(root, count=120, timed=True, seed=0):
    """A graph whose node label follows its feature and its neighbours."""
    generator = numpy.random.default_rng(seed)
    root.mkdir(parents=True)
    signal = generator.normal(size=count)
    labels = (signal > 0).astype(int)
    nodes = pandas.DataFrame(
        {
            "node": [f"{index:03d}" for index in range(count)],
            "signal": signal + generator.normal(0, 0.3, count),
            "label": labels,
        }
    )
    nodes.to_csv(root / "nodes.csv", index=False)
    edges = []
    for index in range(1, count):
        for other in generator.choice(index, min(index, 2), replace=False):
            record = {"source": f"{index:03d}", "target": f"{other:03d}"}
            if timed:
                record["time"] = (START + timedelta(hours=index)).isoformat()
            edges.append(record)
    pandas.DataFrame(edges).to_csv(root / "edges.csv", index=False)
    return root


def config(uri, target="label", **sections):
    return tasks.Config(
        data=tasks.Dataset(uri=str(uri), target=target, modality="graph"),
        **sections,
    )


def test_the_graph_kind_is_a_kind():
    assert isinstance(graph.GRAPH, kinds.Kind)
    assert graph.GRAPH.name == "graph"


def test_nodes_become_rows_with_structure_and_their_first_seen_key(tmp_path):
    root = network(tmp_path / "g", count=10)
    frame, stream = graph.GRAPH.read(config(root))
    assert stream is None
    assert len(frame) == 10
    assert {"signal", "label", *graph.STRUCTURE, rows.KEY} == set(frame.columns)
    assert frame[rows.KEY].iloc[0] == pandas.Timestamp(
        START + timedelta(hours=1)
    )
    assert frame[rows.KEY].is_monotonic_increasing


def test_structure_counts_only_the_edges_known_when_a_node_is_seen(tmp_path):
    root = tmp_path / "g"
    root.mkdir()
    pandas.DataFrame({"node": ["a", "b", "c"], "label": [0, 1, 0]}).to_csv(
        root / "nodes.csv", index=False
    )
    pandas.DataFrame(
        {
            "source": ["a", "b", "a"],
            "target": ["b", "c", "c"],
            "weight": [1.0, 2.0, 5.0],
            "time": [
                "2026-01-01T00:00:00Z",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00Z",
            ],
        }
    ).to_csv(root / "edges.csv", index=False)
    frame = graph.GRAPH.read(config(root))[0]
    kept = frame.set_index(frame["label"].index)
    # a and b are first seen on day 1: only a->b is known then.
    assert kept.loc[0, ["outdegree", "indegree", "strength"]].tolist() == [
        1.0,
        0.0,
        1.0,
    ]
    assert kept.loc[1, "neighbourhood"] == 1.0
    # c is first seen on day 2: b->c is known, the later a->c is not.
    assert kept.loc[2, ["outdegree", "indegree", "strength"]].tolist() == [
        0.0,
        1.0,
        2.0,
    ]
    assert kept.loc[2, "neighbourhood"] == 2.0


def test_an_untimed_graph_counts_every_edge_and_has_no_key(tmp_path):
    root = network(tmp_path / "g", count=10, timed=False)
    frame = graph.GRAPH.read(config(root))[0]
    assert rows.KEY not in frame.columns
    assert frame["indegree"].sum() == frame["outdegree"].sum()


def test_mm7_node_ids_keep_their_text_across_tables(tmp_path):
    root = tmp_path / "g"
    root.mkdir()
    (root / "nodes.csv").write_text(
        "node,label\n007,1\nNA,0\n7,1\n", encoding="utf-8"
    )
    (root / "edges.jsonl").write_text(
        '{"source": "007", "target": "NA"}\n{"source": "NA", "target": "7"}\n',
        encoding="utf-8",
    )
    frame = graph.GRAPH.read(config(root))[0]
    assert frame["outdegree"].tolist() == [1.0, 1.0, 0.0]
    assert frame["indegree"].tolist() == [0.0, 1.0, 1.0]


@pytest.mark.parametrize(
    ("nodes", "edges", "match"),
    [
        ("label\n1\n", "source,target\n", "needs a 'node' column"),
        ("node,label\na,1\na,0\n", "source,target\n", "repeat"),
        ("node,label\na,1\n,0\n", "source,target\n", "missing node id"),
        ("node,label\na,1\n", "source,target\na,b\n", "not in the node"),
        ("node,outdegree,label\na,1,1\n", "source,target\n", "clash"),
        ("node,label\na,1\n", "source,target\na,\n", "missing endpoint"),
    ],
)
def test_an_unusable_graph_is_refused_with_a_data_error(
    tmp_path, nodes, edges, match
):
    root = tmp_path / "g"
    root.mkdir()
    (root / "nodes.csv").write_text(nodes, encoding="utf-8")
    (root / "edges.csv").write_text(edges, encoding="utf-8")
    with pytest.raises(errors.DataError, match=match):
        graph.GRAPH.read(config(root))


def test_a_missing_directory_or_file_is_refused(tmp_path):
    with pytest.raises(errors.DataError, match="directory"):
        graph.GRAPH.read(config(tmp_path / "absent"))
    (tmp_path / "g").mkdir()
    with pytest.raises(errors.DataError, match="no nodes file"):
        graph.GRAPH.read(config(tmp_path / "g"))


def test_a_partly_timed_graph_asks_for_a_declared_node_time(tmp_path):
    root = tmp_path / "g"
    root.mkdir()
    (root / "nodes.csv").write_text(
        "node,label,seen\na,1,2026-01-01\nb,0,2026-01-02\nc,1,2026-01-03\n",
        encoding="utf-8",
    )
    (root / "edges.csv").write_text(
        "source,target,time\na,b,2026-01-01T00:00:00Z\nb,c,\n",
        encoding="utf-8",
    )
    with pytest.raises(errors.DataError, match=r"declare data\.time_column"):
        graph.GRAPH.read(config(root))
    spec = tasks.Config(
        data=tasks.Dataset(
            uri=str(root), target="label", modality="graph", time_column="seen"
        )
    )
    frame = graph.GRAPH.read(spec)[0]
    assert rows.KEY not in frame.columns
    assert frame["seen"].tolist() == ["2026-01-01", "2026-01-02", "2026-01-03"]


def test_mm11_the_split_holds_out_the_latest_seen_nodes(tmp_path):
    root = network(tmp_path / "g")
    spec = config(root)
    frame = graph.GRAPH.read(spec)[0]
    seen = frame[rows.KEY]
    split = graph.GRAPH.constrain(spec, tasks.Task.BINARY).split(frame)
    assert split.manifest.strategy == "temporal"
    assert rows.KEY not in split.features.columns
    assert rows.KEY not in split.test.features.columns
    train = seen.iloc[list(split.manifest.training)]
    test = seen.iloc[list(split.manifest.test)]
    assert train.max() < test.min()
    kept = numpy.asarray(split.manifest.training)
    for fit, check in split.manifest.folds:
        assert (
            seen.iloc[kept[list(fit)]].max()
            <= seen.iloc[kept[list(check)]].min()
        )


def test_an_untimed_graph_splits_as_declared(tmp_path):
    root = network(tmp_path / "g", timed=False)
    spec = config(root)
    frame = graph.GRAPH.read(spec)[0]
    split = graph.GRAPH.constrain(spec, tasks.Task.BINARY).split(frame)
    assert split.manifest.strategy == "stratified"


def test_the_survey_names_the_order_and_the_unconnected_nodes(tmp_path):
    root = network(tmp_path / "g", count=40)
    spec = config(root)
    frame = graph.GRAPH.read(spec)[0]
    manifest = tabular.manifested(frame, "label", str(root), "graph")
    portrait = graph.GRAPH.survey(spec, frame, manifest, tasks.Task.BINARY)
    assert rows.KEY not in portrait.statistics
    assert set(graph.STRUCTURE) & set(portrait.plan.numeric)
    notes = " ".join(portrait.quality)
    assert "latest-seen nodes are held out" in notes


def test_the_featuriser_cost_decoder_and_capabilities(tmp_path):
    assert graph.GRAPH.decode(None) is None
    assert graph.GRAPH.cost(1000, 4, 2, 600.0) == pytest.approx(
        profiler.estimate(1000, 4, 2, 600.0) + 1000 * graph.SECONDS
    )
    declared = graph.GRAPH.declare()
    assert {item.key.modality for item in declared} == {
        capabilities.Modality.GRAPH
    }
    assert {item.status for item in declared} == {
        capabilities.Status.EXPERIMENTAL
    }
    root = network(tmp_path / "g", count=20)
    frame = graph.GRAPH.read(config(root))[0]
    plan = (
        profiler.Profiler(config(root))
        .profile(
            frame.drop(columns=[rows.KEY]),
            tabular.manifested(frame, "label", str(root), "graph"),
            tasks.Task.BINARY,
        )
        .plan
    )
    values = graph.GRAPH.featurise(
        plan, scmle.contracts.data.Preprocessing()
    ).fit_transform(frame)
    assert values.shape[0] == 20


def test_a_graph_run_trains_tests_explains_and_packages(tmp_path, monkeypatch):
    root = network(tmp_path / "g", count=200)
    run = project.Project(
        str(root),
        "label",
        budget=120,
        modality="graph",
        models=support.optin(tasks.Task.BINARY),
        enforcement="cooperative",
        directory=tmp_path / "runs",
    ).run()
    assert run.state == "COMPLETED"
    manifest = json.loads(
        (run.directory / "validation_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["strategy"] == "temporal"
    assert len(set(run.plan.candidates)) > 2
    assert (run.directory / "explanations").is_dir()


def test_a_numeric_node_label_is_a_regression(tmp_path, monkeypatch):
    root = network(tmp_path / "g", count=150, timed=False)
    nodes = pandas.read_csv(root / "nodes.csv", dtype={"node": str})
    nodes["label"] = nodes["signal"] * 2.0 + 0.5
    nodes = nodes.rename(columns={"signal": "measure"})
    nodes.to_csv(root / "nodes.csv", index=False)
    run = project.Project(
        str(root),
        "label",
        budget=120,
        modality="graph",
        models=support.optin(tasks.Task.REGRESSION),
        enforcement="cooperative",
        directory=tmp_path / "runs",
    ).run()
    assert run.state == "COMPLETED"
    assert run.plan.task == "regression"
