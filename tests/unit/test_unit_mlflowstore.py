"""Behavioural tests for the MLflow store, run against an in-memory client.

MLflow is an optional extra; the stand-in client below records every call
and answers the tag searches the store issues, so the mapping from scmle
records to MLflow runs is checked whether or not MLflow is installed.
"""

import json
import pathlib
import re
import sys
import types

import pytest

from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.storage import mlflow_store

CLAUSE = re.compile(r"tags\.`([^`]+)` = '([^']*)'")


class Run:
    def __init__(self, run_id, name, tags, start):
        self.info = types.SimpleNamespace(run_id=run_id, start_time=start)
        self.data = types.SimpleNamespace(tags=dict(tags), metrics={})
        self.name = name
        self.params = {}
        self.texts = {}
        self.status = "RUNNING"


class Client:
    """The parts of `mlflow.tracking.MlflowClient` the store calls."""

    experiments: dict = {}
    created: list = []

    def __init__(self, tracking_uri=None):
        self.tracking_uri = tracking_uri
        self.runs = {}
        self.searches = 0

    def get_experiment_by_name(self, name):
        found = Client.experiments.get(name)
        return (
            None
            if found is None
            else types.SimpleNamespace(experiment_id=found)
        )

    def create_experiment(self, name, artifact_location=None):
        Client.experiments[name] = f"e{len(Client.experiments)}"
        Client.created.append((name, artifact_location))
        return Client.experiments[name]

    def create_run(self, experiment_id, run_name=None, tags=None):
        run = Run(f"r{len(self.runs)}", run_name, tags or {}, len(self.runs))
        self.runs[run.info.run_id] = run
        return run

    def set_tag(self, run_id, key, value):
        self.runs[run_id].data.tags[key] = value

    def log_metric(self, run_id, key, value):
        self.runs[run_id].data.metrics[key] = value

    def log_param(self, run_id, key, value):
        self.runs[run_id].params[key] = value

    def log_text(self, run_id, text, path):
        self.runs[run_id].texts[path] = text

    def set_terminated(self, run_id, status):
        self.runs[run_id].status = status

    def search_runs(self, experiment_ids, filter_string, order_by=None):
        self.searches += 1
        wanted = dict(CLAUSE.findall(filter_string))
        found = [
            run
            for run in self.runs.values()
            if all(run.data.tags.get(k) == v for k, v in wanted.items())
        ]
        if order_by:
            found.sort(key=lambda run: run.info.start_time, reverse=True)
        return found


@pytest.fixture
def library(monkeypatch):
    uris = []
    Client.experiments = {}
    Client.created = []
    tracking = types.ModuleType("mlflow.tracking")
    tracking.MlflowClient = Client
    module = types.ModuleType("mlflow")
    module.set_tracking_uri = uris.append
    module.tracking = tracking
    monkeypatch.setitem(sys.modules, "mlflow", module)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tracking)
    return uris


def make(library, uri=None, experiment="scmle"):
    return mlflow_store.MlflowExperimentStore(uri, experiment)


def trial(identifier="t1", status="ok", score=0.5, **kwargs):
    return experiments.Trial(
        trial_id=identifier,
        model_name="ridge",
        status=experiments.Termination(status),
        metric="auc",
        score_mean=score,
        score_std=0.1,
        fit_seconds=2.0,
        **kwargs,
    )


def test_the_store_needs_mlflow(monkeypatch):
    monkeypatch.setitem(sys.modules, "mlflow", None)
    with pytest.raises(errors.ConfigError, match="'tracking' extra"):
        mlflow_store.MlflowExperimentStore()


def test_the_store_creates_or_reuses_its_experiment(library, tmp_path):
    uri = f"sqlite:///{tmp_path}/m.db"
    store = make(library, uri)
    assert library == [uri]
    assert store.experiment_id == "e0"
    assert store.client.tracking_uri == uri
    assert Client.created == [
        ("scmle", (tmp_path / "m-artifacts").resolve().as_uri())
    ]
    again = make(library, None)
    assert again.experiment_id == "e0"
    assert library == [uri]
    assert len(Client.created) == 1
    assert again.parents == {}


def test_locate_puts_artifacts_beside_a_sqlite_file(tmp_path):
    locate = mlflow_store.MlflowExperimentStore.locate
    assert locate(None) is None
    assert locate("http://server") is None
    assert locate("sqlite://relative.db") is None
    assert locate(f"sqlite:///{tmp_path}/x/runs.db") == (
        pathlib.Path(tmp_path, "x", "runs-artifacts").resolve().as_uri()
    )


def test_commence_creates_the_parent_once(library):
    store = make(library)
    store.commence("run1", "hash", "/dir", "2026-01-01")
    parent = store.parents["run1"]
    run = store.client.runs[parent]
    assert run.name == "run1"
    assert run.data.tags == {
        "scmle.run_id": "run1",
        "scmle.kind": "run",
        "scmle.config_hash": "hash",
        "scmle.run_dir": "/dir",
        "scmle.started_at": "2026-01-01",
        "scmle.state": "CREATED",
    }
    store.client.set_tag(parent, "scmle.state", "RUNNING")
    store.commence("run1", "hash", "/dir", "2026-01-01")
    assert len(store.client.runs) == 1
    assert run.data.tags["scmle.state"] == "CREATED"


def test_climb_finds_a_parent_from_another_process(library):
    first = make(library)
    first.commence("run1", "h", "d", "s")
    second = make(library)
    second.client = first.client
    assert second.climb("run1") == first.parents["run1"]
    searches = second.client.searches
    assert second.climb("run1") == first.parents["run1"]
    assert second.client.searches == searches
    assert second.climb("missing") is None
    assert "missing" not in second.parents


def test_revise_maps_fields_to_metrics_tags_and_artifacts(library):
    store = make(library)
    store.commence("run1", "h", "d", "s")
    store.revise(
        "run1",
        cv_score="0.75",
        test_metrics=json.dumps({"auc": 0.9, "loss": float("nan")}),
        config_json={"a": 1},
        artifacts="list",
        dataset={"rows": 3},
        task="binary",
        state="RUNNING",
    )
    run = store.client.runs[store.parents["run1"]]
    assert run.data.metrics == {"cv_score": 0.75, "test.auc": 0.9}
    assert run.texts == {
        "config_json.json": '{"a": 1}',
        "artifacts.json": "list",
    }
    assert run.data.tags["scmle.dataset"] == '{"rows": 3}'
    assert run.data.tags["scmle.task"] == "binary"
    assert run.data.tags["scmle.state"] == "RUNNING"
    assert run.status == "RUNNING"
    store.revise("run1", cv_score=None, test_metrics={"f1": 1})
    assert run.data.metrics["test.f1"] == 1.0
    assert run.data.metrics["cv_score"] == 0.75


@pytest.mark.parametrize(
    ("state", "status"),
    [
        ("COMPLETED", "FINISHED"),
        ("FAILED", "FAILED"),
        ("CANCELLED", "KILLED"),
        ("BLOCKED", "FAILED"),
    ],
)
def test_revise_ends_the_run_on_a_terminal_state(library, state, status):
    store = make(library)
    store.commence("run1", "h", "d", "s")
    store.revise("run1", state=state)
    assert store.client.runs[store.parents["run1"]].status == status


def test_revise_and_transcribe_ignore_unknown_runs(library):
    store = make(library)
    store.revise("ghost", state="COMPLETED")
    store.transcribe("ghost", trial())
    assert store.client.runs == {}


def test_transcribe_logs_a_child_run(library):
    store = make(library)
    store.commence("run1", "h", "d", "s")
    store.transcribe(
        "run1",
        trial(params={"alpha": "x" * 300}, error="e" * 300, strategy="smote"),
    )
    child = store.client.runs["r1"]
    assert child.name == "t1"
    assert child.data.tags["mlflow.parentRunId"] == store.parents["run1"]
    assert child.data.tags["scmle.kind"] == "trial"
    assert child.data.tags["scmle.status"] == "ok"
    assert child.data.tags["scmle.strategy"] == "smote"
    assert len(child.data.tags["scmle.error"]) == 250
    assert child.params == {
        "model": "ridge",
        "strategy": "smote",
        "param.alpha": "x" * 250,
    }
    assert child.data.metrics == {
        "auc": 0.5,
        "score_std": 0.1,
        "fit_seconds": 2.0,
    }
    assert child.status == "FINISHED"


@pytest.mark.parametrize(
    ("status", "ended"),
    [
        ("ok", "FINISHED"),
        ("pruned", "FINISHED"),
        ("failed", "FAILED"),
        ("skipped", "FAILED"),
    ],
)
def test_transcribe_maps_the_trial_status(library, status, ended):
    store = make(library)
    store.commence("run1", "h", "d", "s")
    store.transcribe("run1", trial(status=status))
    assert store.client.runs["r1"].status == ended


def test_transcribe_skips_a_missing_score(library):
    store = make(library)
    store.commence("run1", "h", "d", "s")
    store.transcribe("run1", trial(score=float("nan")))
    assert store.client.runs["r1"].data.metrics == {"fit_seconds": 2.0}
    unnamed = experiments.Trial(
        "t2", "ridge", experiments.Termination.OK, score_mean=0.3
    )
    store.transcribe("run1", unnamed)
    assert store.client.runs["r2"].data.metrics["score"] == 0.3


def test_runs_and_trials_list_their_records(library):
    store = make(library)
    store.commence("run1", "h", "d", "s")
    store.commence("run2", "h", "d", "s")
    store.revise("run2", cv_score=0.8, state="COMPLETED")
    store.transcribe("run1", trial("t9"))
    store.transcribe("run1", trial("t3"))
    runs = store.runs()
    assert [r["run_id"] for r in runs] == ["run2", "run1"]
    assert runs[0]["state"] == "COMPLETED"
    assert runs[0]["cv_score"] == 0.8
    assert runs[0]["mlflow_run_id"] == store.parents["run2"]
    rows = store.trials("run1")
    assert [row["trial_id"] for row in rows] == ["t3", "t9"]
    assert rows[0]["model_name"] == "ridge"
    assert rows[0]["status"] == "ok"
    assert rows[0]["auc"] == 0.5
    assert store.trials("run2") == []


def test_history_keeps_completed_fingerprinted_runs(library):
    store = make(library)
    for run_id in ("a", "b", "c"):
        store.commence(run_id, "h", "d", "s")
    store.revise(
        "a",
        state="COMPLETED",
        fingerprint={"rows": 1.0},
        selected_model="ridge",
        cv_score=0.7,
        task="binary",
    )
    store.revise("b", state="COMPLETED", selected_model="ridge")
    store.revise("c", fingerprint={"rows": 2.0}, selected_model="ridge")
    history = store.history()
    assert len(history) == 1
    record = history[0]
    assert record.run_id == "a"
    assert record.fingerprint == {"rows": 1.0}
    assert record.selected_strategy == "default"
    assert record.cv_score == 0.7
    assert record.task == "binary"
    assert record.dataset_id == ""
    store.revise("a", cv_score=None)
    store.client.runs[store.parents["a"]].data.metrics.clear()
    assert store.history()[0].cv_score != store.history()[0].cv_score


def test_the_module_constants():
    assert mlflow_store.RUN_TAG == "scmle.run_id"
    assert mlflow_store.PARENT_TAG == "mlflow.parentRunId"
    assert (
        frozenset({"config_json", "artifacts"})
        == mlflow_store.LINEAGE_ARTIFACTS
    )
