"""Concurrency tests: the public API is safe to call from several threads.

Every test starts its workers behind a barrier so they begin together, then
asserts a deterministic outcome (no exception, no lost update, results equal
to a sequential run). The interpreter is free-threaded, so these are real
data races, not simulated ones.
"""

import collections
from collections.abc import Callable
import copy
import dataclasses
import json
import os
import sqlite3
import sys
import threading
import time
from typing import Any

import numpy
import pandas
import pytest
from sklearn import naive_bayes
import support

import scmle.api.predict
import scmle.artifacts.bundle
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import observability
from scmle.contracts import tasks
from scmle.core import limits
from scmle.evaluation import metrics
import scmle.evaluation.metrics
from scmle.experimentation import stores
from scmle.governance import keyring
from scmle.governance import lifecycle
import scmle.governance.registry
import scmle.models.custom
from scmle.models.deeplearning import worker
import scmle.models.registry
from scmle.monitoring import alerts
from scmle.monitoring import events
from scmle.monitoring import incidents
from scmle.monitoring import shadow
from scmle.monitoring import slo

THREADS = 8
APPROVAL_KEY = "thread-safety-key"


def run_together(
    count: int, work: Callable[[int], Any]
) -> tuple[list[Any], list[BaseException]]:
    """Runs `work(index)` in `count` threads released by one barrier.

    Returns:
        Results (None for a failed worker) and the exceptions raised.
    """
    barrier = threading.Barrier(count)
    results: list[Any] = [None] * count
    failures: list[BaseException] = []
    guard = threading.Lock()

    def target(index: int) -> None:
        try:
            barrier.wait(timeout=30)
            results[index] = work(index)
        except BaseException as error:  # Reported to the test, not lost.
            with guard:
                failures.append(error)

    threads = [
        threading.Thread(target=target, args=(index,)) for index in range(count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
    assert not any(thread.is_alive() for thread in threads), "deadlock"
    return results, failures


def failures_of(count: int, work: Callable[[int], Any]) -> list[BaseException]:
    """Runs `work` in `count` threads together; returns only the failures."""
    return run_together(count, work)[1]


def frame(rows: int = 200, seed: int = 0) -> pandas.DataFrame:
    """A small separable binary problem."""
    rng = numpy.random.default_rng(seed)
    data = pandas.DataFrame(
        {"a": rng.normal(size=rows), "b": rng.normal(size=rows)}
    )
    data["y"] = (data.a + data.b + rng.normal(0, 0.5, rows) > 0).astype(int)
    return data


@pytest.fixture(autouse=True)
def clean_registries():
    """Removes whatever these tests register."""
    yield
    for name in list(scmle.models.registry.USER_ADAPTERS):
        if name.startswith("ts_"):
            scmle.models.custom.unregister_estimator(name)
    for name in list(metrics.CATALOG.metrics):
        if name.startswith("ts_"):
            metrics.CATALOG.unregister(name)


# --- estimator and metric registries -------------------------------------


def test_concurrent_register_estimator_keeps_every_registration():
    def work(index):
        name = f"ts_nb_{index}"
        scmle.models.custom.register_estimator(
            name, classifier=naive_bayes.GaussianNB
        )
        return name

    names, failures = run_together(THREADS, work)

    assert not failures
    built = scmle.models.registry.Catalog.with_builtins()
    assert set(names) <= set(built.adapters)


def test_register_same_estimator_name_has_exactly_one_winner():
    def work(index):
        scmle.models.custom.register_estimator(
            "ts_same", classifier=naive_bayes.GaussianNB
        )

    for round_number in range(5):
        scmle.models.custom.unregister_estimator("ts_same")
        failures = failures_of(THREADS, work)
        winners = THREADS - len(failures)
        assert winners == 1, (round_number, winners)
        assert all(isinstance(f, errors.ConfigError) for f in failures)


def test_builtin_name_check_does_not_hide_registered_estimators():
    scmle.models.custom.register_estimator(
        "ts_persist", classifier=naive_bayes.GaussianNB
    )
    stop = threading.Event()

    def churn(index):
        if index == 0:
            for round_number in range(40):  # noqa: B007
                scmle.models.registry.builtin_names()
            stop.set()
            return True
        missing = 0
        while not stop.is_set():
            adapters = scmle.models.registry.Catalog.with_builtins().adapters
            missing += "ts_persist" not in adapters
        return missing

    results, failures = run_together(4, churn)

    assert not failures
    assert results[1:] == [0, 0, 0]


def test_register_unregister_and_lookup_never_raise():
    def work(index):
        name = f"ts_churn_{index}"
        for round_number in range(15):  # noqa: B007
            scmle.models.custom.register_estimator(
                name, classifier=naive_bayes.GaussianNB, replace=True
            )
            scmle.models.registry.Catalog.with_builtins().get(name)
            scmle.models.custom.unregister_estimator(name)

    failures = failures_of(4, work)

    assert not failures


def score(truth, predicted):
    """A module-level metric function."""
    return float(numpy.mean(numpy.asarray(truth) == numpy.asarray(predicted)))


def test_concurrent_register_metric_and_lookup():
    def work(index):
        name = f"ts_metric_{index}"
        scmle.evaluation.metrics.register_metric(
            name, score, greater_is_better=True
        )
        for round_number in range(50):  # noqa: B007
            sorted(metrics.CATALOG.metrics)
            metrics.CATALOG.resolve(name, tasks.Task.BINARY)
        return name

    names, failures = run_together(THREADS, work)

    assert not failures
    for name in names:
        assert name in metrics.CATALOG.report_names[tasks.Task.BINARY]


def test_register_same_metric_name_has_exactly_one_winner():
    def work(index):
        scmle.evaluation.metrics.register_metric(
            "ts_metric_same", score, greater_is_better=True
        )

    for round_number in range(5):  # noqa: B007
        metrics.CATALOG.unregister("ts_metric_same")
        failures = failures_of(THREADS, work)
        assert len(failures) == THREADS - 1


def test_metric_unregister_during_report_names_update_is_consistent():
    def work(index):
        name = f"ts_metric_c{index}"
        for round_number in range(40):  # noqa: B007
            scmle.evaluation.metrics.register_metric(
                name, score, True, replace=True
            )
            metrics.CATALOG.unregister(name)

    failures = failures_of(THREADS, work)

    assert not failures
    for names in metrics.CATALOG.report_names.values():
        assert not [n for n in names if n.startswith("ts_metric_c")]


# --- projects, bundles ----------------------------------------------------


def small_project(data, output_dir):
    """A cooperative project with a small real search."""
    return support.make_project(
        data,
        "y",
        output_dir=output_dir,
        seed=7,
        search=tasks.Search(top_models=1, configs_per_model=2),
        time_budget=300,
    )


@pytest.fixture(scope="module")
def sequential_results(tmp_path_factory):
    """The same two runs, one after another."""
    root = tmp_path_factory.mktemp("sequential")
    out = []
    for index in range(2):
        result = small_project(frame(seed=index), root / str(index)).run()
        out.append(
            (
                result.selected_model,
                result.evaluation["test_metrics"],
                result.predict(frame(60, seed=99)),
            )
        )
    return out


def test_parallel_projects_match_sequential_runs(tmp_path, sequential_results):
    shared = tmp_path / "runs"

    def work(index):
        result = small_project(frame(seed=index), shared).run()
        return (
            result.selected_model,
            result.evaluation["test_metrics"],
            result.predict(frame(60, seed=99)),
            result.run_dir,
        )

    outcomes, failures = run_together(2, work)

    assert not failures
    for index, (model, test, predicted, run_dir) in enumerate(outcomes):  # noqa: B007
        want_model, want_test, want_predicted = sequential_results[index]
        assert model == want_model
        assert test == pytest.approx(want_test)
        numpy.testing.assert_array_equal(predicted, want_predicted)
    run_dirs = {str(outcome[3]) for outcome in outcomes}
    assert len(run_dirs) == 2
    with sqlite3.connect(shared / "experiments.sqlite") as connection:
        rows = connection.execute("SELECT run_id FROM runs").fetchall()
    assert len({row[0] for row in rows}) == 2


def test_parallel_plan_calls_agree():
    project = support.make_project(frame(), "y")
    want = repr(project.plan())

    def work(index):
        return repr(project.plan())

    outcomes, failures = run_together(3, work)

    assert not failures
    assert outcomes == [want] * 3


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """One small fitted run, reused by the read-only tests."""
    out = tmp_path_factory.mktemp("trained")
    return small_project(frame(), out).run()


def test_concurrent_predict_on_one_bundle_matches_sequential(trained):
    bundle = trained.bundle
    batches = [frame(40 + index, seed=50 + index) for index in range(THREADS)]
    want = [bundle.predict(batch) for batch in batches]
    want_proba = [bundle.predict_proba(batch) for batch in batches]

    def work(index):
        out = []
        for round_number in range(10):  # noqa: B007
            out.append(
                (
                    bundle.predict(batches[index]),
                    bundle.predict_proba(batches[index]),
                )
            )
        return out

    outcomes, failures = run_together(THREADS, work)

    assert not failures
    for index, rounds in enumerate(outcomes):
        for labels, probabilities in rounds:
            numpy.testing.assert_array_equal(labels, want[index])
            numpy.testing.assert_allclose(probabilities, want_proba[index])


def test_concurrent_load_model_and_predict(trained, tmp_path):
    directory = trained.save(tmp_path / "export")
    data = frame(50, seed=3)
    want = trained.predict(data)

    def work(index):
        model = scmle.api.predict.load_model(directory, trusted=True)
        return model.predict(data), scmle.api.predict.predict(
            directory, data, trusted=True
        )

    outcomes, failures = run_together(4, work)

    assert not failures
    for first, second in outcomes:
        numpy.testing.assert_array_equal(first, want)
        numpy.testing.assert_array_equal(second, want)


# --- stores ---------------------------------------------------------------


def legacy_database(path):
    """A database from before the later columns were added."""
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE runs (run_id TEXT PRIMARY KEY, "
            "config_hash TEXT NOT NULL, state TEXT NOT NULL, "
            "started_at TEXT NOT NULL, updated_at TEXT NOT NULL, "
            "run_dir TEXT, dataset_id TEXT, data_hash TEXT, "
            "split_fingerprint TEXT, selected_model TEXT, metric TEXT, "
            "cv_score REAL, test_metrics TEXT);"
            "CREATE TABLE trials (run_id TEXT NOT NULL, "
            "trial_id TEXT NOT NULL, model_name TEXT NOT NULL, "
            "status TEXT NOT NULL, metric TEXT, score_mean REAL, "
            "score_std REAL, fit_seconds REAL, params TEXT, "
            "fold_scores TEXT, error TEXT, PRIMARY KEY (run_id, trial_id));"
        )
    sqlite3.connect(path).close()


def test_concurrent_store_open_migrates_once(tmp_path):
    path = tmp_path / "legacy.sqlite"
    legacy_database(path)

    def work(index):
        return stores.SqliteExperimentStore(path)

    opened, failures = run_together(THREADS, work)

    assert not failures
    assert opened[0].runs() == []


def test_concurrent_store_writes_are_all_recorded(tmp_path):
    store = stores.SqliteExperimentStore(tmp_path / "tracking.sqlite")
    per_thread = 20

    def work(index):
        run_id = f"run_{index}"
        store.start_run(run_id, "hash", str(tmp_path), "2026-01-01T00:00:00Z")
        for number in range(per_thread):
            store.record_trial(
                run_id,
                experiments.Trial(
                    trial_id=f"t{number}",
                    model_name="m",
                    status=experiments.TrialStatus.OK,
                    metric="accuracy",
                    score_mean=0.5,
                ),
            )
            store.update_run(run_id, state="RUNNING")
        return run_id

    ids, failures = run_together(THREADS, work)

    assert not failures
    assert sorted(row["run_id"] for row in store.runs()) == sorted(ids)
    for run_id in ids:
        assert len(store.trials(run_id)) == per_thread


@pytest.fixture(scope="module")
def two_bundles(trained, tmp_path_factory):
    """Distinct saved bundles for registry tests."""
    root = tmp_path_factory.mktemp("bundles")
    paths = []
    for index in range(4):
        changed = copy.copy(trained.bundle)
        changed.info = dataclasses.replace(
            trained.bundle.info, model_version=f"v{index}"
        )
        paths.append(changed.save(root / f"b{index}"))
    return paths


def test_concurrent_registry_register_assigns_distinct_versions(
    two_bundles, tmp_path
):
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "registry")

    def work(index):
        return store.register(
            "churn", two_bundles[index], run_id=f"r{index}", trusted=True
        )

    records, failures = run_together(len(two_bundles), work)

    assert not failures
    assert sorted(record.version for record in records) == [1, 2, 3, 4]
    listed = store.versions("churn")
    assert [record.version for record in listed] == [1, 2, 3, 4]
    audit = (
        tmp_path / "registry" / scmle.governance.registry.AUDIT_FILE
    ).read_text()
    assert len(audit.splitlines()) == 4


def test_concurrent_registry_same_bundle_registers_once(two_bundles, tmp_path):
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "registry")

    def work(index):
        return store.register("churn", two_bundles[0], trusted=True)

    records, failures = run_together(THREADS, work)

    assert not failures
    assert {record.version for record in records} == {1}
    assert len(store.versions("churn")) == 1


def test_concurrent_alias_moves_do_not_corrupt_the_index(two_bundles, tmp_path):
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "registry")
    for index in range(len(two_bundles)):
        store.register("churn", two_bundles[index], trusted=True)

    def work(index):
        for round_number in range(10):  # noqa: B007
            store.promote("churn", index + 1, f"stage_{index}")
            store.aliases("churn")

    failures = failures_of(len(two_bundles), work)

    assert not failures
    assert store.aliases("churn") == {
        f"stage_{index}": index + 1 for index in range(len(two_bundles))
    }


def test_concurrent_lifecycle_state_writes(tmp_path):
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "registry")
    manager = lifecycle.Maintainer(store, "churn", str(tmp_path / "w"))

    def work(index):
        for round_number in range(25):  # noqa: B007
            manager.write_state(
                {"last_digest": str(index), "rows_since_attempt": index}
            )
            manager.read_state()

    failures = failures_of(THREADS, work)

    assert not failures
    assert manager.read_state()["rows_since_attempt"] in range(THREADS)


# --- monitoring journals --------------------------------------------------


def breach() -> slo.SLOEvaluation:
    """A breached latency SLO evaluation."""
    target = slo.SLOTarget(
        name="latency",
        metric_name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        maximum=250.0,
        window_days=7,
        minimum_samples=1,
        owner="ml-on-call",
        runbook="https://runbooks.example/latency",
        alert_route="on-call",
    )
    sample = observability.MetricSample(
        name="p95_latency_ms",
        kind=observability.SignalKind.LATENCY,
        value=400.0,
        unit="ms",
        timestamp="2026-10-20T12:00:00+00:00",
    )
    return slo.SLOMonitor().evaluate(
        target, (sample,), observed_at="2026-10-20T12:00:00+00:00"
    )


def test_concurrent_incident_open_keeps_a_valid_chain(tmp_path):
    workflow = incidents.IncidentWorkflow(tmp_path / "incidents.jsonl")
    evaluation = breach()

    def work(index):
        return workflow.open(
            evaluation,
            opened_at="2026-10-20T12:01:00+00:00",
            run_id="run-1",
            rollback_target="previous",
        )

    opened, failures = run_together(THREADS, work)

    assert not failures
    assert len({incident.incident_id for incident in opened}) == 1
    assert len(workflow.read_events()) == 1


def test_concurrent_incident_transitions_keep_a_valid_chain(tmp_path):
    workflow = incidents.IncidentWorkflow(tmp_path / "incidents.jsonl")
    evaluation = breach()
    first = workflow.open(
        evaluation,
        opened_at="2026-10-20T12:01:00+00:00",
        run_id="run-1",
        rollback_target="previous",
    )

    def work(index):
        workflow.acknowledge(
            first.incident_id,
            actor=f"op{index}",
            acknowledged_at="2026-10-20T12:02:00+00:00",
        )

    failures = failures_of(THREADS, work)

    assert not failures
    assert [e["event_type"] for e in workflow.read_events()] == [
        "opened",
        "acknowledged",
    ]


def test_concurrent_alert_dispatch_delivers_once(tmp_path):
    sink_path = tmp_path / "sink.jsonl"
    dispatcher = alerts.AlertDispatcher(
        {"on-call": alerts.JsonlAlertSink(sink_path)},
        tmp_path / "deliveries.jsonl",
    )
    alert = alerts.Alert.from_evidence(
        route="on-call",
        severity="high",
        title="model SLO breached",
        details={"run_id": "run-1"},
        evidence={"slo_status": "breached"},
        created_at="2026-10-20T12:00:00+00:00",
    )

    outcomes, failures = run_together(
        THREADS, lambda index: dispatcher.dispatch(alert)
    )

    assert not failures
    counts = collections.Counter(delivery.status for delivery in outcomes)
    assert counts["delivered"] == 1
    assert len(sink_path.read_text().splitlines()) == 1


def test_concurrent_prediction_log_appends_whole_lines(tmp_path):
    log = events.PredictionLog(tmp_path / "log.jsonl")
    payload = {"event_id": "x", "blob": "z" * 20000}

    def work(index):
        for number in range(20):
            log.append(
                "prediction", {**payload, "event_id": f"{index}-{number}"}
            )

    failures = failures_of(THREADS, work)

    assert not failures
    lines = (tmp_path / "log.jsonl").read_text().splitlines()
    assert len(lines) == THREADS * 20
    assert len({json.loads(line)["event_id"] for line in lines}) == len(lines)


def test_shadow_evaluator_keeps_observations_aligned(trained):
    bundle = trained.bundle
    evaluator = shadow.ShadowEvaluator(bundle, bundle)
    batches = [frame(10 + index, seed=70 + index) for index in range(THREADS)]

    def work(index):
        for round_number in range(10):  # noqa: B007
            evaluator.observe(batches[index])

    failures = failures_of(THREADS, work)

    assert not failures
    sizes = [len(rows) for rows in evaluator.rows]
    assert sizes == [len(out) for out in evaluator.champion_output]
    assert sizes == [len(out) for out in evaluator.challenger_output]
    assert len(evaluator.gaps) == len(sizes)
    assert [len(g) for g in evaluator.gaps] == sizes


def test_key_registry_register_has_exactly_one_winner():
    registry = keyring.SigningKeyRegistry()
    record = keyring.SigningKeyRecord(
        key_id="k1", owner="release", created_at="2026-01-01T00:00:00+00:00"
    )

    def register(index):
        registry.register(record)

    for round_number in range(5):  # noqa: B007
        registry.records.clear()
        failures = failures_of(THREADS, register)
        assert len(failures) == THREADS - 1
        assert all(isinstance(f, ValueError) for f in failures)


# --- process-wide state ---------------------------------------------------


def test_forkserver_environment_restores_the_variable(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.delenv(limits.ACCELERATE_THREADS, raising=False)
    inside = threading.Barrier(THREADS)

    def work(index):
        with limits.forkserver_environment():
            inside.wait(timeout=30)
            time.sleep(0.001 * index)  # Stagger the exits only.

    failures = failures_of(THREADS, work)

    assert not failures
    assert limits.ACCELERATE_THREADS not in os.environ


def test_worker_client_is_a_singleton_per_process(monkeypatch):
    created = []

    class Slow:
        """Counts constructions and widens the race window."""

        def __init__(self):
            created.append(self)
            time.sleep(0.01)

        def shutdown(self):
            """Nothing to stop."""

    monkeypatch.setattr(worker, "Client", Slow)
    monkeypatch.setattr(worker, "CLIENTS", {})

    clients, failures = run_together(THREADS, lambda index: worker.client())

    assert not failures
    assert len(created) == 1
    assert all(client is clients[0] for client in clients)


def test_bundle_remember_does_not_lose_rows():
    bundle = object.__new__(scmle.artifacts.bundle.Bundle)
    bundle.schema = type(
        "Schema", (), {"series": [1], "group_column": "g", "time_column": "t"}
    )()

    class Builder:
        """Keeps every remembered row as the history."""

        history_columns = ["g", "t", "v"]

        def build_history(self, combined):
            return combined

    bundle.history = None
    bundle.series_builder = lambda: Builder()
    bundle.read_frame = lambda data: data
    bundle.history_lock = threading.RLock()

    def work(index):
        for number in range(30):
            row = index * 1000 + number
            bundle.remember(
                pandas.DataFrame({"g": [row], "t": [row], "v": [row]})
            )

    failures = failures_of(THREADS, work)

    assert not failures
    assert len(bundle.history) == THREADS * 30
