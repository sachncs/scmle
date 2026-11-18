"""Guarantees of docs/ARCHITECTURE.md that no other test enforced."""

import threading

import numpy
import pandas
from sklearn import naive_bayes

from benchmarks.qualification import ingest_worlds
from scmle import durability
from scmle.assurance.ingest import grain
from scmle.assurance.ingest import joins
from scmle.assurance.ingest import profile
from scmle.contracts import experiments
from scmle.contracts import models
from scmle.contracts import tasks
import scmle.contracts.metrics
from scmle.data.readers import tabular
from scmle.evaluation import metrics
import scmle.governance.registry
from scmle.models import custom
from scmle.models import registry
from scmle.operations import lifecycle
from scmle.pipeline import planner
from scmle.pipeline import profiler


def test_the_plan_records_included_rejected_and_deferred_techniques():
    """Enforces docs/ARCHITECTURE.md:57."""
    rng = numpy.random.default_rng(0)
    frame = pandas.DataFrame(
        {"x": rng.normal(size=300), "y": rng.integers(0, 2, 300)}
    )
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y"),
        resources=tasks.Resources(time_budget_seconds=60),
    )
    task = tasks.Task.BINARY
    found = profiler.Profiler(spec).profile(
        frame, tabular.manifested(frame, "y"), task
    )
    plan = planner.Planner(registry.Catalog.stock()).plan(
        planner.Situation(spec, found, task)
    )
    actions = {d.action for d in plan.decisions}
    assert actions == set(experiments.Edict)
    assert all(d.technique and d.reason for d in plan.decisions)
    included = {
        d.technique
        for d in plan.decisions
        if d.action is experiments.Edict.INCLUDE
    }
    assert set(plan.candidates) <= included


def recorded(monkeypatch):
    """Records every statement `profile.Sql` sends and the rows it got."""
    log = []
    fetch = profile.Sql.fetch

    def spy(self, sql, *arguments, **keywords):
        rows = fetch(self, sql, *arguments, **keywords)
        log.append((sql, rows))
        return rows

    monkeypatch.setattr(profile.Sql, "fetch", spy)
    return log


def test_table_statistics_are_aggregate_queries_run_in_the_source(
    monkeypatch,
):
    """Enforces docs/ARCHITECTURE.md:95.

    Counts, distinct counts, keys and inclusion all come back as one row of
    aggregates computed by the database; no query returns the table's rows.
    """
    db = ingest_worlds.new_database("aggregates")
    ingest_worlds.load(
        db, "customer", pandas.DataFrame({"id": numpy.arange(1, 41)})
    )
    ingest_worlds.load(
        db,
        "fact",
        pandas.DataFrame(
            {
                "id": numpy.arange(100),
                "customer_id": numpy.arange(1, 101) % 40 + 1,
            }
        ),
    )
    log = recorded(monkeypatch)
    source = db.source()
    catalog = profile.inventory(source, ["customer", "fact"])
    scans = {t.name: grain.keysearch(source, t) for t in catalog.tables}
    graph = joins.discover(source, catalog, scans)
    assert catalog.table("fact").rows == 100
    assert any(s.keys for s in scans.values())
    assert any(link.child_columns == ("customer_id",) for link in graph.links)
    assert len(log) > 5
    for sql, rows in log:
        assert "COUNT(" in sql.upper(), sql
        assert rows is not None and len(rows) <= 1, sql


class Broken(models.Model):
    """An adapter whose estimator cannot predict."""

    name = "doc_broken_plugin"
    family = "x"
    capabilities = models.Capabilities(frozenset({tasks.Task.BINARY}))

    def build(self, task, seed, params=None):
        class Estimator:
            def fit(self, features, target):
                return self

        return Estimator()


class Entry:
    """A stand-in for an `importlib.metadata` entry point."""

    name = "doc_broken_plugin"

    def load(self):
        return Broken


def test_a_plugin_that_fails_conformance_is_not_registered(monkeypatch):
    """Enforces docs/ARCHITECTURE.md:198 (the entry-point half)."""
    monkeypatch.setattr(
        registry.metadata, "entry_points", lambda group: [Entry()]
    )
    catalog = registry.Catalog()
    assert catalog.load_plugins() == ["doc_broken_plugin"]
    assert "doc_broken_plugin" not in catalog.adapters
    assert "doc_broken_plugin" in catalog.plugin_failures


def score(truth, predicted):
    """A module-level metric function."""
    return float(numpy.mean(numpy.asarray(truth) == numpy.asarray(predicted)))


def test_readers_keep_their_snapshot_while_registrations_change():
    """Enforces docs/ARCHITECTURE.md:217."""
    custom.register_estimator("doc_snapshot", classifier=naive_bayes.GaussianNB)
    try:
        before = registry.Catalog.stock()
        custom.unregister_estimator("doc_snapshot")
        assert "doc_snapshot" in before.adapters
        assert "doc_snapshot" not in registry.Catalog.stock().adapters
    finally:
        custom.unregister_estimator("doc_snapshot")
    catalog = metrics.Repertoire()
    table = catalog.metrics
    names = catalog.report_names
    held = dict(table)
    catalog.register(
        scmle.contracts.metrics.Formula(
            "doc_metric", "", True, frozenset(tasks.STANDARD_TASKS), score
        )
    )
    catalog.unregister("roc_auc")
    assert table == held and "roc_auc" in table
    assert "doc_metric" not in table
    assert "roc_auc" in names[tasks.Task.BINARY]
    assert "roc_auc" not in catalog.metrics


def lock_is_free(lock, wait=1.0):
    """Reports whether another thread can take `lock` within `wait` seconds."""
    taken = []

    def attempt():
        if lock.acquire(timeout=wait):
            lock.release()
            taken.append(True)

    worker = threading.Thread(target=attempt)
    worker.start()
    worker.join()
    return bool(taken)


SEEN = []


class Probing(naive_bayes.GaussianNB):
    """A user estimator that checks the registry lock while it fits."""

    def fit(self, features, target, sample_weight=None):
        SEEN.append(lock_is_free(registry.USER_LOCK))
        return super().fit(features, target, sample_weight)


def test_the_registry_lock_is_free_while_a_user_estimator_fits():
    """Enforces docs/ARCHITECTURE.md:247 (the registry half)."""
    SEEN.clear()
    try:
        custom.register_estimator("doc_probing", classifier=Probing)
    finally:
        custom.unregister_estimator("doc_probing")
    assert SEEN and all(SEEN)


def test_review_holds_the_state_lock_for_the_whole_retrain(tmp_path):
    """Enforces docs/ARCHITECTURE.md:247 (the `Maintainer.review` half).

    The retrain runs inside `locked`; while it runs no other thread can take
    the state file's lock.
    """
    store = scmle.governance.registry.ModelRegistryStore(tmp_path / "r")
    manager = lifecycle.Maintainer(store, "churn", str(tmp_path / "w"))
    seen = []

    def work(fresh, target, retrain, configure):
        seen.append(lock_is_free(durability.latch(manager.statefile), 0.2))
        return "verdict"

    manager.locked = work
    assert manager.review(pandas.DataFrame({"y": [0, 1]}), "y") == "verdict"
    assert seen == [False]
    assert lock_is_free(durability.latch(manager.statefile))
