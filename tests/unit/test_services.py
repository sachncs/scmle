"""The run's frozen services, its locked journal and the board holding them."""

import dataclasses
import datetime
import json
import threading
import types

import pandas
import pytest
from sklearn import dummy

import scmle.artifacts.store
from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import resources
from scmle.evaluation import selection
import scmle.models.registry
from scmle.pipeline import engine
from scmle.pipeline import journal
from scmle.pipeline import services
from scmle.pipeline.stages import cost
from scmle.pipeline.stages import reservation
from scmle.pipeline.stages import training
from scmle.runtime import limits
from scmle.runtime import randomness

THREADS = 8
EACH = 25


def config(budget=1800):
    return tasks.Config(
        data=tasks.Dataset(uri="x.csv", target="y"),
        resources=tasks.Resources(time_budget_seconds=budget),
    )


def provide(spec=None, store=None, log=None, **fields):
    spec = spec or config()
    return services.Services(
        spec,
        scmle.models.registry.Catalog.stock(),
        metrics.CATALOG,
        selection.Averager(spec.selection),
        limits.Governor(spec.resources, 100.0),
        store=store,
        log=log,
        **fields,
    )


def board(root=None, **fields):
    store = log = None
    if root is not None:
        root.mkdir()
        store = scmle.artifacts.store.Workspace("r1", root)
        log = events.Log(root / "events.jsonl")
    return engine.Board.start(provide(store=store, log=log, **fields))


def trial(trial_id):
    return experiments.Trial(
        trial_id,
        "m",
        experiments.Termination.OK,
        "accuracy",
        0.5,
        0.0,
        [0.5],
        1.0,
    )


def lines(root):
    text = (root / "events.jsonl").read_text("utf-8")
    return [json.loads(line) for line in text.splitlines()]


class Fixed(services.Clock):
    """A clock that advances `step` seconds on every read."""

    def __init__(self, step):
        self.step = step
        self.reads = 0

    def monotonic(self):
        self.reads += 1
        return self.reads * self.step

    def now(self):
        return datetime.datetime(2026, 1, 2, tzinfo=datetime.UTC)


def test_services_are_frozen():
    given = provide()
    with pytest.raises(dataclasses.FrozenInstanceError):
        given.spec = config(60)
    with pytest.raises(dataclasses.FrozenInstanceError):
        given.history = ()


def test_the_deadline_counts_from_the_governor_start():
    assert provide().deadline == 1900.0
    assert provide(config(60)).deadline == 160.0


def test_elapsed_reads_the_services_clock_from_the_governor_start():
    given = provide(clock=Fixed(250.0))
    assert given.elapsed() == 150.0
    assert given.elapsed() == 400.0


def test_a_preview_has_no_workspace_and_an_empty_directory():
    given = provide()
    assert given.directory == ""
    with pytest.raises(errors.ScmleError) as caught:
        given.workspace  # noqa: B018 -- the property raises
    assert str(caught.value) == (
        "'store' is unavailable: its stage has not run yet"
    )


def test_start_gives_the_journal_the_services_records(tmp_path):
    held = board(tmp_path / "run")
    assert held.journal.spec is held.services.spec
    assert held.journal.store is held.services.store
    assert held.journal.log is held.services.log
    assert held.journal.tracker is held.services.tracker
    assert held.journal.trials == []


def test_services_start_observes_and_binds_the_resources(tmp_path):
    spec = config()
    store = scmle.artifacts.store.Workspace("r1", tmp_path)
    given = services.start(
        spec,
        scmle.models.registry.Catalog.stock(),
        selection.Averager(spec.selection),
        store=store,
        started=12.0,
    )
    assert given.governor.started == 12.0
    assert given.catalog is metrics.CATALOG
    assert given.observation is not None
    assert given.backend.workers >= 1
    assert given.store is store


def test_a_run_has_its_workspace_and_directory(tmp_path):
    store = scmle.artifacts.store.Workspace("r1", tmp_path)
    given = provide(store=store)
    assert given.workspace is store
    assert given.directory == str(tmp_path)


def test_check_raises_once_cancelled_with_the_run_directory(tmp_path):
    store = scmle.artifacts.store.Workspace("r1", tmp_path)
    given = provide(store=store)
    given.check()
    given.event.set()
    with pytest.raises(errors.CancelledError) as caught:
        given.check()
    assert str(caught.value) == "run cancelled"
    assert caught.value.details == {"run_dir": str(tmp_path)}


def test_the_system_clock_reads_monotonic_and_utc_time():
    clock = services.Clock()
    first = clock.monotonic()
    assert clock.monotonic() >= first
    assert clock.now().tzinfo is datetime.UTC


def test_the_random_streams_are_the_keyed_streams():
    drawn = services.Random().generator(7, "conformal.entity", 2).random(4)
    expected = randomness.spawn(7, "conformal.entity", 2).random(4)
    assert drawn.tolist() == expected.tolist()


def test_a_stage_times_with_the_services_clock():
    held = board(clock=Fixed(10_000.0))
    frame = pandas.DataFrame({"a": [0.0, 1.0, 2.0, 3.0]})
    model = dummy.DummyClassifier().fit(frame, [0, 1, 0, 1])
    cost.meter(held, model, frame)
    assert held.services.clock.reads == 2
    assert [entry["work"] for entry in held.journal.skipped] == [
        "serving measurement"
    ]
    single = cost.FIT_MARGIN * 10_000.0
    assert held.journal.skipped[0]["reason"].startswith(
        f"timed 1 of {resources.LATENCY_ROWS} rows: one prediction takes "
        f"about {single:.2f} s and "
    )


def test_the_journal_records_eight_threads_without_loss(tmp_path):
    held = board(tmp_path / "run")
    record = held.journal

    def work():
        for repeat in range(EACH):
            (name,) = record.allocate(1)
            record.transcribe(trial(name))
            record.enter(name, experiments.Edict.INCLUDE, f"kept {repeat}")

    threads = [threading.Thread(target=work) for index in range(THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    total = THREADS * EACH
    names = [f"trial_{index:03d}" for index in range(1, total + 1)]
    assert record.counter == total
    assert sorted(t.identifier for t in record.trials) == names
    assert len(record.outcomes) == total
    written = lines(tmp_path / "run")
    assert sorted(event["artifacts"][0] for event in written) == [
        f"trials/{name}.json" for name in names
    ]
    assert {event["kind"] for event in written} == {"trial.ok"}


def test_forgo_keeps_the_record_and_announces_it(tmp_path):
    held = board(tmp_path / "run")
    held.journal.forgo("conformal", "needs 9 s")
    assert held.journal.skipped == [
        {"work": "conformal", "reason": "needs 9 s"}
    ]
    (event,) = lines(tmp_path / "run")
    assert event["kind"] == "budget.skipped"
    assert event["reason"] == "conformal skipped or reduced: needs 9 s"
    assert event["data"] == {"work": "conformal", "reason": "needs 9 s"}
    assert event["config_hash"] == config().config_hash()


def test_a_preview_journal_keeps_records_but_writes_nothing():
    record = journal.Journal(config())
    record.transcribe(trial("t1"))
    record.forgo("x", "y")
    record.write("a.json", {})
    record.emit("z")
    assert [t.identifier for t in record.trials] == ["t1"]
    assert record.skipped == [{"work": "x", "reason": "y"}]


def test_a_search_reaches_the_journal_when_applied():
    held = board()
    held.journal.transcribe(trial("trial_001"))
    held.journal.allocate(1)
    made = training.Search.copy(held.journal)
    assert made.trials == [trial("trial_001")]
    assert made.counter == 1
    fresh = board()
    made.apply(fresh)
    assert fresh.journal.trials == [trial("trial_001")]
    assert fresh.journal.counter == 1
    assert fresh.journal.allocate(1) == ["trial_002"]
    assert fresh.get(training.Search) is made


def test_a_product_governor_replaces_the_frozen_services():
    held = board()
    before = held.services
    governor = limits.Governor(held.services.spec.resources, 5.0)
    reservation.Reserve(governor).apply(held)
    assert held.services.governor is governor
    assert held.services is not before
    assert held.services.registry is before.registry
    assert held.regulator is governor
    assert held.get(reservation.Reserve).governor is governor


def test_a_product_that_changes_the_services_needs_a_run_board():
    governor = limits.Governor(config().resources, 5.0)
    with pytest.raises(errors.KindError, match="run's board"):
        reservation.Reserve(governor).apply(types.SimpleNamespace(made={}))


def test_an_unknown_attribute_is_still_an_attribute_error():
    with pytest.raises(AttributeError):
        board().nothing  # type: ignore[attr-defined] # noqa: B018 -- raises


def test_amend_restamps_the_later_events(tmp_path):
    held = board(tmp_path / "run")
    held.journal.emit("before")
    amended = config(60)
    held.amend(amended)
    held.journal.emit("after")
    assert held.services.spec is amended
    first, second = lines(tmp_path / "run")
    assert first["config_hash"] == config().config_hash()
    assert second["config_hash"] == amended.config_hash()


def test_the_board_checks_cancellation_and_caps_threads(tmp_path):
    held = board()
    held.check()
    held.services.event.set()
    with pytest.raises(errors.CancelledError):
        held.check()
    assert held.regulator is held.services.governor
