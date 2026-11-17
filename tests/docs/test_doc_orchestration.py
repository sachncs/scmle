"""Guarantees of docs/ORCHESTRATION.md that no other test enforced."""

import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.workflow import definition
import scmle.workflow.schedule
from tests.unit import test_workflow


@pytest.fixture
def store(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "state")


def test_a_transient_failure_is_retried_and_dependents_are_skipped(store):
    """Enforces docs/ORCHESTRATION.md:31 (retries, skips, independents)."""
    clock = test_workflow.Clock()
    recorder = test_workflow.Recorder(clock)
    executor = test_workflow.executor_for(
        store, test_workflow.pipeline(), recorder, clock
    )

    def flaky(context):
        raise retry.TransientError("blip")

    recorder.behaviour["monitor"] = flaky
    record = executor.start("req-1", "alice")
    attempts = executor.step_record(record["run_id"], "monitor")["attempts"]
    assert len(attempts) == 3
    assert record["steps"]["monitor"] == "failed"
    assert record["steps"]["batch_predict"] == "skipped"
    assert record["steps"]["approve"] == "waiting"


def test_start_never_retries_a_failed_run_and_resume_runs_the_rest(store):
    """Enforces docs/ORCHESTRATION.md:31 (start versus resume)."""
    clock = test_workflow.Clock()
    recorder = test_workflow.Recorder(clock)
    executor = test_workflow.executor_for(
        store, test_workflow.pipeline(), recorder, clock
    )
    calls = {"evaluate": 0}

    def broken(context):
        calls["evaluate"] += 1
        raise errors.DataError("bad metric")

    recorder.behaviour["evaluate"] = broken
    record = executor.start("req-1", "alice")
    assert record["state"] == "failed" and calls["evaluate"] == 1
    recorder.behaviour.pop("evaluate")
    before = list(recorder.calls)
    again = executor.start("req-1", "alice")
    assert again["state"] == "failed" and recorder.calls == before
    healed = executor.resume(record["run_id"])
    assert healed["state"] == "waiting_approval"
    rerun = [name for name, ignored in recorder.calls[len(before) :]]
    assert rerun == ["evaluate", "monitor"]


def test_a_deferred_slot_stays_due_and_runs_on_a_later_tick(store):
    """Enforces docs/ORCHESTRATION.md:43."""
    schedule = definition.Schedule(3600.0, 0.0, definition.CatchUp.LATEST_ONLY)
    base = test_workflow.pipeline()
    flow = definition.Workflow(base.name, base.version, base.steps, schedule)
    clock = test_workflow.Clock(3600.0 + 1.0)
    recorder = test_workflow.Recorder(clock)
    executor = test_workflow.executor_for(store, flow, recorder, clock)
    scheduler = scmle.workflow.schedule.Scheduler(flow, store)
    scheduler.save(0.0, {})
    seen = []

    def during(context):
        seen.append(scheduler.tick(executor))
        return {}

    recorder.behaviour["ingest"] = during
    manual = executor.start("manual", "alice")
    recorder.behaviour.pop("ingest")
    assert seen[0].deferred == (3600.0,) and seen[0].ran == ()
    test_workflow.approve(executor, manual)
    assert executor.resume(manual["run_id"])["state"] == "succeeded"
    later = scheduler.tick(executor)
    assert [slot for slot, ignored in later.ran] == [3600.0]
