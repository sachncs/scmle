"""Workflow definition, local executor, and scheduler."""

import dataclasses
import json
import threading

import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.runners import base
from scmle.runners import local
from scmle.workflow import definition
from scmle.workflow import handlers
import scmle.workflow.executor
import scmle.workflow.schedule
from tests import support_runners

KEYS = {"ml-owner": "owner-signing-key-0001", "second": "second-key-0002"}


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def store(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "state")


def fast():
    return retry.Retries(3, 2.0, 2.0, 60.0)


def pipeline(**changes):
    values = {"approvers": ("ml-owner", "second")}
    values.update(changes)
    flow = definition.standard_workflow(**values)
    steps = tuple(
        dataclasses.replace(
            step,
            retries=fast(),
            expires_seconds=7200.0,
            timeout_seconds=600.0,
        )
        for step in flow.steps
    )
    return definition.Workflow(
        flow.name,
        flow.version,
        steps,
        flow.schedule,
        cost_budget=flow.cost_budget,
    )


class Recorder:
    """Handlers that log calls and return small outputs."""

    def __init__(self, clock):
        self.clock = clock
        self.calls = []
        self.behaviour = {}
        self.lock = threading.Lock()

    def handler(self, name):
        def run(context):
            with self.lock:
                self.calls.append((name, context.attempt))
            action = self.behaviour.get(name)
            if action is not None:
                return action(context)
            return {"step": name, "key": context.idempotency_key[:8]}

        return run

    def handlers(self, workflow):
        return {
            s.name: self.handler(s.name)
            for s in workflow.steps
            if s.kind is not definition.StepKind.APPROVAL
        }

    def names(self):
        return [name for name, ignored in self.calls]


def executor_for(store, workflow, recorder, clock, **options):
    return scmle.workflow.executor.Executor(
        workflow,
        store,
        recorder.handlers(workflow),
        approver_keys=KEYS,
        clock=clock,
        sleep=clock.sleep,
        **options,
    )


def approve(executor, record, who="ml-owner", decision="approve"):
    signed = executor.sign_approval(
        record["run_id"], "approve", who, decision, KEYS[who]
    )
    executor.submit_approval(signed)
    return signed


# ---- definition -----------------------------------------------------------


def test_the_standard_pipeline_is_valid_and_ordered():
    flow = definition.standard_workflow()
    assert flow.order() == (
        "ingest",
        "train",
        "evaluate",
        "approve",
        "monitor",
        "batch_predict",
    )
    assert flow.layers() == (
        ("ingest",),
        ("train",),
        ("evaluate",),
        ("approve", "monitor"),
        ("batch_predict",),
    )
    assert flow.has_approval_upstream("batch_predict")
    assert flow.digest() == definition.standard_workflow().digest()
    assert flow.digest() != definition.standard_workflow(version="2").digest()


def test_invalid_definitions_are_refused():
    s = definition.StepKind
    with pytest.raises(errors.ConfigError, match="cycle"):
        definition.Workflow(
            "w",
            "1",
            (
                definition.Step("a", s.INGEST, ("b",)),
                definition.Step("b", s.TRAIN, ("a",)),
            ),
        )
    with pytest.raises(errors.ConfigError, match="unknown"):
        definition.Workflow(
            "w", "1", (definition.Step("a", s.INGEST, ("zzz",)),)
        )
    with pytest.raises(errors.ConfigError, match="unique"):
        definition.Workflow(
            "w",
            "1",
            (definition.Step("a", s.INGEST), definition.Step("a", s.TRAIN)),
        )
    with pytest.raises(errors.ConfigError, match="approved action"):
        definition.Workflow(
            "w",
            "1",
            (
                definition.Step("a", s.TRAIN),
                definition.Step(
                    "b", s.BATCH_PREDICT, ("a",), effect=definition.APPROVED
                ),
            ),
        )
    with pytest.raises(errors.ConfigError, match="approvers"):
        definition.Step("approve", s.APPROVAL)
    with pytest.raises(errors.ConfigError):
        definition.Step("bad name", s.TRAIN)
    with pytest.raises(errors.ConfigError):
        definition.Step("a", s.TRAIN, ("a",))
    with pytest.raises(errors.ConfigError):
        definition.Schedule(0)
    with pytest.raises(errors.ConfigError):
        definition.Workflow(
            "w", "1", (definition.Step("a", s.TRAIN),), max_concurrent_runs=0
        )


# ---- executor -------------------------------------------------------------


def test_a_run_waits_for_approval_and_the_approved_step_follows_it(store):
    clock = Clock()
    recorder = Recorder(clock)
    flow = pipeline()
    executor = executor_for(store, flow, recorder, clock)
    record = executor.start("req-1", requested_by="alice")
    assert record["state"] == "waiting_approval"
    assert record["steps"]["monitor"] == "succeeded"
    assert record["steps"]["batch_predict"] == "pending"
    assert "batch_predict" not in recorder.names()
    approve(executor, record)
    done = executor.resume(record["run_id"])
    assert done["state"] == "succeeded"
    assert recorder.names().count("ingest") == 1
    assert recorder.names()[-1] == "batch_predict"
    saved = executor.step_record(record["run_id"], "batch_predict")
    assert saved["state"] == "succeeded" and saved["idempotency_key"]


def test_a_repeated_invocation_runs_nothing_and_has_the_same_run_id(store):
    clock = Clock()
    recorder = Recorder(clock)
    executor = executor_for(store, pipeline(), recorder, clock)
    first = executor.start("req-1", "alice")
    approve(executor, first)
    done = executor.resume(first["run_id"])
    calls = list(recorder.calls)
    again = executor.start("req-1", "alice")
    assert again["run_id"] == first["run_id"] == done["run_id"]
    assert again["state"] == "succeeded" and recorder.calls == calls
    other = executor.start("req-2", "alice")
    assert other["run_id"] != first["run_id"]


def test_a_second_executor_cannot_work_on_a_run_in_flight(store):
    clock = Clock()
    recorder = Recorder(clock)
    flow = pipeline()
    second = executor_for(store, flow, Recorder(clock), clock, owner="second")
    seen = []

    def during(context):
        run_id = scmle.workflow.executor.run_id_for(flow, "req-1")
        try:
            second.start("req-1", "alice")
        except scmle.workflow.executor.RunInFlightError as error:
            seen.append(str(error))
        return {"run": run_id}

    recorder.behaviour["train"] = during
    executor_for(store, flow, recorder, clock).start("req-1", "alice")
    assert seen and "held by executor" in seen[0]


def test_max_concurrent_runs_is_enforced(store):
    clock = Clock()
    recorder = Recorder(clock)
    flow = pipeline()
    other = executor_for(store, flow, Recorder(clock), clock, owner="b")
    outcome = []

    def during(context):
        try:
            other.start("req-2", "alice")
        except scmle.workflow.executor.ConcurrencyLimitError as error:
            outcome.append(type(error).__name__)
        return {}

    recorder.behaviour["ingest"] = during
    executor_for(store, flow, recorder, clock).start("req-1", "alice")
    assert outcome == ["ConcurrencyLimitError"]


def test_partial_failure_skips_dependents_and_resume_runs_only_the_rest(store):
    clock = Clock()
    recorder = Recorder(clock)
    flow = pipeline()
    executor = executor_for(store, flow, recorder, clock)

    def broken(context):
        raise errors.DataError("bad metric")

    recorder.behaviour["evaluate"] = broken
    record = executor.start("req-1", "alice")
    assert record["state"] == "failed"
    assert record["steps"] == {
        "ingest": "succeeded",
        "train": "succeeded",
        "evaluate": "failed",
        "approve": "skipped",
        "monitor": "skipped",
        "batch_predict": "skipped",
    }
    failed = executor.step_record(record["run_id"], "evaluate")
    assert "bad metric" in failed["error"] and len(failed["attempts"]) == 1
    recorder.behaviour.pop("evaluate")
    before = list(recorder.calls)
    healed = executor.resume(record["run_id"])
    assert healed["state"] == "waiting_approval"
    new = recorder.calls[len(before) :]
    assert [n for n, ignored in new] == ["evaluate", "monitor"]


def test_interrupted_steps_retry_with_backoff_and_the_same_key(store):
    clock = Clock()
    recorder = Recorder(clock)
    keys = []

    def flaky(context):
        keys.append(context.idempotency_key)
        if context.attempt < 3:
            raise scmle.workflow.executor.StepInterruptedError("preempted")
        return {"ok": True}

    recorder.behaviour["train"] = flaky
    executor = executor_for(store, pipeline(), recorder, clock)
    record = executor.start("req-1", "alice")
    assert record["steps"]["train"] == "succeeded"
    assert len(set(keys)) == 1 and len(keys) == 3
    assert clock.sleeps == [2.0, 4.0]
    saved = executor.step_record(record["run_id"], "train")
    assert [a["state"] for a in saved["attempts"]] == [
        "retryable",
        "retryable",
        "succeeded",
    ]


def test_retries_are_bounded(store):
    clock = Clock()
    recorder = Recorder(clock)

    def always(context):
        raise scmle.workflow.executor.StepInterruptedError("preempted")

    recorder.behaviour["train"] = always
    record = executor_for(store, pipeline(), recorder, clock).start("r", "a")
    assert record["steps"]["train"] == "failed"
    assert recorder.names().count("train") == 3


def test_a_crashed_executor_is_taken_over_after_its_lease(store):
    clock = Clock()
    recorder = Recorder(clock)
    flow = pipeline()
    executor = executor_for(store, flow, recorder, clock, lease_seconds=300.0)

    def dies(context):
        raise support_runners.Killed

    recorder.behaviour["train"] = dies
    with pytest.raises(support_runners.Killed):
        executor.start("req-1", "alice")
    run_id = scmle.workflow.executor.run_id_for(flow, "req-1")
    # A killed process cannot release its lease.
    store.put(
        f"workflows/{flow.name}/runs/{run_id}/lock",
        json.dumps({"owner": "dead", "expires": clock.now + 300}).encode(),
    )
    successor = executor_for(store, flow, recorder, clock, owner="b")
    recorder.behaviour.pop("train")
    with pytest.raises(scmle.workflow.executor.RunInFlightError):
        successor.resume(run_id)
    clock.now += 301
    record = successor.resume(run_id)
    assert record["state"] == "waiting_approval"
    assert recorder.names().count("ingest") == 1
    assert recorder.names().count("train") == 2


def test_timeouts_budgets_and_the_run_budget(store):
    clock = Clock()
    recorder = Recorder(clock)

    def slow(context):
        clock.now += 10_000
        context.check()

    recorder.behaviour["ingest"] = slow
    record = executor_for(store, pipeline(), recorder, clock).start("t", "a")
    assert record["steps"]["ingest"] == "failed"
    assert record["state"] == "failed"

    clock2, recorder2 = Clock(), None
    recorder2 = Recorder(clock2)

    def spender(amount):
        def run(context):
            context.spend(amount)
            return {"spent": amount}

        return run

    recorder2.behaviour["ingest"] = spender(4.0)
    recorder2.behaviour["train"] = spender(7.0)
    flow = pipeline(cost_budget=10.0)
    record = executor_for(store, flow, recorder2, clock2).start("b", "a")
    assert record["state"] == "budget_stopped"
    assert record["cost"] >= 4.0
    assert recorder2.names() == ["ingest", "train"]
    skipped = executor_for(store, flow, recorder2, clock2).step_record(
        record["run_id"], "evaluate"
    )
    assert skipped["state"] == "skipped"


def test_a_step_over_its_own_budget_fails(store):
    clock = Clock()
    recorder = Recorder(clock)
    steps = [
        definition.Step(
            "ingest",
            definition.StepKind.INGEST,
            cost_budget=1.0,
            retries=fast(),
        ),
    ]
    flow = definition.Workflow("solo", "1", tuple(steps))
    recorder.behaviour["ingest"] = lambda context: context.spend(2.0)
    record = executor_for(store, flow, recorder, clock).start("x", "a")
    assert record["state"] == "failed"
    saved = executor_for(store, flow, recorder, clock).step_record(
        record["run_id"], "ingest"
    )
    assert "StepBudgetError" in saved["error"]


def test_independent_steps_run_in_parallel_and_a_group_limit_holds(store):
    clock = Clock()
    s = definition.StepKind
    flow = definition.Workflow(
        "par",
        "1",
        (
            definition.Step("a", s.INGEST, retries=fast()),
            definition.Step("b", s.INGEST, retries=fast()),
            definition.Step(
                "c", s.TRAIN, concurrency_group="gpu", retries=fast()
            ),
            definition.Step(
                "d", s.TRAIN, concurrency_group="gpu", retries=fast()
            ),
        ),
        max_parallel=4,
    )
    barrier = threading.Barrier(2, timeout=10)
    inside = []
    peak = [0]
    guard = threading.Lock()

    def meet(context):
        barrier.wait()
        return {}

    def grouped(context):
        with guard:
            inside.append(1)
            peak[0] = max(peak[0], len(inside))
        threading.Event().wait(0.05)
        with guard:
            inside.pop()
        return {}

    handlers_ = {"a": meet, "b": meet, "c": grouped, "d": grouped}
    record = scmle.workflow.executor.Executor(
        flow, store, handlers_, clock=clock, sleep=clock.sleep
    ).start("p", "a")
    assert record["state"] == "succeeded"
    assert peak[0] == 1


def test_approval_rules_are_enforced(store):
    clock = Clock()
    recorder = Recorder(clock)
    executor = executor_for(store, pipeline(), recorder, clock)
    record = executor.start("req-1", requested_by="ml-owner")
    good = executor.sign_approval(
        record["run_id"], "approve", "second", "approve", KEYS["second"]
    )
    forged = {**good, "signature": "0" * 64}
    with pytest.raises(
        scmle.workflow.executor.ApprovalError, match="signature"
    ):
        executor.submit_approval(forged)
    tampered = {**good, "decision": "reject"}
    with pytest.raises(
        scmle.workflow.executor.ApprovalError, match="signature"
    ):
        executor.submit_approval(tampered)
    own = executor.sign_approval(
        record["run_id"], "approve", "ml-owner", "approve", KEYS["ml-owner"]
    )
    with pytest.raises(
        scmle.workflow.executor.ApprovalError, match="requester"
    ):
        executor.submit_approval(own)
    stranger = executor.sign_approval(
        record["run_id"], "approve", "mallory", "approve", "any-key-at-all"
    )
    with pytest.raises(
        scmle.workflow.executor.ApprovalError, match="not listed"
    ):
        executor.submit_approval(stranger)
    wrong_step = {**good, "step": "monitor"}
    with pytest.raises(scmle.workflow.executor.ApprovalError):
        executor.submit_approval(wrong_step)
    executor.submit_approval(good)
    with pytest.raises(scmle.workflow.executor.ApprovalError, match="already"):
        executor.submit_approval(good)
    assert executor.resume(record["run_id"])["state"] == "succeeded"


def test_a_rejection_or_an_expiry_never_reaches_the_approved_step(store):
    clock = Clock()
    recorder = Recorder(clock)
    executor = executor_for(store, pipeline(), recorder, clock)
    record = executor.start("req-1", requested_by="alice")
    approve(executor, record, "second", "reject")
    done = executor.resume(record["run_id"])
    assert done["state"] == "rejected"
    assert "batch_predict" not in recorder.names()
    late = executor.start("req-2", requested_by="alice")
    assert late["state"] == "waiting_approval"
    clock.now += 7201
    expired = executor.resume(late["run_id"])
    assert expired["state"] == "expired"
    assert "batch_predict" not in recorder.names()


def test_a_changed_definition_cannot_continue_an_old_run(store):
    clock = Clock()
    recorder = Recorder(clock)
    first = executor_for(store, pipeline(), recorder, clock)
    record = first.start("req-1", "alice")
    changed = pipeline()
    changed = definition.Workflow(
        changed.name, "2", changed.steps, changed.schedule
    )
    assert (
        scmle.workflow.executor.run_id_for(changed, "req-1") != record["run_id"]
    )
    other = executor_for(store, changed, recorder, clock)
    with pytest.raises(errors.ConfigError, match="different workflow"):
        other.resume(record["run_id"])


def test_every_non_approval_step_needs_a_handler(store):
    with pytest.raises(errors.ConfigError, match="no handler"):
        scmle.workflow.executor.Executor(
            definition.standard_workflow(), store, {}
        )


# ---- schedule -------------------------------------------------------------


def scheduled(policy, name=None, **options):
    schedule = definition.Schedule(3600.0, 0.0, policy, **options)
    flow = pipeline()
    return definition.Workflow(
        name or flow.name, flow.version, flow.steps, schedule
    )


def test_missed_slots_follow_the_declared_policy(store):
    for policy, expected_runs, expected_missed in (
        (definition.CatchUp.RUN_ALL, 5, 0),
        (definition.CatchUp.LATEST_ONLY, 1, 4),
        (definition.CatchUp.SKIP, 0, 5),
    ):
        flow = scheduled(policy, f"wf-{policy.value}")
        clock = Clock(5 * 3600.0 + 2000.0)
        executor = executor_for(store, flow, Recorder(clock), clock)
        scheduler = scmle.workflow.schedule.Scheduler(flow, store)
        scheduler.save(0.0, {})
        report = scheduler.tick(executor)
        assert len(report.ran) == expected_runs, policy
        assert len(report.missed) == expected_missed, policy
        slots = [slot for slot, ignored in report.ran]
        assert slots == sorted(slots)
    current = scheduled(definition.CatchUp.SKIP, "wf-current")
    clock = Clock(5 * 3600.0 + 10.0)
    executor = executor_for(store, current, Recorder(clock), clock)
    scheduler = scmle.workflow.schedule.Scheduler(current, store)
    scheduler.save(0.0, {})
    report = scheduler.tick(executor)
    assert [slot for slot, ignored in report.ran] == [5 * 3600.0]
    assert len(report.missed) == 4


def test_the_first_tick_does_not_backfill_and_a_second_tick_is_quiet(store):
    flow = scheduled(definition.CatchUp.RUN_ALL)
    clock = Clock(10 * 3600.0 + 1.0)
    recorder = Recorder(clock)
    executor = executor_for(store, flow, recorder, clock)
    scheduler = scmle.workflow.schedule.Scheduler(flow, store)
    first = scheduler.tick(executor)
    assert [s for s, ignored in first.ran] == [10 * 3600.0]
    calls = list(recorder.calls)
    second = scheduler.tick(executor)
    assert second.ran == () and second.missed == () and recorder.calls == calls
    clock.now += 3600.0
    third = scheduler.tick(executor)
    assert [s for s, ignored in third.ran] == [11 * 3600.0]


def test_a_scheduler_that_lost_its_state_does_not_repeat_finished_slots(store):
    flow = scheduled(definition.CatchUp.LATEST_ONLY)
    clock = Clock(3600.0 + 1.0)
    recorder = Recorder(clock)
    executor = executor_for(store, flow, recorder, clock)
    scmle.workflow.schedule.Scheduler(flow, store).tick(executor)
    store.delete(f"workflows/{flow.name}/schedule/state.json")
    calls = list(recorder.calls)
    again = scmle.workflow.schedule.Scheduler(flow, store).tick(executor)
    assert len(again.ran) == 1 and recorder.calls == calls


def test_a_slot_that_cannot_start_is_deferred_and_stays_due(store):
    flow = scheduled(definition.CatchUp.LATEST_ONLY)
    clock = Clock(3600.0 + 1.0)
    recorder = Recorder(clock)
    executor = executor_for(store, flow, recorder, clock)
    rival = executor_for(store, flow, Recorder(clock), clock, owner="rival")
    scheduler = scmle.workflow.schedule.Scheduler(flow, store)
    outcome = []

    def during(context):
        outcome.append(scheduler.tick(rival))
        return {}

    recorder.behaviour["ingest"] = during
    scheduler.tick(executor)
    assert outcome[0].deferred == (3600.0,) and outcome[0].ran == ()


def test_a_schedule_is_required():
    with pytest.raises(errors.ConfigError):
        scmle.workflow.schedule.Scheduler(
            pipeline(), objectstore.LocalObjectStore("/tmp/x-sched")
        )


# ---- runner integration -----------------------------------------------------


def test_a_training_step_runs_through_a_runner_and_resumes_inside_it(tmp_path):
    clock = Clock()
    fake = support_runners.FakeClock()
    durable = objectstore.LocalObjectStore(tmp_path / "jobs")
    runner = local.Runner(
        durable,
        support_runners.toy_workload(fake),
        serving_identity="serve",
        clock=base.Clock(fake.now, fake.sleep),
        poll_seconds=1.0,
        interruptions=lambda attempt, step: attempt == 1 and step == 3,
    )

    def spec_for(context):
        return base.Job(
            run_id="job-" + context.idempotency_key[:16],
            name="train",
            project={"data": "d", "target": "y"},
            identity="train",
            budget=base.CostBudget(10.0, base.CostRate(per_cpu_hour=1.0)),
        )

    recorder = Recorder(clock)
    flow = pipeline()
    handlers_ = recorder.handlers(flow)
    handlers_["train"] = handlers.runner_handler(runner, spec_for)
    executor = scmle.workflow.executor.Executor(
        flow,
        objectstore.LocalObjectStore(tmp_path / "state"),
        handlers_,
        approver_keys=KEYS,
        clock=clock,
        sleep=clock.sleep,
    )
    record = executor.start("req-1", "alice")
    assert record["steps"]["train"] == "succeeded"
    saved = executor.step_record(record["run_id"], "train")
    assert saved["outputs"]["attempts"] == 2
    assert record["cost"] > 0
