"""Workflow definition, local executor, scheduler and emitted definitions."""

import copy
import dataclasses
import json
import threading

import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import cloud
from scmle.data.store import objectstore
from scmle.runners import base
from scmle.runners import local
from scmle.workflow import definition as wd
from scmle.workflow import emit
from scmle.workflow import executor as ex
from scmle.workflow import handlers
from scmle.workflow import schedule as sched
from tests import support_cloud as fakes
from tests import support_runners as sr

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


def make_store(kind, tmp_path):
    if kind == "local":
        return objectstore.LocalObjectStore(tmp_path / "state")
    return cloud.S3ObjectStore(fakes.FakeS3Client(), "bkt", "wf")


@pytest.fixture(params=["local", "s3"])
def store(request, tmp_path):
    return make_store(request.param, tmp_path)


def fast():
    return retry.RetryPolicy(3, 2.0, 2.0, 60.0)


def pipeline(**changes):
    values = {"approvers": ("ml-owner", "second")}
    values.update(changes)
    flow = wd.standard_workflow(**values)
    steps = tuple(
        dataclasses.replace(
            step,
            retries=fast(),
            expires_seconds=7200.0,
            timeout_seconds=600.0,
        )
        for step in flow.steps
    )
    return wd.Workflow(
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
            if s.kind is not wd.StepKind.APPROVAL
        }

    def names(self):
        return [name for name, ignored in self.calls]


def executor_for(store, workflow, recorder, clock, **options):
    return ex.LocalExecutor(
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
    flow = wd.standard_workflow()
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
    assert flow.digest() == wd.standard_workflow().digest()
    assert flow.digest() != wd.standard_workflow(version="2").digest()


def test_invalid_definitions_are_refused():
    s = wd.StepKind
    with pytest.raises(errors.ConfigError, match="cycle"):
        wd.Workflow(
            "w",
            "1",
            (
                wd.Step("a", s.INGEST, ("b",)),
                wd.Step("b", s.TRAIN, ("a",)),
            ),
        )
    with pytest.raises(errors.ConfigError, match="unknown"):
        wd.Workflow("w", "1", (wd.Step("a", s.INGEST, ("zzz",)),))
    with pytest.raises(errors.ConfigError, match="unique"):
        wd.Workflow("w", "1", (wd.Step("a", s.INGEST), wd.Step("a", s.TRAIN)))
    with pytest.raises(errors.ConfigError, match="approved action"):
        wd.Workflow(
            "w",
            "1",
            (
                wd.Step("a", s.TRAIN),
                wd.Step("b", s.BATCH_PREDICT, ("a",), effect=wd.APPROVED),
            ),
        )
    with pytest.raises(errors.ConfigError, match="approvers"):
        wd.Step("approve", s.APPROVAL)
    with pytest.raises(errors.ConfigError):
        wd.Step("bad name", s.TRAIN)
    with pytest.raises(errors.ConfigError):
        wd.Step("a", s.TRAIN, ("a",))
    with pytest.raises(errors.ConfigError):
        wd.Schedule(0)
    with pytest.raises(errors.ConfigError):
        wd.Workflow("w", "1", (wd.Step("a", s.TRAIN),), max_concurrent_runs=0)


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
        run_id = ex.run_id_for(flow, "req-1")
        try:
            second.start("req-1", "alice")
        except ex.RunInFlightError as error:
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
        except ex.ConcurrencyLimitError as error:
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
            raise ex.StepInterruptedError("preempted")
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
        raise ex.StepInterruptedError("preempted")

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
        raise sr.Killed()

    recorder.behaviour["train"] = dies
    with pytest.raises(sr.Killed):
        executor.start("req-1", "alice")
    run_id = ex.run_id_for(flow, "req-1")
    # A killed process cannot release its lease.
    store.put(
        f"workflows/{flow.name}/runs/{run_id}/lock",
        json.dumps({"owner": "dead", "expires": clock.now + 300}).encode(),
    )
    successor = executor_for(store, flow, recorder, clock, owner="b")
    recorder.behaviour.pop("train")
    with pytest.raises(ex.RunInFlightError):
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
        wd.Step("ingest", wd.StepKind.INGEST, cost_budget=1.0, retries=fast()),
    ]
    flow = wd.Workflow("solo", "1", tuple(steps))
    recorder.behaviour["ingest"] = lambda context: context.spend(2.0)
    record = executor_for(store, flow, recorder, clock).start("x", "a")
    assert record["state"] == "failed"
    saved = executor_for(store, flow, recorder, clock).step_record(
        record["run_id"], "ingest"
    )
    assert "StepBudgetError" in saved["error"]


def test_independent_steps_run_in_parallel_and_a_group_limit_holds(store):
    clock = Clock()
    s = wd.StepKind
    flow = wd.Workflow(
        "par",
        "1",
        (
            wd.Step("a", s.INGEST, retries=fast()),
            wd.Step("b", s.INGEST, retries=fast()),
            wd.Step("c", s.TRAIN, concurrency_group="gpu", retries=fast()),
            wd.Step("d", s.TRAIN, concurrency_group="gpu", retries=fast()),
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
    record = ex.LocalExecutor(
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
    with pytest.raises(ex.ApprovalError, match="signature"):
        executor.submit_approval(forged)
    tampered = {**good, "decision": "reject"}
    with pytest.raises(ex.ApprovalError, match="signature"):
        executor.submit_approval(tampered)
    own = executor.sign_approval(
        record["run_id"], "approve", "ml-owner", "approve", KEYS["ml-owner"]
    )
    with pytest.raises(ex.ApprovalError, match="requester"):
        executor.submit_approval(own)
    stranger = executor.sign_approval(
        record["run_id"], "approve", "mallory", "approve", "any-key-at-all"
    )
    with pytest.raises(ex.ApprovalError, match="not listed"):
        executor.submit_approval(stranger)
    wrong_step = {**good, "step": "monitor"}
    with pytest.raises(ex.ApprovalError):
        executor.submit_approval(wrong_step)
    executor.submit_approval(good)
    with pytest.raises(ex.ApprovalError, match="already"):
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
    changed = wd.Workflow(changed.name, "2", changed.steps, changed.schedule)
    assert ex.run_id_for(changed, "req-1") != record["run_id"]
    other = executor_for(store, changed, recorder, clock)
    with pytest.raises(errors.ConfigError, match="different workflow"):
        other.resume(record["run_id"])


def test_every_non_approval_step_needs_a_handler(store):
    with pytest.raises(errors.ConfigError, match="no handler"):
        ex.LocalExecutor(wd.standard_workflow(), store, {})


# ---- schedule -------------------------------------------------------------


def scheduled(policy, name=None, **options):
    schedule = wd.Schedule(3600.0, 0.0, policy, **options)
    flow = pipeline()
    return wd.Workflow(name or flow.name, flow.version, flow.steps, schedule)


def test_missed_slots_follow_the_declared_policy(store):
    for policy, expected_runs, expected_missed in (
        (wd.CatchUp.RUN_ALL, 5, 0),
        (wd.CatchUp.LATEST_ONLY, 1, 4),
        (wd.CatchUp.SKIP, 0, 5),
    ):
        flow = scheduled(policy, f"wf-{policy.value}")
        clock = Clock(5 * 3600.0 + 2000.0)
        executor = executor_for(store, flow, Recorder(clock), clock)
        scheduler = sched.Scheduler(flow, store)
        scheduler.save(0.0, {})
        report = scheduler.tick(executor)
        assert len(report.ran) == expected_runs, policy
        assert len(report.missed) == expected_missed, policy
        slots = [slot for slot, ignored in report.ran]
        assert slots == sorted(slots)
    current = scheduled(wd.CatchUp.SKIP, "wf-current")
    clock = Clock(5 * 3600.0 + 10.0)
    executor = executor_for(store, current, Recorder(clock), clock)
    scheduler = sched.Scheduler(current, store)
    scheduler.save(0.0, {})
    report = scheduler.tick(executor)
    assert [slot for slot, ignored in report.ran] == [5 * 3600.0]
    assert len(report.missed) == 4


def test_the_first_tick_does_not_backfill_and_a_second_tick_is_quiet(store):
    flow = scheduled(wd.CatchUp.RUN_ALL)
    clock = Clock(10 * 3600.0 + 1.0)
    recorder = Recorder(clock)
    executor = executor_for(store, flow, recorder, clock)
    scheduler = sched.Scheduler(flow, store)
    first = scheduler.tick(executor)
    assert [s for s, ignored in first.ran] == [10 * 3600.0]
    calls = list(recorder.calls)
    second = scheduler.tick(executor)
    assert second.ran == () and second.missed == () and recorder.calls == calls
    clock.now += 3600.0
    third = scheduler.tick(executor)
    assert [s for s, ignored in third.ran] == [11 * 3600.0]


def test_a_scheduler_that_lost_its_state_does_not_repeat_finished_slots(store):
    flow = scheduled(wd.CatchUp.LATEST_ONLY)
    clock = Clock(3600.0 + 1.0)
    recorder = Recorder(clock)
    executor = executor_for(store, flow, recorder, clock)
    sched.Scheduler(flow, store).tick(executor)
    store.delete(f"workflows/{flow.name}/schedule/state.json")
    calls = list(recorder.calls)
    again = sched.Scheduler(flow, store).tick(executor)
    assert len(again.ran) == 1 and recorder.calls == calls


def test_a_slot_that_cannot_start_is_deferred_and_stays_due(store):
    flow = scheduled(wd.CatchUp.LATEST_ONLY)
    clock = Clock(3600.0 + 1.0)
    recorder = Recorder(clock)
    executor = executor_for(store, flow, recorder, clock)
    rival = executor_for(store, flow, Recorder(clock), clock, owner="rival")
    scheduler = sched.Scheduler(flow, store)
    outcome = []

    def during(context):
        outcome.append(scheduler.tick(rival))
        return {}

    recorder.behaviour["ingest"] = during
    scheduler.tick(executor)
    assert outcome[0].deferred == (3600.0,) and outcome[0].ran == ()


def test_a_schedule_is_required():
    with pytest.raises(errors.ConfigError):
        sched.Scheduler(
            pipeline(), objectstore.LocalObjectStore("/tmp/x-sched")
        )


# ---- runner integration -----------------------------------------------------


def test_a_training_step_runs_through_a_runner_and_resumes_inside_it(tmp_path):
    clock = Clock()
    fake = sr.FakeClock()
    durable = objectstore.LocalObjectStore(tmp_path / "jobs")
    runner = local.LocalRunner(
        durable,
        sr.toy_workload(fake),
        serving_identity="serve",
        clock=base.Clock(fake.now, fake.sleep),
        poll_seconds=1.0,
        interruptions=lambda attempt, step: attempt == 1 and step == 3,
    )

    def spec_for(context):
        return base.JobSpec(
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
    executor = ex.LocalExecutor(
        flow,
        make_store("local", tmp_path),
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


# ---- emitted definitions ----------------------------------------------------


def test_the_step_functions_document_is_structurally_valid():
    flow = scheduled(wd.CatchUp.LATEST_ONLY)
    emitted = emit.to_step_functions(flow, {"cluster": "arn:aws:ecs:cluster/c"})
    document = emitted["definition"]
    assert emit.validate_step_functions(document) == []
    json.dumps(emitted)
    states = document["States"]
    assert document["StartAt"] == "Layer0"
    assert states["Layer3"]["Type"] == "Parallel"
    assert {b["StartAt"] for b in states["Layer3"]["Branches"]} == {
        "approve",
        "monitor",
    }
    task = states["Layer1"]
    assert task["Retry"][0]["MaxAttempts"] == 2
    assert task["Retry"][0]["BackoffRate"] == 2.0
    assert task["TimeoutSeconds"] == 600
    approval = states["Layer3"]["Branches"][0]["States"]["approve"]
    assert approval["Resource"].endswith("waitForTaskToken")
    assert approval["TimeoutSeconds"] == 7200
    assert emitted["schedule"] == "rate(1 hour)"
    assert emitted["controls"]["approved_steps"] == ["batch_predict"]
    assert emitted["controls"]["max_concurrent_runs"] == 1


def test_the_step_functions_validator_finds_structural_problems():
    good = emit.to_step_functions(wd.standard_workflow())["definition"]
    broken = copy.deepcopy(good)
    broken["States"]["Layer0"]["Next"] = "Nowhere"
    assert any("Nowhere" in p for p in emit.validate_step_functions(broken))
    broken = copy.deepcopy(good)
    broken["StartAt"] = "Missing"
    assert emit.validate_step_functions(broken)
    broken = copy.deepcopy(good)
    broken["States"]["Orphan"] = {"Type": "Succeed"}
    assert any("unreachable" in p for p in emit.validate_step_functions(broken))
    broken = copy.deepcopy(good)
    broken["States"]["Layer0"]["Type"] = "Bogus"
    assert any("Type" in p for p in emit.validate_step_functions(broken))
    broken = copy.deepcopy(good)
    del broken["States"]["Layer1"]["Next"]
    assert any(
        "Next and End" in p for p in emit.validate_step_functions(broken)
    )
    broken = copy.deepcopy(good)
    broken["States"]["Layer1"]["Retry"][0]["BackoffRate"] = 0.5
    assert any("BackoffRate" in p for p in emit.validate_step_functions(broken))
    broken = copy.deepcopy(good)
    broken["States"]["Layer3"]["Branches"][0]["States"]["approve"][
        "Resource"
    ] = "x"
    assert any("branch 0" in p for p in emit.validate_step_functions(broken))
    assert emit.validate_step_functions({"States": {}})


def test_the_cloud_workflows_document_is_structurally_valid():
    flow = scheduled(wd.CatchUp.LATEST_ONLY)
    emitted = emit.to_cloud_workflows(flow, {"approve": "projects/p/topics/t"})
    document = emitted["definition"]
    assert emit.validate_cloud_workflows(document) == []
    json.dumps(emitted)
    steps = document["main"]["steps"]
    names = [next(iter(item)) for item in steps]
    assert names[0] == "init" and names[-1] == "finish"
    parallel = next(
        i for i in steps if any(k.startswith("layer_approve") for k in i)
    )
    assert "parallel" in next(iter(parallel.values()))
    text = json.dumps(document)
    assert "events.await_callback" in text and "approval rejected" in text
    assert "execution_key" in text
    assert emitted["schedule"] == "0 */1 * * *"
    assert emitted["controls"]["approved_steps"] == ["batch_predict"]


def test_the_cloud_workflows_validator_finds_structural_problems():
    good = emit.to_cloud_workflows(wd.standard_workflow())["definition"]
    broken = copy.deepcopy(good)
    broken["main"]["steps"].append(copy.deepcopy(broken["main"]["steps"][0]))
    assert any(
        "more than once" in p for p in emit.validate_cloud_workflows(broken)
    )
    broken = copy.deepcopy(good)
    broken["main"]["steps"][1] = {"ingest": {"nonsense": 1}}
    assert any("keyword" in p for p in emit.validate_cloud_workflows(broken))
    broken = copy.deepcopy(good)
    broken["main"]["steps"].append(
        {"jump": {"assign": [{"a": 1}], "next": "nope"}}
    )
    assert any("nope" in p for p in emit.validate_cloud_workflows(broken))
    broken = copy.deepcopy(good)
    first = next(i for i in broken["main"]["steps"] if "ingest" in i)["ingest"]
    first["retry"]["max_retries"] = -1
    assert any(
        "max_retries" in p for p in emit.validate_cloud_workflows(broken)
    )
    assert emit.validate_cloud_workflows({"main": {}})
    assert emit.validate_cloud_workflows(
        {"main": {"steps": [{"a": 1, "b": 2}]}}
    )


def test_schedules_convert_or_are_refused():
    assert emit.eventbridge_rate(60) == "rate(1 minute)"
    assert emit.eventbridge_rate(900) == "rate(15 minutes)"
    assert emit.eventbridge_rate(86400) == "rate(1 day)"
    assert emit.eventbridge_rate(7200) == "rate(2 hours)"
    assert emit.cron_for(900) == "*/15 * * * *"
    assert emit.cron_for(21600) == "0 */6 * * *"
    assert emit.cron_for(86400) == "0 0 * * *"
    for bad in (30, 90):
        with pytest.raises(errors.ConfigError):
            emit.eventbridge_rate(bad)
    with pytest.raises(errors.ConfigError):
        emit.cron_for(7 * 60)
    with pytest.raises(errors.ConfigError):
        emit.cron_for(5 * 3600)
