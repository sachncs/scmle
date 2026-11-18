"""Behaviour of the workflow executor's modules, one concern at a time."""

import hashlib
import json
import logging
import threading

import pytest

from scmle.contracts import errors
from scmle.contracts import signing
from scmle.runtime import retry
from scmle.runtime import state
from scmle.storage import objectstore
from scmle.workflow import definition
from scmle.workflow.executor import approvals
from scmle.workflow.executor import context
from scmle.workflow.executor import failures
from scmle.workflow.executor import leases
from scmle.workflow.executor import loop
from scmle.workflow.executor import records
from scmle.workflow.executor import runs
from scmle.workflow.executor import status
from scmle.workflow.executor import steps

KEYS = {"boss": "boss-signing-key-0001", "peer": "peer-signing-key-0002"}


class Clock:
    def __init__(self, now=1000.0):
        self.now = now
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def quick(attempts=3):
    return retry.Retries(attempts, 2.0, 3.0, 100.0)


def task(name, *needs, **changes):
    values = {"retries": quick(), "timeout_seconds": 50.0}
    values.update(changes)
    return definition.Step(
        name, definition.StepKind.TRAIN, tuple(needs), **values
    )


def gate(name, *needs, expires=100.0):
    return definition.Step(
        name,
        definition.StepKind.APPROVAL,
        tuple(needs),
        approvers=("boss", "peer"),
        expires_seconds=expires,
    )


def flow(*items, **changes):
    return definition.Workflow("flow", "1", tuple(items), **changes)


@pytest.fixture
def store(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "state")


@pytest.fixture
def clock():
    return Clock()


def build(store, clock, workflow, handlers=None, **options):
    if handlers is None:
        handlers = {
            s.name: (lambda c, n=s.name: {"made": n})
            for s in workflow.steps
            if s.kind is not definition.StepKind.APPROVAL
        }
    executor = runs.Executor(
        workflow,
        store,
        handlers,
        approver_keys=KEYS,
        clock=clock,
        sleep=clock.sleep,
        **options,
    )
    executor.timer = clock
    return executor


# ---- context ---------------------------------------------------------------


def frame(store, clock, step=None, **changes):
    return context.StepContext(
        "run",
        step or task("a", cost_budget=10.0),
        1,
        "key",
        {},
        store,
        clock(),
        clock,
        **changes,
    )


def test_a_context_check_allows_exactly_the_timeout(store, clock):
    current = frame(store, clock)
    clock.now += 50.0
    current.check()
    clock.now += 0.001
    with pytest.raises(failures.DeadlineError, match=r"a passed 50\.0s"):
        current.check()


def test_a_context_spend_accumulates_and_allows_exactly_the_step_budget(
    store, clock
):
    current = frame(store, clock)
    current.spend(4.0)
    current.spend(6.0)
    assert current.spent == 10.0
    with pytest.raises(failures.StepBudgetError, match=r"a cost passed 10\.0"):
        current.spend(0.5)
    assert current.spent == 10.5


def test_a_context_spend_checks_the_run_budget_with_earlier_cost(store, clock):
    current = frame(store, clock, task("a"), run_spent=7.0, run_budget=10.0)
    current.spend(3.0)
    with pytest.raises(failures.StepBudgetError, match="run's cost budget"):
        current.spend(0.01)


def test_a_context_without_budgets_never_refuses_spending(store, clock):
    current = frame(store, clock, task("a"))
    current.spend(1e9)
    assert current.spent == 1e9


# ---- records ---------------------------------------------------------------


def test_a_parse_returns_a_json_object():
    assert records.parse("k", b'{"a": 1}') == {"a": 1}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (b"{not json", "workflow record k is corrupt"),
        (b"\xff\xfe", "workflow record k is corrupt"),
        (b"[1, 2]", "workflow record k is not a JSON object"),
    ],
)
def test_a_parse_refuses_what_is_not_a_json_object(data, message):
    with pytest.raises(errors.ArtifactError, match=message) as caught:
        records.parse("k", data)
    assert caught.value.details["key"] == "k"


def test_a_distill_is_the_sha256_of_sorted_json():

    expected = hashlib.sha256(b'{"a": 1, "b": 2}').hexdigest()
    assert records.distill({"b": 2, "a": 1}) == expected
    assert records.distill({"a": 1, "b": 2}) == expected
    assert len(expected) == 64


def test_a_run_id_is_the_name_and_sixteen_digest_characters():
    workflow = flow(task("a"))
    run = records.run_id_for(workflow, "slot-1")
    digest = records.distill(["flow", "1", "slot-1"])
    assert run == f"flow-{digest[:16]}"
    assert records.run_id_for(workflow, "slot-1") == run
    assert records.run_id_for(workflow, "slot-2") != run


def test_a_run_id_of_a_long_name_is_cut_to_sixty_four_characters():
    workflow = definition.Workflow("w" * 60, "1", (task("a"),))
    assert len(records.run_id_for(workflow, "x")) == 64


def test_records_read_keys_under_the_run_prefix(store, clock):
    executor = build(store, clock, flow(task("a")))
    assert executor.base("r1") == "workflows/flow/runs/r1/"
    assert executor.run_record("r1") is None
    assert executor.step_record("r1", "a") is None
    store.put("workflows/flow/runs/r1/run.json", b'{"state": "x"}')
    store.put("workflows/flow/runs/r1/steps/a.json", b'{"state": "y"}')
    assert executor.run_record("r1") == {"state": "x"}
    assert executor.step_record("r1", "a") == {"state": "y"}
    store.put("workflows/flow/runs/r1/run.json", b"[]")
    with pytest.raises(errors.ArtifactError):
        executor.run_record("r1")


# ---- leases ----------------------------------------------------------------


def lock(store, run="r1"):
    return json.loads(store.get(f"workflows/flow/runs/{run}/lock")[0])


def test_the_lease_constants_are_the_documented_policy():
    assert leases.LEASE_SECONDS == 300.0
    assert leases.LEASE_ATTEMPTS == 3
    assert (
        objectstore.ConflictError,
        objectstore.AbsentError,
    ) == leases.CHANGED


def test_a_lease_body_expires_after_its_validity_or_at_the_floor(store, clock):
    executor = build(store, clock, flow(task("a")), lease_seconds=60.0)
    body = json.loads(executor.lease("r1", 4))
    assert body == {"owner": "executor", "expires": 1060.0, "fence": 4}
    floored = json.loads(executor.lease("r1", 4, floor=5000.0))
    assert floored["expires"] == 5000.0


def test_acquiring_a_free_run_takes_fence_one(store, clock):
    executor = build(store, clock, flow(task("a")), lease_seconds=60.0)
    executor.acquire("r1")
    assert executor.fences == {"r1": 1}
    assert lock(store) == {"owner": "executor", "expires": 1060.0, "fence": 1}


def test_a_valid_lease_held_by_another_refuses(store, clock):
    first = build(store, clock, flow(task("a")), lease_seconds=60.0)
    second = build(store, clock, flow(task("a")), owner="other")
    first.acquire("r1")
    clock.now += 59.9
    with pytest.raises(failures.BusyError, match="held by executor"):
        second.acquire("r1")
    assert "r1" not in second.fences


def test_an_expired_lease_is_taken_over_with_a_higher_fence(store, clock):
    first = build(store, clock, flow(task("a")), lease_seconds=60.0)
    second = build(store, clock, flow(task("a")), owner="other")
    first.acquire("r1")
    clock.now += 60.0
    second.acquire("r1")
    assert second.fences["r1"] == 2
    assert lock(store)["owner"] == "other"
    with pytest.raises(failures.EvictionError, match="now held by other"):
        first.hold("r1")


def test_hold_refuses_a_missing_lease_or_an_unknown_run(store, clock):
    executor = build(store, clock, flow(task("a")))
    with pytest.raises(failures.EvictionError, match="lease on run r1 is gone"):
        executor.hold("r1")
    executor.acquire("r1")
    held, meta = executor.hold("r1")
    assert held["fence"] == 1
    assert meta.key.endswith("/lock")
    executor.fences.clear()
    with pytest.raises(failures.EvictionError, match="ours None"):
        executor.hold("r1")


def test_renewing_a_lease_moves_its_expiry_and_never_back(store, clock):
    executor = build(store, clock, flow(task("a")), lease_seconds=60.0)
    executor.acquire("r1")
    clock.now += 30.0
    executor.renew_lease("r1")
    assert lock(store)["expires"] == 1090.0
    clock.now = 0.0
    executor.renew_lease("r1")
    assert lock(store)["expires"] == 1090.0


def test_relinquish_deletes_our_lease_and_forgets_the_fence(store, clock):
    executor = build(store, clock, flow(task("a")))
    executor.acquire("r1")
    executor.relinquish("r1")
    assert executor.fences == {}
    with pytest.raises(objectstore.AbsentError):
        store.get("workflows/flow/runs/r1/lock")


def test_relinquish_leaves_a_lease_another_now_holds(store, clock, caplog):
    first = build(store, clock, flow(task("a")), lease_seconds=60.0)
    second = build(store, clock, flow(task("a")), owner="other")
    first.acquire("r1")
    clock.now += 61.0
    second.acquire("r1")
    with caplog.at_level(logging.WARNING):
        first.relinquish("r1")
    assert "no longer ours at release" in caplog.text
    assert lock(store)["owner"] == "other"
    assert first.fences == {}


def test_a_fenced_write_carries_the_fence_and_refuses_after_eviction(
    store, clock
):
    first = build(store, clock, flow(task("a")), lease_seconds=60.0)
    second = build(store, clock, flow(task("a")), owner="other")
    first.acquire("r1")
    key = "workflows/flow/runs/r1/run.json"
    first.write(key, {"state": "running"})
    assert json.loads(store.get(key)[0]) == {"fence": 1, "state": "running"}
    clock.now += 61.0
    second.acquire("r1")
    second.write(key, {"state": "mine"})
    with pytest.raises(failures.EvictionError, match="fence 2, ours 1"):
        first.write(key, {"state": "late"})
    assert json.loads(store.get(key)[0])["state"] == "mine"


def test_an_unleased_write_is_stored_plainly(store, clock):
    executor = build(store, clock, flow(task("a")))
    executor.write("elsewhere/x.json", {"b": 1})
    executor.write("workflows/flow/runs/r9/run.json", {"b": 2})
    assert json.loads(store.get("elsewhere/x.json")[0]) == {"b": 1}
    stored = json.loads(store.get("workflows/flow/runs/r9/run.json")[0])
    assert stored == {"b": 2}


def test_scan_lists_only_runs_with_a_valid_lease(store, clock):
    executor = build(store, clock, flow(task("a")), lease_seconds=60.0)
    executor.acquire("r1")
    clock.now += 30.0
    executor.acquire("r2")
    store.put("workflows/flow/runs/r3/run.json", b"{}")
    assert sorted(executor.scan()) == ["r1", "r2"]
    clock.now += 30.0
    assert executor.scan() == ["r2"]


def test_a_heartbeat_ends_when_the_lease_is_lost(store, clock, caplog):
    executor = build(store, clock, flow(task("a")), lease_seconds=0.03)
    executor.acquire("r1")
    store.delete("workflows/flow/runs/r1/lock")
    with caplog.at_level(logging.WARNING), executor.pulse("r1"):
        pass
    executor.beat("r1", threading.Event())
    assert "the lease was lost" in caplog.text


# ---- steps -----------------------------------------------------------------


def test_the_write_retries_are_three_attempts():
    assert retry.Retries(3, 1.0, 2.0, 4.0) == steps.WRITES


def test_an_idempotency_key_hashes_run_step_parameters_and_upstream(
    store, clock
):
    executor = build(store, clock, flow(task("a")))
    step = task("a", parameters={"p": 1})
    key = executor.idempotency_key("r1", step, {"b": {"x": 1}})
    expected = records.distill(
        ["r1", "a", {"p": 1}, {"b": records.distill({"x": 1})}]
    )
    assert key == expected
    assert executor.idempotency_key("r2", step, {"b": {"x": 1}}) != key
    assert executor.idempotency_key("r1", step, {"b": {"x": 2}}) != key


def test_note_cuts_a_plain_failure_to_three_hundred_characters(store, clock):
    executor = build(store, clock, flow(task("a")))
    attempts = []
    text = executor.note(attempts, 2, ValueError("x" * 400), "failed")
    assert len(text) <= 300
    assert attempts[0]["attempt"] == 2
    assert attempts[0]["state"] == "failed"
    assert attempts[0]["error"] == text
    assert "failure" in attempts[0]


def test_note_keeps_a_framework_or_retryable_failure_whole(store, clock):
    executor = build(store, clock, flow(task("a")))
    attempts = []
    whole = executor.note(attempts, 1, errors.ConfigError("y" * 400), "failed")
    assert len(whole) > 300
    kept = executor.note(attempts, 2, ValueError("z" * 400), "retryable")
    assert len(kept) > 300
    cut = executor.note(attempts, 3, errors.InputError("w" * 400), "failed")
    assert len(cut) <= 300


def test_begin_builds_a_context_timed_from_now(store, clock):
    executor = build(store, clock, flow(task("a")))
    current = executor.begin("r1", task("a"), 2, "k", {"u": {}}, 3.5)
    assert current.run_id == "r1"
    assert current.attempt == 2
    assert current.idempotency_key == "k"
    assert current.started == 1000.0
    assert current.run_spent == 3.5
    assert current.run_budget is None


def test_a_transient_failure_is_retried_after_the_policy_delay(store, clock):
    calls = []

    def flaky(current):
        calls.append(current.attempt)
        current.spend(1.0)
        if current.attempt < 3:
            raise retry.TransientError("blip")
        return {"ok": True}

    executor = build(store, clock, flow(task("a")), {"a": flaky})
    executor.acquire("r1")
    state, outputs, cost = executor.run_step("r1", task("a"), {}, 0.0)
    assert (state, outputs, cost) == ("succeeded", {"ok": True}, 3.0)
    assert calls == [1, 2, 3]
    assert clock.sleeps == [2.0, 6.0]
    saved = executor.step_record("r1", "a")
    assert saved["state"] == "succeeded"
    assert saved["cost"] == 3.0
    assert [a["state"] for a in saved["attempts"]] == [
        "retryable",
        "retryable",
        "succeeded",
    ]


def test_a_step_that_keeps_failing_transiently_fails_after_its_attempts(
    store, clock
):
    def always(current):
        raise retry.TransientError("down")

    executor = build(store, clock, flow(task("a")), {"a": always})
    executor.acquire("r1")
    state, outputs, cost = executor.run_step("r1", task("a"), {}, 0.0)
    assert (state, outputs, cost) == ("failed", {}, 0.0)
    assert clock.sleeps == [2.0, 6.0]
    saved = executor.step_record("r1", "a")
    assert saved["state"] == "failed"
    assert len(saved["attempts"]) == 3
    assert "down" in saved["error"]


def test_a_plain_error_fails_the_step_without_a_retry(store, clock):
    def broken(current):
        current.spend(2.0)
        raise RuntimeError("boom")

    executor = build(store, clock, flow(task("a")), {"a": broken})
    executor.acquire("r1")
    state, outputs, cost = executor.run_step("r1", task("a"), {}, 0.0)
    assert (state, outputs, cost) == ("failed", {}, 2.0)
    assert clock.sleeps == []
    saved = executor.step_record("r1", "a")
    assert saved["cost"] == 2.0
    assert len(saved["attempts"]) == 1


def test_a_step_past_its_timeout_fails(store, clock):
    def slow(current):
        clock.now += 51.0
        return {"late": True}

    executor = build(store, clock, flow(task("a")), {"a": slow})
    executor.acquire("r1")
    assert executor.run_step("r1", task("a"), {}, 0.0)[0] == "failed"


def test_a_step_that_succeeded_with_the_same_key_is_not_run_again(store, clock):
    calls = []

    def once(current):
        calls.append(1)
        return {"n": len(calls)}

    executor = build(store, clock, flow(task("a")), {"a": once})
    executor.acquire("r1")
    first = executor.run_step("r1", task("a"), {}, 0.0)
    second = executor.run_step("r1", task("a"), {}, 0.0)
    assert first == ("succeeded", {"n": 1}, 0.0)
    assert second == ("succeeded", {"n": 1}, 0.0)
    assert calls == [1]
    third = executor.run_step("r1", task("a"), {"b": {"new": 1}}, 0.0)
    assert third[1] == {"n": 2}


def test_a_handler_returning_none_succeeds_with_no_outputs(store, clock):
    executor = build(store, clock, flow(task("a")), {"a": lambda c: None})
    executor.acquire("r1")
    assert executor.run_step("r1", task("a"), {}, 0.0) == ("succeeded", {}, 0.0)


# ---- approvals -------------------------------------------------------------


def gated():
    return flow(task("a"), gate("ok", "a"), task("b", "ok", effect="approved"))


def test_admit_needs_succeeded_parents_and_room_in_the_group(store, clock):
    workflow = flow(
        task("a"),
        task("b", "a", concurrency_group="g"),
        group_limits={"g": 2},
    )
    executor = build(store, clock, workflow)
    step = workflow.step("b")
    assert not executor.admit(step, {"a": "pending", "b": "pending"}, {})
    ready = {"a": "succeeded", "b": "pending"}
    assert executor.admit(step, ready, {"g": 1})
    assert not executor.admit(step, ready, {"g": 2})
    assert executor.admit(workflow.step("a"), ready, {"g": 9})


def test_a_decision_is_signed_over_its_payload(store, clock):
    executor = build(store, clock, gated())
    signed = executor.decide("r1", "ok", "boss", "approve", KEYS["boss"])
    payload = {
        "run_id": "r1",
        "workflow_digest": gated().digest(),
        "step": "ok",
        "approver": "boss",
        "decision": "approve",
        "at": 1000.0,
    }
    assert signed == {
        **payload,
        "signature": signing.mac(payload, KEYS["boss"]),
    }
    assert (
        executor.decide("r1", "ok", "boss", "reject", "k", at=5.0)["at"] == 5.0
    )


def test_an_approval_run_waits_then_finishes_once_approved(store, clock):
    executor = build(store, clock, gated())
    record = executor.start("t1", requested_by="dev")
    assert record["state"] == "waiting_approval"
    assert record["steps"] == {
        "a": "succeeded",
        "ok": "waiting",
        "b": "pending",
    }
    signed = executor.decide(
        record["run_id"], "ok", "boss", "approve", KEYS["boss"]
    )
    executor.ratify(signed)
    with pytest.raises(failures.ApprovalError, match="already recorded"):
        executor.ratify(signed)
    done = executor.resume(record["run_id"])
    assert done["state"] == "succeeded"
    assert executor.step_record(record["run_id"], "ok")["outputs"] == {
        "approver": "boss"
    }


def test_a_rejected_approval_ends_the_run_rejected(store, clock):
    executor = build(store, clock, gated())
    record = executor.start("t1", requested_by="dev")
    executor.ratify(
        executor.decide(record["run_id"], "ok", "peer", "reject", KEYS["peer"])
    )
    done = executor.resume(record["run_id"])
    assert done["state"] == "rejected"
    assert done["steps"]["b"] == "skipped"
    assert executor.step_record(record["run_id"], "ok")["approver"] == "peer"


def test_an_approval_expires_only_after_its_window(store, clock):
    executor = build(store, clock, gated())
    record = executor.start("t1", requested_by="dev")
    since = executor.step_record(record["run_id"], "ok")["waiting_since"]
    assert since == 1000.0
    clock.now += 100.0
    assert executor.resume(record["run_id"])["state"] == "waiting_approval"
    clock.now += 0.5
    assert executor.resume(record["run_id"])["state"] == "expired"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"run_id": "nope"}, "no such run"),
        ({"approver": "stranger"}, "not listed"),
        ({"approver": "dev"}, "not listed"),
        ({"signature": "forged"}, "does not verify"),
    ],
)
def test_ratify_refuses_bad_decisions(store, clock, change, message):
    executor = build(store, clock, gated())
    record = executor.start("t1", requested_by="dev")
    signed = executor.decide(
        record["run_id"], "ok", "boss", "approve", KEYS["boss"]
    )
    with pytest.raises(failures.ApprovalError, match=message):
        executor.ratify({**signed, **change})


def test_ratify_refuses_the_requester_and_odd_decisions(store, clock):
    executor = build(store, clock, gated())
    record = executor.start("t1", requested_by="boss")
    run = record["run_id"]
    own = executor.decide(run, "ok", "boss", "approve", KEYS["boss"])
    with pytest.raises(failures.ApprovalError, match="their own run"):
        executor.ratify(own)
    odd = executor.decide(run, "ok", "peer", "maybe", KEYS["peer"])
    with pytest.raises(failures.ApprovalError, match="approve or reject"):
        executor.ratify(odd)
    plain = executor.decide(run, "a", "peer", "approve", KEYS["peer"])
    with pytest.raises(failures.ApprovalError, match="not listed"):
        executor.ratify(plain)


def test_a_decision_from_an_approver_without_a_key_does_not_verify(
    store, clock
):
    workflow = gated()
    executor = runs.Executor(
        workflow,
        store,
        {"a": lambda c: {}, "b": lambda c: {}},
        approver_keys={"peer": KEYS["peer"]},
        clock=clock,
        sleep=clock.sleep,
    )
    record = executor.start("t1", requested_by="dev")
    signed = executor.decide(record["run_id"], "ok", "boss", "approve", "k")
    with pytest.raises(failures.ApprovalError, match="does not verify"):
        executor.ratify(signed)


def test_approvals_is_the_base_of_the_loop():
    assert issubclass(loop.Loop, approvals.Approvals)
    assert issubclass(approvals.Approvals, steps.Steps)


# ---- loop ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("states", "expected"),
    [
        (
            {"a": "succeeded", "b": "waiting", "c": "rejected"},
            "waiting_approval",
        ),
        ({"a": "rejected", "b": "expired"}, "rejected"),
        ({"a": "expired", "b": "failed"}, "expired"),
        ({"a": "succeeded", "b": "succeeded"}, "succeeded"),
        ({"a": "succeeded", "b": "skipped"}, "failed"),
        ({"a": "failed"}, "failed"),
    ],
)
def test_derive_ranks_the_run_state(store, clock, states, expected):
    executor = build(store, clock, flow(task("a")))
    assert executor.derive(states) == status.Progress(expected)


def test_a_failed_step_skips_its_dependents_but_not_independent_steps(
    store, clock
):
    def broken(current):
        raise RuntimeError("no")

    workflow = flow(task("a"), task("b", "a"), task("c", "b"), task("d"))
    handlers = {
        "a": broken,
        "b": lambda c: {},
        "c": lambda c: {},
        "d": lambda c: {"d": 1},
    }
    executor = build(store, clock, workflow, handlers)
    record = executor.start("t1")
    assert record["state"] == "failed"
    assert record["steps"] == {
        "a": "failed",
        "b": "skipped",
        "c": "skipped",
        "d": "succeeded",
    }
    assert executor.step_record(record["run_id"], "b")["reason"] == "upstream"
    assert record["ended_at"] == 1000.0


def test_outputs_flow_downstream_and_cost_is_tallied(store, clock):
    seen = {}

    def first(current):
        current.spend(1.25)
        return {"v": 7}

    def second(current):
        seen.update(current.upstream)
        current.spend(0.5)
        return {}

    workflow = flow(task("a"), task("b", "a"))
    executor = build(store, clock, workflow, {"a": first, "b": second})
    record = executor.start("t1")
    assert record["state"] == "succeeded"
    assert seen == {"a": {"v": 7}}
    assert record["cost"] == 1.75
    assert executor.tally(record["run_id"]) == 1.75


def test_a_run_over_its_budget_stops_the_remaining_steps(store, clock):
    def costly(current):
        current.spend(4.0)
        return {}

    workflow = flow(task("a"), task("b", "a"), cost_budget=3.0)
    executor = build(store, clock, workflow, {"a": costly, "b": costly})
    record = executor.start("t1")
    assert record["state"] == "budget_stopped"
    assert record["steps"]["b"] == "pending"
    assert executor.step_record(record["run_id"], "b")["reason"] == "run budget"


def test_a_failed_run_resumes_from_its_failed_steps(store, clock):
    calls = []
    broken = {"on": True}

    def first(current):
        calls.append("a")
        return {}

    def second(current):
        calls.append("b")
        if broken["on"]:
            raise RuntimeError("no")
        return {}

    workflow = flow(task("a"), task("b", "a"))
    executor = build(store, clock, workflow, {"a": first, "b": second})
    record = executor.start("t1")
    assert record["state"] == "failed"
    assert executor.start("t1")["state"] == "failed"
    broken["on"] = False
    done = executor.resume(record["run_id"])
    assert done["state"] == "succeeded"
    assert calls == ["a", "b", "b"]
    states, outputs = executor.restore(record["run_id"])
    assert states == {"a": "succeeded", "b": "succeeded"}
    assert outputs == {"a": {}, "b": {}}


# ---- runs ------------------------------------------------------------------


def test_an_executor_needs_a_handler_for_every_working_step(store, clock):
    with pytest.raises(
        errors.ConfigError, match=r"no handler for steps \['b'\]"
    ):
        runs.Executor(gated(), store, {"a": lambda c: {}})


def test_an_executor_keeps_its_settings(store, clock):
    executor = runs.Executor(flow(task("a")), store, {"a": lambda c: {}})
    assert executor.lease_seconds == 300.0
    assert executor.owner == "executor"
    assert executor.approver_keys == {}
    assert executor.fences == {}


def test_starting_a_finished_trigger_returns_its_record(store, clock):
    calls = []
    workflow = flow(task("a"))
    executor = build(store, clock, workflow, {"a": lambda c: calls.append(1)})
    first = executor.start("t1", requested_by="dev")
    assert first["requested_by"] == "dev"
    assert first["trigger_key"] == "t1"
    assert first["workflow_version"] == "1"
    assert first["started_at"] == 1000.0
    again = executor.start("t1")
    assert again["state"] == "succeeded"
    assert calls == [1]
    assert executor.resume(first["run_id"])["state"] == "succeeded"
    assert calls == [1]


def test_resume_of_an_unknown_run_is_a_config_error(store, clock):
    executor = build(store, clock, flow(task("a")))
    with pytest.raises(errors.ConfigError, match="no run ghost"):
        executor.resume("ghost")


def test_a_run_started_with_another_definition_is_refused(store, clock):
    executor = build(store, clock, gated())
    record = executor.start("t1", requested_by="dev")
    changed = definition.Workflow(
        "flow",
        "1",
        (
            task("a", parameters={"x": 1}),
            gate("ok", "a"),
            task("b", "ok", effect="approved"),
        ),
    )
    other = build(store, clock, changed)
    with pytest.raises(errors.ConfigError, match="different workflow"):
        other.resume(record["run_id"])
    assert other.fences == {}


def test_concurrent_runs_beyond_the_limit_are_refused(store, clock):
    executor = build(store, clock, flow(task("a")))
    holder = build(store, clock, flow(task("a")), owner="other")
    holder.acquire("someone")
    with pytest.raises(
        failures.ConcurrencyLimitError, match="1 runs of flow are in flight"
    ):
        executor.start("t1")
    roomy = build(store, clock, flow(task("a"), max_concurrent_runs=2))
    assert roomy.start("t1")["state"] == "succeeded"


def test_a_run_held_by_another_executor_is_busy(store, clock):
    workflow = flow(task("a"), max_concurrent_runs=2)
    executor = build(store, clock, workflow)
    holder = build(store, clock, workflow, owner="other")
    run = records.run_id_for(workflow, "t1")
    holder.acquire(run)
    with pytest.raises(failures.BusyError, match="held by other"):
        executor.start("t1")


def test_status_finished_states_exclude_running_and_waiting():
    assert status.Progress.RUNNING not in status.FINISHED
    assert status.Progress.WAITING_APPROVAL not in status.FINISHED
    assert len(status.FINISHED) == 5


# ---- the run table and the bucket -----------------------------------------


def test_the_run_table_resumes_only_a_failed_or_waiting_run():
    table = status.TABLE
    assert set(table) == set(status.Progress)
    # A run left running by a crashed executor is continued (running ->
    # running); every other state is where a drive ends.
    assert table[status.Progress.RUNNING] == frozenset(status.Progress)
    assert table[status.Progress.FAILED] == {status.Progress.RUNNING}
    assert table[status.Progress.WAITING_APPROVAL] == {status.Progress.RUNNING}
    ends = {progress for progress, targets in table.items() if not targets}
    assert ends == set(status.FINISHED) - {status.Progress.FAILED}
    machine = state.Machine(table, status.Progress.SUCCEEDED)
    with pytest.raises(state.TransitionError, match="succeeded is terminal"):
        machine.transition(status.Progress.RUNNING)


def test_a_settled_run_is_returned_as_it_is_by_resume(store, clock):
    executor = build(store, clock, flow(task("a")))
    for settled in ("succeeded", "rejected", "expired", "budget_stopped"):
        record = {"state": settled, "trigger_key": "t", "requested_by": "x"}
        store.put(
            f"workflows/flow/runs/{settled}/run.json",
            json.dumps(record).encode(),
        )
        assert executor.resume(settled) == record


def test_a_bucket_keeps_a_runs_records_under_its_prefix(store, clock):
    executor = build(store, clock, flow(task("a"), task("b", "a")))
    bucket = executor.bucket("r1")
    assert bucket.base == "workflows/flow/runs/r1/"
    assert bucket.fetch("run.json") is None
    assert bucket.restore(0, "a", None, {}) is None
    kept = bucket.record(
        0, "a", status.Phase.SUCCEEDED, 1.5, {"v": 1}, None, cost=2.0
    )
    stored = executor.step_record("r1", "a")
    assert stored == {
        "step": "a",
        "state": "succeeded",
        "outputs": {"v": 1},
        "cost": 2.0,
    }
    assert kept == {**stored, "product": records.distill({"v": 1})}
    assert bucket.restore(0, "a", None, {}) == (
        {"v": 1},
        records.distill({"v": 1}),
    )
    failed = bucket.record(1, "b", status.Phase.FAILED, 0.0, error="no")
    assert failed["product"] is None
    assert bucket.restore(1, "b", None, {}) is None
    bucket.keep("run.json", {"state": "running"})
    assert executor.run_record("r1") == {"state": "running"}
    before = sorted(meta.key for meta in store.list_objects("workflows/"))
    bucket.finish()
    assert sorted(m.key for m in store.list_objects("workflows/")) == before


def test_a_bucket_writes_through_the_runs_fenced_write(store, clock):
    executor = build(store, clock, flow(task("a")))
    executor.acquire("r1")
    executor.bucket("r1").record(0, "a", status.Phase.RUNNING, 0.0)
    assert executor.step_record("r1", "a")["fence"] == 1
    holder = build(store, clock, flow(task("a")), owner="other")
    clock.now += executor.lease_seconds + 1
    holder.acquire("r1")
    with pytest.raises(failures.EvictionError):
        executor.bucket("r1").record(0, "a", status.Phase.FAILED, 0.0)
