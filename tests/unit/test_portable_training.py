"""Runner interface: local, AWS and GCP adapters, resume, retries, budget."""

import json

import pytest

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import cloud
from scmle.data.store import objectstore
from scmle.runners import aws
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import entrypoint
from scmle.runners import gcp
from scmle.runners import local
from scmle.runners import reproducibility
from scmle.runners import roles
from tests import support_cloud as fakes
from tests import support_runners as sr

TRAIN = "arn:aws:iam::1:role/scmle-training"
SERVE = "arn:aws:iam::1:role/scmle-serving"
RATE = base.CostRate(per_cpu_hour=3.6)


def make_store(kind, tmp_path):
    if kind == "local":
        return objectstore.LocalObjectStore(tmp_path / "durable")
    if kind == "s3":
        return cloud.S3ObjectStore(fakes.FakeS3Client(), "bkt", "runs-area")
    return cloud.GcsObjectStore(fakes.FakeGcsClient(), "bkt", "runs-area")


@pytest.fixture(params=["local", "s3", "gcs"])
def store(request, tmp_path):
    return make_store(request.param, tmp_path)


def spec_for(run_id="run-1", **changes):
    values = {
        "run_id": run_id,
        "name": "churn",
        "project": {"data": "data/t.csv", "target": "y"},
        "identity": TRAIN,
        "resources": base.Resources(cpu=1.0, memory_mb=1024),
        "retries": retry.RetryPolicy(3, 10.0, 2.0, 60.0),
    }
    values.update(changes)
    return base.JobSpec(**values)


def local_runner(store, clock, workload=None, **options):
    workload = workload or sr.toy_workload(clock)
    return local.LocalRunner(
        store,
        workload,
        serving_identity=SERVE,
        clock=base.Clock(clock.now, clock.sleep),
        poll_seconds=1.0,
        **options,
    )


def reference_result(tmp_path):
    clock = sr.FakeClock()
    store = objectstore.LocalObjectStore(tmp_path / "reference")
    local_runner(store, clock).run(spec_for("reference"))
    return json.loads(store.get("runs/reference/result.json")[0])


def test_a_run_writes_spec_checkpoints_result_and_manifest(store):
    clock = sr.FakeClock()
    manifest = local_runner(store, clock).run(spec_for())
    assert manifest.state == "succeeded" and len(manifest.attempts) == 1
    assert [c["step"] for c in manifest.checkpoints] == [1, 2, 3, 4, 5, 6]
    assert manifest.result_sha256
    keys = {e["key"] for e in manifest.evidence}
    assert {"runs/run-1/spec.json", "runs/run-1/result.json"} <= keys
    stored = json.loads(store.get("runs/run-1/manifest.json")[0])
    assert stored["state"] == "succeeded" and stored["format"] == 1
    assert stored["spec_digest"] == spec_for().digest()
    assert stored["identity"] == TRAIN


def test_an_interrupted_job_resumes_from_its_checkpoint_and_matches_the_reference(
    store, tmp_path
):
    clock = sr.FakeClock()
    seen = []
    runner = local_runner(
        store,
        clock,
        sr.toy_workload(clock, log=seen),
        interruptions=lambda attempt, step: attempt == 1 and step == 3,
    )
    manifest = runner.run(spec_for())
    assert manifest.state == "succeeded"
    assert [a.state for a in manifest.attempts] == ["interrupted", "succeeded"]
    assert seen == [(1, 0), (2, 3)]
    assert 10.0 in clock.sleeps
    result = json.loads(store.get("runs/run-1/result.json")[0])
    assert result == reference_result(tmp_path)


def test_retries_are_bounded(store):
    clock = sr.FakeClock()
    runner = local_runner(store, clock, interruptions=lambda a, s: True)
    manifest = runner.run(spec_for())
    assert manifest.state == "retries_exhausted"
    assert len(manifest.attempts) == 3
    assert [s for s in clock.sleeps if s >= 10] == [10.0, 20.0]


def test_transient_failures_retry_and_permanent_ones_do_not(store):
    clock = sr.FakeClock()
    calls = []

    def flaky(context):
        calls.append(context.attempt)
        if context.attempt < 3:
            raise retry.TransientError("node lost")
        context.finish({"metrics": {}})

    manifest = local_runner(store, clock, flaky).run(spec_for("flaky"))
    assert manifest.state == "succeeded" and calls == [1, 2, 3]

    def broken(context):
        raise ValueError("bad column")

    manifest = local_runner(store, clock, broken).run(spec_for("broken"))
    assert manifest.state == "failed" and len(manifest.attempts) == 1
    assert "ValueError: bad column" in manifest.attempts[0].reason


def test_a_run_that_ends_without_a_result_is_a_failure(store):
    clock = sr.FakeClock()
    manifest = local_runner(store, clock, lambda context: None).run(
        spec_for("silent")
    )
    assert manifest.state == "failed"


def test_the_cost_budget_stops_a_runaway_job(store):
    clock = sr.FakeClock()
    slow = sr.toy_workload(clock, steps=1000)
    budget = base.CostBudget(0.2, RATE)
    manifest = local_runner(store, clock, slow).run(
        spec_for("runaway", budget=budget)
    )
    assert manifest.state == "budget_stopped"
    assert len(manifest.attempts) == 1
    assert manifest.total_cost > 0.2
    assert manifest.total_cost <= 0.2 + RATE.cost(base.Resources(1, 1024), 60)
    steps = [c["step"] for c in manifest.checkpoints]
    assert max(steps) < 1000 and steps == sorted(steps)
    again = local_runner(store, clock, slow).run(
        spec_for("runaway", budget=budget)
    )
    assert again.state == "budget_stopped" and len(again.attempts) == 1


def test_the_budget_accumulates_across_attempts(store):
    clock = sr.FakeClock()
    runner = local_runner(
        store,
        clock,
        interruptions=lambda attempt, step: step % 2 == 0 and attempt < 3,
    )
    manifest = runner.run(spec_for("spend", budget=base.CostBudget(0.3, RATE)))
    assert manifest.state == "budget_stopped"
    assert [a.state for a in manifest.attempts] == [
        "interrupted",
        "interrupted",
        "cancelled",
    ]
    assert manifest.attempts[0].cost == pytest.approx(0.12, abs=0.001)
    assert manifest.total_cost == pytest.approx(0.36, abs=0.001)


def test_a_timeout_is_final(store):
    clock = sr.FakeClock()
    manifest = local_runner(store, clock).run(
        spec_for("slow", timeout_seconds=100.0)
    )
    assert manifest.state == "timed_out" and len(manifest.attempts) == 1


def test_a_finished_run_id_is_idempotent_and_a_different_spec_is_refused(
    store,
):
    clock = sr.FakeClock()
    log = []
    runner = local_runner(store, clock, sr.toy_workload(clock, log=log))
    first = runner.run(spec_for())
    again = runner.run(spec_for())
    assert log == [(1, 0)] and again.state == first.state
    assert again.attempts == first.attempts
    with pytest.raises(base.RunConflictError):
        runner.run(spec_for(project={"data": "data/other.csv", "target": "y"}))


def test_a_crashed_supervisor_continues_the_numbering(store):
    clock = sr.FakeClock()
    runner = local_runner(store, clock)
    spec = spec_for("resumed")
    manifest = runner.register(spec)
    manifest.attempts.append(
        base.Attempt(
            1, "local-resumed-1", "running", "2026-01-01T00:00:00+00:00"
        )
    )
    runner.save(manifest)
    checkpoints = checkpoint.CheckpointStore(store, "resumed")
    checkpoints.save(2, {"w": 2.25})
    done = runner.run(spec)
    assert done.state == "succeeded" and done.attempts[-1].attempt == 2


def test_training_cannot_run_as_serving_or_touch_released_bundles(store):
    clock = sr.FakeClock()
    runner = local_runner(store, clock)
    with pytest.raises(errors.ConfigError, match="serving identity"):
        runner.run(spec_for("bad", identity=SERVE))

    def activating(context):
        context.store.put("bundles/model/v2", b"x")

    manifest = local_runner(store, clock, activating).run(spec_for("act"))
    assert manifest.state == "failed"
    assert "RoleDeniedError" in manifest.attempts[0].reason
    assert not store.exists("bundles/model/v2")


def test_role_tables_are_separate():
    roles.check_separation(roles.TRAINING_ROLE, roles.SERVING_ROLE)
    both = roles.Role("serving", roles.SERVING_ROLE.reads, ("runs/",))
    with pytest.raises(errors.ConfigError):
        roles.check_separation(roles.TRAINING_ROLE, both)
    with pytest.raises(errors.ConfigError):
        roles.check_separation(
            roles.Role("training", (), ("bundles/",)), roles.SERVING_ROLE
        )
    with pytest.raises(errors.ConfigError):
        roles.check_separation(
            roles.TRAINING_ROLE, roles.Role("training", (), ("audit/",))
        )
    wrapped = roles.RoleStore(
        objectstore.LocalObjectStore("/tmp/unused-role-store"),
        roles.SERVING_ROLE,
    )
    with pytest.raises(roles.RoleDeniedError):
        wrapped.put("runs/x/manifest.json", b"")
    with pytest.raises(roles.RoleDeniedError):
        wrapped.get("data/train.csv")


def test_specialised_compute_needs_a_declared_reason_and_specs_are_validated():
    with pytest.raises(errors.ConfigError, match="reason"):
        base.Resources(gpu=1)
    assert base.Resources(gpu=1, gpu_reason="transformer fine-tuning").gpu == 1
    for bad in (
        {"run_id": "has space"},
        {"run_id": ""},
        {"timeout_seconds": 0},
        {"environment": {"API_TOKEN": "x"}},
        {"environment": {"NOTE": "-----BEGIN PRIVATE"}},
    ):
        with pytest.raises(errors.ConfigError):
            spec_for(**bad)
    with pytest.raises(errors.ConfigError):
        base.CostBudget(0, RATE)


def test_checkpoints_skip_a_damaged_newest_one(store):
    checkpoints = checkpoint.CheckpointStore(store, "c")
    checkpoints.save(1, {"w": 1})
    checkpoints.save(2, {"w": 2})
    store.put(
        checkpoints.key(3), b'{"step": 3, "state": {"w": 9}, "sha256": "x"}'
    )
    store.put(checkpoints.key(4), b"not json")
    latest = checkpoints.latest()
    assert latest.step == 2 and latest.state == {"w": 2}
    assert len(checkpoints.skipped) == 2
    assert checkpoint.CheckpointStore(store, "empty").latest() is None
    with pytest.raises(checkpoint.CheckpointError):
        checkpoints.save(5, {"bad": object()})
    with pytest.raises(checkpoint.CheckpointError):
        checkpoints.save(-1, {})


# ---- AWS ------------------------------------------------------------------


def aws_runner(store, ecs, clock, **options):
    return aws.AwsContainerJobRunner(
        ecs,
        store,
        cluster="scmle",
        task_definition="scmle-train:7",
        container="trainer",
        subnets=["subnet-1"],
        security_groups=["sg-1"],
        store_uri="s3://bkt/runs-area",
        serving_identity=SERVE,
        clock=base.Clock(clock.now, clock.sleep),
        poll_seconds=1.0,
        **options,
    )


def test_aws_runs_the_same_spec_and_resumes_after_a_spot_interruption(
    tmp_path,
):
    clock = sr.FakeClock()
    store = make_store("local", tmp_path)
    killed = []

    def first(context):
        found = context.resume()
        start = found.step if found else 0
        weight = found.state["w"] if found else 0.0
        for step in range(start + 1, 7):
            weight += 0.5 * (3.0 - weight)
            context.step(step, {"w": weight})
            if step == 3 and not killed:
                killed.append(True)
                raise sr.Killed()
        context.finish({"metrics": {"w": weight}, "steps": 6})

    containers = [sr.ecs_container(store, first)] * 2
    ecs = sr.FakeEcs(store, containers)
    manifest = aws_runner(store, ecs, clock).run(spec_for())
    assert manifest.state == "succeeded"
    assert [a.state for a in manifest.attempts] == ["interrupted", "succeeded"]
    assert [c["step"] for c in manifest.checkpoints] == [1, 2, 3, 4, 5, 6]
    result = json.loads(store.get("runs/run-1/result.json")[0])
    assert result == reference_result(tmp_path)
    request = ecs.requests[0]
    assert request["overrides"]["taskRoleArn"] == TRAIN
    assert (
        request["clientToken"] == "run-1-1"
        and request["launchType"] == "FARGATE"
    )
    assert request["overrides"]["containerOverrides"][0]["command"] == [
        "python",
        "-m",
        "scmle.runners.entrypoint",
    ]
    assert SERVE not in json.dumps(request)


def test_aws_budget_and_timeout_stop_the_task(tmp_path):
    clock = sr.FakeClock()
    store = make_store("local", tmp_path)

    def forever(context):
        raise AssertionError("not reached")

    ecs = sr.FakeEcs(store, [lambda env: {"lastStatus": "RUNNING"}])
    manifest = aws_runner(store, ecs, clock).run(
        spec_for("cost", budget=base.CostBudget(0.01, RATE))
    )
    assert manifest.state == "budget_stopped"
    assert ecs.stopped and ecs.stopped[0][1].startswith("scmle: budget_stopped")
    ecs = sr.FakeEcs(store, [lambda env: {"lastStatus": "RUNNING"}])
    manifest = aws_runner(store, ecs, clock).run(
        spec_for("late", timeout_seconds=30.0)
    )
    assert manifest.state == "timed_out" and ecs.stopped


def test_aws_requests_carry_gpu_capacity_and_a_stable_spec_digest(tmp_path):
    clock = sr.FakeClock()
    store = make_store("local", tmp_path)
    ecs = sr.FakeEcs(store, [])
    runner = aws_runner(store, ecs, clock, capacity_provider="FARGATE_SPOT")
    spec = spec_for(resources=base.Resources(2, 4096, 1, "deep tabular model"))
    request = runner.request(spec, 1)
    assert request["capacityProviderStrategy"][0]["capacityProvider"] == (
        "FARGATE_SPOT"
    )
    assert "launchType" not in request
    override = request["overrides"]["containerOverrides"][0]
    assert override["resourceRequirements"] == [{"type": "GPU", "value": "1"}]
    env = {e["name"]: e["value"] for e in override["environment"]}
    assert env["SCMLE_SPEC_DIGEST"] == spec.digest()
    assert not any("KEY" in name or "TOKEN" in name for name in env)


def test_aws_maps_task_outcomes():
    runner = aws_runner(None, None, sr.FakeClock())
    table = [
        (sr.stopped(0), "succeeded", False),
        (sr.stopped(143, reason="SIGTERM"), "interrupted", False),
        (sr.stopped(137, "SpotInterruption", "spot"), "interrupted", False),
        (sr.stopped(137, reason="OutOfMemoryError"), "failed", False),
        (sr.stopped(1), "failed", False),
        (sr.stopped(None, "UserInitiated", "stop"), "cancelled", False),
        (
            {
                "lastStatus": "STOPPED",
                "stopCode": "TaskFailedToStart",
                "stoppedReason": "RESOURCE:CPU capacity unavailable",
            },
            "failed",
            True,
        ),
        ({"lastStatus": "PENDING"}, "pending", False),
        ({"lastStatus": "RUNNING"}, "running", False),
    ]
    for task, state, retryable in table:
        status = runner.map_task(task)
        assert (status.state.value, status.retryable) == (state, retryable), (
            task
        )


def test_aws_capacity_failures_at_placement_are_retried(tmp_path):
    clock = sr.FakeClock()
    store = make_store("local", tmp_path)

    def work(context):
        context.finish({"metrics": {}})

    ecs = sr.FakeEcs(store, [sr.ecs_container(store, work)])
    ecs.placement_failures = ["RESOURCE:FARGATE capacity unavailable"]
    manifest = aws_runner(store, ecs, clock).run(spec_for("place"))
    assert manifest.state == "succeeded" and len(manifest.attempts) == 2
    assert manifest.attempts[0].state == "failed"


# ---- GCP ------------------------------------------------------------------


JOB = "projects/p/locations/europe-west1/jobs/scmle-train"
GCP_TRAIN = "scmle-training@p.iam.gserviceaccount.com"
GCP_SERVE = "scmle-serving@p.iam.gserviceaccount.com"


def gcp_runner(store, client, clock, **options):
    values = {
        "job_name": JOB,
        "job_service_account": GCP_TRAIN,
        "job_cpu": 2.0,
        "job_memory_mb": 4096,
        "store_uri": "gs://bkt/runs-area",
        "serving_identity": GCP_SERVE,
        "clock": base.Clock(clock.now, clock.sleep),
        "poll_seconds": 1.0,
    }
    values.update(options)
    return gcp.GcpJobRunner(client, store, **values)


def test_gcp_resumes_after_a_preemption_and_matches_the_reference(tmp_path):
    clock = sr.FakeClock()
    store = make_store("local", tmp_path)
    killed = []

    def work(context):
        found = context.resume()
        start = found.step if found else 0
        weight = found.state["w"] if found else 0.0
        for step in range(start + 1, 7):
            weight += 0.5 * (3.0 - weight)
            context.step(step, {"w": weight})
            if step == 4 and not killed:
                killed.append(True)
                raise sr.Killed()
        context.finish({"metrics": {"w": weight}, "steps": 6})

    client = sr.FakeCloudRun([sr.run_v2_container(store, work)] * 2)
    manifest = gcp_runner(store, client, clock).run(
        spec_for(identity=GCP_TRAIN)
    )
    assert manifest.state == "succeeded"
    assert [a.state for a in manifest.attempts] == ["interrupted", "succeeded"]
    assert json.loads(store.get("runs/run-1/result.json")[0]) == (
        reference_result(tmp_path)
    )
    name, overrides = client.started[0]
    assert name == JOB and overrides["task_count"] == 1
    assert overrides["timeout"] == "3600s"
    env = {
        e["name"]: e["value"]
        for e in overrides["container_overrides"][0]["env"]
    }
    assert env["SCMLE_SPEC_DIGEST"] == spec_for(identity=GCP_TRAIN).digest()


def test_gcp_refuses_mismatched_identity_resources_and_the_serving_account(
    tmp_path,
):
    clock = sr.FakeClock()
    store = make_store("local", tmp_path)
    client = sr.FakeCloudRun([])
    runner = gcp_runner(store, client, clock)
    with pytest.raises(errors.ConfigError, match="own service account"):
        runner.submit(spec_for(identity="someone@else"), 1)
    with pytest.raises(errors.ConfigError, match="more than the job"):
        runner.submit(
            spec_for(identity=GCP_TRAIN, resources=base.Resources(8, 1024)), 1
        )
    with pytest.raises(errors.ConfigError, match="more than the job"):
        runner.submit(
            spec_for(
                identity=GCP_TRAIN, resources=base.Resources(1, 1024, 1, "x")
            ),
            1,
        )
    with pytest.raises(errors.ConfigError, match="serving identity"):
        gcp_runner(store, client, clock, job_service_account=GCP_SERVE)
    assert client.started == []


def test_gcp_budget_cancels_the_execution_and_outcomes_are_mapped(tmp_path):
    clock = sr.FakeClock()
    store = make_store("local", tmp_path)
    client = sr.FakeCloudRun([lambda env: {"running_count": 1}])
    manifest = gcp_runner(store, client, clock).run(
        spec_for(identity=GCP_TRAIN, budget=base.CostBudget(0.01, RATE))
    )
    assert manifest.state == "budget_stopped" and client.cancelled
    runner = gcp_runner(store, client, clock)

    def handle():
        return base.JobHandle("e", 1)

    table = [
        ({"succeeded_count": 1}, "succeeded", False),
        ({"cancelled_count": 1}, "cancelled", False),
        ({"running_count": 1}, "running", False),
        ({"running_count": 0}, "pending", False),
        (
            {"failed_count": 1, "conditions": [{"reason": "Preempted"}]},
            "interrupted",
            False,
        ),
        (
            {"failed_count": 1, "conditions": [{"message": "Quota exceeded"}]},
            "failed",
            True,
        ),
        (
            {
                "failed_count": 1,
                "conditions": [{"message": "Memory limit exceeded"}],
            },
            "failed",
            False,
        ),
    ]
    for execution, state, retryable in table:
        client.executions["e"] = {"polls": 5, "done": execution}
        status = runner.poll(handle())
        assert (status.state.value, status.retryable) == (state, retryable)


def test_run_v2_client_wraps_the_sdk_shapes():
    class Operation:
        class metadata:  # noqa: N801
            name = "exec-1"

    class Jobs:
        def run_job(self, request):
            self.request = request
            return Operation()

    class Condition:
        reason = "Preempted"
        message = "SIGTERM"

    class Execution:
        running_count = 0
        succeeded_count = 0
        failed_count = 1
        cancelled_count = 0
        completion_time = "t"
        conditions = [Condition()]

    class Executions:
        def get_execution(self, name):
            return Execution()

        def cancel_execution(self, request):
            self.cancelled = request

    wrapped = gcp.RunV2Client(Jobs(), Executions())
    assert wrapped.run_job("job", {"task_count": 1}) == "exec-1"
    assert wrapped.get_execution("e")["conditions"][0]["reason"] == "Preempted"
    wrapped.cancel_execution("e")
    assert wrapped.executions.cancelled == {"name": "e"}

    def missing(name):
        raise ImportError(name)

    with pytest.raises(errors.ConfigError):
        gcp.run_v2_client(missing)
    with pytest.raises(errors.ConfigError):
        aws.ecs_client(missing)


# ---- entrypoint, reproducibility -----------------------------------------


def test_the_entrypoint_checks_the_spec_digest_and_the_store_scheme(tmp_path):
    store = make_store("local", tmp_path)
    clock = sr.FakeClock()
    runner = local_runner(store, clock)
    spec = spec_for("entry")
    runner.register(spec)

    def work(context):
        context.finish({"metrics": {"ok": 1}})

    env = {"SCMLE_RUN_ID": "entry", "SCMLE_SPEC_DIGEST": spec.digest()}
    assert sr.run_container(store, env, work) == 0
    assert json.loads(store.get("runs/entry/result.json")[0]) == {
        "metrics": {"ok": 1}
    }
    wrong = {**env, "SCMLE_SPEC_DIGEST": "0" * 64}
    assert sr.run_container(store, wrong, work) == 1
    assert sr.run_container(store, {"SCMLE_RUN_ID": "nope"}, work) == 1
    assert entrypoint.open_store(tmp_path.as_uri()).scheme == "file"
    with pytest.raises(errors.ConfigError):
        entrypoint.open_store("ftp://host/x")
    with pytest.raises(errors.ConfigError):
        entrypoint.load_workload("no_colon")
    assert callable(entrypoint.load_workload("tests.support_runners:current"))


def test_reproducibility_is_reported_against_declared_tolerances():
    tolerances = reproducibility.Tolerances({"auc": 0.01}, default=0.0)
    report = reproducibility.compare(
        {"auc": 0.90, "loss": 0.30},
        {"auc": 0.905, "loss": 0.31},
        tolerances,
        place="aws",
    )
    rows = {row["metric"]: row for row in report["metrics"]}
    assert rows["auc"]["within"] and not rows["loss"]["within"]
    assert not report["within_tolerance"] and report["place"] == "aws"
    missing = reproducibility.compare({"a": 1}, {}, tolerances, place="gcp")
    assert not missing["within_tolerance"]
    with pytest.raises(errors.ConfigError):
        reproducibility.Tolerances({"a": -1})
