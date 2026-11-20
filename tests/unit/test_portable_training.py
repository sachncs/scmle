"""Runner interface: local, AWS and GCP adapters, resume, retries, budget."""

import json

import pytest

from scmle.contracts import errors
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import entrypoint
from scmle.runners import local
from scmle.runners import reproducibility
from scmle.runners import roles
from scmle.runtime import retry
from scmle.storage import objectstore
from tests import support_runners

TRAIN = "scmle-training"
SERVE = "scmle-serving"
RATE = base.Rate(per_cpu_hour=3.6)


def make_store(tmp_path):
    return objectstore.Disk(tmp_path / "durable")


@pytest.fixture
def store(tmp_path):
    return make_store(tmp_path)


def spec_for(run_id="run-1", **changes):
    values = {
        "identifier": run_id,
        "name": "churn",
        "project": {"data": "data/t.csv", "target": "y"},
        "identity": TRAIN,
        "resources": base.Resources(cpu=1.0, memory_mb=1024),
        "retries": retry.Retries(3, 10.0, 2.0, 60.0),
    }
    values.update(changes)
    return base.Job(**values)


def local_runner(store, clock, workload=None, **options):
    workload = workload or support_runners.toy_workload(clock)
    return local.Runner(
        store,
        workload,
        identity=SERVE,
        clock=base.Clock(clock.now, clock.sleep),
        interval=1.0,
        **options,
    )


def reference_result(tmp_path):
    clock = support_runners.FakeClock()
    store = objectstore.Disk(tmp_path / "reference")
    local_runner(store, clock).run(spec_for("reference"))
    return json.loads(store.get("runs/reference/result.json")[0])


def test_a_run_writes_spec_checkpoints_result_and_manifest(store):
    clock = support_runners.FakeClock()
    manifest = local_runner(store, clock).run(spec_for())
    assert manifest.state == "succeeded" and len(manifest.attempts) == 1
    assert [c["step"] for c in manifest.checkpoints] == [1, 2, 3, 4, 5, 6]
    assert manifest.digests["result"]
    keys = {e["key"] for e in manifest.evidence}
    assert {"runs/run-1/spec.json", "runs/run-1/result.json"} <= keys
    stored = json.loads(store.get("runs/run-1/manifest.json")[0])
    assert stored["state"] == "succeeded" and stored["format"] == 2
    assert stored["digests"]["spec"] == spec_for().digest()
    assert stored["identity"] == TRAIN


def test_an_interrupted_job_resumes_from_its_checkpoint_and_matches_the_reference(
    store, tmp_path
):
    clock = support_runners.FakeClock()
    seen = []
    runner = local_runner(
        store,
        clock,
        support_runners.toy_workload(clock, log=seen),
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
    clock = support_runners.FakeClock()
    runner = local_runner(store, clock, interruptions=lambda a, s: True)
    manifest = runner.run(spec_for())
    assert manifest.state == "retries_exhausted"
    assert len(manifest.attempts) == 3
    assert [s for s in clock.sleeps if s >= 10] == [10.0, 20.0]


def test_transient_failures_retry_and_permanent_ones_do_not(store):
    clock = support_runners.FakeClock()
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
    clock = support_runners.FakeClock()
    manifest = local_runner(store, clock, lambda context: None).run(
        spec_for("silent")
    )
    assert manifest.state == "failed"


def test_the_cost_budget_stops_a_runaway_job(store):
    clock = support_runners.FakeClock()
    slow = support_runners.toy_workload(clock, steps=1000)
    budget = base.Budget(0.2, RATE)
    manifest = local_runner(store, clock, slow).run(
        spec_for("runaway", budget=budget)
    )
    assert manifest.state == "budget_stopped"
    assert len(manifest.attempts) == 1
    assert manifest.cost > 0.2
    assert manifest.cost <= 0.2 + RATE.cost(base.Resources(1, 1024), 60)
    steps = [c["step"] for c in manifest.checkpoints]
    assert max(steps) < 1000 and steps == sorted(steps)
    again = local_runner(store, clock, slow).run(
        spec_for("runaway", budget=budget)
    )
    assert again.state == "budget_stopped" and len(again.attempts) == 1


def test_the_budget_accumulates_across_attempts(store):
    clock = support_runners.FakeClock()
    runner = local_runner(
        store,
        clock,
        interruptions=lambda attempt, step: step % 2 == 0 and attempt < 3,
    )
    manifest = runner.run(spec_for("spend", budget=base.Budget(0.3, RATE)))
    assert manifest.state == "budget_stopped"
    assert [a.state for a in manifest.attempts] == [
        "interrupted",
        "interrupted",
        "cancelled",
    ]
    assert manifest.attempts[0].cost == pytest.approx(0.12, abs=0.001)
    assert manifest.cost == pytest.approx(0.36, abs=0.001)


def test_a_timeout_is_final(store):
    clock = support_runners.FakeClock()
    manifest = local_runner(store, clock).run(
        spec_for("slow", timeout=100.0)
    )
    assert manifest.state == "timed_out" and len(manifest.attempts) == 1


def test_a_finished_run_id_is_idempotent_and_a_different_spec_is_refused(
    store,
):
    clock = support_runners.FakeClock()
    log = []
    runner = local_runner(
        store, clock, support_runners.toy_workload(clock, log=log)
    )
    first = runner.run(spec_for())
    again = runner.run(spec_for())
    assert log == [(1, 0)] and again.state == first.state
    assert again.attempts == first.attempts
    with pytest.raises(base.ConflictError):
        runner.run(spec_for(project={"data": "data/other.csv", "target": "y"}))


def test_a_crashed_supervisor_continues_the_numbering(store):
    clock = support_runners.FakeClock()
    runner = local_runner(store, clock)
    spec = spec_for("resumed")
    manifest = runner.register(spec)
    manifest.attempts.append(
        base.Attempt(
            1, "local-resumed-1", "running", "2026-01-01T00:00:00+00:00"
        )
    )
    runner.save(manifest)
    checkpoints = checkpoint.Store(store, "resumed")
    checkpoints.save(2, {"w": 2.25})
    done = runner.run(spec)
    assert done.state == "succeeded" and done.attempts[-1].attempt == 2


def test_training_cannot_run_as_serving_or_touch_released_bundles(store):
    # Enforces docs/PORTABLE_TRAINING.md:44 and docs/PORTABLE_TRAINING.md:45.
    # Enforces website/src/pages/docs/portable-training.astro:29.
    clock = support_runners.FakeClock()
    runner = local_runner(store, clock)
    with pytest.raises(errors.ConfigError, match="serving identity"):
        runner.run(spec_for("bad", identity=SERVE))

    def activating(context):
        context.store.put("bundles/model/v2", b"x")

    manifest = local_runner(store, clock, activating).run(spec_for("act"))
    assert manifest.state == "failed"
    assert "DeniedError" in manifest.attempts[0].reason
    assert not store.exists("bundles/model/v2")


def test_role_tables_are_separate():
    # Enforces docs/TENANT_ISOLATION.md:56.
    # Enforces website/src/pages/docs/tenant-isolation.astro:34.
    roles.separate(roles.TRAINING_ROLE, roles.SERVING_ROLE)
    both = roles.Role("serving", roles.SERVING_ROLE.reads, ("runs/",))
    with pytest.raises(errors.ConfigError):
        roles.separate(roles.TRAINING_ROLE, both)
    with pytest.raises(errors.ConfigError):
        roles.separate(
            roles.Role("training", (), ("bundles/",)), roles.SERVING_ROLE
        )
    with pytest.raises(errors.ConfigError):
        roles.separate(
            roles.TRAINING_ROLE, roles.Role("training", (), ("audit/",))
        )
    wrapped = roles.Store(
        objectstore.Disk("/tmp/unused-role-store"),
        roles.SERVING_ROLE,
    )
    with pytest.raises(roles.DeniedError):
        wrapped.put("runs/x/manifest.json", b"")
    with pytest.raises(roles.DeniedError):
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
        base.Budget(0, RATE)


def test_checkpoints_skip_a_damaged_newest_one(store):
    checkpoints = checkpoint.Store(store, "c")
    checkpoints.save(1, {"w": 1})
    checkpoints.save(2, {"w": 2})
    store.put(
        checkpoints.key(3), b'{"step": 3, "state": {"w": 9}, "sha256": "x"}'
    )
    store.put(checkpoints.key(4), b"not json")
    latest = checkpoints.latest()
    assert latest.step == 2 and latest.state == {"w": 2}
    assert len(checkpoints.skipped) == 2
    assert checkpoint.Store(store, "empty").latest() is None
    with pytest.raises(checkpoint.CheckpointError):
        checkpoints.save(5, {"bad": object()})
    with pytest.raises(checkpoint.CheckpointError):
        checkpoints.save(-1, {})


# ---- entrypoint, reproducibility -----------------------------------------


def test_the_entrypoint_checks_the_spec_digest_and_the_store_scheme(tmp_path):
    # Enforces docs/PORTABLE_TRAINING.md:64.
    # Enforces website/src/pages/docs/portable-training.astro:33.
    store = make_store(tmp_path)
    clock = support_runners.FakeClock()
    runner = local_runner(store, clock)
    spec = spec_for("entry")
    runner.register(spec)

    def work(context):
        context.finish({"metrics": {"ok": 1}})

    env = {"SCMLE_RUN_ID": "entry", "SCMLE_SPEC_DIGEST": spec.digest()}
    assert support_runners.run_container(store, env, work) == 0
    assert json.loads(store.get("runs/entry/result.json")[0]) == {
        "metrics": {"ok": 1}
    }
    wrong = {**env, "SCMLE_SPEC_DIGEST": "0" * 64}
    assert support_runners.run_container(store, wrong, work) == 1
    assert (
        support_runners.run_container(store, {"SCMLE_RUN_ID": "nope"}, work)
        == 1
    )
    assert entrypoint.mount(tmp_path.as_uri()).scheme == "file"
    with pytest.raises(errors.ConfigError):
        entrypoint.mount("ftp://host/x")
    with pytest.raises(errors.ConfigError):
        entrypoint.resolve("no_colon")
    assert callable(entrypoint.resolve("tests.support_runners:current"))


def test_reproducibility_is_reported_against_declared_tolerances():
    tolerances = reproducibility.Tolerances({"auc": 0.01}, default=0.0)
    report = reproducibility.compare(
        {"auc": 0.90, "loss": 0.30},
        {"auc": 0.905, "loss": 0.31},
        tolerances,
        place="cloud-a",
    )
    rows = {row["metric"]: row for row in report["metrics"]}
    assert rows["auc"]["within"] and not rows["loss"]["within"]
    assert not report["within_tolerance"] and report["place"] == "cloud-a"
    missing = reproducibility.compare({"a": 1}, {}, tolerances, place="cloud-b")
    assert not missing["within_tolerance"]
    with pytest.raises(errors.ConfigError):
        reproducibility.Tolerances({"a": -1})


def test_a_checkpoint_load_tells_absent_damaged_and_outage_apart(
    store, monkeypatch, caplog
):
    # ERR-10: absent, corrupt and backend-down all returned None, so an
    # outage looked like "no checkpoint" and the work restarted from zero.
    held = checkpoint.Store(store, "run-x")
    absent = held.load(3)
    assert absent.condition is checkpoint.Condition.ABSENT
    assert absent.checkpoint is None
    held.save(3, {"epoch": 3})
    loaded = held.load(3)
    assert loaded.condition is checkpoint.Condition.LOADED
    assert loaded.checkpoint.state == {"epoch": 3}
    store.put(held.key(4), b"{not json")
    with caplog.at_level("WARNING", logger=checkpoint.__name__):
        damaged = held.load(4)
    assert damaged.condition is checkpoint.Condition.CORRUPT
    assert damaged.checkpoint is None and "malformed" in damaged.reason
    assert damaged.key == held.key(4)
    assert any("malformed" in r.getMessage() for r in caplog.records)

    def down(key):
        raise objectstore.TransientError("backend unavailable")

    monkeypatch.setattr(store, "get", down)
    with pytest.raises(objectstore.TransientError):
        held.load(3)


def test_a_resume_past_damaged_checkpoints_leaves_a_record(store):
    # ERR-10: a restart from zero (or an older step) past damaged
    # checkpoints is recorded in durable storage, not only logged.
    checkpoints = checkpoint.Store(store, "d")
    store.put(checkpoints.key(1), b"not json")
    context = local.Context(
        roles.Store(store, roles.TRAINING_ROLE),
        "d",
        2,
        spec_for("d"),
        checkpoints,
    )
    assert context.resume() is None
    assert [r.key for r in checkpoints.skipped] == [checkpoints.key(1)]
    record = json.loads(store.get(checkpoints.ledger)[0])
    assert record["attempt"] == 2 and record["resumed_from"] is None
    assert record["damaged"][0]["key"] == checkpoints.key(1)
    assert "malformed" in record["damaged"][0]["reason"]
    clean = checkpoint.Store(store, "e")
    clean.save(1, {"w": 1})
    local.Context(
        roles.Store(store, roles.TRAINING_ROLE),
        "e",
        1,
        spec_for("e"),
        clean,
    ).resume()
    assert not store.exists(clean.ledger)


def test_the_manifest_names_damaged_checkpoints(store):
    # ERR-10: `finish` dropped a damaged checkpoint from the manifest
    # without a trace.
    clock = support_runners.FakeClock()

    def damaging(context):
        context.step(1, {"w": 1})
        context.checkpoints.store.put(context.checkpoints.key(2), b"x")
        context.finish({"metrics": {"w": 1}})

    manifest = local_runner(store, clock, damaging).run(spec_for("m"))
    assert manifest.state == "succeeded"
    assert manifest.checkpoints[0]["step"] == 1
    assert manifest.checkpoints[0]["sha256"]
    damaged = manifest.checkpoints[1]
    assert damaged["step"] == 2 and damaged["sha256"] is None
    assert "malformed" in damaged["damage"]
