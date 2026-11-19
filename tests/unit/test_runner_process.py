"""The process and container runners: a real child process per attempt."""

import json
import pathlib
import signal
import sys
import threading
import time

import pytest

from scmle.contracts import errors
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import local
from scmle.runtime import retry
from scmle.storage import objectstore
from scmle.storage import tenancy
from tests import support_runners

ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAIN = "arn:train"
SERVE = "arn:serve"


@pytest.fixture
def store(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "store")


def job(run_id, workload_project=None, **changes):
    values = {
        "run_id": run_id,
        "name": "count",
        "project": workload_project or {},
        "identity": TRAIN,
        "retries": retry.Retries(3, 0.01, 1.0, 0.01),
        "timeout_seconds": 60.0,
    }
    values.update(changes)
    return base.Job(**values)


def process(store, workload, **options):
    return local.Process(
        store,
        serving_identity=SERVE,
        workload=f"tests.support_jobs:{workload}",
        directory=ROOT,
        poll_seconds=0.02,
        **options,
    )


def states(manifest):
    return [attempt.state for attempt in manifest.attempts]


def test_exit_statuses_map_to_attempt_states():
    assert base.Exit.status(0) == base.Status(base.State.SUCCEEDED)
    assert base.Exit.status(1, "boom") == base.Status(
        base.State.FAILED, "exit 1: boom"
    )
    assert base.Exit.status(75) == base.Status(
        base.State.FAILED, "exit 75", retryable=True
    )
    for code in (143, 137, -15, -9):
        assert base.Exit.status(code) == base.Status(
            base.State.INTERRUPTED, f"signal (exit {code})"
        )
    assert base.Exit.status(125).outcome() is base.Progress.FAILED


def test_a_job_spec_reads_back_from_either_layout():
    spec = job(
        "read-1", {"steps": 2}, budget=base.CostBudget(5.0, base.CostRate(1.0))
    )
    assert base.Job.read(spec.to_dict()) == spec
    assert base.Job.read(spec.anchor()) == spec
    with pytest.raises(errors.ConfigError, match="malformed job spec"):
        base.Job.read({"name": "x"})


def test_a_process_run_checkpoints_and_leaves_its_result(store):
    manifest = process(store, "count").run(job("count-1", {"steps": 3}))
    assert manifest.state == "succeeded"
    assert manifest.runner == "process"
    assert states(manifest) == ["succeeded"]
    assert manifest.attempts[0].job_id == "scmle-process-count-1-1"
    assert json.loads(store.get("runs/count-1/result.json")[0]) == {
        "steps": 3,
        "attempt": 1,
        "resumed": 0,
    }
    assert checkpoint.CheckpointStore(store, "count-1").latest().step == 3
    assert store.get("runs/count-1/logs/1.txt")[0] == b""


def test_a_transient_failure_is_retried_and_a_permanent_one_is_final(store):
    flaky = process(store, "flaky").run(job("flaky-1"))
    assert flaky.state == "succeeded"
    assert states(flaky) == ["failed", "succeeded"]
    assert flaky.attempts[0].reason == (
        "exit 75: scmle.runtime.retry.TransientError: throttled"
    )
    broken = process(store, "broken").run(job("broken-1"))
    assert broken.state == "failed"
    assert states(broken) == ["failed"]
    assert broken.attempts[0].reason == (
        "exit 1: scmle.contracts.errors.InputError: bad column"
    )
    log = store.get("runs/broken-1/logs/1.txt")[0].decode()
    assert "Traceback" in log and "bad column" in log


@pytest.mark.skipif(sys.platform == "win32", reason="SIGTERM is POSIX only")
def test_a_preempted_process_resumes_from_its_checkpoint(store):
    runner = process(store, "count")
    spec = job("preempt-1", {"steps": 12, "pause": 0.05})
    done = []
    thread = threading.Thread(target=lambda: done.append(runner.run(spec)))
    thread.start()
    checkpoints = checkpoint.CheckpointStore(store, "preempt-1")
    deadline = time.monotonic() + 30
    while checkpoints.latest() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    runner.jobs["scmle-process-preempt-1-1"].process.send_signal(signal.SIGTERM)
    thread.join(60)
    (manifest,) = done
    assert manifest.state == "succeeded"
    assert states(manifest) == ["interrupted", "succeeded"]
    assert manifest.attempts[0].reason == "signal (exit 143)"
    result = json.loads(store.get("runs/preempt-1/result.json")[0])
    assert result["steps"] == 12 and result["attempt"] == 2
    assert 1 <= result["resumed"] < 12
    assert (
        "interrupted: signal"
        in store.get("runs/preempt-1/logs/1.txt")[0].decode()
    )


@pytest.mark.parametrize("deaf", [False, True])
def test_a_timeout_stops_the_process_and_is_final(store, deaf):
    if deaf and sys.platform == "win32":
        pytest.skip("SIGTERM is POSIX only")
    runner = process(store, "stubborn", grace=0.5)
    spec = job(f"stop-{int(deaf)}", {"deaf": deaf}, timeout_seconds=1.5)
    manifest = runner.run(spec)
    assert manifest.state == "timed_out"
    assert states(manifest) == ["cancelled"]
    assert manifest.attempts[0].reason == "timed_out"
    child = runner.jobs[f"scmle-process-stop-{int(deaf)}-1"]
    assert child.process.returncode is not None
    assert child.log.closed


def test_the_process_runner_needs_a_store_the_job_can_open(tmp_path):
    memory = objectstore.RetryingStore(
        objectstore.LocalObjectStore(tmp_path / "s"), retry.Retries()
    )
    with pytest.raises(errors.ConfigError, match="needs a LocalObjectStore"):
        local.Process(memory, serving_identity=SERVE)
    with pytest.raises(errors.ConfigError, match="module:function"):
        local.Process(
            objectstore.LocalObjectStore(tmp_path / "s"),
            serving_identity=SERVE,
            workload="nocolon",
        )
    runner = local.Container(
        objectstore.LocalObjectStore(tmp_path / "s"),
        serving_identity=SERVE,
        image="scmle:1",
        client=str(tmp_path / "no-docker"),
    )
    with pytest.raises(errors.ConfigError, match="cannot start"):
        runner.submit(job("nodocker-1"), 1)


def test_the_container_command_is_hardened_and_carries_no_values(tmp_path):
    store = objectstore.LocalObjectStore(tmp_path / "s")
    runner = local.Container(
        store,
        serving_identity=SERVE,
        image="scmle@sha256:" + "a" * 64,
        user="1000:1000",
    )
    spec = job(
        "box-1",
        resources=base.Resources(
            cpu=2, memory_mb=1024, gpu=1, gpu_reason="cnn"
        ),
        environment={"MODE": "fast"},
    )
    assert runner.command(spec, 2) == [
        "docker",
        "run",
        "--name",
        "scmle-container-box-1-2",
        "--init",
        "--read-only",
        "--tmpfs",
        "/tmp",
        "--workdir",
        "/tmp",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        "1000:1000",
        "--cpus",
        "2",
        "--memory",
        "1024m",
        "--gpus",
        "1",
        "--mount",
        f"type=bind,source={store.root},target=/store",
        "--env",
        "HOME=/tmp",
        "--env",
        "MODE",
        "--env",
        "SCMLE_ATTEMPT",
        "--env",
        "SCMLE_RUN_ID",
        "--env",
        "SCMLE_SPEC_DIGEST",
        "--env",
        "SCMLE_STORE_URI",
        "--env",
        "SCMLE_TENANT",
        "--env",
        "SCMLE_WORKLOAD",
        "scmle@sha256:" + "a" * 64,
        "python",
        "-m",
        "scmle.runners.entrypoint",
    ]
    assert runner.environment(spec, 2) == {
        "MODE": "fast",
        "SCMLE_TENANT": "",
        "SCMLE_RUN_ID": "box-1",
        "SCMLE_ATTEMPT": "2",
        "SCMLE_STORE_URI": "file:///store",
        "SCMLE_SPEC_DIGEST": spec.digest(),
        "SCMLE_WORKLOAD": "scmle.runners.project:run_project",
    }


def test_the_entrypoint_exit_status_tells_transient_from_interrupted(store):
    runner = process(store, "count")
    spec = job("codes-1")
    runner.register(spec)
    env = {"SCMLE_RUN_ID": "codes-1"}

    def preempted(context):
        raise local.PreemptedError("reclaimed")

    def throttled(context):
        raise retry.TransientError("later")

    assert support_runners.run_container(store, env, preempted) == 143
    assert support_runners.run_container(store, env, throttled) == 75
    assert support_runners.run_container(store, env, lambda c: None) == 0
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL


def test_the_command_line_runs_a_spec_in_a_process(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.delenv(tenancy.VARIABLE, raising=False)
    spec = job("cli-1", {"steps": 2})
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec.to_dict()))
    data = tmp_path / "train.csv"
    data.write_bytes(b"a,y\n1,0\n")
    arguments = [
        "--store",
        str(tmp_path / "store"),
        "--spec",
        str(path),
        "--serving",
        SERVE,
        "--workload",
        "tests.support_jobs:count",
        "--poll",
        "0.02",
        "--put",
        f"data/train.csv={data}",
    ]
    assert local.main(arguments) == 0
    assert capsys.readouterr().out == "cli-1: succeeded\n"
    store = objectstore.LocalObjectStore(tmp_path / "store")
    assert store.get("data/train.csv")[0] == b"a,y\n1,0\n"
    assert local.main([*arguments[:-2], "--put", "nokey"]) == 1
    assert "--put needs KEY=PATH" in capsys.readouterr().err
    assert local.main([*arguments, "--serving", TRAIN]) == 1


def test_a_tenant_runner_runs_its_job_in_the_tenant_namespace(tmp_path):
    shared = objectstore.LocalObjectStore(tmp_path / "store")
    confined = tenancy.Tenancy(shared, "acme")
    runner = process(confined, "count")
    assert runner.root == shared.root
    manifest = runner.run(job("tenant-1", {"steps": 2}))
    assert manifest.state == "succeeded"
    assert (
        json.loads(confined.get("runs/tenant-1/result.json")[0])["steps"] == 2
    )
    assert shared.head("tenants/acme/runs/tenant-1/result.json") is not None
    assert shared.list_objects("runs/") == []


def test_the_job_environment_names_the_runner_tenant_over_the_spec(store):
    spec = job("names-1", environment={"SCMLE_TENANT": "other"})
    confined = process(tenancy.Tenancy(store, "acme"), "count")
    assert confined.environment(spec, 1)["SCMLE_TENANT"] == "acme"
    assert process(store, "count").environment(spec, 1)["SCMLE_TENANT"] == ""


def test_a_tenancy_over_a_store_the_job_cannot_open_is_refused(tmp_path):
    remote = objectstore.RetryingStore(
        objectstore.LocalObjectStore(tmp_path / "s"), retry.Retries()
    )
    with pytest.raises(errors.ConfigError, match="needs a LocalObjectStore"):
        local.Process(tenancy.Tenancy(remote, "acme"), serving_identity=SERVE)


def test_the_command_line_confines_the_store_to_its_tenant(
    tmp_path, capsys, monkeypatch
):
    monkeypatch.setenv(tenancy.VARIABLE, "acme")
    spec = job("cli-tenant-1", {"steps": 1})
    path = tmp_path / "job.json"
    path.write_text(json.dumps(spec.to_dict()))
    data = tmp_path / "train.csv"
    data.write_bytes(b"a,y\n1,0\n")
    arguments = [
        "--store",
        str(tmp_path / "store"),
        "--spec",
        str(path),
        "--serving",
        SERVE,
        "--workload",
        "tests.support_jobs:count",
        "--poll",
        "0.02",
        "--put",
        f"data/train.csv={data}",
    ]
    assert local.main(arguments) == 0
    assert capsys.readouterr().out == "cli-tenant-1: succeeded\n"
    shared = objectstore.LocalObjectStore(tmp_path / "store")
    assert shared.get("tenants/acme/data/train.csv")[0] == b"a,y\n1,0\n"
    assert shared.head("tenants/acme/runs/cli-tenant-1/result.json")
    assert shared.list_objects("data/") == []
