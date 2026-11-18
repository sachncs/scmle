"""The job spec, the runner template, the project workload and Ray."""

import hashlib
import json
import sys
import types

import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.runners import project
from scmle.runners.base import job
from scmle.runners.base import runner
from scmle.runtime import ray_backend
from scmle.runtime import retry
from scmle.storage import objectstore


def spec(**fields):
    return job.Job(
        **{
            "run_id": "run-1",
            "name": "train",
            "project": {"b": 1, "a": 2},
            "identity": "trainer",
            **fields,
        }
    )


def test_resources_must_be_positive_and_a_gpu_needs_a_reason():
    assert job.Resources() == job.Resources(1.0, 2048, 0, "")
    for bad in ({"cpu": 0}, {"memory_mb": 0}, {"gpu": -1}):
        with pytest.raises(errors.ConfigError, match="must be positive"):
            job.Resources(**bad)
    with pytest.raises(errors.ConfigError, match="gpu_reason"):
        job.Resources(gpu=1, gpu_reason="  ")
    assert job.Resources(gpu=1, gpu_reason="vision").gpu == 1


def test_a_cost_rate_prices_cpu_memory_and_gpu_by_the_hour():
    rate = job.CostRate(per_cpu_hour=2.0, per_gb_hour=1.0, per_gpu_hour=10.0)
    resources = job.Resources(cpu=2, memory_mb=2048, gpu=1, gpu_reason="x")
    assert rate.cost(resources, 3600) == 16.0
    assert rate.cost(resources, 1800) == 8.0
    assert rate.cost(resources, -5) == 0.0


def test_a_cost_budget_must_be_positive():
    with pytest.raises(errors.ConfigError, match="max_cost must be positive"):
        job.CostBudget(0, job.CostRate())
    assert job.CostBudget(0.01, job.CostRate()).max_cost == 0.01


def test_the_run_id_must_be_short_and_safe():
    assert spec(run_id="a" * 64).run_id == "a" * 64
    for bad in ("", "a" * 65, "a/b", "a b", "ä"):
        with pytest.raises(errors.ConfigError, match="run_id must be 1-64"):
            spec(run_id=bad)


def test_a_job_needs_a_timeout_and_an_identity():
    for bad in ({"timeout_seconds": 0}, {"identity": ""}):
        with pytest.raises(errors.ConfigError, match="timeout and an identity"):
            spec(**bad)


def test_the_environment_may_not_hold_secrets():
    for word in job.SECRET_WORDS:
        with pytest.raises(errors.ConfigError, match="looks like a secret"):
            spec(environment={f"my_{word.lower()}": "x"})
    with pytest.raises(errors.ConfigError, match="environment PEM holds a key"):
        spec(environment={"PEM": "-----BEGIN RSA"})
    assert spec(environment={"MODE": "fast"}).environment == {"MODE": "fast"}


def test_the_project_digest_is_of_the_canonical_json():
    text = json.dumps({"a": 2, "b": 1}, separators=(",", ":"))
    assert spec().project_digest == hashlib.sha256(text.encode()).hexdigest()


def test_the_digest_is_anchored_at_the_version_one_layout():
    made = spec()
    anchor = made.anchor()
    assert anchor["run_id"] == "run-1"
    assert anchor["timeout_seconds"] == 3600.0
    assert anchor["budget"] is None
    assert anchor["project_digest"] == made.project_digest
    assert (
        made.digest()
        == hashlib.sha256(
            json.dumps(anchor, sort_keys=True).encode()
        ).hexdigest()
    )
    budgeted = spec(budget=job.CostBudget(5.0, job.CostRate(per_cpu_hour=1.0)))
    assert budgeted.anchor()["budget"] == {
        "max_cost": 5.0,
        "rate": {"per_cpu_hour": 1.0, "per_gb_hour": 0.0, "per_gpu_hour": 0.0},
    }
    assert budgeted.digest() != made.digest()


def test_the_stored_form_uses_the_current_keys():
    stored = spec().to_dict()
    assert stored["identifier"] == "run-1"
    assert stored["timeout"] == 3600.0
    assert stored["digests"]["project"] == spec().project_digest
    assert "run_id" not in stored


def test_a_status_says_how_it_ends_the_run():
    assert job.Status(job.State.SUCCEEDED).outcome() is job.Progress.SUCCEEDED
    assert job.Status(job.State.INTERRUPTED).outcome() is None
    assert job.Status(job.State.FAILED, retryable=True).outcome() is None
    assert job.Status(job.State.FAILED).outcome() is job.Progress.FAILED
    assert job.Status(job.State.CANCELLED).outcome() is job.Progress.FAILED


def test_terminal_and_final_states():
    assert {
        job.State.SUCCEEDED,
        job.State.FAILED,
        job.State.INTERRUPTED,
        job.State.CANCELLED,
    } == job.TERMINAL
    assert job.Progress.STARTED not in job.FINAL
    assert len(job.FINAL) == len(job.Progress) - 1


def test_the_meter_adds_attempt_costs_and_stops_at_the_ceiling():
    budget = job.CostBudget(2.0, job.CostRate(per_cpu_hour=3600.0))
    meter = job.Meter(spec(budget=budget, timeout_seconds=10), spent=1.0)
    assert meter.estimate(0.5) == 1.5
    assert meter.exceeded(1.0) is False
    assert meter.exceeded(1.01) is True
    assert meter.verdict(1.01) is job.Progress.BUDGET_STOPPED
    assert meter.close(0.5) == 1.5
    assert meter.spent == 1.5


def test_the_meter_without_a_budget_only_times_out():
    meter = job.Meter(spec(timeout_seconds=10), spent=3.0)
    assert meter.estimate(1e9) == 3.0
    assert meter.exceeded(1e9) is False
    assert meter.verdict(10) is None
    assert meter.verdict(10.5) is job.Progress.TIMED_OUT


def test_stamp_is_utc_to_the_second():
    text = job.stamp()
    assert text.endswith("+00:00")
    assert len(text) == len("2026-01-01T00:00:00+00:00")


class Clock:
    def __init__(self):
        self.seconds = 0.0
        self.sleeps = []

    def now(self):
        return self.seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds


class Scripted(runner.Runner):
    """Plays back statuses; each poll takes the next one."""

    kind = "scripted"

    def __init__(self, store, statuses, result=True):
        self.clock_source = Clock()
        super().__init__(
            store,
            serving_identity="server",
            clock=job.Clock(self.clock_source.now, self.clock_source.sleep),
            poll_seconds=1.0,
        )
        self.statuses = list(statuses)
        self.submitted = []
        self.cancelled = []
        self.result = result

    def submit(self, spec, attempt):
        self.submitted.append(attempt)
        if self.result:
            self.store.put(self.result_key(spec.run_id), b"{}")
        return job.Ticket(f"job-{attempt}", attempt)

    def poll(self, handle):
        if len(self.statuses) > 1:
            return self.statuses.pop(0)
        return self.statuses[0]

    def cancel(self, handle, reason):
        self.cancelled.append((handle.job_id, reason))


@pytest.fixture
def store(tmp_path):
    return objectstore.LocalObjectStore(tmp_path / "store")


def quick(**fields):
    return spec(
        retries=retry.Retries(
            max_attempts=3, base_delay_seconds=2.0, factor=2.0
        ),
        **fields,
    )


def test_the_keys_of_a_run():
    made = Scripted(None, [])
    assert made.locate("r") == "runs/r/manifest.json"
    assert made.spec_key("r") == "runs/r/spec.json"
    assert made.result_key("r") == "runs/r/result.json"
    assert made.load is not None


def test_a_successful_attempt_ends_the_run(store):
    made = Scripted(
        store, [job.Status(job.State.RUNNING), job.Status(job.State.SUCCEEDED)]
    )
    manifest = made.run(quick())
    assert manifest.state == job.Progress.SUCCEEDED.value
    assert made.submitted == [1]
    assert len(manifest.attempts) == 1
    assert manifest.attempts[0].state == "succeeded"
    assert manifest.attempts[0].seconds == 1.0
    assert made.clock_source.sleeps == [1.0]
    assert made.load("run-1").state == "succeeded"


def test_a_finished_run_starts_nothing_again(store):
    made = Scripted(store, [job.Status(job.State.SUCCEEDED)])
    made.run(quick())
    again = Scripted(store, [job.Status(job.State.FAILED)])
    assert again.run(quick()).state == "succeeded"
    assert again.submitted == []


def test_interrupted_attempts_retry_with_backoff_until_exhausted(store):
    made = Scripted(store, [job.Status(job.State.INTERRUPTED, "preempted")])
    manifest = made.run(quick())
    assert manifest.state == job.Progress.RETRIES_EXHAUSTED.value
    assert made.submitted == [1, 2, 3]
    assert made.clock_source.sleeps == [2.0, 4.0]
    assert [a.reason for a in manifest.attempts] == ["preempted"] * 3


def test_a_success_without_a_result_is_recorded_as_failed(store):
    made = Scripted(store, [job.Status(job.State.SUCCEEDED)], result=False)
    manifest = made.run(quick())
    assert manifest.state == job.Progress.FAILED.value
    assert manifest.attempts[-1].reason == "job ended without a result"


def test_a_timeout_cancels_the_attempt_and_confirms_it(store):
    statuses = [job.Status(job.State.RUNNING)] * 3 + [
        job.Status(job.State.CANCELLED, "stopped")
    ]
    made = Scripted(store, statuses)
    manifest = made.run(quick(timeout_seconds=1.5))
    assert manifest.state == job.Progress.TIMED_OUT.value
    assert made.cancelled == [("job-1", "timed_out")]
    assert manifest.attempts[0].state == "cancelled"
    assert manifest.attempts[0].reason == "timed_out"


def test_confirm_gives_up_after_six_polls(store):
    made = Scripted(store, [job.Status(job.State.RUNNING)])
    status = made.confirm(job.Ticket("j", 1))
    assert status == job.Status(job.State.CANCELLED, "cancel requested")
    assert made.clock_source.sleeps == [1.0] * 5


def test_a_clashing_spec_or_the_serving_identity_is_refused(store):
    made = Scripted(store, [job.Status(job.State.SUCCEEDED)])
    made.run(quick())
    with pytest.raises(job.ConflictError, match="different spec"):
        made.register(quick(name="other"))
    with pytest.raises(errors.ConfigError, match="serving identity"):
        made.register(quick(identity="server"))


def test_a_budget_stop_is_recorded_with_its_cost(store):
    budget = job.CostBudget(1.0, job.CostRate(per_cpu_hour=3600.0))
    made = Scripted(
        store,
        [job.Status(job.State.RUNNING)] * 3 + [job.Status(job.State.CANCELLED)],
    )
    manifest = made.run(quick(budget=budget))
    assert manifest.state == job.Progress.BUDGET_STOPPED.value
    assert made.cancelled == [("job-1", "budget_stopped")]
    assert manifest.cost == pytest.approx(manifest.attempts[0].cost)
    assert manifest.cost > 1.0


def test_the_mirror_pushes_only_changed_files_and_restores(store, tmp_path):
    local = tmp_path / "local"
    (local / "sub").mkdir(parents=True)
    (local / "a.txt").write_text("a")
    (local / "sub" / "b.txt").write_text("b")
    mirror = project.Mirror(store, "w/", local)
    assert mirror.push() == 2
    assert mirror.push() == 0
    (local / "a.txt").write_text("changed")
    assert mirror.push() == 1
    assert mirror.uploaded["sub/b.txt"] == hashlib.sha256(b"b").hexdigest()
    empty = tmp_path / "empty"
    back = project.Mirror(store, "w/", empty)
    assert back.restore() == 2
    assert (empty / "sub" / "b.txt").read_text() == "b"
    assert (empty / "a.txt").read_text() == "changed"
    assert back.push() == 0


def test_close_stops_the_thread_and_pushes(store, tmp_path):
    local = tmp_path / "local"
    local.mkdir()
    mirror = project.Mirror(store, "w/", local)
    mirror.start(3600)
    (local / "late.txt").write_text("x")
    mirror.close()
    assert mirror.stop.is_set()
    assert not mirror.thread.is_alive()
    assert store.get("w/late.txt")[0] == b"x"
    assert project.SYNC_SECONDS == 30.0


def test_locate_finds_the_first_run_configuration(tmp_path):
    assert project.locate(tmp_path) is None
    (tmp_path / "b").mkdir()
    (tmp_path / "a").mkdir()
    (tmp_path / "b" / "config.json").write_text("{}")
    (tmp_path / "a" / "spec.json").write_text("{}")
    (tmp_path / "a" / "other.json").write_text("{}")
    assert project.locate(tmp_path) == tmp_path / "a"
    assert project.CONFIG_NAMES == ("config.json", "spec.json")


def test_configure_builds_the_sections_and_renames_derivation():
    built = project.configure(
        {
            "time_budget": 60,
            "search": {"configs_per_model": 2},
            "derivation": {},
        }
    )
    assert built["time_budget"] == 60
    assert isinstance(built["search"], tasks.Search)
    assert built["search"].configs_per_model == 2
    assert isinstance(built["feature_graph"], tasks.Derivation)
    assert "derivation" not in built
    kept = project.configure({"models": tasks.Candidates()})
    assert isinstance(kept["models"], tasks.Candidates)


def test_summarize_reports_the_run():
    result = types.SimpleNamespace(
        run_id="r",
        state="complete",
        selected_model="ridge",
        evaluation={
            "metric": "rmse",
            "selected": {"score": 1.5},
            "metrics": {"rmse": 1.4},
        },
        trials=[
            types.SimpleNamespace(
                model_name="ridge", status=types.SimpleNamespace(value="ok")
            ),
            types.SimpleNamespace(model_name="dummy", status="failed"),
        ],
        summary=lambda: "done",
    )
    assert project.summarize(result, True) == {
        "run_id": "r",
        "state": "complete",
        "resumed": True,
        "selected_model": "ridge",
        "metric": "rmse",
        "selected_cv_score": 1.5,
        "test_metrics": {"rmse": 1.4},
        "trials": [
            {"model": "ridge", "status": "ok"},
            {"model": "dummy", "status": "failed"},
        ],
        "summary": "done",
    }


def test_a_project_without_data_or_target_is_refused():
    context = types.SimpleNamespace(
        spec=types.SimpleNamespace(project={"data": "x"})
    )
    with pytest.raises(errors.ConfigError, match="'data' and 'target'"):
        project.run_project(context)


class RayError(Exception):
    pass


class FakeRay(types.ModuleType):
    def __init__(self, initialized=False):
        super().__init__("ray")
        self.initialized = initialized
        self.settings = None
        self.retries = None
        self.exceptions = types.SimpleNamespace(RayError=RayError)

    def is_initialized(self):
        return self.initialized

    def init(self, **settings):
        self.settings = settings

    def remote(self, max_retries):
        self.retries = max_retries

        def wrap(function):
            return types.SimpleNamespace(remote=lambda item: (function, item))

        return wrap

    def get(self, reference):
        function, item = reference
        return function(item)


def test_remote_starts_a_local_runtime_with_the_workers(monkeypatch):
    fake = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    remote = ray_backend.Remote(0)
    assert remote.workers == 1
    assert remote.retries == ray_backend.DEFAULT_RETRIES == 1
    assert fake.settings["num_cpus"] == 1
    assert fake.settings["include_dashboard"] is False
    assert "address" not in fake.settings
    assert remote.name == "ray"


def test_remote_joins_a_cluster_and_reuses_a_running_runtime(monkeypatch):
    fake = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    ray_backend.Remote(4, address="auto")
    assert fake.settings["address"] == "auto"
    assert "num_cpus" not in fake.settings
    running = FakeRay(initialized=True)
    monkeypatch.setitem(sys.modules, "ray", running)
    ray_backend.Remote(4)
    assert running.settings is None


def test_remote_maps_in_order_and_hands_failures_over(monkeypatch):
    fake = FakeRay()
    monkeypatch.setitem(sys.modules, "ray", fake)
    remote = ray_backend.Remote(2, retries=3)
    assert remote.map(lambda x: x * 2, []) == []
    assert remote.map(lambda x: x * 2, [1, 2, 3]) == [2, 4, 6]
    assert fake.retries == 3

    def fails(item):
        if item == 2:
            raise RayError("lost")
        return item

    assert remote.map(fails, [1, 2], lambda item, error: f"{item}:{error}") == [
        1,
        "2:lost",
    ]
    with pytest.raises(RayError):
        remote.map(fails, [2])


def test_remote_without_ray_asks_for_the_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "ray", None)
    with pytest.raises(errors.ConfigError, match="'distributed' extra"):
        ray_backend.Remote(1)
