"""The fault catalogue, the runner's verdicts and the telemetry spool."""

import json

import pytest

from scmle.contracts import observability
from scmle.data.store import objectstore
from scmle.operations import recovery
from scmle.operations import spool


@pytest.fixture(autouse=True)
def full_speed(monkeypatch):
    """CI sets a time scale for slow runners; these tests assume none."""
    monkeypatch.delenv(recovery.TIME_SCALE_ENV, raising=False)


STAMP = "2026-11-05T10:00:00+00:00"


def passing(fault_id, rto=1.0, lost=0, component="service"):
    def scenario():
        result = recovery.Result(fault_id, environment="test")
        result.checks = {"it held": True}
        result.measurements = [
            recovery.Measurement(component, rto, lost, "test")
        ]
        return result

    return scenario


# ---- catalogue ---------------------------------------------------------


def test_the_catalogue_covers_every_declared_fault_once():
    ids = [fault.id for fault in recovery.CATALOGUE]
    assert len(ids) == len(set(ids))
    required = {
        "process_kill",
        "unhealthy_container_rolling_restart",
        "invalid_credentials",
        "lost_object_store_access",
        "dependency_outage",
        "corrupted_bundle",
        "incompatible_schema",
        "bad_release",
        "job_interruption",
        "rollback_to_managed_revision",
        "audit_log_continuity",
        "secret_rotation_recovery",
    }
    assert required <= set(ids)
    for fault in recovery.CATALOGUE:
        assert fault.cloud_needs and fault.local_limits and fault.components
        for component in fault.components:
            assert component in recovery.OBJECTIVE_OF


def test_every_objective_is_declared_with_a_scope_and_none_loses_data():
    assert {o.component for o in recovery.OBJECTIVES} == {
        "service",
        "ledger",
        "registry_state",
        "object_store",
        "secrets",
        "telemetry_spool",
        "job_state",
    }
    for goal in recovery.OBJECTIVES:
        assert goal.rto_seconds > 0 and goal.rpo_lost == 0 and goal.scope


# ---- runner ------------------------------------------------------------


def test_a_fault_with_no_scenario_is_not_run_never_passed():
    report = recovery.Runner({"process_kill": passing("process_kill")}).run()
    by_id = {r["fault_id"]: r for r in report["results"]}
    assert by_id["process_kill"]["status"] == recovery.PASSED
    assert by_id["bad_release"]["status"] == recovery.NOT_RUN
    assert "no scenario" in by_id["bad_release"]["problems"][0]
    managed = by_id["rollback_to_managed_revision"]
    assert managed["status"] == recovery.NOT_RUN
    assert "real cloud" in managed["problems"][0]
    assert report["summary"]["passed"] == 1
    assert report["summary"]["not_run"] == len(recovery.CATALOGUE) - 1
    for provider in recovery.PROVIDERS:
        assert report["cloud"][provider]["status"] == recovery.NOT_RUN


def test_the_runner_refuses_scenarios_it_cannot_honestly_run():
    with pytest.raises(ValueError, match="not in the catalogue"):
        recovery.Runner({"nonsense": passing("nonsense")})
    with pytest.raises(ValueError, match="cannot run locally"):
        recovery.Runner(
            {
                "rollback_to_managed_revision": passing(
                    "rollback_to_managed_revision"
                )
            }
        )


def test_measurements_outside_the_declared_objective_fail_the_fault():
    slow = passing(
        "process_kill", rto=recovery.OBJECTIVE_OF["service"].rto_seconds + 1
    )
    lossy = passing("bad_release", lost=1)
    never = passing("corrupted_bundle", rto=None)
    report = recovery.Runner(
        {"process_kill": slow, "bad_release": lossy, "corrupted_bundle": never}
    ).run()
    by_id = {r["fault_id"]: r for r in report["results"]}
    assert by_id["process_kill"]["status"] == recovery.FAILED
    assert "exceeds" in by_id["process_kill"]["problems"][0]
    assert "lost" in by_id["bad_release"]["problems"][0]
    assert "did not recover" in by_id["corrupted_bundle"]["problems"][0]
    assert report["all_local_passed"] is False


def test_a_failed_check_or_a_crash_is_a_failure_not_a_pass():
    def false_check():
        result = recovery.Result("process_kill", environment="test")
        result.checks = {"the thing held": False}
        result.measurements = [recovery.Measurement("service", 1.0, 0, "t")]
        return result

    def crash():
        raise RuntimeError("boom")

    report = recovery.Runner(
        {"process_kill": false_check, "bad_release": crash}
    ).run()
    by_id = {r["fault_id"]: r for r in report["results"]}
    assert by_id["process_kill"]["status"] == recovery.FAILED
    assert "check failed: the thing held" in by_id["process_kill"]["problems"]
    assert by_id["bad_release"]["status"] == recovery.FAILED
    assert "RuntimeError: boom" in by_id["bad_release"]["problems"][0]


def test_a_selected_subset_marks_the_rest_not_selected_and_the_report_is_json():
    scenarios = {
        "process_kill": passing("process_kill"),
        "bad_release": passing("bad_release"),
    }
    report = recovery.Runner(scenarios).run(["bad_release"])
    by_id = {r["fault_id"]: r for r in report["results"]}
    assert by_id["bad_release"]["status"] == recovery.PASSED
    assert by_id["process_kill"]["problems"] == ["not selected in this run"]
    assert report["all_local_passed"] is True
    json.dumps(report)


# ---- spool -------------------------------------------------------------


def sample(n):
    return observability.MetricSample(
        f"m{n}", observability.SignalKind.THROUGHPUT, float(n), "count", STAMP
    )


class Destination:
    def __init__(self):
        self.up = False
        self.received = []

    def __call__(self, samples):
        if not self.up:
            raise ConnectionError("destination down")
        self.received.extend(s.name for s in samples)
        return len(samples)


def test_samples_survive_an_outage_and_a_restart_and_arrive_in_order(tmp_path):
    destination = Destination()
    path = tmp_path / "spool.jsonl"
    first = spool.MetricsSpool(path, destination, batch_size=4)
    first.add([sample(n) for n in range(10)])
    down = first.flush()
    assert (
        down.sent == 0
        and down.remaining == 10
        and "ConnectionError" in down.error
    )
    again = spool.MetricsSpool(path, destination, batch_size=4)
    assert again.pending() == 10
    destination.up = True
    done = again.flush()
    assert done.sent == 10 and done.remaining == 0 and done.error is None
    assert destination.received == [f"m{n}" for n in range(10)]
    assert again.pending() == 0


def test_a_partial_failure_keeps_exactly_what_was_not_accepted(tmp_path):
    calls = {"count": 0}

    def flaky(samples):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("reset")
        return len(samples)

    held = spool.MetricsSpool(tmp_path / "s.jsonl", flaky, batch_size=3)
    held.add([sample(n) for n in range(7)])
    result = held.flush()
    assert result.sent == 3 and result.remaining == 4
    assert [json.loads(line)["name"] for line in held.lines()] == [
        "m3",
        "m4",
        "m5",
        "m6",
    ]


def test_a_destination_that_accepts_fewer_than_sent_is_not_trusted(tmp_path):
    held = spool.MetricsSpool(tmp_path / "s.jsonl", lambda s: len(s) - 1)
    held.add([sample(1), sample(2)])
    result = held.flush()
    assert result.sent == 0 and result.remaining == 2
    assert "accepted 1 of 2" in result.error


def test_an_empty_spool_and_bad_batch_sizes():
    with pytest.raises(ValueError):
        spool.MetricsSpool("x", lambda s: 0, batch_size=0)


# ---- a finding from the job-interruption fault --------------------------


def test_a_killed_writers_temp_file_does_not_make_the_store_unlistable(
    tmp_path,
):
    store = objectstore.LocalObjectStore(tmp_path)
    store.put("runs/a/step.json", b"{}")
    (tmp_path / "runs" / "a" / ".step.json.q1w2e3r4.tmp").write_bytes(b"half")
    (tmp_path / "runs" / "a" / ".put-abcdef").write_bytes(b"half")
    assert [m.key for m in store.list_objects("runs/")] == ["runs/a/step.json"]
    assert objectstore.is_temporary(".x.json.abc.tmp")
    assert not objectstore.is_temporary("x.tmp")
    assert not objectstore.is_temporary("step.json")


def test_the_time_scale_defaults_to_one_and_is_validated(monkeypatch):
    monkeypatch.delenv(recovery.TIME_SCALE_ENV, raising=False)
    assert recovery.time_scale() == 1.0
    for bad in ("0", "-1", "fast", "nan", "inf"):
        monkeypatch.setenv(recovery.TIME_SCALE_ENV, bad)
        with pytest.raises(ValueError, match=recovery.TIME_SCALE_ENV):
            recovery.time_scale()


def test_a_scaled_run_widens_the_limit_but_not_the_declared_objective(
    monkeypatch,
):
    slow = recovery.Measurement("service", 40.0, 0, "t")
    monkeypatch.delenv(recovery.TIME_SCALE_ENV, raising=False)
    assert slow.problems() == ["service: RTO 40.00s exceeds 30s"]
    monkeypatch.setenv(recovery.TIME_SCALE_ENV, "2")
    assert slow.problems() == []
    assert recovery.Measurement("service", 61.0, 0, "t").problems()
    record = slow.to_dict()
    assert record["time_scale"] == 2.0
    assert record["rto_objective_seconds"] == 30.0
    assert recovery.OBJECTIVE_OF["service"].rto_seconds == 30.0


def test_the_report_states_its_time_scale(monkeypatch):
    monkeypatch.setenv(recovery.TIME_SCALE_ENV, "2.5")
    assert recovery.Runner({}).run([])["time_scale"] == 2.5
