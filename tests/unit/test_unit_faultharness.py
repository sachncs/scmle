"""Behaviour of the fault harness, the outage scenarios and the catalogue."""

import json
import pathlib
import subprocess
import sys

import pytest

from scmle.artifacts import bundle
from scmle.deploy import mocks
from scmle.operations import recovery
from scmle.rehearsal.faults import catalogue
from scmle.rehearsal.faults import crashes
from scmle.rehearsal.faults import harness
from scmle.rehearsal.faults import outages
from scmle.rehearsal.faults import release
from scmle.rehearsal.faults import service

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def golden_paths(monkeypatch):
    """The harness reads the golden files relative to the repository root."""
    monkeypatch.chdir(ROOT)


# ---- harness ---------------------------------------------------------------


def test_harness_constants_are_the_documented_policy():
    assert harness.MODEL == "recovery-model"
    assert harness.ENVIRONMENT == "production"
    assert harness.READY_SECONDS == 60.0
    assert harness.SPOOLED_SAMPLES == 50
    assert harness.FULL_TRAFFIC_PERCENT == 100
    assert harness.LABEL_NOISE_RATE == 0.05
    assert harness.INTERRUPTED_STEP_RUNS == 2
    assert harness.ROTATED_KEY != harness.WRONG_KEY


def test_records_returns_the_first_golden_records():
    loaded = json.loads((ROOT / "deploy/local/golden/records.json").read_text())
    assert harness.records() == loaded["records"][:3]
    assert harness.records(1) == loaded["records"][:1]
    assert len(harness.records(5)) == 5


def test_invert_negates_every_coefficient():
    spec = harness.specify()
    wrong = harness.invert()
    assert wrong["model_version"] == "golden-inverted"
    assert wrong["intercept"] == -spec["intercept"]
    assert wrong["coefficients"] == {
        k: -v for k, v in spec["coefficients"].items()
    }
    assert harness.specify() == spec


def test_corrupt_appends_to_the_model_file_and_changes_its_digest(tmp_path):
    folder = harness.export_service(tmp_path / "svc", harness.specify())
    assert (folder / "serve.py").is_file()
    before = harness.digest(folder)
    assert before.startswith("sha256:")
    broken = harness.corrupt(folder, tmp_path / "broken")
    assert broken == tmp_path / "broken"
    assert harness.digest(broken) != before
    assert harness.digest(folder) == before
    assert (
        (broken / "model" / bundle.MODEL_FILE)
        .read_bytes()
        .endswith(b"tampered")
    )


def test_an_instance_names_its_base_and_liveness():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    made = harness.Instance(child, 8123, pathlib.Path())
    assert made.base == "http://127.0.0.1:8123"
    assert made.alive() is False


def test_a_fleet_kills_its_children_and_counts_strays():
    fleet = harness.Fleet()
    child = fleet.child("import time; time.sleep(60)")
    assert fleet.children == [child]
    assert fleet.close() == 0
    assert child.poll() is not None
    assert child.stdout.closed


def test_a_fleet_stops_its_emulators(tmp_path):
    fleet = harness.Fleet()
    store = fleet.serve(mocks.Depot(tmp_path / "objects"))
    assert store.httpd is not None
    assert fleet.close() == 0
    assert store.httpd is None


def test_interrupt_returns_the_last_acknowledged_unit(tmp_path):
    fleet = harness.Fleet()
    code = "import sys\nfor n in range(1000):\n    print(n, flush=True)\n"
    try:
        assert harness.interrupt(fleet, code, tmp_path / "x", 5) == 4
        assert harness.interrupt(fleet, "pass", tmp_path / "y", 5) == -1
    finally:
        fleet.close()


def test_predict_and_failing_report_an_unreachable_service():
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    port = harness.demo.unused()
    instance = harness.Instance(child, port, pathlib.Path())
    assert harness.predict(instance.base) == (0, {})
    seen = harness.failing(instance, timeout=0.3)
    assert seen == {
        "readyz": 0,
        "phase": None,
        "reason": None,
        "livez": 0,
        "predict": 0,
    }
    assert harness.boot(instance, timeout=0.3) is None


def run_broken_stage(kept):
    with harness.stage("t", "fault", "here") as (fleet, base, result):
        kept["base"] = base
        kept["result"] = result
        kept["existed"] = base.is_dir()
        fleet.child("import time; time.sleep(60)")
        raise RuntimeError("scenario broke")


def test_stage_closes_the_fleet_and_removes_the_folder_even_on_error():
    kept = {}
    with pytest.raises(RuntimeError, match="scenario broke"):
        run_broken_stage(kept)
    assert kept["existed"] is True
    assert kept["base"].name.startswith("scmle-recovery-t-")
    assert not kept["base"].exists()
    assert kept["result"].fault_id == "fault"
    assert kept["result"].environment == "here"
    assert kept["result"].detail["processes_still_alive_after_close"] == 0


@pytest.mark.parametrize(
    ("livez", "readyz", "predict", "expected"),
    [
        (200, 503, 503, True),
        (0, 503, 503, False),
        (200, 200, 503, False),
        (200, 503, 200, False),
    ],
)
def test_unready_means_alive_but_refusing(livez, readyz, predict, expected):
    verdict = {"livez": livez, "readyz": readyz, "predict": predict}
    assert harness.unready(verdict) is expected


def test_a_real_service_boots_and_predicts(tmp_path):
    folder = harness.export_service(tmp_path / "svc", harness.specify())
    fleet = harness.Fleet()
    try:
        running = fleet.start(folder)
        assert running.alive()
        assert harness.boot(running) is not None
        status, body = harness.predict(running.base, 2)
        assert status == 200
        assert len(body["predictions"]) == 2
    finally:
        assert fleet.close() == 0


# ---- outages ---------------------------------------------------------------


def passed(result):
    assert result.checks, result
    assert all(result.checks.values()), result.checks
    assert result.detail["processes_still_alive_after_close"] == 0


def test_the_object_store_outage_recovers_without_loss():
    result = outages.object_store_loss()
    passed(result)
    assert result.fault_id == "lost_object_store_access"
    assert result.environment == "local emulator + subprocess"
    measured = result.measurements[0]
    assert measured.component == "object_store"
    assert measured.rpo_lost == 0
    assert measured.rto_seconds > 0
    assert result.detail["objects"] > 0


def test_a_metrics_outage_spools_and_drains_every_sample():
    result = outages.blackout()
    passed(result)
    assert result.fault_id == "dependency_outage"
    measured = result.measurements[0]
    assert measured.component == "telemetry_spool"
    assert measured.rpo_lost == 0
    down = result.detail["flush_while_down"]
    assert down["sent"] == 0
    assert down["remaining"] == harness.SPOOLED_SAMPLES


# ---- catalogue -------------------------------------------------------------


def test_the_catalogue_maps_every_local_fault_to_its_scenario():
    assert catalogue.SCENARIOS["process_kill"] is crashes.process_kill
    assert catalogue.SCENARIOS["bad_release"] is release.bad_release
    assert catalogue.SCENARIOS["dependency_outage"] is outages.blackout
    assert catalogue.SCENARIOS["secret_rotation_recovery"] is service.rekey
    assert set(catalogue.SCENARIOS) == {
        f.id for f in recovery.CATALOGUE if f.local
    }


def test_run_all_reports_unselected_faults_as_not_run():
    report = catalogue.run_all(["no-such-fault"])
    statuses = {r["fault"]: r["status"] for r in report["results"]}
    assert set(statuses.values()) == {recovery.NOT_RUN}
    assert set(statuses) == {f.id for f in recovery.CATALOGUE}


def test_run_all_runs_only_the_selected_fault():
    report = catalogue.run_all(["job_interruption"])
    found = {r["fault"]: r for r in report["results"]}
    assert found["job_interruption"]["status"] == recovery.PASSED
    assert found["process_kill"]["status"] == recovery.NOT_RUN
