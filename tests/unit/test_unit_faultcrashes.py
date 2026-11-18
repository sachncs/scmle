"""Behaviour of the crash scenarios: killed services, jobs and audit writers."""

import json
import pathlib

import pytest

from scmle.rehearsal.faults import crashes
from scmle.rehearsal.faults import harness
from scmle.runtime import limits

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def golden_paths(monkeypatch):
    """The scenarios read the golden files relative to the repository root."""
    monkeypatch.chdir(ROOT)


def passed(result):
    assert result.checks, result
    assert all(result.checks.values()), result.checks
    assert result.detail.get("processes_still_alive_after_close", 0) == 0


def test_the_killed_return_code_is_the_signal_on_posix():
    expected = (
        int(limits.KILL_SIGNAL) if limits.WINDOWS else -int(limits.KILL_SIGNAL)
    )
    assert expected == crashes.KILLED_RETURN_CODE


def test_the_writers_acknowledge_each_unit_on_stdout():
    for code in (crashes.STATE_WRITER, crashes.LEDGER_WRITER):
        assert "print(n, flush=True)" in code
        compile(code, "writer", "exec")
    compile(crashes.JOB_CHILD, "job", "exec")


def test_stress_loses_no_acknowledged_atomic_write(tmp_path):
    fleet = harness.Fleet()
    try:
        state = crashes.stress(fleet, tmp_path)
    finally:
        assert fleet.close() == 0
    assert state["trials"] == 10
    assert state["acknowledged_writes_lost"] == 0
    assert state["unreadable_files"] == 0
    assert state["read_seconds_max"] >= 0.0
    for trial in range(10):
        counter = json.loads((tmp_path / f"state-{trial}.json").read_text())
        assert counter["counter"] >= 0


def test_a_killed_job_resumes_without_repeating_finished_steps():
    result = crashes.job_interruption()
    passed(result)
    assert result.fault_id == "job_interruption"
    assert result.detail["handler_calls"] == [
        "ingest",
        "train",
        "train",
        "evaluate",
    ]
    assert result.detail["lease_seconds"] == 1.0
    assert result.detail["state_objects"] > 0
    measured = result.measurements[0]
    assert measured.component == "job_state"
    assert measured.rpo_lost == 0
    assert measured.rto_seconds >= 1.5


def test_a_killed_audit_writer_reopens_with_a_verified_chain():
    result = crashes.resilience()
    passed(result)
    assert result.fault_id == "audit_log_continuity"
    assert result.detail == {
        "trials": 5,
        "torn_or_broken": 0,
        "processes_still_alive_after_close": 0,
    }
    measured = result.measurements[0]
    assert measured.component == "ledger"
    assert measured.rpo_lost == 0


def test_a_killed_service_restarts_with_the_same_answers():
    result = crashes.process_kill()
    passed(result)
    assert result.fault_id == "process_kill"
    assert len(result.detail["service_rto_seconds_per_trial"]) == 3
    assert result.detail["state_trials"]["trials"] == 10
    service, state = result.measurements
    assert (service.component, service.rpo_lost) == ("service", 0)
    assert (state.component, state.rpo_lost) == ("registry_state", 0)
    assert round(service.rto_seconds, 3) == max(
        result.detail["service_rto_seconds_per_trial"]
    )
