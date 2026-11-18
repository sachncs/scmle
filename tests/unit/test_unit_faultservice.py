"""Behaviour of the service fault scenarios: keys, bundles, schemas, restarts."""

import pathlib
import subprocess
import sys
import threading
import time

import pytest

from scmle.rehearsal import demo
from scmle.rehearsal.faults import harness
from scmle.rehearsal.faults import service

ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def golden_paths(monkeypatch):
    """The scenarios read the golden files relative to the repository root."""
    monkeypatch.chdir(ROOT)


def passed(result):
    assert result.checks, result
    assert all(result.checks.values()), result.checks
    assert result.detail["processes_still_alive_after_close"] == 0


def test_the_drain_policy_is_three_requests_or_ten_seconds():
    assert service.DRAIN_REQUESTS == 3
    assert service.DRAIN_SECONDS == 10.0


def test_a_client_counts_failures_against_a_dead_instance():
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    instance = harness.Instance(dead, demo.unused(), pathlib.Path())
    client = service.Client({"current": instance})
    assert (client.sent, client.failed, client.longest_gap) == (0, 0, 0.0)
    client.start()
    time.sleep(0.3)
    client.stop_flag.set()
    client.join(timeout=10)
    assert client.sent >= 1
    assert client.failed == client.sent
    assert client.longest_gap == 0.0


def test_drain_gives_up_after_its_deadline(monkeypatch):
    monkeypatch.setattr(service, "DRAIN_SECONDS", 0.2)
    client = service.Client({})
    started = time.monotonic()
    client.drain()
    assert 0.15 <= time.monotonic() - started < 5


def test_drain_returns_once_enough_requests_were_answered():
    client = service.Client({})
    client.sent = 5

    def answer():
        time.sleep(0.1)
        client.sent += service.DRAIN_REQUESTS

    helper = threading.Thread(target=answer)
    helper.start()
    started = time.monotonic()
    client.drain()
    helper.join()
    assert time.monotonic() - started < service.DRAIN_SECONDS


def test_a_rolling_restart_never_routes_to_an_unready_copy():
    result = service.restart()
    passed(result)
    assert result.fault_id == "unhealthy_container_rolling_restart"
    assert result.detail["requests_failed"] == 0
    assert result.detail["requests_sent"] > 0
    assert harness.unready(result.detail["unready_replacement"])
    measured = result.measurements[0]
    assert (measured.component, measured.rpo_lost) == ("service", 0)


def test_a_wrong_key_leaves_the_service_unready_without_leaking_it():
    result = service.invalid_credentials()
    passed(result)
    assert harness.WRONG_KEY not in str(result.to_dict())
    assert result.measurements[0].component == "secrets"
    assert harness.unready(result.detail["with_wrong_key"])


def test_a_rotated_key_serves_and_is_audited():
    result = service.rekey()
    passed(result)
    assert result.fault_id == "secret_rotation_recovery"
    assert result.measurements[0].component == "secrets"
    assert harness.ROTATED_KEY not in str(result.to_dict())


def test_a_corrupted_bundle_is_refused_and_restored():
    result = service.corrupted_bundle()
    passed(result)
    assert result.fault_id == "corrupted_bundle"
    assert harness.unready(result.detail["tampered_verdict"])
    assert result.measurements[0].component == "service"


def test_an_incompatible_schema_is_refused_and_recovered():
    result = service.incompatible_schema()
    passed(result)
    assert result.detail["statuses"] == {
        "expected_schema_digest": 409,
        "schema_version": 409,
    }
    assert result.detail["classification"]["verdict"] == "data_quality"
    measured = result.measurements[0]
    assert (measured.component, measured.rpo_lost) == ("service", 0)
