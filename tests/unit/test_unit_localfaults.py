"""Behaviour of each local fault scenario and the evidence it records."""

import errno
import hashlib
import json

import pytest

from scmle.contracts import errors
from scmle.contracts import reliability
from scmle.rehearsal.local_faults import executor
from scmle.rehearsal.local_faults import interruptions
from scmle.rehearsal.local_faults import provenance
from scmle.rehearsal.local_faults import resources
from scmle.rehearsal.local_faults import workspace


@pytest.fixture
def runner(tmp_path):
    return executor.Executor(tmp_path / "faults")


def facts(artifacts):
    """Each artifact's evidence type with its recorded details."""
    found = {}
    for item in artifacts:
        record = json.loads(item.content)
        found[item.evidence_type] = record["details"]
    return found


# ---- workspace -------------------------------------------------------------


def test_a_workspace_creates_and_resolves_its_root(tmp_path):
    made = workspace.Workspace(tmp_path / "a" / ".." / "b")
    assert made.root == (tmp_path / "b").resolve()
    assert made.root.is_dir()


def test_an_artifact_is_compact_sorted_json_marked_safe(tmp_path):
    made = workspace.Workspace(tmp_path)
    item = made.artifact(
        reliability.Contingency.TIMEOUT, "kind", "seen", {"b": 1, "a": "é"}
    )
    assert item.content == (
        b'{"details":{"a":"\\u00e9","b":1},"evidence_type":"kind",'
        b'"observation":"seen","scenario":"timeout"}'
    )
    assert item.evidence_type == "kind"
    assert item.observation == "seen"
    assert item.observed_outcome is reliability.Outcome.SAFE_OUTCOME_OBSERVED


def test_write_json_writes_compact_sorted_json(tmp_path):
    made = workspace.Workspace(tmp_path)
    made.write_json(tmp_path / "x.json", {"b": [1], "a": None})
    assert (tmp_path / "x.json").read_bytes() == b'{"a":null,"b":[1]}'


def test_halt_terminates_a_sleeping_child(tmp_path):
    # Enforces website/src/pages/docs/operations.astro:194.
    terminated, code = workspace.Workspace(tmp_path).halt()
    assert terminated is True
    assert code != 0


# ---- interruptions ---------------------------------------------------------


def test_unreachable_always_fails_like_a_connector():
    with pytest.raises(OSError, match="controlled connector failure"):
        interruptions.unreachable()


def test_process_death_keeps_the_checkpoint(runner):
    found = facts(runner.run_process_death())
    assert list(found) == [
        "process_termination",
        "checkpoint_resume",
        "artifact_integrity",
    ]
    assert found["process_termination"]["terminated"] is True
    assert found["checkpoint_resume"] == {"restored_step": 3}
    checkpoint = runner.root / "process-death-checkpoint.json"
    assert json.loads(checkpoint.read_text()) == {
        "run_id": "local-process-death",
        "step": 3,
    }
    assert (
        found["artifact_integrity"]["sha256"]
        == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    )


def test_a_stale_checkpoint_is_rejected(runner):
    # Enforces website/src/pages/docs/operations.astro:19.
    # Enforces website/src/pages/docs/recovery.astro:11.
    found = facts(runner.run_stale_checkpoint())
    assert found == {
        "checkpoint_identity": {"source": "source-old"},
        "stale_checkpoint_rejection": {"rejected": True},
        "run_state": {"state": "blocked_before_resume"},
    }
    assert (runner.root / "stale-checkpoint.json").is_file()


def test_worker_loss_writes_one_event(runner):
    found = facts(runner.run_worker_loss())
    assert found["idempotent_retry"] == {"event_count": 1}
    assert found["event_deduplication"] == {"event_ids": ["event-1"]}
    assert found["worker_termination"]["terminated"] is True
    output = json.loads((runner.root / "worker-output.json").read_text())
    assert output == {"event_ids": ["event-1"]}


def test_a_timeout_is_cancelled_and_publishes_nothing(runner):
    # Enforces website/src/pages/docs/operations.astro:194.
    found = facts(runner.run_timeout())
    assert found["timeout_injection"] == {"terminated": True}
    assert found["cancellation_record"] == {"state": "cancelled"}
    assert found["resource_error"] == {"public_result_exists": False}
    terminal = json.loads((runner.root / "timeout-terminal.json").read_text())
    assert terminal["state"] == "cancelled"


def test_retries_are_exhausted_after_three_attempts(runner):
    # Enforces website/src/pages/docs/operations.astro:194.
    found = facts(runner.run_retry_exhaustion())
    assert found == {
        "retry_history": {"attempts": 3},
        "terminal_event": {"state": "retry_exhausted"},
        "partial_publication_check": {"public_result_exists": False},
    }
    terminal = json.loads((runner.root / "retry-terminal.json").read_text())
    assert terminal == {"attempts": 3, "state": "retry_exhausted"}


# ---- resources -------------------------------------------------------------


def test_resource_budgets_are_the_documented_constants():
    assert resources.DISK_BUDGET_BYTES == 8
    assert resources.MEMORY_BUDGET_BYTES == 1024
    assert resources.REQUIRED_DEPENDENCY == "numpy==2.3.0"
    assert resources.OBSERVED_DEPENDENCY == "numpy==2.4.0"


def test_cap_refuses_only_a_payload_over_the_disk_budget():
    resources.cap(b"x" * 8)
    with pytest.raises(OSError) as caught:
        resources.cap(b"x" * 9)
    assert caught.value.errno == errno.ENOSPC


def test_ration_refuses_only_an_allocation_over_the_budget():
    resources.ration(bytearray(1024))
    with pytest.raises(MemoryError, match="memory budget exhausted"):
        resources.ration(bytearray(1025))


def test_a_full_disk_publishes_nothing(runner):
    found = facts(runner.run_disk_full())
    assert found == {
        "disk_capacity_exhaustion": {"errno": errno.ENOSPC, "rejected": True},
        "artifact_integrity": {"public_artifact_exists": False},
        "resource_error": {"error": "ENOSPC"},
    }


def test_a_corrupt_artifact_is_detected_and_quarantined(runner):
    found = facts(runner.run_corrupt_artifact())
    assert found["corruption_injection"] == {"changed": True}
    assert found["digest_mismatch"] == {
        "expected": hashlib.sha256(b"valid-artifact").hexdigest(),
        "actual": hashlib.sha256(b"corrupted-artifact").hexdigest(),
    }
    assert found["quarantine_record"] == {"quarantined": True}
    assert not (runner.root / "artifact.bin").exists()


def test_memory_exhaustion_stops_just_past_the_budget(runner):
    found = facts(runner.run_memory_exhaustion())
    assert found["memory_pressure"] == {"rejected": True, "bytes": 1280}
    assert found["publication_absence"] == {"public_result_exists": False}
    assert found["resource_error"] == {"error": "MemoryError"}


def test_a_partial_write_never_reaches_the_public_path(runner):
    # Enforces website/src/pages/docs/recovery.astro:11.
    found = facts(runner.run_partial_write())
    assert found == {
        "write_interruption": {"temporary_exists": True},
        "atomicity_check": {"public_exists": False},
        "artifact_integrity": {"public_bytes": 0},
    }
    assert list(runner.root.iterdir()) == []


def test_a_dependency_mismatch_blocks_the_run(runner):
    found = facts(runner.run_dependency_mismatch())
    assert found == {
        "environment_identity": {
            "required": "numpy==2.3.0",
            "observed": "numpy==2.4.0",
        },
        "dependency_rejection": {"rejected": True},
        "run_state": {"state": "blocked_before_execution"},
    }


# ---- provenance ------------------------------------------------------------


def test_a_tampered_signature_is_quarantined(runner):
    # Enforces website/src/pages/docs/operations.astro:194.
    # Enforces website/src/pages/docs/recovery.astro:11.
    found = facts(runner.run_invalid_signature())
    assert found == {
        "signature_tampering": {"rejected": True},
        "verification_failure": {"verified": False},
        "quarantine_record": {"quarantined": True},
    }
    record = json.loads((runner.root / "quarantine-signature.json").read_text())
    assert record == {"rejected": True, "state": "quarantined"}


def test_a_duplicate_event_is_committed_once(runner):
    # Enforces website/src/pages/docs/operations.astro:179.
    found = facts(runner.run_duplicate_event())
    assert found["duplicate_event"]["first"] == ["event-1"]
    assert found["duplicate_event"]["second"] == ["event-1"]
    assert found["idempotency_key"] == {"accepted": ["event-1"]}
    assert found["state_transition"]["committed"] == ["event-1"]


def test_an_incomplete_lineage_blocks_release(runner):
    # Enforces website/src/pages/docs/operations.astro:194.
    found = facts(runner.run_incomplete_lineage())
    assert found == {
        "lineage_gap": {"rejected": True},
        "release_block": {"release_ready": False},
        "audit_record": {"node": "curation"},
    }


def test_a_rejected_approval_blocks_release(runner):
    # Enforces website/src/pages/docs/operations.astro:194.
    found = facts(runner.run_rejected_approval())
    assert found == {
        "approval_rejection": {"rejected": True},
        "release_block": {"release_ready": False},
        "audit_record": {"reviewer": "independent-reviewer"},
    }


def test_a_changed_snapshot_is_rejected(runner):
    # Enforces website/src/pages/docs/operations.astro:179.
    # Enforces website/src/pages/docs/recovery.astro:11.
    # Enforces website/src/pages/docs/recovery.astro:29.
    found = facts(runner.run_changed_snapshot())
    assert found == {
        "snapshot_digest_change": {"snapshot_id": "snapshot-1"},
        "run_invalidation": {"rejected": True},
        "lineage_record": {"release_ready": False},
    }


def test_provenance_scenarios_live_on_the_workspace():
    assert issubclass(provenance.Provenance, workspace.Workspace)


# ---- executor --------------------------------------------------------------


@pytest.mark.parametrize("scenario", list(reliability.Contingency))
def test_execute_returns_exactly_the_planned_evidence(runner, scenario):
    artifacts = runner.execute(scenario)
    expected = next(
        plan.requirements
        for plan in reliability.drills()
        if plan.scenario is scenario
    )
    names = [item.evidence_type for item in artifacts]
    assert sorted(names) == sorted(expected)
    assert len(names) == len(set(names))
    for item in artifacts:
        assert json.loads(item.content)["scenario"] == scenario.value


def test_execute_refuses_a_value_that_is_not_a_scenario(runner):
    with pytest.raises(errors.KindError, match="scenario must be Contingency"):
        runner.execute("timeout")


def test_a_handler_failure_is_reported_as_a_runtime_error(tmp_path):
    class Broken(executor.Executor):
        __slots__ = ()

        def run_timeout(self):
            raise OSError("boom")

    with pytest.raises(
        RuntimeError, match="local fault scenario failed: timeout"
    ):
        Broken(tmp_path).execute(reliability.Contingency.TIMEOUT)


def test_evidence_other_than_the_plan_is_refused(tmp_path):
    class Short(executor.Executor):
        __slots__ = ()

        def run_timeout(self):
            return super().run_timeout()[:2]

    with pytest.raises(ValueError, match="scenario timeout returned evidence"):
        Short(tmp_path).execute(reliability.Contingency.TIMEOUT)


def test_repeated_evidence_is_refused(tmp_path):
    class Twice(executor.Executor):
        __slots__ = ()

        def run_timeout(self):
            found = super().run_timeout()
            return (*found, found[0])

    with pytest.raises(ValueError, match="expected"):
        Twice(tmp_path).execute(reliability.Contingency.TIMEOUT)
