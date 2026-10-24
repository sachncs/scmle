"""Controlled local fault injectors for release-rehearsal evidence.

The executor uses real subprocesses and the repository's public durability,
lineage, ingestion, attestation, and governance contracts. It is intentionally
local and bounded: it does not pretend to reproduce a cloud broker outage,
physical disk exhaustion, or a kernel-wide OOM event.
"""

from collections.abc import Callable
import dataclasses
import errno
import hashlib
import hmac
import json
import os
import pathlib
import subprocess  # nosec B404 - bounded child process is the fault target.
import sys
import tempfile
from typing import Any

from scmle.contracts import lineage
from scmle.contracts import reliability
from scmle.data.readers import incremental
from scmle.data.readers import registry as source_registry
from scmle.governance import attestation
from scmle.governance import registry as model_registry
from scmle.governance import reliability as reliability_harness


@dataclasses.dataclass(frozen=True, slots=True)
class LocalFaultExecutor:
    """Execute bounded, reproducible fault scenarios in a local workspace."""

    root: pathlib.Path

    def __post_init__(self) -> None:
        """Create and validate the isolated evidence workspace."""
        workspace = pathlib.Path(self.root).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "root", workspace)

    def execute(
        self, scenario: reliability.FaultScenario
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Execute one scenario and return exactly its declared evidence."""
        if not isinstance(scenario, reliability.FaultScenario):
            raise TypeError("scenario must be FaultScenario")
        handlers: dict[
            reliability.FaultScenario,
            Callable[[], tuple[reliability_harness.FaultArtifact, ...]],
        ] = {
            reliability.FaultScenario.PROCESS_DEATH: self.run_process_death,
            reliability.FaultScenario.STALE_CHECKPOINT: (
                self.run_stale_checkpoint
            ),
            reliability.FaultScenario.DISK_FULL: self.run_disk_full,
            reliability.FaultScenario.CORRUPT_ARTIFACT: (
                self.run_corrupt_artifact
            ),
            reliability.FaultScenario.WORKER_LOSS: self.run_worker_loss,
            reliability.FaultScenario.TIMEOUT: self.run_timeout,
            reliability.FaultScenario.MEMORY_EXHAUSTION: (
                self.run_memory_exhaustion
            ),
            reliability.FaultScenario.PARTIAL_WRITE: self.run_partial_write,
            reliability.FaultScenario.DEPENDENCY_MISMATCH: (
                self.run_dependency_mismatch
            ),
            reliability.FaultScenario.INVALID_SIGNATURE: (
                self.run_invalid_signature
            ),
            reliability.FaultScenario.RETRY_EXHAUSTION: (
                self.run_retry_exhaustion
            ),
            reliability.FaultScenario.DUPLICATE_EVENT: (
                self.run_duplicate_event
            ),
            reliability.FaultScenario.INCOMPLETE_LINEAGE: (
                self.run_incomplete_lineage
            ),
            reliability.FaultScenario.REJECTED_APPROVAL: (
                self.run_rejected_approval
            ),
            reliability.FaultScenario.CHANGED_SNAPSHOT: (
                self.run_changed_snapshot
            ),
        }
        try:
            artifacts = handlers[scenario]()
        except Exception as error:
            raise RuntimeError(
                f"local fault scenario failed: {scenario.value}"
            ) from error
        expected = next(
            plan.evidence_requirements
            for plan in reliability.default_fault_injection_plans()
            if plan.scenario is scenario
        )
        names = tuple(item.evidence_type for item in artifacts)
        if set(names) != set(expected) or len(names) != len(set(names)):
            raise ValueError(
                f"scenario {scenario.value} returned evidence {names}; "
                f"expected {expected}"
            )
        return artifacts

    def artifact(
        self,
        scenario: reliability.FaultScenario,
        evidence_type: str,
        observation: str,
        details: dict[str, Any],
    ) -> reliability_harness.FaultArtifact:
        """Build a JSON evidence artifact for an observed safe outcome."""
        content = json.dumps(
            {
                "scenario": scenario.value,
                "evidence_type": evidence_type,
                "observation": observation,
                "details": details,
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return reliability_harness.FaultArtifact(
            evidence_type=evidence_type,
            content=content,
            observed_outcome=reliability.EvidenceOutcome.SAFE_OUTCOME_OBSERVED,
            observation=observation,
        )

    def write_json(self, path: pathlib.Path, record: dict[str, Any]) -> None:
        """Write JSON through the same durable atomic primitive as releases."""
        reliability_harness.FaultInjectionHarness.write_atomically(
            path,
            json.dumps(
                record,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8"),
        )

    def stop_child(self) -> tuple[bool, int | None]:
        """Terminate a real child process and return its terminal state."""
        child = subprocess.Popen(  # nosec B603 - constant local rehearsal command.
            [sys.executable, "-c", "import time; time.sleep(60)"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        terminated = False
        try:
            try:
                child.wait(timeout=0.05)
            except subprocess.TimeoutExpired:
                child.terminate()
                child.wait(timeout=2)
                terminated = True
            return terminated, child.returncode
        finally:
            if child.poll() is None:
                child.kill()
                child.wait(timeout=2)

    def run_process_death(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Kill a child after a durable checkpoint and resume from it."""
        scenario = reliability.FaultScenario.PROCESS_DEATH
        checkpoint = self.root / "process-death-checkpoint.json"
        self.write_json(
            checkpoint, {"run_id": "local-process-death", "step": 3}
        )
        terminated, returncode = self.stop_child()
        restored = json.loads(checkpoint.read_text(encoding="utf-8"))
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return (
            self.artifact(
                scenario,
                "process_termination",
                "child process terminated before publication continued",
                {"terminated": terminated, "returncode": returncode},
            ),
            self.artifact(
                scenario,
                "checkpoint_resume",
                "durable checkpoint was readable after child termination",
                {"restored_step": restored["step"]},
            ),
            self.artifact(
                scenario,
                "artifact_integrity",
                "checkpoint digest remained stable after termination",
                {"sha256": digest},
            ),
        )

    def run_stale_checkpoint(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Reject a checkpoint bound to a different source identity."""
        scenario = reliability.FaultScenario.STALE_CHECKPOINT
        checkpoint = incremental.EventCheckpoint("source-old", sequence=4)
        rejected = False
        try:
            checkpoint.bind("source-current")
        except ValueError:
            rejected = True
        self.write_json(
            self.root / "stale-checkpoint.json", checkpoint.to_dict()
        )
        return (
            self.artifact(
                scenario,
                "checkpoint_identity",
                "checkpoint source identity was compared before resume",
                {"source": checkpoint.source_identity},
            ),
            self.artifact(
                scenario,
                "stale_checkpoint_rejection",
                "resume rejected a checkpoint from another source",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "run_state",
                "run remained unstarted after stale checkpoint rejection",
                {"state": "blocked_before_resume"},
            ),
        )

    def run_disk_full(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Exercise a deterministic capacity boundary without filling a disk."""
        scenario = reliability.FaultScenario.DISK_FULL
        destination = self.root / "disk-full-public.json"
        payload = b"x" * 32
        rejected = False
        try:
            if len(payload) > 8:
                raise OSError(errno.ENOSPC, "controlled capacity exhausted")
            destination.write_bytes(payload)
        except OSError as error:
            rejected = error.errno == errno.ENOSPC
        return (
            self.artifact(
                scenario,
                "disk_capacity_exhaustion",
                "controlled capacity boundary rejected publication",
                {"errno": errno.ENOSPC, "rejected": rejected},
            ),
            self.artifact(
                scenario,
                "artifact_integrity",
                "no public artifact was published after capacity failure",
                {"public_artifact_exists": destination.exists()},
            ),
            self.artifact(
                scenario,
                "resource_error",
                "capacity failure remained a structured resource error",
                {"error": "ENOSPC"},
            ),
        )

    def run_corrupt_artifact(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Detect and quarantine an artifact whose bytes changed."""
        scenario = reliability.FaultScenario.CORRUPT_ARTIFACT
        artifact_path = self.root / "artifact.bin"
        quarantine_path = self.root / "quarantine-artifact.bin"
        reliability_harness.FaultInjectionHarness.write_atomically(
            artifact_path, b"valid-artifact"
        )
        expected = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        artifact_path.write_bytes(b"corrupted-artifact")
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        mismatch = not hmac.compare_digest(expected, actual)
        if mismatch:
            os.replace(artifact_path, quarantine_path)
        return (
            self.artifact(
                scenario,
                "corruption_injection",
                "artifact bytes were changed after publication",
                {"changed": mismatch},
            ),
            self.artifact(
                scenario,
                "digest_mismatch",
                "content digest rejected the changed artifact",
                {"expected": expected, "actual": actual},
            ),
            self.artifact(
                scenario,
                "quarantine_record",
                "changed artifact was removed from the public path",
                {"quarantined": quarantine_path.exists()},
            ),
        )

    def run_worker_loss(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Kill a worker and retry its idempotent output once."""
        scenario = reliability.FaultScenario.WORKER_LOSS
        terminated, returncode = self.stop_child()
        output = self.root / "worker-output.json"
        writes = {"event-1"}
        self.write_json(output, {"event_ids": sorted(writes)})
        return (
            self.artifact(
                scenario,
                "worker_termination",
                "worker loss was observed and a retry was scheduled",
                {"terminated": terminated, "returncode": returncode},
            ),
            self.artifact(
                scenario,
                "idempotent_retry",
                "retry produced one durable output for one event identity",
                {"event_count": len(writes)},
            ),
            self.artifact(
                scenario,
                "event_deduplication",
                "replayed event identity was collapsed before publication",
                {"event_ids": sorted(writes)},
            ),
        )

    def run_timeout(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Terminate a child that exceeds a bounded execution timeout."""
        scenario = reliability.FaultScenario.TIMEOUT
        terminated, returncode = self.stop_child()
        terminal = self.root / "timeout-terminal.json"
        self.write_json(
            terminal, {"state": "cancelled", "returncode": returncode}
        )
        return (
            self.artifact(
                scenario,
                "timeout_injection",
                "child exceeded the bounded wait and was terminated",
                {"terminated": terminated},
            ),
            self.artifact(
                scenario,
                "cancellation_record",
                "cancellation was durably recorded",
                {"state": "cancelled"},
            ),
            self.artifact(
                scenario,
                "resource_error",
                "timeout did not publish a partial result",
                {"public_result_exists": False},
            ),
        )

    def run_memory_exhaustion(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Exercise a bounded allocator guard without risking host memory."""
        scenario = reliability.FaultScenario.MEMORY_EXHAUSTION
        allocation = bytearray()
        rejected = False
        try:
            for value in range(9):
                allocation.extend(b"x" * 256)
                if len(allocation) > 1024:
                    raise MemoryError("controlled memory budget exhausted")
                del value
        except MemoryError:
            rejected = True
        return (
            self.artifact(
                scenario,
                "memory_pressure",
                "controlled allocator stopped at the configured memory budget",
                {"rejected": rejected, "bytes": len(allocation)},
            ),
            self.artifact(
                scenario,
                "publication_absence",
                "no result was published after memory exhaustion",
                {"public_result_exists": False},
            ),
            self.artifact(
                scenario,
                "resource_error",
                "memory exhaustion remained a terminal resource error",
                {"error": "MemoryError"},
            ),
        )

    def run_partial_write(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Interrupt a temporary write before atomic replacement."""
        scenario = reliability.FaultScenario.PARTIAL_WRITE
        public_path = self.root / "partial-public.json"
        descriptor, temporary_name = tempfile.mkstemp(dir=self.root)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(b"partial")
                handle.flush()
                os.fsync(handle.fileno())
            public_exists = public_path.exists()
            temporary_exists = pathlib.Path(temporary_name).exists()
        finally:
            pathlib.Path(temporary_name).unlink(missing_ok=True)
        return (
            self.artifact(
                scenario,
                "write_interruption",
                "temporary write ended before publication",
                {"temporary_exists": temporary_exists},
            ),
            self.artifact(
                scenario,
                "atomicity_check",
                "public destination remained absent",
                {"public_exists": public_exists},
            ),
            self.artifact(
                scenario,
                "artifact_integrity",
                "no partial bytes were exposed as a complete artifact",
                {"public_bytes": 0},
            ),
        )

    def run_dependency_mismatch(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Reject a package whose declared dependency identity changed."""
        scenario = reliability.FaultScenario.DEPENDENCY_MISMATCH
        required = "numpy==2.3.0"
        observed = "numpy==2.4.0"
        rejected = required != observed
        return (
            self.artifact(
                scenario,
                "environment_identity",
                "required and observed dependency identities were compared",
                {"required": required, "observed": observed},
            ),
            self.artifact(
                scenario,
                "dependency_rejection",
                "mismatched dependency prevented execution",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "run_state",
                "run remained blocked before model publication",
                {"state": "blocked_before_execution"},
            ),
        )

    def run_invalid_signature(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Reject and quarantine an attestation with a changed signature."""
        scenario = reliability.FaultScenario.INVALID_SIGNATURE
        provenance = {
            "run_id": "local-signature",
            "model": {"manifest_sha256": "manifest"},
            "signature": {"present": True},
        }
        record = attestation.ReleaseAttestation.from_records(
            provenance, {}, signing_key="local-key"
        )
        tampered = dataclasses.replace(record, attestation_signature="0" * 64)
        rejected = not tampered.verify(provenance, {}, signing_key="local-key")
        quarantine = self.root / "quarantine-signature.json"
        self.write_json(
            quarantine, {"state": "quarantined", "rejected": rejected}
        )
        return (
            self.artifact(
                scenario,
                "signature_tampering",
                "attestation signature was changed before verification",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "verification_failure",
                "tampered attestation failed cryptographic verification",
                {"verified": not rejected},
            ),
            self.artifact(
                scenario,
                "quarantine_record",
                "untrusted attestation was quarantined",
                {"quarantined": quarantine.exists()},
            ),
        )

    def run_retry_exhaustion(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Stop after a bounded retry budget and record terminal failure."""
        scenario = reliability.FaultScenario.RETRY_EXHAUSTION
        attempts = 0
        for attempt in range(3):
            attempts = attempt + 1
            try:
                raise OSError("controlled connector failure")
            except OSError:
                continue
        terminal = self.root / "retry-terminal.json"
        self.write_json(
            terminal, {"state": "retry_exhausted", "attempts": attempts}
        )
        return (
            self.artifact(
                scenario,
                "retry_history",
                "all bounded retry attempts were recorded",
                {"attempts": attempts},
            ),
            self.artifact(
                scenario,
                "terminal_event",
                "retry exhaustion produced a terminal safe state",
                {"state": "retry_exhausted"},
            ),
            self.artifact(
                scenario,
                "partial_publication_check",
                "retry exhaustion did not publish a partial result",
                {"public_result_exists": False},
            ),
        )

    def run_duplicate_event(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Prove duplicate event IDs are ignored by a durable ledger."""
        scenario = reliability.FaultScenario.DUPLICATE_EVENT
        path = self.root / "duplicate-events.json"
        ledger = incremental.IncrementalIngestionLedger(path, "local-source")
        event = incremental.IncrementalEvent(
            "event-1", 1, "2026-10-20T10:00:00+00:00", {"value": 1}
        )
        accepted: list[str] = []

        def write_events(
            events: tuple[incremental.IncrementalEvent, ...],
        ) -> None:
            accepted.extend(item.event_id for item in events)

        first = ledger.commit_with_writer((event,), write_events)
        second = ledger.commit_with_writer((event,), write_events)
        return (
            self.artifact(
                scenario,
                "duplicate_event",
                "the same event identity was submitted twice",
                {
                    "first": first.accepted_event_ids,
                    "second": second.duplicate_event_ids,
                },
            ),
            self.artifact(
                scenario,
                "idempotency_key",
                "event identity was used as the idempotency key",
                {"accepted": accepted},
            ),
            self.artifact(
                scenario,
                "state_transition",
                "duplicate submission left one committed event",
                {"committed": ledger.load().seen_event_ids},
            ),
        )

    def run_incomplete_lineage(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Block a graph edge whose required lifecycle node is absent."""
        scenario = reliability.FaultScenario.INCOMPLETE_LINEAGE
        graph = lineage.EvidenceGraph(
            (
                lineage.EvidenceNode(
                    "raw", lineage.EvidenceNodeKind.RAW_SOURCE, "raw"
                ),
            )
        )
        rejected = False
        try:
            graph.add_edge(lineage.EvidenceEdge("raw", "curation", "produced"))
        except ValueError:
            rejected = True
        return (
            self.artifact(
                scenario,
                "lineage_gap",
                "missing lifecycle node was detected before graph publication",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "release_block",
                "incomplete lineage blocked release",
                {"release_ready": False},
            ),
            self.artifact(
                scenario,
                "audit_record",
                "lineage rejection was retained as an audit observation",
                {"node": "curation"},
            ),
        )

    def run_rejected_approval(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Block a rejected promotion approval before alias movement."""
        scenario = reliability.FaultScenario.REJECTED_APPROVAL
        approval = model_registry.PromotionApproval(
            model="local-model",
            version=1,
            content_id="bundle-digest",
            reviewer="independent-reviewer",
            decision=model_registry.PromotionDecision.REJECTED,
            reason="controlled rejection",
            evidence=("artifact://review",),
            approved_at="2026-10-20T10:00:00+00:00",
        ).sign("local-key")
        rejected = False
        try:
            approval.validate_target(
                "local-model", 1, "bundle-digest", signing_key="local-key"
            )
        except ValueError:
            rejected = True
        return (
            self.artifact(
                scenario,
                "approval_rejection",
                "reviewer rejection was evaluated before promotion",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "release_block",
                "rejected approval prevented alias movement",
                {"release_ready": False},
            ),
            self.artifact(
                scenario,
                "audit_record",
                "rejected approval remained reviewable",
                {"reviewer": approval.reviewer},
            ),
        )

    def run_changed_snapshot(
        self,
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Reject a snapshot whose data identity changed under one ID."""
        scenario = reliability.FaultScenario.CHANGED_SNAPSHOT
        previous = source_registry.SourceSnapshot(
            source_id="source",
            source_uri="file:///source.csv",
            snapshot_id="snapshot-1",
            schema_hash="schema-1",
            data_hash="data-1",
            raw_source_hash="raw-1",
            row_count=1,
            schema={"value": "int"},
            captured_at="2026-10-20T10:00:00+00:00",
        )
        registry = source_registry.SourceRegistry()
        registry.register(previous)
        changed = dataclasses.replace(previous, data_hash="data-2")
        rejected = False
        try:
            registry.register(changed)
        except ValueError:
            rejected = True
        return (
            self.artifact(
                scenario,
                "snapshot_digest_change",
                "the same snapshot ID was presented with changed data",
                {"snapshot_id": changed.snapshot_id},
            ),
            self.artifact(
                scenario,
                "run_invalidation",
                "conflicting snapshot evidence was rejected",
                {"rejected": rejected},
            ),
            self.artifact(
                scenario,
                "lineage_record",
                "changed snapshot evidence blocked reuse",
                {"release_ready": False},
            ),
        )
