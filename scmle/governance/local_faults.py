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
        """Creates and validates the isolated evidence workspace."""
        workspace = pathlib.Path(self.root).expanduser().resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        object.__setattr__(self, "root", workspace)

    def execute(
        self, scenario: reliability.FaultScenario
    ) -> tuple[reliability_harness.FaultArtifact, ...]:
        """Executes one scenario and checks its evidence against its plan.

        Args:
            scenario: The fault scenario to rehearse.

        Returns:
            The artifacts the scenario's handler produced, in its order;
            their evidence types are exactly those its default plan requires,
            each once.

        Raises:
            RuntimeError: If the scenario's handler raises any exception.
            TypeError: If `scenario` is not a `FaultScenario`.
            ValueError: If the evidence types produced differ from those the
                plan requires or repeat.
        """
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
        """Builds an evidence artifact that records a safe outcome as observed.

        The outcome is always marked as observed; nothing here checks that
        `details` supports it.

        Args:
            scenario: The scenario the evidence belongs to.
            evidence_type: The kind of evidence.
            observation: What was observed.
            details: Facts recorded with the observation; they must be JSON
                serializable.

        Returns:
            The artifact: compact, key-sorted ASCII JSON bytes holding the
            scenario value, evidence type, observation and details, with the
            safe-outcome-observed outcome and the observation text.
        """
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
        """Writes JSON with the evidence harness's atomic, synced write.

        Args:
            path: The destination file.
            record: The content, written as compact sorted JSON.
        """
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
        """Starts a sleeping child process, terminates it and reports how.

        The child is a Python interpreter that sleeps for 60 seconds. It is
        given 0.05 seconds to exit on its own, then terminated and waited for
        up to 2 seconds. A child that outlives that is killed and
        `subprocess.TimeoutExpired` propagates.

        Returns:
            A pair: whether the child had to be terminated, and its exit code
            (negative signal number on POSIX after a termination).
        """
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
        """Writes a checkpoint, terminates a child and reads it back.

        The child is unrelated to the checkpoint. The checkpoint
        ``process-death-checkpoint.json`` is written under the root before
        the child is stopped and read back afterwards.

        Returns:
            The three evidence artifacts, in order: process termination,
            checkpoint resume (the step read back) and artifact integrity
            (the checkpoint's SHA-256).
        """
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
        """Checks that a checkpoint cannot be bound to a different source.

        Binds a checkpoint of one source to another source identity and
        records whether `ValueError` was raised; the checkpoint is also
        written to ``stale-checkpoint.json`` under the root. The run-state
        artifact is a fixed statement.

        Returns:
            The three evidence artifacts, in order: checkpoint identity,
            stale-checkpoint rejection and run state.
        """
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
        """Simulates a full disk without filling one.

        A 32-byte payload exceeds a fixed 8-byte budget, so `OSError` with
        ENOSPC is raised before anything is written, and the method records
        that no file exists at the destination.

        Returns:
            The three evidence artifacts, in order: disk capacity
            exhaustion, artifact integrity and resource error.

        Raises:
            OSError: The simulated ENOSPC; it is caught inside and never
                reaches the caller.
        """
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
        """Corrupts a written artifact, detects it by digest and quarantines it.

        Writes ``artifact.bin`` under the root, records its SHA-256,
        overwrites it with other bytes, compares the digests and, on a
        mismatch, renames it to ``quarantine-artifact.bin``.

        Returns:
            The three evidence artifacts, in order: corruption injection,
            digest mismatch (expected and actual digests) and quarantine
            record.
        """
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
        """Terminates a child as a stand-in worker and writes one event output.

        No retry or replay is performed: a single output file
        ``worker-output.json`` with one event id is written under the root,
        and the evidence reports that one event.

        Returns:
            The three evidence artifacts, in order: worker termination,
            idempotent retry and event deduplication.
        """
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
        """Terminates a child that outlives a short wait and records it.

        The wait is the 0.05 seconds of `stop_child`; the cancellation is
        written to ``timeout-terminal.json`` under the root. The resource
        error artifact is a fixed statement that nothing was published.

        Returns:
            The three evidence artifacts, in order: timeout injection,
            cancellation record and resource error.
        """
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
        """Simulates memory exhaustion with a tiny allocation budget.

        A bytearray grows by 256 bytes at a time until it passes 1024 bytes,
        at which point `MemoryError` is raised by the method itself.

        Returns:
            The three evidence artifacts, in order: memory pressure (whether
            it was rejected and the bytes held), publication absence and
            resource error.

        Raises:
            MemoryError: The simulated exhaustion; it is caught inside and
                never reaches the caller.
        """
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
        """Writes a temporary file and never replaces the public path.

        A temporary file in the root receives and syncs a few bytes; the
        public destination is checked to be absent, and the temporary file is
        then deleted. Nothing is interrupted and no replacement is attempted.

        Returns:
            The three evidence artifacts, in order: write interruption
            (whether the temporary file existed), atomicity check (whether
            the public path existed) and artifact integrity.
        """
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
        """Compares two fixed dependency identities and records the mismatch.

        The identities are the constants ``numpy==2.3.0`` and
        ``numpy==2.4.0``; no installed package is inspected.

        Returns:
            The three evidence artifacts, in order: environment identity,
            dependency rejection and run state.
        """
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
        """Tampers with an attestation signature and records the failed check.

        Builds a signed release attestation, replaces its signature with
        zeros, verifies it with the same key, and writes a quarantine record
        ``quarantine-signature.json`` under the root.

        Returns:
            The three evidence artifacts, in order: signature tampering,
            verification failure and quarantine record.
        """
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
        """Simulates three failing attempts and records the terminal state.

        Each of three attempts raises `OSError` by itself and is counted;
        the terminal state is then written to ``retry-terminal.json`` under
        the root.

        Returns:
            The three evidence artifacts, in order: retry history (the
            attempt count), terminal event and partial-publication check.

        Raises:
            OSError: The simulated connector failure of each attempt; it is
                caught inside and never reaches the caller.
        """
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
        """Commits one event twice to an ingestion ledger and records both.

        Uses an incremental ingestion ledger stored in
        ``duplicate-events.json`` under the root. The evidence reports the
        accepted and duplicate identities of the two commits, the events the
        writer received, and the identities the ledger holds afterwards.

        Returns:
            The three evidence artifacts, in order: duplicate event,
            idempotency key and state transition.
        """
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
        """Adds a lineage edge to a missing node and records the rejection.

        The graph holds one raw-source node; an edge to an absent node is
        added and `ValueError` is expected. The release-block and audit
        artifacts are fixed statements.

        Returns:
            The three evidence artifacts, in order: lineage gap, release
            block and audit record.
        """
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
        """Validates a signed rejected approval and records the refusal.

        Builds and signs a reviewer approval whose decision is rejected and
        checks it against its own target; `ValueError` is expected. No
        registry or alias is involved.

        Returns:
            The three evidence artifacts, in order: approval rejection,
            release block and audit record.
        """
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
        """Registers a snapshot, then one with the same id and other data.

        The second registration in an in-memory source registry is expected
        to raise `ValueError`; the outcome is recorded. The lineage artifact
        is a fixed statement.

        Returns:
            The three evidence artifacts, in order: snapshot digest change,
            run invalidation and lineage record.
        """
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
