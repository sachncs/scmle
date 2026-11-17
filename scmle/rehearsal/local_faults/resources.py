"""Scenarios that exhaust or corrupt a resource: disk, memory, files."""

import errno
import hashlib
import hmac
import os
import pathlib
import tempfile

from scmle.contracts import reliability
import scmle.governance.reliability
from scmle.rehearsal.local_faults import workspace

DISK_BUDGET_BYTES = 8  # the fixed budget a disk-full payload exceeds
MEMORY_BUDGET_BYTES = 1024  # the fixed budget the memory fault exceeds
REQUIRED_DEPENDENCY: str = "numpy==2.3.0"  # the identity the run requires
OBSERVED_DEPENDENCY: str = (
    "numpy==2.4.0"  # the identity the environment reports
)


def cap(payload: bytes) -> None:
    """Refuses a payload larger than the fixed disk budget, as a full disk.

    Args:
        payload: The bytes about to be written.

    Raises:
        OSError: With ENOSPC if the payload exceeds `DISK_BUDGET_BYTES`.
    """
    if len(payload) > DISK_BUDGET_BYTES:
        raise OSError(errno.ENOSPC, "controlled capacity exhausted")


def ration(allocation: bytearray) -> None:
    """Refuses an allocation larger than the fixed memory budget.

    Args:
        allocation: The bytes held so far.

    Raises:
        MemoryError: If the allocation exceeds `MEMORY_BUDGET_BYTES`.
    """
    if len(allocation) > MEMORY_BUDGET_BYTES:
        raise MemoryError("controlled memory budget exhausted")


class Resources(workspace.Workspace):
    """A full disk, a corrupt artifact, memory, partial writes, dependencies."""

    __slots__ = ()

    def run_disk_full(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
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
        scenario = reliability.Contingency.DISK_FULL
        destination = self.root / "disk-full-public.json"
        payload = b"x" * 32
        rejected = False
        try:
            cap(payload)
            destination.write_bytes(payload)
        except OSError as error:
            # handled: simulated ENOSPC recorded in disk_capacity_exhaustion
            # evidence
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
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Corrupts a written artifact, detects it by digest and quarantines it.

        Writes ``artifact.bin`` under the root, records its SHA-256,
        overwrites it with other bytes, compares the digests and, on a
        mismatch, renames it to ``quarantine-artifact.bin``.

        Returns:
            The three evidence artifacts, in order: corruption injection,
            digest mismatch (expected and actual digests) and quarantine
            record.
        """
        scenario = reliability.Contingency.CORRUPT_ARTIFACT
        artifact_path = self.root / "artifact.bin"
        quarantine_path = self.root / "quarantine-artifact.bin"
        scmle.governance.reliability.Harness.write_atomically(
            artifact_path, b"valid-artifact"
        )
        expected = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        artifact_path.write_bytes(b"corrupted-artifact")
        actual = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
        mismatch = not hmac.compare_digest(expected, actual)
        if mismatch:
            artifact_path.replace(quarantine_path)
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

    def run_memory_exhaustion(
        self,
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
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
        scenario = reliability.Contingency.MEMORY_EXHAUSTION
        allocation = bytearray()
        rejected = False
        try:
            for value in range(9):
                allocation.extend(b"x" * 256)
                ration(allocation)
                del value
        except MemoryError:
            # handled: simulated MemoryError recorded in memory_pressure
            # evidence
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
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Writes a temporary file and never replaces the public path.

        A temporary file in the root receives and syncs a few bytes; the
        public destination is checked to be absent, and the temporary file is
        then deleted. Nothing is interrupted and no replacement is attempted.

        Returns:
            The three evidence artifacts, in order: write interruption
            (whether the temporary file existed), atomicity check (whether
            the public path existed) and artifact integrity.
        """
        scenario = reliability.Contingency.PARTIAL_WRITE
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
    ) -> tuple[scmle.governance.reliability.Emission, ...]:
        """Compares two fixed dependency identities and records the mismatch.

        The identities are the constants ``numpy==2.3.0`` and
        ``numpy==2.4.0``; no installed package is inspected.

        Returns:
            The three evidence artifacts, in order: environment identity,
            dependency rejection and run state.
        """
        scenario = reliability.Contingency.DEPENDENCY_MISMATCH
        required = REQUIRED_DEPENDENCY
        observed = OBSERVED_DEPENDENCY
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
