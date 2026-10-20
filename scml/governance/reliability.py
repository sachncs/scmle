"""Artifact-backed execution boundary for reliability fault evidence."""

from collections.abc import Callable
import contextlib
import dataclasses
import hashlib
import os
import pathlib
import tempfile

from scml import durability
from scml.contracts import errors
from scml.contracts import reliability


@dataclasses.dataclass(frozen=True, slots=True)
class FaultArtifact:
    """One artifact emitted by an injected reliability scenario.

    The harness does not inject process death, disk pressure, or worker loss
    itself. The caller supplies the scenario-specific executor. This boundary
    makes the executor's output durable, content-addressed, and compatible
    with the release-blocking reliability registry.
    """

    evidence_type: str
    content: bytes
    observed_outcome: reliability.EvidenceOutcome
    observation: str

    def __post_init__(self) -> None:
        """Reject evidence that cannot be persisted and reviewed."""
        if not self.evidence_type.strip():
            raise ValueError("evidence_type must not be empty")
        if "/" in self.evidence_type or "\\" in self.evidence_type:
            raise ValueError("evidence_type must be a path-safe name")
        if not isinstance(self.content, bytes) or not self.content:
            raise ValueError("content must be non-empty bytes")
        if not isinstance(self.observed_outcome, reliability.EvidenceOutcome):
            raise TypeError("observed_outcome must be EvidenceOutcome")
        if not self.observation.strip():
            raise ValueError("observation must not be empty")


class FaultInjectionHarness:
    """Persist executor output and update one reliability plan scenario."""

    def __init__(
        self,
        registry: reliability.FaultInjectionPlanRegistry,
        artifact_root: str | pathlib.Path,
        execution_id: str,
        reviewer: str,
        recorded_at: str,
    ) -> None:
        """Initialize a harness with explicit evidence identity."""
        if not isinstance(registry, reliability.FaultInjectionPlanRegistry):
            raise TypeError("registry must be FaultInjectionPlanRegistry")
        for name, value in (
            ("execution_id", execution_id),
            ("reviewer", reviewer),
            ("recorded_at", recorded_at),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if "/" in execution_id or "\\" in execution_id:
            raise ValueError("execution_id must be a path-safe name")
        self.registry = registry
        self.artifact_root = pathlib.Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.execution_id = execution_id
        self.reviewer = reviewer
        self.recorded_at = recorded_at

    def run(
        self,
        scenario: reliability.FaultScenario,
        executor: Callable[
            [reliability.FaultScenario], tuple[FaultArtifact, ...]
        ],
    ) -> reliability.FaultScenarioPlan:
        """Run an injected executor and record durable evidence.

        The executor must return one artifact for each evidence requirement
        declared by the scenario. Missing or unexpected evidence remains
        ``INCONCLUSIVE`` or raises from the registry; it never becomes a
        passing release claim by omission.
        """
        if not isinstance(scenario, reliability.FaultScenario):
            raise TypeError("scenario must be FaultScenario")
        try:
            artifacts = executor(scenario)
        except Exception as error:
            raise errors.ReliabilityError(
                "fault executor failed before evidence was recorded",
                {"scenario": scenario.value, "execution_id": self.execution_id},
            ) from error
        if not isinstance(artifacts, tuple) or not all(
            isinstance(item, FaultArtifact) for item in artifacts
        ):
            raise TypeError("executor must return FaultArtifact values")
        evidence: list[reliability.FaultEvidence] = []
        for artifact in artifacts:
            path = self.artifact_root / (
                f"{self.execution_id}-{scenario.value}-{artifact.evidence_type}.bin"
            )
            self.write_atomically(path, artifact.content)
            evidence.append(
                reliability.FaultEvidence(
                    evidence_type=artifact.evidence_type,
                    execution_id=self.execution_id,
                    artifact_uri=path.as_uri(),
                    artifact_digest=self.digest(path),
                    observed_outcome=artifact.observed_outcome,
                    recorded_at=self.recorded_at,
                    reviewer=self.reviewer,
                    observation=artifact.observation,
                )
            )
        return self.registry.record_evidence(scenario, tuple(evidence))

    @staticmethod
    def digest(path: str | pathlib.Path) -> str:
        """Return the SHA-256 digest of a persisted evidence artifact."""
        digest = hashlib.sha256()
        with pathlib.Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def write_atomically(path: pathlib.Path, content: bytes) -> None:
        """Write an evidence artifact using flush, fsync, and replace."""
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, path)
            durability.sync_directory(path.parent)
        except Exception:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(temporary_name)
            raise
