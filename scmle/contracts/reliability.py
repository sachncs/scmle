"""Evidence-first contracts for production reliability fault scenarios."""

from collections.abc import Mapping
import dataclasses
import enum
import json
from typing import Any
from typing import Self


class FaultScenario(enum.StrEnum):
    """Fault condition that must have a safe, reviewable response."""

    PROCESS_DEATH = "process_death"
    STALE_CHECKPOINT = "stale_checkpoint"
    DISK_FULL = "disk_full"
    CORRUPT_ARTIFACT = "corrupt_artifact"
    WORKER_LOSS = "worker_loss"
    TIMEOUT = "timeout"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    PARTIAL_WRITE = "partial_write"
    DEPENDENCY_MISMATCH = "dependency_mismatch"
    INVALID_SIGNATURE = "invalid_signature"
    RETRY_EXHAUSTION = "retry_exhaustion"
    DUPLICATE_EVENT = "duplicate_event"
    INCOMPLETE_LINEAGE = "incomplete_lineage"
    REJECTED_APPROVAL = "rejected_approval"
    CHANGED_SNAPSHOT = "changed_snapshot"


class SafeOutcome(enum.StrEnum):
    """Safe outcome expected after a fault is injected."""

    RESUME_DURABLE_CHECKPOINT = "resume_durable_checkpoint"
    REJECT_STALE_CHECKPOINT = "reject_stale_checkpoint"
    PRESERVE_DURABLE_ARTIFACT = "preserve_durable_artifact"
    QUARANTINE_CORRUPT_ARTIFACT = "quarantine_corrupt_artifact"
    RETRY_IDEMPOTENT_WORK = "retry_idempotent_work"
    CANCEL_AND_RECORD_TERMINAL_FAILURE = "cancel_and_record_terminal_failure"
    ABORT_BEFORE_PUBLICATION = "abort_before_publication"
    REJECT_INCOMPATIBLE_DEPENDENCY = "reject_incompatible_dependency"
    REJECT_UNTRUSTED_ARTIFACT = "reject_untrusted_artifact"
    TERMINATE_AFTER_RETRY_EXHAUSTION = "terminate_after_retry_exhaustion"
    IGNORE_DUPLICATE_EVENT = "ignore_duplicate_event"
    BLOCK_RELEASE_FOR_MISSING_LINEAGE = "block_release_for_missing_lineage"
    BLOCK_RELEASE_FOR_REJECTED_APPROVAL = "block_release_for_rejected_approval"
    INVALIDATE_CHANGED_SNAPSHOT = "invalidate_changed_snapshot"


class Recoverability(enum.StrEnum):
    """Level of recovery promised by a reliability scenario."""

    AUTOMATIC = "automatic"
    OPERATOR_ASSISTED = "operator_assisted"
    TERMINAL_SAFE_FAILURE = "terminal_safe_failure"


class VerificationStatus(enum.StrEnum):
    """Evidence state of one fault scenario."""

    PLANNED = "planned"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class EvidenceOutcome(enum.StrEnum):
    """Observed result recorded by a fault-injection execution."""

    SAFE_OUTCOME_OBSERVED = "safe_outcome_observed"
    UNSAFE_OUTCOME_OBSERVED = "unsafe_outcome_observed"
    INCONCLUSIVE = "inconclusive"


@dataclasses.dataclass(frozen=True, slots=True)
class FaultEvidence:
    """Evidence produced by an actual fault-injection execution.

    The contract requires an execution identity and a content-addressed
    artifact reference. A plan can be declared before execution, but callers
    cannot represent a verification result without this evidence envelope.

    Attributes:
        evidence_type: Stable name of the evidence item, such as
            ``checkpoint_resume`` or ``artifact_digest``.
        execution_id: Identity of the run that performed the injection.
        artifact_uri: URI of the durable evidence artifact.
        artifact_digest: Digest of the evidence artifact contents.
        observed_outcome: Result observed by the execution.
        recorded_at: Timestamp supplied by the execution recorder.
        reviewer: Human or automated reviewer that verified the evidence.
        observation: Concrete observation captured in the evidence record.
    """

    evidence_type: str
    execution_id: str
    artifact_uri: str
    artifact_digest: str
    observed_outcome: EvidenceOutcome
    recorded_at: str
    reviewer: str
    observation: str

    def __post_init__(self) -> None:
        """Reject evidence records that cannot be independently reviewed.

        Raises:
            TypeError: If `observed_outcome` is not an `EvidenceOutcome`.
            ValueError: If a required text field is blank.
        """
        for name in (
            "evidence_type",
            "execution_id",
            "artifact_uri",
            "artifact_digest",
            "recorded_at",
            "reviewer",
            "observation",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty")
        if not isinstance(self.observed_outcome, EvidenceOutcome):
            raise TypeError("observed_outcome must be EvidenceOutcome")

    def to_dict(self) -> dict[str, str]:
        """Return a deterministic JSON-compatible evidence record."""
        return {
            "evidence_type": self.evidence_type,
            "execution_id": self.execution_id,
            "artifact_uri": self.artifact_uri,
            "artifact_digest": self.artifact_digest,
            "observed_outcome": self.observed_outcome.value,
            "recorded_at": self.recorded_at,
            "reviewer": self.reviewer,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild evidence from its JSON-compatible representation.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The evidence.
        """
        return cls(
            evidence_type=str(record["evidence_type"]),
            execution_id=str(record["execution_id"]),
            artifact_uri=str(record["artifact_uri"]),
            artifact_digest=str(record["artifact_digest"]),
            observed_outcome=EvidenceOutcome(record["observed_outcome"]),
            recorded_at=str(record["recorded_at"]),
            reviewer=str(record["reviewer"]),
            observation=str(record["observation"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FaultScenarioPlan:
    """Declaration and verification state for one fault scenario.

    ``PLANNED`` is the only status permitted without execution evidence.
    Every other status requires evidence, and ``PASSED`` additionally requires
    safe evidence for every declared evidence requirement.

    Attributes:
        scenario: Fault condition being tested.
        expected_safe_outcome: Safe behavior required by the release policy.
        recoverability: Recovery promise made to operators and users.
        evidence_requirements: Evidence types required for a passing result.
        release_blocking: Whether an unsafe or unverified result blocks release.
        verification_status: Current evidence state.
        evidence: Evidence records from actual executions.
    """

    scenario: FaultScenario
    expected_safe_outcome: SafeOutcome
    recoverability: Recoverability
    evidence_requirements: tuple[str, ...]
    release_blocking: bool
    verification_status: VerificationStatus = VerificationStatus.PLANNED
    evidence: tuple[FaultEvidence, ...] = ()

    def __post_init__(self) -> None:
        """Validate declaration completeness and evidence/status consistency.

        Raises:
            TypeError: If an enumerated field has the wrong type or
                `release_blocking` is not a bool.
            ValueError: If the evidence requirements are empty, blank or
                repeated, evidence types repeat, a status other than planned has
                no evidence, or a passed scenario lacks a required evidence type
                or has evidence of an unsafe outcome.
        """
        enum_fields = (
            ("scenario", self.scenario, FaultScenario),
            ("expected_safe_outcome", self.expected_safe_outcome, SafeOutcome),
            ("recoverability", self.recoverability, Recoverability),
            (
                "verification_status",
                self.verification_status,
                VerificationStatus,
            ),
        )
        for name, value, expected in enum_fields:
            if not isinstance(value, expected):
                raise TypeError(f"{name} must be {expected.__name__}")
        if not self.evidence_requirements:
            raise ValueError("evidence_requirements must not be empty")
        if any(
            not isinstance(requirement, str) or not requirement.strip()
            for requirement in self.evidence_requirements
        ):
            raise ValueError("evidence_requirements must contain text")
        if len(set(self.evidence_requirements)) != len(
            self.evidence_requirements
        ):
            raise ValueError("evidence_requirements must be unique")
        if not isinstance(self.release_blocking, bool):
            raise TypeError("release_blocking must be bool")
        evidence_types = [item.evidence_type for item in self.evidence]
        if len(set(evidence_types)) != len(evidence_types):
            raise ValueError("evidence types must be unique per scenario")
        if (
            self.verification_status is not VerificationStatus.PLANNED
            and not self.evidence
        ):
            raise ValueError("verification status requires execution evidence")
        if self.verification_status is VerificationStatus.PASSED:
            missing = set(self.evidence_requirements) - set(evidence_types)
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise ValueError(
                    "passed scenario is missing evidence requirements: "
                    f"{missing_text}"
                )
            if any(
                item.observed_outcome
                is not EvidenceOutcome.SAFE_OUTCOME_OBSERVED
                for item in self.evidence
            ):
                raise ValueError(
                    "passed scenario requires safe outcome evidence only"
                )

    def to_dict(self) -> dict[str, Any]:
        """Return a deterministic JSON-compatible scenario record."""
        return {
            "scenario": self.scenario.value,
            "expected_safe_outcome": self.expected_safe_outcome.value,
            "recoverability": self.recoverability.value,
            "evidence_requirements": list(self.evidence_requirements),
            "release_blocking": self.release_blocking,
            "verification_status": self.verification_status.value,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild a scenario plan from its JSON-compatible representation.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The scenario plan.
        """
        return cls(
            scenario=FaultScenario(record["scenario"]),
            expected_safe_outcome=SafeOutcome(record["expected_safe_outcome"]),
            recoverability=Recoverability(record["recoverability"]),
            evidence_requirements=tuple(record["evidence_requirements"]),
            release_blocking=bool(record["release_blocking"]),
            verification_status=VerificationStatus(
                record["verification_status"]
            ),
            evidence=tuple(
                FaultEvidence.from_dict(item) for item in record["evidence"]
            ),
        )


@dataclasses.dataclass
class FaultInjectionPlanRegistry:
    """Deterministic registry of the complete production fault-injection plan.

    The registry accepts planned declarations without pretending they have
    executed. Serialization and deserialization reject a registry that does
    not declare every required production scenario.

    Attributes:
        plans: Scenario plans keyed by their stable scenario enum.
    """

    plans: dict[FaultScenario, FaultScenarioPlan] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        """Validate and normalize plans supplied at construction time."""
        supplied = tuple(self.plans.values())
        self.plans = {}
        for plan in supplied:
            self.register(plan)

    @classmethod
    def production_beta(cls) -> Self:
        """Build the complete reliability plan for the production beta.

        Returns:
            A registry holding a plan for every required scenario.
        """
        registry = cls()
        for plan in default_fault_injection_plans():
            registry.register(plan)
        registry.validate_complete()
        return registry

    def register(self, plan: FaultScenarioPlan) -> None:
        """Register one complete scenario declaration.

        Args:
            plan: Scenario declaration to register.

        Raises:
            TypeError: If ``plan`` is not a scenario plan.
            ValueError: If the scenario is already registered.
        """
        if not isinstance(plan, FaultScenarioPlan):
            raise TypeError("plan must be FaultScenarioPlan")
        if plan.scenario in self.plans:
            raise ValueError(f"scenario already registered: {plan.scenario}")
        self.plans[plan.scenario] = plan

    def record_evidence(
        self,
        scenario: FaultScenario,
        evidence: tuple[FaultEvidence, ...],
    ) -> FaultScenarioPlan:
        """Record one execution and derive its verification status.

        Safe evidence for every declared requirement produces ``PASSED``.
        Unsafe evidence produces ``FAILED`` even when other requirements are
        missing. A run with no unsafe evidence but incomplete coverage is
        ``INCONCLUSIVE``. The registry is updated only after the complete
        replacement plan validates, so malformed evidence cannot overwrite a
        planned scenario. A scenario that is not registered fails with a
        `KeyError`.

        Args:
            scenario: Scenario whose execution produced the evidence.
            evidence: Unique evidence records from one execution.

        Returns:
            The updated scenario plan.

        Raises:
            TypeError: If the scenario or evidence shape is invalid.
            ValueError: If evidence is empty, comes from more than one
                execution, or belongs to another scenario's requirements.
        """
        if not isinstance(scenario, FaultScenario):
            raise TypeError("scenario must be FaultScenario")
        if not isinstance(evidence, tuple) or not evidence:
            raise ValueError("evidence must be a non-empty tuple")
        if not all(isinstance(item, FaultEvidence) for item in evidence):
            raise TypeError("evidence must contain FaultEvidence values")
        execution_ids = {item.execution_id for item in evidence}
        if len(execution_ids) != 1:
            raise ValueError(
                "evidence for one scenario must share an execution_id"
            )
        plan = self.plans[scenario]
        evidence_types = {item.evidence_type for item in evidence}
        unexpected = evidence_types - set(plan.evidence_requirements)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise ValueError(f"evidence has unexpected requirements: {names}")
        missing = set(plan.evidence_requirements) - evidence_types
        has_unsafe = any(
            item.observed_outcome is EvidenceOutcome.UNSAFE_OUTCOME_OBSERVED
            for item in evidence
        )
        if has_unsafe:
            status = VerificationStatus.FAILED
        elif missing:
            status = VerificationStatus.INCONCLUSIVE
        else:
            status = VerificationStatus.PASSED
        updated = dataclasses.replace(
            plan,
            verification_status=status,
            evidence=evidence,
        )
        self.plans[scenario] = updated
        return updated

    def validate_complete(self) -> None:
        """Reject a registry that omits any required production scenario.

        Raises:
            ValueError: If a scenario has no plan.
        """
        missing = set(FaultScenario) - set(self.plans)
        if missing:
            missing_text = ", ".join(
                item.value
                for item in sorted(missing, key=lambda value: value.value)
            )
            raise ValueError(
                f"fault-injection plan is incomplete: {missing_text}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the complete plan in deterministic dictionary form."""
        self.validate_complete()
        ordered_plans = sorted(
            self.plans.values(), key=lambda plan: plan.scenario.value
        )
        return {
            "schema_version": 1,
            "scenarios": [plan.to_dict() for plan in ordered_plans],
        }

    def to_json(self) -> str:
        """Serialize the complete plan using canonical JSON ordering.

        Returns:
            The plan as compact JSON with sorted keys.
        """
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuild and validate a complete registry from a mapping.

        Args:
            record: A mapping with schema version 1 and a list of scenarios.

        Returns:
            The registry.

        Raises:
            TypeError: If a scenario is not a mapping.
            ValueError: If the schema version is unsupported or the scenarios
                are not a list.
        """
        if record.get("schema_version") != 1:
            raise ValueError("unsupported fault-injection plan schema version")
        scenarios = record.get("scenarios")
        if not isinstance(scenarios, list):
            raise ValueError("scenarios must be a list")
        registry = cls()
        for item in scenarios:
            if not isinstance(item, Mapping):
                raise TypeError("each scenario must be a mapping")
            registry.register(FaultScenarioPlan.from_dict(item))
        registry.validate_complete()
        return registry

    @classmethod
    def from_json(cls, serialized: str) -> Self:
        """Rebuild and validate a complete registry from canonical JSON.

        Args:
            serialized: The JSON text.

        Returns:
            The registry.

        Raises:
            TypeError: If the JSON does not hold an object.
        """
        record = json.loads(serialized)
        if not isinstance(record, Mapping):
            raise TypeError("serialized plan must contain a JSON object")
        return cls.from_dict(record)


def default_fault_injection_plans() -> tuple[FaultScenarioPlan, ...]:
    """Return the complete, unexecuted production-beta fault plan.

    These declarations describe what must be injected and what evidence a
    release needs. They intentionally remain ``PLANNED`` until an external
    execution harness supplies real evidence records.
    """
    scenarios = (
        (
            FaultScenario.PROCESS_DEATH,
            SafeOutcome.RESUME_DURABLE_CHECKPOINT,
            Recoverability.AUTOMATIC,
            ("process_termination", "checkpoint_resume", "artifact_integrity"),
        ),
        (
            FaultScenario.STALE_CHECKPOINT,
            SafeOutcome.REJECT_STALE_CHECKPOINT,
            Recoverability.OPERATOR_ASSISTED,
            ("checkpoint_identity", "stale_checkpoint_rejection", "run_state"),
        ),
        (
            FaultScenario.DISK_FULL,
            SafeOutcome.PRESERVE_DURABLE_ARTIFACT,
            Recoverability.OPERATOR_ASSISTED,
            (
                "disk_capacity_exhaustion",
                "artifact_integrity",
                "resource_error",
            ),
        ),
        (
            FaultScenario.CORRUPT_ARTIFACT,
            SafeOutcome.QUARANTINE_CORRUPT_ARTIFACT,
            Recoverability.OPERATOR_ASSISTED,
            ("corruption_injection", "digest_mismatch", "quarantine_record"),
        ),
        (
            FaultScenario.WORKER_LOSS,
            SafeOutcome.RETRY_IDEMPOTENT_WORK,
            Recoverability.AUTOMATIC,
            ("worker_termination", "idempotent_retry", "event_deduplication"),
        ),
        (
            FaultScenario.TIMEOUT,
            SafeOutcome.CANCEL_AND_RECORD_TERMINAL_FAILURE,
            Recoverability.OPERATOR_ASSISTED,
            ("timeout_injection", "cancellation_record", "resource_error"),
        ),
        (
            FaultScenario.MEMORY_EXHAUSTION,
            SafeOutcome.ABORT_BEFORE_PUBLICATION,
            Recoverability.OPERATOR_ASSISTED,
            ("memory_pressure", "publication_absence", "resource_error"),
        ),
        (
            FaultScenario.PARTIAL_WRITE,
            SafeOutcome.PRESERVE_DURABLE_ARTIFACT,
            Recoverability.AUTOMATIC,
            ("write_interruption", "atomicity_check", "artifact_integrity"),
        ),
        (
            FaultScenario.DEPENDENCY_MISMATCH,
            SafeOutcome.REJECT_INCOMPATIBLE_DEPENDENCY,
            Recoverability.OPERATOR_ASSISTED,
            ("environment_identity", "dependency_rejection", "run_state"),
        ),
        (
            FaultScenario.INVALID_SIGNATURE,
            SafeOutcome.REJECT_UNTRUSTED_ARTIFACT,
            Recoverability.OPERATOR_ASSISTED,
            (
                "signature_tampering",
                "verification_failure",
                "quarantine_record",
            ),
        ),
        (
            FaultScenario.RETRY_EXHAUSTION,
            SafeOutcome.TERMINATE_AFTER_RETRY_EXHAUSTION,
            Recoverability.TERMINAL_SAFE_FAILURE,
            ("retry_history", "terminal_event", "partial_publication_check"),
        ),
        (
            FaultScenario.DUPLICATE_EVENT,
            SafeOutcome.IGNORE_DUPLICATE_EVENT,
            Recoverability.AUTOMATIC,
            ("duplicate_event", "idempotency_key", "state_transition"),
        ),
        (
            FaultScenario.INCOMPLETE_LINEAGE,
            SafeOutcome.BLOCK_RELEASE_FOR_MISSING_LINEAGE,
            Recoverability.TERMINAL_SAFE_FAILURE,
            ("lineage_gap", "release_block", "audit_record"),
        ),
        (
            FaultScenario.REJECTED_APPROVAL,
            SafeOutcome.BLOCK_RELEASE_FOR_REJECTED_APPROVAL,
            Recoverability.TERMINAL_SAFE_FAILURE,
            ("approval_rejection", "release_block", "audit_record"),
        ),
        (
            FaultScenario.CHANGED_SNAPSHOT,
            SafeOutcome.INVALIDATE_CHANGED_SNAPSHOT,
            Recoverability.OPERATOR_ASSISTED,
            ("snapshot_digest_change", "run_invalidation", "lineage_record"),
        ),
    )
    return tuple(
        FaultScenarioPlan(
            scenario=scenario,
            expected_safe_outcome=expected_safe_outcome,
            recoverability=recoverability,
            evidence_requirements=evidence_requirements,
            release_blocking=True,
        )
        for (
            scenario,
            expected_safe_outcome,
            recoverability,
            evidence_requirements,
        ) in scenarios
    )
