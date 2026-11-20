"""Evidence-first contracts for production reliability fault scenarios."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
import json
from typing import Any
from typing import ClassVar
from typing import Self

from scmle.contracts import errors
from scmle.contracts import records


class Contingency(enum.StrEnum):
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


class Expectation(enum.StrEnum):
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


class Verification(enum.StrEnum):
    """Evidence state of one fault scenario."""

    PLANNED = "planned"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class Outcome(enum.StrEnum):
    """Observed result recorded by a fault-injection execution."""

    SAFE_OUTCOME_OBSERVED = "safe_outcome_observed"
    UNSAFE_OUTCOME_OBSERVED = "unsafe_outcome_observed"
    INCONCLUSIVE = "inconclusive"


@dataclasses.dataclass(frozen=True, slots=True)
class Evidence:
    """Evidence produced by an actual fault-injection execution.

    The contract requires non-blank text for an execution identity, an
    artifact URI and an artifact digest, among other fields; it does not
    check the digest against the artifact. A plan can be declared before
    execution, but callers cannot represent a verification result without
    this evidence envelope.

    Attributes:
        kind: Stable name of the evidence item, such as
            ``checkpoint_resume`` or ``artifact_digest``.
        execution: Identity of the run that performed the injection.
        uri: URI of the durable evidence artifact.
        digests: Digests by the hashed thing; ``artifact`` is the digest of
            the evidence artifact contents.
        outcome: Result observed by the execution.
        recorded: Timestamp supplied by the execution recorder.
        reviewer: Human or automated reviewer that verified the evidence.
        observation: Concrete observation captured in the evidence record.
        migrations: Version steps of the stored evidence, which is versioned
            by the registry that holds it.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "evidence_type": "kind",
                "execution_id": "execution",
                "artifact_uri": "uri",
                "observed_outcome": "outcome",
                "recorded_at": "recorded",
            },
            groups={"digests": {"artifact_digest": "artifact"}},
        ),
    )

    kind: str
    execution: str
    uri: str
    digests: Mapping[str, str]
    outcome: Outcome
    recorded: str
    reviewer: str
    observation: str

    def __post_init__(self) -> None:
        """Rejects evidence records that cannot be independently reviewed.

        Raises:
            TypeError: If `outcome` is not an `Outcome`.
            ValueError: If a required text field, or the artifact digest, is
                blank.
        """
        for name, value in (
            ("kind", self.kind),
            ("execution", self.execution),
            ("uri", self.uri),
            ("digests.artifact", self.digests.get("artifact")),
            ("recorded", self.recorded),
            ("reviewer", self.reviewer),
            ("observation", self.observation),
        ):
            if not isinstance(value, str) or not value.strip():
                raise errors.InputError(f"{name} must not be empty")
        object.__setattr__(self, "digests", dict(self.digests))
        if not isinstance(self.outcome, Outcome):
            raise errors.KindError("outcome must be Outcome")

    def to_dict(self) -> records.Layout:
        """Returns a deterministic JSON-compatible evidence record."""
        return {
            "kind": self.kind,
            "execution": self.execution,
            "uri": self.uri,
            "digests": dict(sorted(self.digests.items())),
            "outcome": self.outcome.value,
            "recorded": self.recorded,
            "reviewer": self.reviewer,
            "observation": self.observation,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds evidence from its JSON-compatible representation.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The evidence.
        """
        return cls(
            kind=str(record["kind"]),
            execution=str(record["execution"]),
            uri=str(record["uri"]),
            digests={
                str(key): str(value)
                for key, value in dict(record["digests"]).items()
            },
            outcome=Outcome(record["outcome"]),
            recorded=str(record["recorded"]),
            reviewer=str(record["reviewer"]),
            observation=str(record["observation"]),
        )


def require(requirements: tuple[str, ...]) -> None:
    """Checks the evidence types a fault scenario requires.

    Args:
        requirements: The required evidence types.

    Raises:
        ValueError: If there are none, one is blank or not text, or one is
            repeated.
    """
    if not requirements:
        raise errors.InputError("requirements must not be empty")
    supplied: Iterable[object] = requirements  # may hold non-strings
    if any(
        not isinstance(requirement, str) or not requirement.strip()
        for requirement in supplied
    ):
        raise errors.InputError("requirements must contain text")
    if len(set(requirements)) != len(requirements):
        raise errors.InputError("requirements must be unique")


@dataclasses.dataclass(frozen=True, slots=True)
class Plan:
    """Declaration and verification state for one fault scenario.

    ``PLANNED`` is the only status permitted without execution evidence.
    Every other status requires evidence, and ``PASSED`` additionally requires
    safe evidence for every declared evidence requirement.

    Attributes:
        scenario: Fault condition being tested.
        outcome: Expected safe behavior required by the release policy.
        recoverability: Recovery promise made to operators and users.
        requirements: Evidence types required for a passing result.
        blocking: Whether an unsafe or unverified result blocks release.
        verification: Current evidence state.
        evidence: Evidence records from actual executions.
        migrations: Version steps of the stored plan and its evidence, which
            the registry that holds them versions.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            renames={
                "expected_safe_outcome": "outcome",
                "evidence_requirements": "requirements",
                "release_blocking": "blocking",
                "verification_status": "verification",
            },
            parts={"evidence": Evidence.migrations[0]},
        ),
    )

    scenario: Contingency
    outcome: Expectation
    recoverability: Recoverability
    requirements: tuple[str, ...]
    blocking: bool
    verification: Verification = Verification.PLANNED
    evidence: tuple[Evidence, ...] = ()

    def __post_init__(self) -> None:
        """Validates declaration completeness and evidence/status consistency.

        Raises:
            TypeError: If an enumerated field has the wrong type or
                `blocking` is not a bool.
            ValueError: If the evidence requirements are empty, blank or
                repeated, evidence types repeat, a status other than planned has
                no evidence, or a passed scenario lacks a required evidence type
                or has any evidence whose outcome is not
                ``safe_outcome_observed``.
        """
        enum_fields = (
            ("scenario", self.scenario, Contingency),
            ("outcome", self.outcome, Expectation),
            ("recoverability", self.recoverability, Recoverability),
            ("verification", self.verification, Verification),
        )
        for name, value, expected in enum_fields:
            if not isinstance(value, expected):
                raise errors.KindError(f"{name} must be {expected.__name__}")
        require(self.requirements)
        if not isinstance(self.blocking, bool):
            raise errors.KindError("blocking must be bool")
        evidence_types = [item.kind for item in self.evidence]
        if len(set(evidence_types)) != len(evidence_types):
            raise errors.InputError(
                "evidence types must be unique per scenario"
            )
        if self.verification is not Verification.PLANNED and not self.evidence:
            raise errors.InputError(
                "verification status requires execution evidence"
            )
        if self.verification is Verification.PASSED:
            missing = set(self.requirements) - set(evidence_types)
            if missing:
                missing_text = ", ".join(sorted(missing))
                raise errors.InputError(
                    "passed scenario is missing evidence requirements: "
                    f"{missing_text}"
                )
            if any(
                item.outcome is not Outcome.SAFE_OUTCOME_OBSERVED
                for item in self.evidence
            ):
                raise errors.InputError(
                    "passed scenario requires safe outcome evidence only"
                )

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible scenario record."""
        return {
            "scenario": self.scenario.value,
            "outcome": self.outcome.value,
            "recoverability": self.recoverability.value,
            "requirements": list(self.requirements),
            "blocking": self.blocking,
            "verification": self.verification.value,
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a scenario plan from its JSON-compatible representation.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The scenario plan.
        """
        return cls(
            scenario=Contingency(record["scenario"]),
            outcome=Expectation(record["outcome"]),
            recoverability=Recoverability(record["recoverability"]),
            requirements=tuple(record["requirements"]),
            blocking=bool(record["blocking"]),
            verification=Verification(record["verification"]),
            evidence=tuple(
                Evidence.from_dict(item) for item in record["evidence"]
            ),
        )


VERSION = records.ORIGIN + len(Plan.migrations)


@dataclasses.dataclass
class Registry:
    """Deterministic registry of the complete production fault-injection plan.

    The registry accepts planned declarations without pretending they have
    executed. Serialization and deserialization reject a registry that does
    not declare every required production scenario.

    Attributes:
        plans: Scenario plans keyed by their stable scenario enum.
    """

    plans: dict[Contingency, Plan] = dataclasses.field(default_factory=dict)

    def __post_init__(self) -> None:
        """Re-registers the plans supplied at construction time.

        The mapping is rebuilt through `register`, so each key becomes the
        plan's own scenario.
        """
        supplied = tuple(self.plans.values())
        self.plans = {}
        for plan in supplied:
            self.register(plan)

    @classmethod
    def prerelease(cls) -> Self:
        """Builds the complete reliability plan for the production beta.

        Returns:
            A registry holding a plan for every required scenario.
        """
        registry = cls()
        for plan in drills():
            registry.register(plan)
        registry.insist()
        return registry

    def register(self, plan: Plan) -> None:
        """Registers one complete scenario declaration.

        Args:
            plan: Scenario declaration to register.

        Raises:
            TypeError: If ``plan`` is not a scenario plan.
            ValueError: If the scenario is already registered.
        """
        if not isinstance(plan, Plan):
            raise errors.KindError("plan must be Plan")
        if plan.scenario in self.plans:
            raise errors.InputError(
                f"scenario already registered: {plan.scenario}"
            )
        self.plans[plan.scenario] = plan

    def log(
        self,
        scenario: Contingency,
        evidence: tuple[Evidence, ...],
    ) -> Plan:
        """Records one execution and derives its verification status.

        Safe evidence for every declared requirement produces ``PASSED``.
        Unsafe evidence produces ``FAILED`` even when other requirements are
        missing. A run with no unsafe evidence but incomplete coverage is
        ``INCONCLUSIVE``. The registry is updated only after the complete
        replacement plan validates, so malformed evidence cannot overwrite a
        planned scenario. A scenario that is not registered fails with a
        `KeyError`. Evidence for every requirement but with an
        ``inconclusive`` outcome is not downgraded: validating the
        scenario record rejects it with `ValueError`.

        Args:
            scenario: Scenario whose execution produced the evidence.
            evidence: A non-empty tuple of evidence records from one
                execution, with distinct evidence types.

        Returns:
            The updated scenario plan.

        Raises:
            TypeError: If `scenario` is not a `Contingency` or an item is
                not a `Evidence`.
            ValueError: If `evidence` is not a non-empty tuple, comes from
                more than one execution, repeats an evidence type, has a type
                outside the scenario's requirements, or would give a passed
                plan an unsafe or inconclusive outcome.
        """
        if not isinstance(scenario, Contingency):
            raise errors.KindError("scenario must be Contingency")
        supplied: object = evidence  # callers may pass a non-tuple
        if not isinstance(supplied, tuple) or not supplied:
            raise errors.InputError("evidence must be a non-empty tuple")
        if not all(isinstance(item, Evidence) for item in evidence):
            raise errors.KindError("evidence must contain Evidence values")
        executions = {item.execution for item in evidence}
        if len(executions) != 1:
            raise errors.InputError(
                "evidence for one scenario must share an execution"
            )
        plan = self.plans[scenario]
        evidence_types = {item.kind for item in evidence}
        unexpected = evidence_types - set(plan.requirements)
        if unexpected:
            names = ", ".join(sorted(unexpected))
            raise errors.InputError(
                f"evidence has unexpected requirements: {names}"
            )
        missing = set(plan.requirements) - evidence_types
        has_unsafe = any(
            item.outcome is Outcome.UNSAFE_OUTCOME_OBSERVED for item in evidence
        )
        if has_unsafe:
            status = Verification.FAILED
        elif missing:
            status = Verification.INCONCLUSIVE
        else:
            status = Verification.PASSED
        updated = dataclasses.replace(
            plan,
            verification=status,
            evidence=evidence,
        )
        self.plans[scenario] = updated
        return updated

    def insist(self) -> None:
        """Rejects a registry that omits any required production scenario.

        Raises:
            ValueError: If a scenario has no plan.
        """
        missing = set(Contingency) - set(self.plans)
        if missing:
            missing_text = ", ".join(
                item.value
                for item in sorted(missing, key=lambda value: value.value)
            )
            raise errors.InputError(
                f"fault-injection plan is incomplete: {missing_text}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns the complete plan in deterministic dictionary form.

        Returns:
            A dict with the current ``schema_version`` and ``scenarios``, the
            scenario
            records ordered by scenario value. Raises `ValueError` if a
            required scenario has no plan.
        """
        self.insist()
        ordered_plans = sorted(
            self.plans.values(), key=lambda plan: plan.scenario.value
        )
        return {
            "schema_version": VERSION,
            "scenarios": [plan.to_dict() for plan in ordered_plans],
        }

    def to_json(self) -> str:
        """Serializes the complete plan using canonical JSON ordering.

        Returns:
            The plan as compact, ASCII-only JSON with sorted keys. Raises
            `ValueError` if a required scenario has no plan.
        """
        return json.dumps(
            self.to_dict(),
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds and validates a complete registry from a mapping.

        A registry stored at an older schema version has its scenarios
        migrated to the current layout first.

        Args:
            record: A mapping with a supported ``schema_version`` and a
                ``scenarios`` list of scenario mappings.

        Returns:
            The registry.

        Raises:
            TypeError: If a scenario is not a mapping.
            ValueError: If the schema version is unsupported, the scenarios
                are not a list, a scenario repeats, a scenario record is
                invalid, or a required scenario is missing.
        """
        version = record.get("schema_version")
        if (
            isinstance(version, bool)
            or not isinstance(version, int)
            or not records.ORIGIN <= version <= VERSION
        ):
            raise errors.InputError(
                "unsupported fault-injection plan schema version"
            )
        scenarios = record.get("scenarios")
        if not isinstance(scenarios, list):
            raise errors.InputError("scenarios must be a list")
        registry = cls()
        for item in scenarios:
            if not isinstance(item, Mapping):
                raise errors.KindError("each scenario must be a mapping")
            registry.register(
                Plan.from_dict(
                    records.upgrade(Plan.migrations, dict(item), version)
                )
            )
        registry.insist()
        return registry

    @classmethod
    def parse(cls, serialized: str) -> Self:
        """Rebuilds and validates a complete registry from canonical JSON.

        Args:
            serialized: The JSON text.

        Returns:
            The registry.

        Raises:
            TypeError: If the JSON does not hold an object.
            ValueError: If the text is not valid JSON, or as `from_dict`.
        """
        record = json.loads(serialized)
        if not isinstance(record, Mapping):
            raise errors.KindError("serialized plan must contain a JSON object")
        return cls.from_dict(record)


SCENARIOS: tuple[
    tuple[Contingency, Expectation, Recoverability, tuple[str, ...]], ...
] = (
    (
        Contingency.PROCESS_DEATH,
        Expectation.RESUME_DURABLE_CHECKPOINT,
        Recoverability.AUTOMATIC,
        ("process_termination", "checkpoint_resume", "artifact_integrity"),
    ),
    (
        Contingency.STALE_CHECKPOINT,
        Expectation.REJECT_STALE_CHECKPOINT,
        Recoverability.OPERATOR_ASSISTED,
        ("checkpoint_identity", "stale_checkpoint_rejection", "run_state"),
    ),
    (
        Contingency.DISK_FULL,
        Expectation.PRESERVE_DURABLE_ARTIFACT,
        Recoverability.OPERATOR_ASSISTED,
        (
            "disk_capacity_exhaustion",
            "artifact_integrity",
            "resource_error",
        ),
    ),
    (
        Contingency.CORRUPT_ARTIFACT,
        Expectation.QUARANTINE_CORRUPT_ARTIFACT,
        Recoverability.OPERATOR_ASSISTED,
        ("corruption_injection", "digest_mismatch", "quarantine_record"),
    ),
    (
        Contingency.WORKER_LOSS,
        Expectation.RETRY_IDEMPOTENT_WORK,
        Recoverability.AUTOMATIC,
        ("worker_termination", "idempotent_retry", "event_deduplication"),
    ),
    (
        Contingency.TIMEOUT,
        Expectation.CANCEL_AND_RECORD_TERMINAL_FAILURE,
        Recoverability.OPERATOR_ASSISTED,
        ("timeout_injection", "cancellation_record", "resource_error"),
    ),
    (
        Contingency.MEMORY_EXHAUSTION,
        Expectation.ABORT_BEFORE_PUBLICATION,
        Recoverability.OPERATOR_ASSISTED,
        ("memory_pressure", "publication_absence", "resource_error"),
    ),
    (
        Contingency.PARTIAL_WRITE,
        Expectation.PRESERVE_DURABLE_ARTIFACT,
        Recoverability.AUTOMATIC,
        ("write_interruption", "atomicity_check", "artifact_integrity"),
    ),
    (
        Contingency.DEPENDENCY_MISMATCH,
        Expectation.REJECT_INCOMPATIBLE_DEPENDENCY,
        Recoverability.OPERATOR_ASSISTED,
        ("environment_identity", "dependency_rejection", "run_state"),
    ),
    (
        Contingency.INVALID_SIGNATURE,
        Expectation.REJECT_UNTRUSTED_ARTIFACT,
        Recoverability.OPERATOR_ASSISTED,
        (
            "signature_tampering",
            "verification_failure",
            "quarantine_record",
        ),
    ),
    (
        Contingency.RETRY_EXHAUSTION,
        Expectation.TERMINATE_AFTER_RETRY_EXHAUSTION,
        Recoverability.TERMINAL_SAFE_FAILURE,
        ("retry_history", "terminal_event", "partial_publication_check"),
    ),
    (
        Contingency.DUPLICATE_EVENT,
        Expectation.IGNORE_DUPLICATE_EVENT,
        Recoverability.AUTOMATIC,
        ("duplicate_event", "idempotency_key", "state_transition"),
    ),
    (
        Contingency.INCOMPLETE_LINEAGE,
        Expectation.BLOCK_RELEASE_FOR_MISSING_LINEAGE,
        Recoverability.TERMINAL_SAFE_FAILURE,
        ("lineage_gap", "release_block", "audit_record"),
    ),
    (
        Contingency.REJECTED_APPROVAL,
        Expectation.BLOCK_RELEASE_FOR_REJECTED_APPROVAL,
        Recoverability.TERMINAL_SAFE_FAILURE,
        ("approval_rejection", "release_block", "audit_record"),
    ),
    (
        Contingency.CHANGED_SNAPSHOT,
        Expectation.INVALIDATE_CHANGED_SNAPSHOT,
        Recoverability.OPERATOR_ASSISTED,
        ("snapshot_digest_change", "run_invalidation", "lineage_record"),
    ),
)
"""The production-beta fault plan: scenario, safe outcome, recoverability
and the evidence each drill must record."""


def drills() -> tuple[Plan, ...]:
    """Returns the complete, unexecuted production-beta fault plan.

    These declarations describe what must be injected and what evidence a
    release needs. They intentionally remain ``PLANNED`` until an external
    execution harness supplies real evidence records.
    """
    return tuple(
        Plan(
            scenario=scenario,
            outcome=outcome,
            recoverability=recoverability,
            requirements=requirements,
            blocking=True,
        )
        for (
            scenario,
            outcome,
            recoverability,
            requirements,
        ) in SCENARIOS
    )
