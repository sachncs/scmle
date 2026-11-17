"""Data-quality findings and the evidence that supports them."""

from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import math
from typing import Any
from typing import Self

from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import explanations


def normalise(
    evidence: Mapping[str, Any], field_name: str = "quality_evidence"
) -> dict[str, object]:
    """Validates and normalizes one JSON-native quality evidence record.

    A floating-point value that is not finite is refused with a
    `ValueError` by the JSON validation.

    Args:
        evidence: Mapping containing only JSON-native values.
        field_name: Path used in validation errors.

    Returns:
        A detached dictionary containing JSON-native values.

    Raises:
        TypeError: If a key or value is not JSON-native, or the record does
            not serialize to an object.
    """
    normalized = explanations.sanitise(evidence, field_name)
    if not isinstance(normalized, dict):
        raise errors.KindError(f"{field_name} must serialize to an object")
    return normalized


@dataclasses.dataclass(frozen=True, slots=True)
class Observation:
    """One quality check result and its reproducible evidence.

    Attributes:
        note: Human-readable finding text, if the check raised one.
        evidence: JSON-native method, threshold, sampling, metric, and
            limitation details for the check.
    """

    note: str | None
    evidence: dict[str, object]

    def __post_init__(self) -> None:
        """Validates and detaches the evidence from mutable caller state.

        The evidence is replaced by a normalized copy; invalid evidence is
        rejected as described for `normalise`.

        Raises:
            ValueError: If the note is given but blank.
        """
        if self.note is not None and not self.note.strip():
            raise errors.InputError("note must not be empty")
        object.__setattr__(
            self,
            "evidence",
            normalise(self.evidence, "evidence"),
        )


class FindingSeverity(enum.StrEnum):
    """Severity assigned to a curation finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class Disposition(enum.StrEnum):
    """Review state assigned to a curation finding."""

    OPEN = "open"
    ACCEPTED = "accepted"
    RESOLVED = "resolved"
    WAIVED = "waived"
    FALSE_POSITIVE = "false_positive"


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One reviewable, reproducible observation about a dataset.

    Attributes:
        finding_id: Stable identifier for the rule finding.
        rule_version: Version of the rule that produced the finding.
        severity: Operational impact if the finding is not addressed.
        confidence: Confidence in the finding, from zero through one.
        affected_scope: Columns, rows, partitions, or objects affected.
        evidence: Reproducible measurements or references supporting it.
        remediation: Action required to resolve or assess the finding.
        owner: Person or team responsible for the finding.
        reviewer: Person who reviewed it, if review has happened.
        disposition: Current review decision.
        blocking: Whether the finding blocks training or release.
        timestamp: UTC timestamp when the finding was observed.
    """

    finding_id: str
    rule_version: str
    severity: FindingSeverity
    confidence: float
    affected_scope: Mapping[str, Any]
    evidence: Mapping[str, Any]
    remediation: str
    owner: str
    reviewer: str | None
    disposition: Disposition
    blocking: bool
    timestamp: str = dataclasses.field(default_factory=events.stamp)

    def __post_init__(self) -> None:
        """Validates fields that determine finding identity and decisions.

        A severity or disposition given as a string is converted to its enum.

        Raises:
            ValueError: If the id, rule version, remediation, owner or timestamp
                is blank, the confidence is not a finite number in [0, 1], or
                the severity or disposition is not a valid value.
        """
        if not self.finding_id.strip():
            raise errors.InputError("finding_id must not be empty")
        if not self.rule_version.strip():
            raise errors.InputError("rule_version must not be empty")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise errors.InputError(
                "confidence must be finite and between zero and one"
            )
        if not self.remediation.strip():
            raise errors.InputError("remediation must not be empty")
        if not self.owner.strip():
            raise errors.InputError("owner must not be empty")
        if not self.timestamp.strip():
            raise errors.InputError("timestamp must not be empty")
        severity: Any = self.severity  # may be a raw value at runtime
        if not isinstance(severity, FindingSeverity):
            object.__setattr__(self, "severity", FindingSeverity(severity))
        disposition: Any = self.disposition  # may be a raw value at runtime
        if not isinstance(disposition, Disposition):
            object.__setattr__(self, "disposition", Disposition(disposition))

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON- and Arrow-compatible mapping."""
        return {
            "id": self.finding_id,
            "rule_version": self.rule_version,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "affected_scope": dict(self.affected_scope),
            "evidence": dict(self.evidence),
            "remediation": self.remediation,
            "owner": self.owner,
            "reviewer": self.reviewer,
            "disposition": self.disposition.value,
            "blocking": self.blocking,
            "timestamp": self.timestamp,
        }

    @classmethod
    def glean(
        cls,
        message: str,
        category: str,
        severity: FindingSeverity = FindingSeverity.WARNING,
    ) -> Self:
        """Creates a structured finding from a pipeline observation.

        The profiling layer may produce a human-readable observation while a
        rule is being developed. This constructor gives that observation a
        stable identity and evidence envelope before it is persisted. It is a
        canonical contract conversion, not a second finding representation.

        Args:
            message: Human-readable observation emitted by a profiling rule.
            category: Stable category such as ``quality`` or ``risk``.
            severity: Operational severity assigned by the caller.

        Returns:
            A reviewable finding with a deterministic ID.

        Raises:
            ValueError: If the message or category is empty.
        """
        observation = message.strip()
        finding_category = category.strip().lower()
        if not observation:
            raise errors.InputError("message must not be empty")
        if not finding_category:
            raise errors.InputError("category must not be empty")
        digest = hashlib.sha256(observation.encode("utf-8")).hexdigest()
        return cls(
            finding_id=f"{finding_category}.{digest[:16]}",
            rule_version="pipeline-observation.v1",
            severity=severity,
            confidence=1.0,
            affected_scope={"category": finding_category},
            evidence={"message": observation},
            remediation="Review the observation and record a disposition.",
            owner="pipeline",
            reviewer=None,
            disposition=Disposition.OPEN,
            blocking=False,
        )

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a finding from its serialized representation.

        A missing required key raises `KeyError`, and an invalid value is
        rejected by the constructor. Only `reviewer` may be absent.

        Args:
            record: A mapping as produced by `to_dict`.

        Returns:
            The finding.
        """
        return cls(
            finding_id=str(record["id"]),
            rule_version=str(record["rule_version"]),
            severity=FindingSeverity(record["severity"]),
            confidence=float(record["confidence"]),
            affected_scope=dict(record["affected_scope"]),
            evidence=dict(record["evidence"]),
            remediation=str(record["remediation"]),
            owner=str(record["owner"]),
            reviewer=record.get("reviewer"),
            disposition=Disposition(record["disposition"]),
            blocking=bool(record["blocking"]),
            timestamp=str(record["timestamp"]),
        )


def vet(findings: tuple[Any, ...]) -> None:
    """Checks that every entry of a record's findings is a finding.

    Args:
        findings: The findings of a manifest.

    Raises:
        TypeError: If an entry is not a `Finding`.
    """
    if any(not isinstance(finding, Finding) for finding in findings):
        raise errors.KindError("findings must contain Finding records")
