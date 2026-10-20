"""Shared curation evidence for non-tabular dataset foundations."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import hashlib
from typing import Any

from scml.contracts import data as data_contracts


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityCurationReport:
    """JSON-native data-understanding evidence before modelling."""

    modality: str
    dataset_hash: str
    sample_count: int
    profile: Mapping[str, Any]
    quality_findings: tuple[data_contracts.CurationFinding, ...] = ()
    risk_findings: tuple[data_contracts.CurationFinding, ...] = ()
    schema_version: int = 1

    @property
    def unresolved_blocking_findings(
        self,
    ) -> tuple[data_contracts.CurationFinding, ...]:
        """Return blocking findings without a safe review disposition."""
        safe_dispositions = {
            data_contracts.FindingDisposition.RESOLVED,
            data_contracts.FindingDisposition.WAIVED,
            data_contracts.FindingDisposition.FALSE_POSITIVE,
        }
        return tuple(
            finding
            for finding in (*self.quality_findings, *self.risk_findings)
            if finding.blocking and finding.disposition not in safe_dispositions
        )

    @property
    def release_ready(self) -> bool:
        """Return whether curation has no unresolved blocking finding."""
        return not self.unresolved_blocking_findings

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible report."""
        return {
            "schema_version": self.schema_version,
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "sample_count": self.sample_count,
            "profile": dict(self.profile),
            "quality_findings": [
                finding.to_dict() for finding in self.quality_findings
            ],
            "risk_findings": [
                finding.to_dict() for finding in self.risk_findings
            ],
            "release_ready": self.release_ready,
            "unresolved_blocking_findings": [
                finding.finding_id
                for finding in self.unresolved_blocking_findings
            ],
        }


def dataset_hash(identities: Iterable[str]) -> str:
    """Return an order-independent digest for modality sample identities."""
    canonical = "\n".join(sorted(str(identity) for identity in identities))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def label_counts(labels: Iterable[Any]) -> dict[str, int]:
    """Return stable label buckets without interpreting task semantics."""
    counts: dict[str, int] = {}
    for label in labels:
        bucket = (
            "unlabeled" if label is None else f"{type(label).__name__}:{label}"
        )
        counts[bucket] = counts.get(bucket, 0) + 1
    return dict(sorted(counts.items()))


def build_finding(
    finding_id: str,
    severity: data_contracts.FindingSeverity,
    scope: Mapping[str, Any],
    evidence: Mapping[str, Any],
    remediation: str,
    blocking: bool,
) -> data_contracts.CurationFinding:
    """Build one modality curation finding with a shared rule version."""
    return data_contracts.CurationFinding(
        finding_id=finding_id,
        rule_version="multimodal-curation.v1",
        severity=severity,
        confidence=1.0,
        affected_scope=scope,
        evidence=evidence,
        remediation=remediation,
        owner="data-curation",
        reviewer=None,
        disposition=data_contracts.FindingDisposition.OPEN,
        blocking=blocking,
    )


__all__ = [
    "ModalityCurationReport",
    "build_finding",
    "dataset_hash",
    "label_counts",
]
