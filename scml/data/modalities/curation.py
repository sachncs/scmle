"""Shared curation evidence for non-tabular dataset foundations."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any
from typing import Self

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


@dataclasses.dataclass(frozen=True, slots=True)
class ModalityDatasetCard:
    """Human-reviewable card generated from modality curation evidence.

    The card keeps data understanding separate from model quality. Privacy
    and provenance are explicit status fields because a modality adapter
    cannot infer consent, ownership, retention, or access policy from bytes.
    """

    modality: str
    dataset_hash: str
    sample_count: int
    curation: Mapping[str, Any]
    intended_use: str
    limitations: tuple[str, ...]
    provenance: Mapping[str, str]
    privacy: Mapping[str, str]
    manifest: data_contracts.ModalityManifest | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        """Reject a card that is detached from its curation report."""
        if not self.modality.strip():
            raise ValueError("modality must not be empty")
        if not self.dataset_hash.strip():
            raise ValueError("dataset_hash must not be empty")
        if isinstance(self.sample_count, bool) or self.sample_count < 1:
            raise ValueError("sample_count must be positive")
        if not self.intended_use.strip():
            raise ValueError("intended_use must not be empty")
        if not self.limitations or any(
            not item.strip() for item in self.limitations
        ):
            raise ValueError("limitations must contain non-empty values")
        try:
            normalized = json.loads(
                json.dumps(dict(self.curation), allow_nan=False, sort_keys=True)
            )
        except (TypeError, ValueError) as error:
            raise ValueError("curation must be JSON-native") from error
        if not isinstance(normalized, dict):
            raise ValueError("curation must serialize to an object")
        object.__setattr__(self, "curation", normalized)
        if self.curation.get("dataset_hash") != self.dataset_hash:
            raise ValueError("curation report does not match dataset_hash")
        if self.curation.get("sample_count") != self.sample_count:
            raise ValueError("curation report does not match sample_count")
        for name, values in (
            ("provenance", self.provenance),
            ("privacy", self.privacy),
        ):
            if not values or any(
                not isinstance(key, str)
                or not isinstance(value, str)
                or not key.strip()
                or not value.strip()
                for key, value in values.items()
            ):
                raise ValueError(f"{name} must contain non-empty values")
        if self.manifest is not None:
            if not isinstance(self.manifest, data_contracts.ModalityManifest):
                raise TypeError("manifest must be ModalityManifest")
            if self.manifest.content_hash != self.dataset_hash:
                raise ValueError("manifest does not match dataset_hash")
            if self.manifest.sample_count != self.sample_count:
                raise ValueError("manifest does not match sample_count")
            if self.manifest.modality != self.modality:
                raise ValueError("manifest does not match modality")

    @classmethod
    def from_report(
        cls,
        modality: str,
        report: Mapping[str, Any],
        limitations: Iterable[str],
        manifest: data_contracts.ModalityManifest | None = None,
    ) -> Self:
        """Build a card without inventing missing governance decisions."""
        dataset_hash_value = report.get("dataset_hash")
        sample_count_value = report.get("sample_count")
        if not isinstance(dataset_hash_value, str):
            raise ValueError("curation report must contain dataset_hash")
        if not isinstance(sample_count_value, int):
            raise ValueError("curation report must contain sample_count")
        if manifest is None:
            manifest = data_contracts.ModalityManifest(
                dataset_id=f"{modality}-{dataset_hash_value[:16]}",
                snapshot_id=dataset_hash_value,
                modality=modality,
                task="classification",
                sample_count=sample_count_value,
                content_hash=dataset_hash_value,
                findings=tuple(
                    data_contracts.CurationFinding.from_dict(finding)
                    for finding in (
                        *report.get("quality_findings", ()),
                        *report.get("risk_findings", ()),
                    )
                ),
            )
        if manifest.content_hash != dataset_hash_value:
            raise ValueError("manifest does not match dataset_hash")
        if manifest.sample_count != sample_count_value:
            raise ValueError("manifest does not match sample_count")
        if manifest.modality != modality:
            raise ValueError("manifest does not match modality")
        declared_provenance = {
            "status": "declared",
            "manifest": "source, lineage, and ownership are in manifest",
        }
        declared_privacy = {
            "status": "declared",
            "manifest": "privacy, consent, and retention are in manifest",
        }
        if (
            not manifest.source.owner
            and not manifest.source.acquisition_method
            and manifest.privacy.classification == "unclassified"
        ):
            declared_provenance = {
                "status": "adapter_source_metadata_not_declared",
                "required_action": (
                    "record source ownership and lineage before promotion"
                ),
            }
            declared_privacy = {
                "status": "not_assessed",
                "required_action": (
                    "record consent, PII classification, retention, and "
                    "access policy before promotion"
                ),
            }
        return cls(
            modality=modality,
            dataset_hash=dataset_hash_value,
            sample_count=sample_count_value,
            curation=dict(report),
            intended_use="bounded lifecycle validation and baseline comparison",
            limitations=tuple(limitations),
            provenance=declared_provenance,
            privacy=declared_privacy,
            manifest=manifest,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return deterministic JSON-native dataset-card evidence."""
        return {
            "schema_version": self.schema_version,
            "modality": self.modality,
            "dataset_hash": self.dataset_hash,
            "sample_count": self.sample_count,
            "curation": dict(self.curation),
            "intended_use": self.intended_use,
            "limitations": list(self.limitations),
            "provenance": dict(sorted(self.provenance.items())),
            "privacy": dict(sorted(self.privacy.items())),
            "manifest": (
                self.manifest.to_dict() if self.manifest is not None else None
            ),
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


def conflicting_label_groups(
    items: Iterable[tuple[str, str, Any]],
) -> tuple[dict[str, object], ...]:
    """Return duplicate identities carrying more than one known label.

    Each item contains an identity, source, and optional label. Unlabeled
    samples do not create a conflict by themselves; disagreement between two
    known labels does.
    """
    grouped: dict[str, dict[str, Any]] = {}
    for identity, source, label in items:
        if label is None:
            continue
        bucket = next(iter(label_counts((label,))))
        record = grouped.setdefault(
            identity, {"sources": set(), "labels": set()}
        )
        record["sources"].add(source)
        record["labels"].add(bucket)
    conflicts: list[dict[str, object]] = []
    for identity, record in grouped.items():
        labels = record["labels"]
        if len(labels) > 1:
            conflicts.append(
                {
                    "identity": identity,
                    "sources": sorted(record["sources"]),
                    "labels": sorted(labels),
                }
            )
    return tuple(
        sorted(conflicts, key=lambda conflict: str(conflict["identity"]))
    )


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
    "ModalityDatasetCard",
    "build_finding",
    "conflicting_label_groups",
    "dataset_hash",
    "label_counts",
]
