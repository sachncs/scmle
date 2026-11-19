"""Quality and risk findings of a document dataset."""

from collections import defaultdict
from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any
from typing import TypedDict

from scmle.contracts import data
from scmle.data.modalities.document.elements import Layout
from scmle.data.modalities.document.elements import pii
from scmle.data.modalities.document.elements import Profile
from scmle.data.modalities.document.sample import bucketize
from scmle.data.modalities.document.sample import Sample


def flaw(
    finding_id: str,
    severity: data.FindingSeverity,
    scope: dict[str, Any],
    evidence: dict[str, Any],
    remediation: str,
    blocking: bool,
) -> data.Finding:
    """Creates a document-specific finding with one stable rule.

    Args:
        finding_id: Identifier of the finding.
        severity: Severity of the finding.
        scope: Description of what the finding affects.
        evidence: Evidence supporting the finding.
        remediation: Action that would resolve the finding.
        blocking: Whether the finding blocks release until resolved.

    Returns:
        An open finding with confidence 1.0 and owner ``data-curation``.
    """
    return data.Finding(
        identifier=finding_id,
        rule="document-curation.v1",
        severity=severity,
        confidence=1.0,
        scope=scope,
        evidence=evidence,
        remediation=remediation,
        owner="data-curation",
        reviewer=None,
        disposition=data.Disposition.OPEN,
        blocking=blocking,
    )


def legibility(
    samples: Sequence[Sample], layouts: Sequence[Layout]
) -> list[data.Finding]:
    """Builds the finding on layouts that carry no OCR confidence.

    Args:
        samples: The samples of a dataset.
        layouts: The layouts that samples carry.

    Returns:
        One warning when a layout has no OCR confidence; otherwise nothing.
    """
    findings: list[data.Finding] = []
    missing_ocr_confidence = sum(
        layout.confidence is None for layout in layouts
    )
    if missing_ocr_confidence:
        findings.append(
            flaw(
                "quality.document-ocr-confidence-missing",
                data.FindingSeverity.WARNING,
                {
                    "document_ids": [
                        sample.identifier
                        for sample in samples
                        if sample.layout is not None
                    ]
                },
                {"sample_count": missing_ocr_confidence},
                "Record OCR confidence or an explicit reason it is "
                "unavailable before making extraction-quality claims.",
                blocking=False,
            )
        )
    return findings


def traceability(samples: Sequence[Sample]) -> list[data.Finding]:
    """Builds the finding on samples with only half of a source identity.

    Args:
        samples: The samples of a dataset.

    Returns:
        One warning when a sample has a source URI without a source digest,
        or the reverse; otherwise nothing.
    """
    findings: list[data.Finding] = []
    partially_traceable = tuple(
        sample
        for sample in samples
        if bool(sample.uri.strip()) != (sample.digest is not None)
    )
    if partially_traceable:
        findings.append(
            flaw(
                "quality.document-source-traceability-incomplete",
                data.FindingSeverity.WARNING,
                {
                    "document_ids": [
                        sample.identifier for sample in partially_traceable
                    ]
                },
                {"sample_count": len(partially_traceable)},
                "Provide both the source URI and raw-source digest before "
                "claiming citation traceability.",
                blocking=False,
            )
        )
    return findings


class Layering(TypedDict):
    """The layout and OCR fields of a document `Curation`.

    Attributes:
        layout_providers: Sorted names of the layout providers.
        layout_element_count: Total number of layout elements.
        tables: Total number of table elements.
        ocr_confidence_min: Smallest OCR confidence, or None.
        ocr_confidence_max: Largest OCR confidence, or None.
    """

    layout_providers: tuple[str, ...]
    layout_element_count: int
    tables: int
    ocr_confidence_min: float | None
    ocr_confidence_max: float | None


def layering(layouts: Sequence[Layout]) -> Layering:
    """Summarises the layout and OCR evidence of a dataset.

    Args:
        layouts: The layouts that samples carry.

    Returns:
        The `Curation` fields ``layout_providers``,
        ``layout_element_count``, ``tables``,
        ``ocr_confidence_min`` and ``ocr_confidence_max``.
    """
    ocr_confidences = [
        layout.confidence for layout in layouts if layout.confidence is not None
    ]
    return {
        "layout_providers": tuple(
            sorted({layout.provider for layout in layouts})
        ),
        "layout_element_count": sum(len(layout.elements) for layout in layouts),
        "tables": sum(layout.tables for layout in layouts),
        "ocr_confidence_min": (
            min(ocr_confidences) if ocr_confidences else None
        ),
        "ocr_confidence_max": (
            max(ocr_confidences) if ocr_confidences else None
        ),
    }


class Review:
    """Sensitive-content, quality and risk review of a document dataset.

    Attributes:
        samples: The samples, in caller order.
        content_hash: Order-independent content hash of the dataset.
    """

    __slots__ = ()

    samples: tuple[Sample, ...]
    digest: str

    def sensitivity(self) -> tuple[dict[str, int], list[str]]:
        """Counts sensitive-value matches across the corpus.

        Returns:
            The match count of each sensitive-value name, and the ids of the
            documents that have any match, in sample order.
        """
        sensitive_counts: defaultdict[str, int] = defaultdict(int)
        sensitive_document_ids: list[str] = []
        for sample in self.samples:
            matches = pii(sample.content)
            if matches:
                sensitive_document_ids.append(sample.identifier)
                for name, count in matches.items():
                    sensitive_counts[name] += count
        return sensitive_counts, sensitive_document_ids

    def inspect(
        self,
        profiles: tuple[Profile, ...],
        duplicate_groups: tuple[tuple[str, ...], ...],
        layouts: list[Layout],
        max_characters: int,
        max_tokens: int,
    ) -> list[data.Finding]:
        """Builds the data-quality findings of the corpus.

        Args:
            profiles: One profile per sample.
            duplicate_groups: Groups of document ids with identical content.
            layouts: The layouts that samples carry.
            max_characters: Maximum characters inspected per sample.
            max_tokens: Maximum tokens retained per sample.

        Returns:
            The findings for missing OCR confidence, duplicate content,
            truncated profiles, unknown language and incomplete source
            traceability, in that order.
        """
        findings: list[data.Finding] = []
        findings.extend(legibility(self.samples, layouts))

        if duplicate_groups:
            findings.append(
                flaw(
                    "quality.duplicate-document-content",
                    data.FindingSeverity.WARNING,
                    {"groups": [list(group) for group in duplicate_groups]},
                    {"group_count": len(duplicate_groups)},
                    "Confirm duplicates are intentional or deduplicate "
                    "before training.",
                    blocking=False,
                )
            )

        truncated_ids = tuple(
            profile.document_id for profile in profiles if profile.truncated
        )
        if truncated_ids:
            findings.append(
                flaw(
                    "quality.profile-truncated",
                    data.FindingSeverity.INFO,
                    {"document_ids": list(truncated_ids)},
                    {
                        "max_characters": max_characters,
                        "max_tokens": max_tokens,
                    },
                    "Increase profiling bounds or run a full-content audit "
                    "before release.",
                    blocking=False,
                )
            )

        unknown_ids = tuple(
            profile.document_id
            for profile in profiles
            if profile.language == "unknown"
        )
        if unknown_ids:
            findings.append(
                flaw(
                    "quality.unknown-language",
                    data.FindingSeverity.WARNING,
                    {"document_ids": list(unknown_ids)},
                    {"count": len(unknown_ids)},
                    "Confirm language coverage and configure an appropriate "
                    "tokenizer.",
                    blocking=False,
                )
            )

        findings.extend(traceability(self.samples))
        return findings

    def assess(
        self,
        label_counts: Mapping[str, int],
        sensitive_counts: Mapping[str, int],
        sensitive_document_ids: list[str],
    ) -> list[data.Finding]:
        """Builds the risk findings of the corpus.

        Identical content is grouped by its hash over the samples themselves,
        so two samples that repeat one document id are still compared
        (MM-16).

        Args:
            label_counts: Samples per label bucket.
            sensitive_counts: Match count of each sensitive-value name.
            sensitive_document_ids: Ids of the documents with a match.

        Returns:
            The findings for sensitive content, conflicting labels among
            samples of identical content and a mix of labelled and
            unlabelled samples, in that order.
        """
        findings: list[data.Finding] = []
        copies: dict[str, list[Sample]] = {}
        for sample in self.samples:
            copies.setdefault(sample.content_hash, []).append(sample)
        if sensitive_counts:
            findings.append(
                flaw(
                    "risk.document-sensitive-content",
                    data.FindingSeverity.ERROR,
                    {"document_ids": sorted(sensitive_document_ids)},
                    {"match_counts": dict(sorted(sensitive_counts.items()))},
                    "Redact or explicitly govern detected sensitive content "
                    "before training or promotion.",
                    blocking=True,
                )
            )
        conflicts = sorted(
            (
                tuple(sorted(sample.identifier for sample in members)),
                {bucketize(sample.label) for sample in members},
            )
            for members in copies.values()
            if len(members) > 1
        )
        for group, labels in conflicts:
            if len(labels) > 1:
                findings.append(
                    flaw(
                        "risk.conflicting-duplicate-labels",
                        data.FindingSeverity.ERROR,
                        {"document_ids": list(group)},
                        {"labels": sorted(labels)},
                        "Resolve the label conflict or explicitly exclude "
                        "the affected content.",
                        blocking=True,
                    )
                )
        mixed_label_state = (
            "unlabeled" in label_counts and len(label_counts) > 1
        )
        if mixed_label_state:
            findings.append(
                flaw(
                    "risk.mixed-label-state",
                    data.FindingSeverity.WARNING,
                    {"dataset": self.digest},
                    {"label_counts": dict(label_counts)},
                    "Declare whether unlabeled samples are valid for the "
                    "intended task.",
                    blocking=False,
                )
            )
        return findings
