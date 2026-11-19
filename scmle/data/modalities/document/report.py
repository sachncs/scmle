"""The curation report of a document dataset."""

import dataclasses
from typing import Any
from typing import ClassVar

from scmle.contracts import data
from scmle.contracts import records
from scmle.data.modalities.document.elements import Profile

CURATION = records.nest(
    {
        "sample_count": "samples",
        "document_count": "documents",
        "duplicate_groups": "duplicates",
        "max_characters": "characters",
        "max_tokens": "tokens",
        "table_element_count": "tables",
        "source_traceability_coverage": "traceability",
        "citation_count": "citations",
        "release_ready": "ready",
        "unresolved_blocking_findings": "blockers",
    },
    groups={
        "digests": {"dataset_hash": "dataset"},
        "counts": {
            "language_counts": "languages",
            "format_counts": "formats",
            "label_counts": "labels",
            "sensitive_match_counts": "sensitive",
        },
        "findings": {"quality_findings": "quality", "risk_findings": "risk"},
        "layout": {
            "layout_coverage": "coverage",
            "layout_providers": "providers",
            "layout_element_count": "elements",
        },
        "ocr": {"ocr_confidence_min": "low", "ocr_confidence_max": "high"},
    },
)
"""Version 2 of the document curation report: single-word keys."""


@dataclasses.dataclass(frozen=True, slots=True)
class Curation:
    """Reviewable data-understanding evidence for a document dataset.

    The report describes the observed corpus before any modelling decision.
    It intentionally keeps findings separate from model quality: a document
    corpus can be useful for retrieval or unsupervised work even when labels
    are absent, while conflicting labels on identical content remain a risk
    that must be reviewed.

    Attributes:
        dataset_hash: Order-independent digest of the dataset.
        samples: Number of samples.
        documents: Number of distinct document identities.
        profiles: One profile per sample, in sample order.
        duplicates: Groups of document ids with identical content.
        language_counts: Samples per language code, sorted by code.
        format_counts: Samples per document format, sorted by name.
        label_counts: Samples per label bucket (``unlabeled`` or
            ``<type>:<label>``), sorted by bucket.
        quality_findings: Findings about data quality.
        risk_findings: Findings about risk such as sensitive content and
            conflicting labels.
        characters: Character bound used for profiling.
        tokens: Token bound used for profiling.
        layout_coverage: Share of samples with layout evidence.
        layout_providers: Sorted names of the layout providers.
        layout_element_count: Total number of layout elements.
        tables: Total number of table elements.
        ocr_confidence_min: Smallest OCR confidence, or None.
        ocr_confidence_max: Largest OCR confidence, or None.
        sensitive_match_counts: Matches per sensitive-value pattern.
        traceability: Share of samples with both a source URI
            and a source digest.
        citations: Number of citations over the traceable samples.
        migrations: Version steps of the stored report; version 2 writes
            single-word keys (`digests.dataset`, `samples`, `documents`,
            `duplicates`, `counts.{languages, formats, labels, sensitive}`,
            `findings.{quality, risk}`, `characters`, `tokens`,
            `layout.{coverage, providers, elements}`, `tables`,
            `ocr.{low, high}`, `traceability`, `citations`, `ready` and
            `blockers`); the fields keep their names.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (CURATION,)

    dataset_hash: str
    samples: int
    documents: int
    profiles: tuple[Profile, ...]
    duplicates: tuple[tuple[str, ...], ...]
    language_counts: dict[str, int]
    format_counts: dict[str, int]
    label_counts: dict[str, int]
    quality_findings: tuple[data.Finding, ...]
    risk_findings: tuple[data.Finding, ...]
    characters: int
    tokens: int
    layout_coverage: float = 0.0
    layout_providers: tuple[str, ...] = ()
    layout_element_count: int = 0
    tables: int = 0
    ocr_confidence_min: float | None = None
    ocr_confidence_max: float | None = None
    sensitive_match_counts: dict[str, int] = dataclasses.field(
        default_factory=dict
    )
    traceability: float = 0.0
    citations: int = 0

    @property
    def blockers(
        self,
    ) -> tuple[data.Finding, ...]:
        """The blocking findings without a safe review disposition.

        A disposition is safe when it is resolved, waived or a false
        positive. Quality findings come before risk findings.
        """
        safe_dispositions = {
            data.Disposition.RESOLVED,
            data.Disposition.WAIVED,
            data.Disposition.FALSE_POSITIVE,
        }
        return tuple(
            finding
            for finding in (*self.quality_findings, *self.risk_findings)
            if finding.blocking and finding.disposition not in safe_dispositions
        )

    @property
    def release(self) -> bool:
        """Whether no unresolved blocking finding remains."""
        return not self.blockers

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native evidence for a dataset card or run artifact.

        Returns:
            A dict with every report field (profiles and findings as nested
            dicts) in the version-2 layout, the schema version, the derived
            `ready` flag and the ids of the unresolved blocking findings
            under `blockers`.
        """
        layout = {
            "dataset_hash": self.dataset_hash,
            "sample_count": self.samples,
            "document_count": self.documents,
            "profiles": [
                dataclasses.asdict(profile) for profile in self.profiles
            ],
            "duplicate_groups": [list(group) for group in self.duplicates],
            "language_counts": dict(self.language_counts),
            "format_counts": dict(self.format_counts),
            "label_counts": dict(self.label_counts),
            "quality_findings": [
                finding.to_dict() for finding in self.quality_findings
            ],
            "risk_findings": [
                finding.to_dict() for finding in self.risk_findings
            ],
            "max_characters": self.characters,
            "max_tokens": self.tokens,
            "layout_coverage": self.layout_coverage,
            "layout_providers": list(self.layout_providers),
            "layout_element_count": self.layout_element_count,
            "table_element_count": self.tables,
            "ocr_confidence_min": self.ocr_confidence_min,
            "ocr_confidence_max": self.ocr_confidence_max,
            "sensitive_match_counts": dict(self.sensitive_match_counts),
            "source_traceability_coverage": self.traceability,
            "citation_count": self.citations,
            "release_ready": self.release,
            "unresolved_blocking_findings": [
                finding.identifier for finding in self.blockers
            ],
        }
        return {
            records.KEY: records.ORIGIN + len(self.migrations),
            **records.upgrade(self.migrations, layout, records.ORIGIN),
        }
