"""Document data contracts and dependency-light ingestion.

This module deliberately keeps parsing small and explicit. UTF-8 text
and Markdown are supported by the standard library. PDF support is an
optional injected reader so the core package never imports a document parser
implicitly or silently changes its dependency surface.
"""

from collections import defaultdict
from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import enum
import hashlib
import math
import pathlib
import random
import re
from typing import Any

from scmle.contracts import data as data_contracts
from scmle.contracts import errors


class DocumentFormat(enum.StrEnum):
    """Document formats understood by :class:`DocumentReader`."""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"


class DocumentValidationError(errors.DataError):
    """A document violates an ingestion or contract invariant."""

    category = "document_validation"

    def __init__(
        self,
        message: str,
        code: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a structured document validation failure.

        Args:
            message: Human-readable description of the failure.
            code: Machine-readable failure code, stored in the context.
            details: Extra context entries merged into the error context.
        """
        context = {"code": code}
        if details:
            context.update(details)
        super().__init__(message, context)


class DocumentDependencyError(errors.DataError):
    """An optional parser is required but is not available."""

    category = "document_dependency"

    def __init__(
        self,
        message: str,
        dependency: str,
        extra: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Initializes a structured optional-dependency failure.

        Args:
            message: Human-readable description of the failure.
            dependency: Name of the missing parser dependency.
            extra: Name of the package extra that provides it.
            details: Extra context entries merged into the error context.
        """
        context = {
            "code": "missing_dependency",
            "dependency": dependency,
            "extra": extra,
        }
        if details:
            context.update(details)
        super().__init__(message, context)


SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(
        r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE
    ),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})"
        r"[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def sensitive_matches(content: str) -> dict[str, int]:
    """Counts supported sensitive-value patterns without returning values.

    Args:
        content: Text to scan for email addresses, phone numbers and social
            security numbers.

    Returns:
        A dict from pattern name (`email`, `phone`, `ssn`) to the number of
        matches, sorted by name. Patterns with no match are omitted.
    """
    return {
        name: len(pattern.findall(content))
        for name, pattern in sorted(SENSITIVE_PATTERNS.items())
        if pattern.search(content) is not None
    }


def redact_sensitive_text(content: str) -> str:
    """Replace supported sensitive values with type-only placeholders.

    Args:
        content: Text to redact.

    Returns:
        The text with each match replaced by ``[REDACTED_<NAME>]``, for
        example ``[REDACTED_EMAIL]``.
    """
    redacted = content
    for name, pattern in sorted(SENSITIVE_PATTERNS.items()):
        redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
    return redacted


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentSpan:
    """A character range with simple document provenance.

    Attributes:
        kind: Either ``page`` or ``section``.
        start: Offset of the first character; not negative.
        end: Offset after the last character; greater than ``start``.
        label: Human-readable label, such as a heading text.
        page_number: One-based page number; required for page spans.
    """

    kind: str
    start: int
    end: int
    label: str = ""
    page_number: int | None = None

    def __post_init__(self) -> None:
        """Validates the span and its provenance kind.

        Raises:
            ValueError: If kind is not ``page`` or ``section``, the range is
                empty or starts below zero, or a page span lacks a positive
                page number.
        """
        if self.kind not in {"page", "section"}:
            raise ValueError("kind must be 'page' or 'section'")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must have a positive character range")
        if self.kind == "page" and (
            self.page_number is None or self.page_number < 1
        ):
            raise ValueError("page spans require a positive page_number")


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentCitation:
    """A source-backed locator for a document character span.

    Attributes:
        document_id: Non-empty identity of the document.
        source_uri: Non-empty URI of the raw source.
        source_digest: Lowercase SHA-256 hex digest of the raw source.
        start: Offset of the first character; not negative.
        end: Offset after the last character; greater than ``start``.
        locator: Non-empty human-readable locator.
        page_number: Optional one-based page number.
    """

    document_id: str
    source_uri: str
    source_digest: str
    start: int
    end: int
    locator: str
    page_number: int | None = None

    def __post_init__(self) -> None:
        """Validates the source identity and character locator.

        Raises:
            ValueError: If document_id, source_uri or locator is blank,
                source_digest is not a lowercase SHA-256 digest, the range is
                empty or starts below zero, or page_number is below one.
        """
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.source_uri.strip():
            raise ValueError("source_uri must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("citation must have a positive character range")
        if not self.locator.strip():
            raise ValueError("locator must not be empty")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be positive")

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native citation evidence.

        Returns:
            A dict of the citation fields.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentElement:
    """One extracted layout, OCR, or table element with source offsets.

    Attributes:
        kind: One of ``text``, ``heading``, ``table``, ``image`` or ``ocr``.
        start: Offset of the first character; not negative.
        end: Offset after the last character; greater than ``start``.
        page_number: One-based page number.
        confidence: Optional confidence in [0, 1].
        table_rows: Number of table rows; required for tables.
        table_columns: Number of table columns; required for tables.
    """

    kind: str
    start: int
    end: int
    page_number: int
    confidence: float | None = None
    table_rows: int | None = None
    table_columns: int | None = None

    def __post_init__(self) -> None:
        """Validates element type, offsets, page, confidence, and table shape.

        Raises:
            ValueError: If kind is unsupported, the range is empty or starts
                below zero, page_number is below one, the confidence is a
                bool, not finite or outside [0, 1], a table lacks row or
                column counts, or a given count is below one.
        """
        if self.kind not in {"text", "heading", "table", "image", "ocr"}:
            raise ValueError("unsupported document element kind")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("document element must have a positive range")
        if self.page_number < 1:
            raise ValueError("document elements require a positive page_number")
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise ValueError("document element confidence must be in [0, 1]")
        if self.kind == "table" and (
            self.table_rows is None or self.table_columns is None
        ):
            raise ValueError("table elements require row and column counts")
        if self.table_rows is not None and self.table_rows < 1:
            raise ValueError("table_rows must be positive")
        if self.table_columns is not None and self.table_columns < 1:
            raise ValueError("table_columns must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentLayout:
    """Caller-owned extraction provenance for OCR, layout, and tables.

    Attributes:
        source: Non-blank identifier of the extracted source.
        provider: Non-blank name of the extraction provider.
        provider_version: Non-blank version of the provider.
        elements: Extracted elements.
        ocr_confidence: Optional overall OCR confidence in [0, 1].
        source_digest: Optional SHA-256 hex digest of the raw source the
            layout was extracted from.
    """

    source: str
    provider: str
    provider_version: str
    elements: tuple[DocumentElement, ...]
    ocr_confidence: float | None = None
    source_digest: str | None = None

    def __post_init__(self) -> None:
        """Validates the extraction identity and optional OCR confidence.

        Raises:
            ValueError: If source, provider or provider_version is blank, the
                OCR confidence is a bool, not finite or outside [0, 1], or
                source_digest is not a lowercase SHA-256 digest.
            TypeError: If elements is not a tuple of DocumentElement.
        """
        for field_name, value in (
            ("source", self.source),
            ("provider", self.provider),
            ("provider_version", self.provider_version),
        ):
            if not value.strip():
                raise ValueError(
                    f"document layout {field_name} must not be empty"
                )
        if not isinstance(self.elements, tuple):
            raise TypeError("document layout elements must be a tuple")
        if not all(
            isinstance(element, DocumentElement) for element in self.elements
        ):
            raise TypeError("elements must contain DocumentElement values")
        if self.ocr_confidence is not None and (
            isinstance(self.ocr_confidence, bool)
            or not math.isfinite(self.ocr_confidence)
            or not 0.0 <= self.ocr_confidence <= 1.0
        ):
            raise ValueError("OCR confidence must be in [0, 1]")
        if self.source_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.source_digest
        ):
            raise ValueError("source_digest must be a lowercase SHA-256 digest")

    @property
    def table_count(self) -> int:
        """The number of extracted table elements."""
        return sum(element.kind == "table" for element in self.elements)

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native extraction evidence.

        Returns:
            A dict of the layout fields, with elements as nested dicts.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentProfile:
    """Bounded lexical profile for one document.

    Attributes:
        document_id: Non-blank identity of the document.
        character_count: Number of characters inspected.
        token_count: Number of word tokens, capped at the token bound.
        language: Language code from a script-ratio heuristic, or
            ``unknown``.
        truncated: Whether the character or token bound cut the document.
    """

    document_id: str
    character_count: int
    token_count: int
    language: str
    truncated: bool

    def __post_init__(self) -> None:
        """Validates profile counts and identity.

        Raises:
            ValueError: If document_id or language is blank or a count is
                negative.
        """
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if self.character_count < 0 or self.token_count < 0:
            raise ValueError("profile counts must not be negative")
        if not self.language.strip():
            raise ValueError("language must not be empty")


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentCurationReport:
    """Reviewable data-understanding evidence for a document dataset.

    The report describes the observed corpus before any modelling decision.
    It intentionally keeps findings separate from model quality: a document
    corpus can be useful for retrieval or unsupervised work even when labels
    are absent, while conflicting labels on identical content remain a risk
    that must be reviewed.

    Attributes:
        dataset_hash: Order-independent digest of the dataset.
        sample_count: Number of samples.
        document_count: Number of distinct document identities.
        profiles: One profile per sample, in sample order.
        duplicate_groups: Groups of document ids with identical content.
        language_counts: Samples per language code, sorted by code.
        format_counts: Samples per document format, sorted by name.
        label_counts: Samples per label bucket (``unlabeled`` or
            ``<type>:<label>``), sorted by bucket.
        quality_findings: Findings about data quality.
        risk_findings: Findings about risk such as sensitive content and
            conflicting labels.
        max_characters: Character bound used for profiling.
        max_tokens: Token bound used for profiling.
        layout_coverage: Share of samples with layout evidence.
        layout_providers: Sorted names of the layout providers.
        layout_element_count: Total number of layout elements.
        table_element_count: Total number of table elements.
        ocr_confidence_min: Smallest OCR confidence, or None.
        ocr_confidence_max: Largest OCR confidence, or None.
        sensitive_match_counts: Matches per sensitive-value pattern.
        source_traceability_coverage: Share of samples with both a source URI
            and a source digest.
        citation_count: Number of citations over the traceable samples.
    """

    dataset_hash: str
    sample_count: int
    document_count: int
    profiles: tuple[DocumentProfile, ...]
    duplicate_groups: tuple[tuple[str, ...], ...]
    language_counts: dict[str, int]
    format_counts: dict[str, int]
    label_counts: dict[str, int]
    quality_findings: tuple[data_contracts.CurationFinding, ...]
    risk_findings: tuple[data_contracts.CurationFinding, ...]
    max_characters: int
    max_tokens: int
    layout_coverage: float = 0.0
    layout_providers: tuple[str, ...] = ()
    layout_element_count: int = 0
    table_element_count: int = 0
    ocr_confidence_min: float | None = None
    ocr_confidence_max: float | None = None
    sensitive_match_counts: dict[str, int] = dataclasses.field(
        default_factory=dict
    )
    source_traceability_coverage: float = 0.0
    citation_count: int = 0

    @property
    def unresolved_blocking_findings(
        self,
    ) -> tuple[data_contracts.CurationFinding, ...]:
        """The blocking findings without a safe review disposition.

        A disposition is safe when it is resolved, waived or a false
        positive. Quality findings come before risk findings.
        """
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
        """Whether no unresolved blocking finding remains."""
        return not self.unresolved_blocking_findings

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-native evidence for a dataset card or run artifact.

        Returns:
            A dict with every report field (profiles and findings as nested
            dicts), schema version 1, the derived `release_ready` flag and
            the ids of the unresolved blocking findings.
        """
        return {
            "schema_version": 1,
            "dataset_hash": self.dataset_hash,
            "sample_count": self.sample_count,
            "document_count": self.document_count,
            "profiles": [
                dataclasses.asdict(profile) for profile in self.profiles
            ],
            "duplicate_groups": [
                list(group) for group in self.duplicate_groups
            ],
            "language_counts": dict(self.language_counts),
            "format_counts": dict(self.format_counts),
            "label_counts": dict(self.label_counts),
            "quality_findings": [
                finding.to_dict() for finding in self.quality_findings
            ],
            "risk_findings": [
                finding.to_dict() for finding in self.risk_findings
            ],
            "max_characters": self.max_characters,
            "max_tokens": self.max_tokens,
            "layout_coverage": self.layout_coverage,
            "layout_providers": list(self.layout_providers),
            "layout_element_count": self.layout_element_count,
            "table_element_count": self.table_element_count,
            "ocr_confidence_min": self.ocr_confidence_min,
            "ocr_confidence_max": self.ocr_confidence_max,
            "sensitive_match_counts": dict(self.sensitive_match_counts),
            "source_traceability_coverage": self.source_traceability_coverage,
            "citation_count": self.citation_count,
            "release_ready": self.release_ready,
            "unresolved_blocking_findings": [
                finding.finding_id
                for finding in self.unresolved_blocking_findings
            ],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentSample:
    """Immutable decoded document content and its identity.

    Attributes:
        document_id: Non-blank document identity.
        content: Non-blank decoded text; pages are separated by form feeds.
        format: Document format; a string is converted to the enum.
        source_uri: URI of the raw source; may be empty.
        encoding: Source encoding; only UTF-8 is accepted.
        source_digest: Optional lowercase SHA-256 hex digest of the raw
            source bytes.
        label: Optional task label.
        provenance: Page and section spans; derived from the content when
            empty.
        layout: Optional layout evidence; no element may extend past the
            content.
        content_hash: SHA-256 hex digest of the UTF-8 content; set on
            construction.
        size_bytes: Length of the UTF-8 content in bytes; set on
            construction.
    """

    document_id: str
    content: str
    format: DocumentFormat = DocumentFormat.TEXT
    source_uri: str = ""
    encoding: str = "utf-8"
    source_digest: str | None = None
    label: str | int | float | bool | None = None
    provenance: tuple[DocumentSpan, ...] = ()
    layout: DocumentLayout | None = None
    content_hash: str = dataclasses.field(init=False)
    size_bytes: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates content and derives immutable identity fields.

        Raises:
            DocumentValidationError: If the document id or content is blank,
                the encoding is not UTF-8, the source digest is not a
                lowercase SHA-256 digest, a layout element extends past the
                content, or the layout's source digest differs from the
                sample's.
        """
        if not self.document_id.strip():
            raise DocumentValidationError(
                "document_id must not be empty",
                "empty_document_id",
            )
        if not self.content.strip():
            raise DocumentValidationError(
                "document content must not be empty",
                "empty_content",
                {"document_id": self.document_id},
            )
        if self.encoding.lower().replace("-", "") != "utf8":
            raise DocumentValidationError(
                "document encoding must be UTF-8",
                "unsupported_encoding",
                {"encoding": self.encoding},
            )
        if self.source_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.source_digest
        ):
            raise DocumentValidationError(
                "source_digest must be a lowercase SHA-256 digest",
                "invalid_source_digest",
                {"document_id": self.document_id},
            )
        if not isinstance(self.format, DocumentFormat):
            object.__setattr__(self, "format", DocumentFormat(self.format))
        normalized_provenance = tuple(self.provenance)
        if not normalized_provenance:
            normalized_provenance = provenance_for_content(self.content)
        object.__setattr__(self, "provenance", normalized_provenance)
        if self.layout is not None and any(
            element.end > len(self.content) for element in self.layout.elements
        ):
            raise DocumentValidationError(
                "document layout element exceeds decoded content",
                "layout_out_of_range",
                {"document_id": self.document_id},
            )
        if (
            self.layout is not None
            and self.layout.source_digest is not None
            and self.source_digest is not None
            and self.layout.source_digest != self.source_digest
        ):
            raise DocumentValidationError(
                "document layout source digest does not match the document",
                "layout_source_mismatch",
                {"document_id": self.document_id},
            )
        content_bytes = self.content.encode("utf-8")
        object.__setattr__(
            self,
            "content_hash",
            hashlib.sha256(content_bytes).hexdigest(),
        )
        object.__setattr__(self, "size_bytes", len(content_bytes))

    def citations(self) -> tuple[DocumentCitation, ...]:
        """Returns source-backed citations for the document provenance spans.

        A citation is intentionally unavailable when the caller created an
        in-memory sample without a raw-source identity. The method never
        invents a URI or treats the decoded content digest as a raw-source
        digest.

        Returns:
            One citation per provenance span, or an empty tuple when the
            sample has no source URI or no source digest. The locator is the
            span label, or ``<kind>:<start>-<end>`` when the label is empty.
        """
        if not self.source_uri.strip() or self.source_digest is None:
            return ()
        return tuple(
            DocumentCitation(
                document_id=self.document_id,
                source_uri=self.source_uri,
                source_digest=self.source_digest,
                start=span.start,
                end=span.end,
                locator=span.label or f"{span.kind}:{span.start}-{span.end}",
                page_number=span.page_number,
            )
            for span in self.provenance
        )

    def profile(
        self,
        max_characters: int = 100_000,
        max_tokens: int = 20_000,
    ) -> DocumentProfile:
        """Builds a bounded lexical profile without external dependencies.

        Args:
            max_characters: Maximum characters inspected.
            max_tokens: Maximum tokens retained in the profile.

        Returns:
            A deterministic bounded profile. Tokens are word characters
            runs; the language comes from `language_heuristic` over the
            inspected characters.

        Raises:
            ValueError: If either bound is not positive.
        """
        if max_characters < 1 or max_tokens < 1:
            raise ValueError("profiling bounds must be positive")
        sample = self.content[:max_characters]
        tokens = re.findall(r"\b\w+\b", sample, flags=re.UNICODE)
        truncated = (
            len(self.content) > max_characters or len(tokens) > max_tokens
        )
        return DocumentProfile(
            document_id=self.document_id,
            character_count=len(sample),
            token_count=min(len(tokens), max_tokens),
            language=language_heuristic(sample),
            truncated=truncated,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentSplit:
    """Leakage-aware document-level split.

    Attributes:
        train: Samples assigned to training.
        validation: Samples assigned to validation.
        test: Samples assigned to testing.
    """

    train: tuple[DocumentSample, ...]
    validation: tuple[DocumentSample, ...]
    test: tuple[DocumentSample, ...]

    def document_ids(self) -> dict[str, tuple[str, ...]]:
        """Returns sorted document IDs assigned to each split.

        Returns:
            A dict from partition name (`train`, `validation`, `test`) to
            the sorted distinct document ids.
        """
        return {
            "train": tuple(
                sorted({sample.document_id for sample in self.train})
            ),
            "validation": tuple(
                sorted({sample.document_id for sample in self.validation})
            ),
            "test": tuple(sorted({sample.document_id for sample in self.test})),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentDataset:
    """Immutable collection of documents with identity and split operations.

    Attributes:
        samples: The samples, in caller order.
        content_hash: SHA-256 hex digest of the sorted document ids and
            content hashes, so independent of sample order; set on
            construction.
    """

    samples: tuple[DocumentSample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates the collection and derives order-independent identity.

        Raises:
            DocumentValidationError: If there are no samples.
            TypeError: If an element is not a DocumentSample.
        """
        normalized_samples = tuple(self.samples)
        if not normalized_samples:
            raise DocumentValidationError(
                "document dataset must contain at least one sample",
                "empty_dataset",
            )
        if any(
            not isinstance(sample, DocumentSample)
            for sample in normalized_samples
        ):
            raise TypeError("samples must contain only DocumentSample values")
        identities = sorted(
            f"{sample.document_id}\0{sample.content_hash}"
            for sample in normalized_samples
        )
        dataset_hash = hashlib.sha256(
            "\n".join(identities).encode()
        ).hexdigest()
        object.__setattr__(self, "samples", normalized_samples)
        object.__setattr__(self, "content_hash", dataset_hash)

    def duplicate_groups(self) -> tuple[tuple[str, ...], ...]:
        """Returns deterministic groups of samples with identical content.

        Returns:
            Groups of two or more document ids that share a content hash.
            Each group is sorted, and the groups are sorted.
        """
        groups: defaultdict[str, list[str]] = defaultdict(list)
        for sample in self.samples:
            groups[sample.content_hash].append(sample.document_id)
        duplicate_groups = [
            tuple(sorted(document_ids))
            for document_ids in groups.values()
            if len(document_ids) > 1
        ]
        return tuple(sorted(duplicate_groups))

    def profile(
        self,
        max_characters: int = 100_000,
        max_tokens: int = 20_000,
    ) -> tuple[DocumentProfile, ...]:
        """Profiles every sample using the same bounded limits.

        Args:
            max_characters: Maximum characters inspected per sample.
            max_tokens: Maximum tokens retained per sample.

        Returns:
            One profile per sample, in sample order. A bound that is not
            positive raises ValueError from `DocumentSample.profile`.
        """
        return tuple(
            sample.profile(max_characters, max_tokens)
            for sample in self.samples
        )

    def curate(
        self,
        max_characters: int = 100_000,
        max_tokens: int = 20_000,
    ) -> DocumentCurationReport:
        """Builds bounded, reviewable curation evidence for the corpus.

        Duplicate content is reported separately from conflicting labels. A
        duplicate is often expected for replicated sources, while identical
        content with different labels is a blocking risk for supervised work.
        The method does not silently remove samples or infer a modelling task.

        Args:
            max_characters: Maximum characters inspected per sample.
            max_tokens: Maximum tokens retained per sample.

        Returns:
            The curation report: profiles, duplicate groups, language,
            format and label counts, layout and OCR statistics, sensitive
            value matches, source traceability, and findings.

        Raises:
            ValueError: If either bound is not positive.
        """
        if max_characters < 1 or max_tokens < 1:
            raise ValueError("curation bounds must be positive")

        profiles = self.profile(max_characters, max_tokens)
        duplicate_groups = self.duplicate_groups()
        language_counts: dict[str, int] = defaultdict(int)
        format_counts: dict[str, int] = defaultdict(int)
        label_counts: dict[str, int] = defaultdict(int)
        sample_by_id = {sample.document_id: sample for sample in self.samples}

        def label_key(label: str | int | float | bool | None) -> str:
            """Returns a stable human-readable label bucket.

            Args:
                label: A sample label, possibly None.

            Returns:
                ``unlabeled`` for None, otherwise ``<type name>:<label>``.
            """
            if label is None:
                return "unlabeled"
            return f"{type(label).__name__}:{label}"

        for sample, profile in zip(self.samples, profiles, strict=True):
            language_counts[profile.language] += 1
            format_counts[sample.format.value] += 1
            label_counts[label_key(sample.label)] += 1

        quality_findings: list[data_contracts.CurationFinding] = []
        risk_findings: list[data_contracts.CurationFinding] = []
        layouts = [
            sample.layout
            for sample in self.samples
            if sample.layout is not None
        ]
        ocr_confidences = [
            layout.ocr_confidence
            for layout in layouts
            if layout.ocr_confidence is not None
        ]
        missing_ocr_confidence = sum(
            layout.ocr_confidence is None for layout in layouts
        )
        sensitive_counts: defaultdict[str, int] = defaultdict(int)
        sensitive_document_ids: list[str] = []
        for sample in self.samples:
            matches = sensitive_matches(sample.content)
            if matches:
                sensitive_document_ids.append(sample.document_id)
                for name, count in matches.items():
                    sensitive_counts[name] += count

        def create_finding(
            finding_id: str,
            severity: data_contracts.FindingSeverity,
            scope: dict[str, Any],
            evidence: dict[str, Any],
            remediation: str,
            blocking: bool,
        ) -> data_contracts.CurationFinding:
            """Creates a document-specific finding with one stable rule.

            Args:
                finding_id: Identifier of the finding.
                severity: Severity of the finding.
                scope: Description of what the finding affects.
                evidence: Evidence supporting the finding.
                remediation: Action that would resolve the finding.
                blocking: Whether the finding blocks release until resolved.

            Returns:
                An open finding with confidence 1.0 and owner
                ``data-curation``.
            """
            return data_contracts.CurationFinding(
                finding_id=finding_id,
                rule_version="document-curation.v1",
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

        if missing_ocr_confidence:
            quality_findings.append(
                create_finding(
                    "quality.document-ocr-confidence-missing",
                    data_contracts.FindingSeverity.WARNING,
                    {
                        "document_ids": [
                            sample.document_id
                            for sample in self.samples
                            if sample.layout is not None
                        ]
                    },
                    {"sample_count": missing_ocr_confidence},
                    "Record OCR confidence or an explicit reason it is "
                    "unavailable before making extraction-quality claims.",
                    False,
                )
            )

        if duplicate_groups:
            quality_findings.append(
                create_finding(
                    "quality.duplicate-document-content",
                    data_contracts.FindingSeverity.WARNING,
                    {"groups": [list(group) for group in duplicate_groups]},
                    {"group_count": len(duplicate_groups)},
                    "Confirm duplicates are intentional or deduplicate "
                    "before training.",
                    False,
                )
            )

        if sensitive_counts:
            risk_findings.append(
                create_finding(
                    "risk.document-sensitive-content",
                    data_contracts.FindingSeverity.ERROR,
                    {"document_ids": sorted(sensitive_document_ids)},
                    {"match_counts": dict(sorted(sensitive_counts.items()))},
                    "Redact or explicitly govern detected sensitive content "
                    "before training or promotion.",
                    True,
                )
            )

        truncated_ids = tuple(
            profile.document_id for profile in profiles if profile.truncated
        )
        if truncated_ids:
            quality_findings.append(
                create_finding(
                    "quality.profile-truncated",
                    data_contracts.FindingSeverity.INFO,
                    {"document_ids": list(truncated_ids)},
                    {
                        "max_characters": max_characters,
                        "max_tokens": max_tokens,
                    },
                    "Increase profiling bounds or run a full-content audit "
                    "before release.",
                    False,
                )
            )

        unknown_ids = tuple(
            profile.document_id
            for profile in profiles
            if profile.language == "unknown"
        )
        if unknown_ids:
            quality_findings.append(
                create_finding(
                    "quality.unknown-language",
                    data_contracts.FindingSeverity.WARNING,
                    {"document_ids": list(unknown_ids)},
                    {"count": len(unknown_ids)},
                    "Confirm language coverage and configure an appropriate "
                    "tokenizer.",
                    False,
                )
            )

        for group in duplicate_groups:
            labels = {
                label_key(sample_by_id[document_id].label)
                for document_id in group
            }
            if len(labels) > 1:
                risk_findings.append(
                    create_finding(
                        "risk.conflicting-duplicate-labels",
                        data_contracts.FindingSeverity.ERROR,
                        {"document_ids": list(group)},
                        {"labels": sorted(labels)},
                        "Resolve the label conflict or explicitly exclude "
                        "the affected content.",
                        True,
                    )
                )
        mixed_label_state = (
            "unlabeled" in label_counts and len(label_counts) > 1
        )
        if mixed_label_state:
            risk_findings.append(
                create_finding(
                    "risk.mixed-label-state",
                    data_contracts.FindingSeverity.WARNING,
                    {"dataset": self.content_hash},
                    {"label_counts": dict(label_counts)},
                    "Declare whether unlabeled samples are valid for the "
                    "intended task.",
                    False,
                )
            )

        traceable_samples = tuple(
            sample
            for sample in self.samples
            if sample.source_uri.strip() and sample.source_digest is not None
        )
        partially_traceable = tuple(
            sample
            for sample in self.samples
            if bool(sample.source_uri.strip())
            != (sample.source_digest is not None)
        )
        if partially_traceable:
            quality_findings.append(
                create_finding(
                    "quality.document-source-traceability-incomplete",
                    data_contracts.FindingSeverity.WARNING,
                    {
                        "document_ids": [
                            sample.document_id for sample in partially_traceable
                        ]
                    },
                    {"sample_count": len(partially_traceable)},
                    "Provide both the source URI and raw-source digest before "
                    "claiming citation traceability.",
                    False,
                )
            )

        return DocumentCurationReport(
            dataset_hash=self.content_hash,
            sample_count=len(self.samples),
            document_count=len({sample.document_id for sample in self.samples}),
            profiles=profiles,
            duplicate_groups=duplicate_groups,
            language_counts=dict(sorted(language_counts.items())),
            format_counts=dict(sorted(format_counts.items())),
            label_counts=dict(sorted(label_counts.items())),
            layout_coverage=len(layouts) / len(self.samples),
            layout_providers=tuple(
                sorted({layout.provider for layout in layouts})
            ),
            layout_element_count=sum(
                len(layout.elements) for layout in layouts
            ),
            table_element_count=sum(layout.table_count for layout in layouts),
            ocr_confidence_min=(
                min(ocr_confidences) if ocr_confidences else None
            ),
            ocr_confidence_max=(
                max(ocr_confidences) if ocr_confidences else None
            ),
            sensitive_match_counts=dict(sorted(sensitive_counts.items())),
            source_traceability_coverage=(
                len(traceable_samples) / len(self.samples)
            ),
            citation_count=sum(
                len(sample.citations()) for sample in traceable_samples
            ),
            quality_findings=tuple(quality_findings),
            risk_findings=tuple(risk_findings),
            max_characters=max_characters,
            max_tokens=max_tokens,
        )

    def split(
        self,
        validation_fraction: float = 0.2,
        test_fraction: float = 0.2,
        seed: int = 0,
    ) -> DocumentSplit:
        """Splits samples without placing one document ID in two partitions.

        Samples are shuffled at the document-ID group level, then emitted in
        their original order within each partition. This supports chunked
        documents and prevents content from one document leaking into a test
        partition through another chunk.

        Args:
            validation_fraction: Share of document ids for validation, in
                [0, 1).
            test_fraction: Share of document ids for testing, in [0, 1).
            seed: Seed of the shuffle of the sorted document ids.

        Returns:
            The split. With at least three document ids a non-zero fraction
            gets at least one id; the remaining samples form the training
            set.

        Raises:
            ValueError: If a fraction is outside [0, 1) or the fractions sum
                to 1 or more.
        """
        if not 0 <= validation_fraction < 1:
            raise ValueError("validation_fraction must be in [0, 1)")
        if not 0 <= test_fraction < 1:
            raise ValueError("test_fraction must be in [0, 1)")
        if validation_fraction + test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be < 1")

        grouped: defaultdict[str, list[DocumentSample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.document_id].append(sample)
        document_ids = sorted(grouped)
        randomizer = random.Random(seed)  # nosec B311 - seeded data split
        randomizer.shuffle(document_ids)

        test_count = int(len(document_ids) * test_fraction)
        validation_count = int(len(document_ids) * validation_fraction)
        if test_fraction and test_count == 0 and len(document_ids) >= 3:
            test_count = 1
        if (
            validation_fraction
            and validation_count == 0
            and len(document_ids) >= 3
        ):
            validation_count = 1
        while test_count + validation_count >= len(document_ids):
            if validation_count > 0:
                validation_count -= 1
            elif test_count > 0:
                test_count -= 1
            else:
                break

        test_ids = set(document_ids[:test_count])
        validation_ids = set(
            document_ids[test_count : test_count + validation_count]
        )
        train = tuple(
            sample
            for sample in self.samples
            if sample.document_id not in test_ids | validation_ids
        )
        validation = tuple(
            sample
            for sample in self.samples
            if sample.document_id in validation_ids
        )
        test = tuple(
            sample for sample in self.samples if sample.document_id in test_ids
        )
        return DocumentSplit(train=train, validation=validation, test=test)


DocumentSource = bytes | str | pathlib.Path
PdfPageReader = Callable[[bytes], Sequence[str]]
LayoutReader = Callable[[bytes, str, DocumentFormat], DocumentLayout | None]


class DocumentReader:
    """Reads UTF-8 text, Markdown, and optionally injected PDF pages.

    Attributes:
        max_bytes: Largest encoded source size accepted.
        pdf_reader: Optional callable that returns page text from PDF bytes.
        layout_reader: Optional callable that returns layout evidence.
    """

    def __init__(
        self,
        max_bytes: int = 10_000_000,
        pdf_reader: PdfPageReader | None = None,
        layout_reader: LayoutReader | None = None,
    ) -> None:
        """Initializes a bounded document reader.

        Args:
            max_bytes: Maximum encoded source size accepted by the reader.
            pdf_reader: Optional callable supplied by a PDF extra. It receives
                PDF bytes and returns page text in page order.
            layout_reader: Optional caller-owned extractor. It receives raw
                bytes, the stable document ID, and the resolved format, and
                returns layout/OCR/table provenance or ``None``.

        Raises:
            ValueError: If ``max_bytes`` is not positive.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self.pdf_reader = pdf_reader
        self.layout_reader = layout_reader

    def format_for(
        self,
        source: str | pathlib.Path,
        document_format: DocumentFormat | str | None = None,
    ) -> DocumentFormat:
        """Resolves a format from an explicit value or a path suffix.

        Args:
            source: Path whose suffix is used when no format is given.
            document_format: Explicit format, as the enum or its value.

        Returns:
            The explicit format when given; otherwise markdown for ``.md``
            and ``.markdown``, pdf for ``.pdf`` and text for ``.txt``,
            ``.text`` or no suffix.

        Raises:
            DocumentValidationError: If no format is given and the suffix is
                not recognised. An explicit value that is not a known format
                raises ValueError from the enum.
        """
        if document_format is not None:
            return DocumentFormat(document_format)
        suffix = pathlib.Path(source).suffix.lower()
        if suffix in {".md", ".markdown"}:
            return DocumentFormat.MARKDOWN
        if suffix == ".pdf":
            return DocumentFormat.PDF
        if suffix in {".txt", ".text", ""}:
            return DocumentFormat.TEXT
        raise DocumentValidationError(
            f"unsupported document suffix '{suffix}'",
            "unsupported_format",
            {"suffix": suffix},
        )

    def load(
        self,
        source: DocumentSource,
        document_id: str | None = None,
        document_format: DocumentFormat | str | None = None,
        label: str | int | float | bool | None = None,
        layout: DocumentLayout | None = None,
    ) -> DocumentSample:
        """Loads one document and validates its source bytes and content.

        Args:
            source: A local path or already-acquired bytes.
            document_id: Stable document identity. Bytes receive a digest-based
                identity when omitted; paths receive their file name.
            document_format: Explicit format. Bytes default to text; for a
                path it is inferred from the suffix when omitted.
            label: Optional task label carried without interpretation.
            layout: Optional caller-owned OCR, layout, and table evidence.

        Returns:
            An immutable decoded document sample. A PDF without a supplied
            reader raises DocumentDependencyError from `read_pdf`.

        Raises:
            DocumentValidationError: If a path is not an existing file, the
                size exceeds ``max_bytes``, the suffix is not recognised, or
                the bytes or content are invalid.
        """
        path: pathlib.Path | None = None
        if isinstance(source, bytes):
            raw_bytes = source
            resolved_format = DocumentFormat(
                document_format or DocumentFormat.TEXT
            )
            source_uri = "memory://"
        else:
            path = pathlib.Path(source)
            if not path.exists() or not path.is_file():
                raise DocumentValidationError(
                    f"document not found: {path}",
                    "source_not_found",
                    {"source": str(path)},
                )
            resolved_format = self.format_for(path, document_format)
            source_size = path.stat().st_size
            if source_size > self.max_bytes:
                raise DocumentValidationError(
                    "document exceeds the configured size limit",
                    "size_limit",
                    {
                        "actual_bytes": source_size,
                        "max_bytes": self.max_bytes,
                        "source": str(path),
                    },
                )
            raw_bytes = path.read_bytes()
            source_uri = path.resolve().as_uri()

        if len(raw_bytes) > self.max_bytes:
            raise DocumentValidationError(
                "document exceeds the configured size limit",
                "size_limit",
                {"actual_bytes": len(raw_bytes), "max_bytes": self.max_bytes},
            )
        resolved_id = document_id or self.default_document_id(raw_bytes, path)
        if resolved_format is DocumentFormat.PDF:
            content = self.read_pdf(raw_bytes)
        else:
            content = self.decode_utf8(raw_bytes, resolved_id)
        resolved_layout = layout
        if resolved_layout is None and self.layout_reader is not None:
            resolved_layout = self.read_layout(
                raw_bytes, resolved_id, resolved_format
            )
        return DocumentSample(
            document_id=resolved_id,
            content=content,
            format=resolved_format,
            source_uri=source_uri,
            source_digest=hashlib.sha256(raw_bytes).hexdigest(),
            label=label,
            layout=resolved_layout,
        )

    def decode_utf8(self, raw_bytes: bytes, document_id: str) -> str:
        """Decodes strict UTF-8 and reports malformed byte offsets.

        Args:
            raw_bytes: The encoded source.
            document_id: Identity recorded in the error context.

        Returns:
            The decoded text.

        Raises:
            DocumentValidationError: If the bytes are not valid UTF-8; the
                context holds the offsets of the malformed bytes.
        """
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DocumentValidationError(
                "document is not valid UTF-8",
                "invalid_encoding",
                {
                    "document_id": document_id,
                    "encoding": "utf-8",
                    "start": error.start,
                    "end": error.end,
                },
            ) from error
        return content

    def read_pdf(self, raw_bytes: bytes) -> str:
        """Reads PDF pages through the optional injected parser.

        Args:
            raw_bytes: The PDF file.

        Returns:
            The page texts joined by form feeds.

        Raises:
            DocumentDependencyError: If no PDF reader was supplied or the
                reader fails with ImportError.
            DocumentValidationError: If the reader fails with any other
                exception, returns no pages, or returns a page that is not
                text.
        """
        if self.pdf_reader is None:
            raise DocumentDependencyError(
                "PDF input requires the optional document parser extra",
                dependency="pypdf",
                extra="documents",
                details={"format": DocumentFormat.PDF.value},
            )
        try:
            pages = tuple(self.pdf_reader(raw_bytes))
        except ImportError as error:
            raise DocumentDependencyError(
                "PDF input requires the optional document parser extra",
                dependency="pypdf",
                extra="documents",
                details={"format": DocumentFormat.PDF.value},
            ) from error
        except Exception as error:
            raise DocumentValidationError(
                "PDF parser could not decode the source",
                "pdf_parse_error",
                {"format": DocumentFormat.PDF.value},
            ) from error
        if not pages or any(not isinstance(page, str) for page in pages):
            raise DocumentValidationError(
                "PDF parser returned no valid page text",
                "pdf_empty_or_invalid",
                {"page_count": len(pages)},
            )
        return "\f".join(pages)

    def read_layout(
        self,
        raw_bytes: bytes,
        document_id: str,
        document_format: DocumentFormat,
    ) -> DocumentLayout | None:
        """Reads optional layout, OCR, or table evidence through a callback.

        Args:
            raw_bytes: The raw source.
            document_id: Stable document identity passed to the callback.
            document_format: Resolved format passed to the callback.

        Returns:
            The layout the callback returns, or None when no layout reader
            is set or the callback returns None.

        Raises:
            DocumentDependencyError: If the callback fails with ImportError.
            DocumentValidationError: If the callback fails with any other
                exception or returns something other than a DocumentLayout
                or None.
        """
        if self.layout_reader is None:
            return None
        try:
            layout = self.layout_reader(raw_bytes, document_id, document_format)
        except ImportError as error:
            raise DocumentDependencyError(
                "document layout extraction requires its optional parser extra",
                dependency="document-layout-parser",
                extra="documents",
                details={"format": document_format.value},
            ) from error
        except Exception as error:
            raise DocumentValidationError(
                "document layout extractor could not decode the source",
                "layout_parse_error",
                {"document_id": document_id, "format": document_format.value},
            ) from error
        if layout is not None and not isinstance(layout, DocumentLayout):
            raise DocumentValidationError(
                "document layout extractor returned an invalid contract",
                "layout_invalid_result",
                {"document_id": document_id},
            )
        return layout

    def default_document_id(
        self, raw_bytes: bytes, path: pathlib.Path | None
    ) -> str:
        """Returns a deterministic fallback identity for an unlabelled source.

        The identity is stable for the same path name or source bytes.

        Args:
            raw_bytes: The raw source bytes.
            path: Path of the source, or None for in-memory bytes.

        Returns:
            The file name of ``path``, or ``document-`` followed by the first
            16 hex digits of the SHA-256 of the bytes.
        """
        if path is not None:
            return path.name
        return f"document-{hashlib.sha256(raw_bytes).hexdigest()[:16]}"


def language_heuristic(content: str) -> str:
    """Returns a conservative language code using Unicode script ratios.

    Args:
        content: Text to classify.

    Returns:
        ``hi`` if at least 25% of the letters are Devanagari, else ``en`` if
        at least 60% are ASCII, else ``unknown`` (also when there are no
        letters).
    """
    letters = [character for character in content if character.isalpha()]
    if not letters:
        return "unknown"
    devanagari = sum("\u0900" <= character <= "\u097f" for character in letters)
    ascii_letters = sum(character.isascii() for character in letters)
    if devanagari / len(letters) >= 0.25:
        return "hi"
    if ascii_letters / len(letters) >= 0.60:
        return "en"
    return "unknown"


def provenance_for_content(content: str) -> tuple[DocumentSpan, ...]:
    """Builds page and Markdown-heading spans from character offsets.

    Args:
        content: Document text; form feeds separate pages.

    Returns:
        Page spans for every non-empty page, followed by section spans for
        each Markdown heading (levels 1 to 6). A section runs from its
        heading to the next heading or the end, and carries the page number
        of its heading.
    """
    spans: list[DocumentSpan] = []
    page_starts = [0]
    page_ends: list[int] = []
    for match in re.finditer("\f", content):
        page_ends.append(match.start())
        page_starts.append(match.end())
    page_ends.append(len(content))
    for page_number, (start, end) in enumerate(
        zip(page_starts, page_ends, strict=True), start=1
    ):
        if end > start:
            spans.append(
                DocumentSpan(
                    kind="page",
                    start=start,
                    end=end,
                    label=f"page {page_number}",
                    page_number=page_number,
                )
            )

    headings = list(re.finditer(r"(?m)^\s{0,3}#{1,6}\s+(.+?)\s*$", content))
    for index, heading in enumerate(headings):
        end = (
            headings[index + 1].start()
            if index + 1 < len(headings)
            else len(content)
        )
        label = heading.group(1).strip()
        page_number = content.count("\f", 0, heading.start()) + 1
        spans.append(
            DocumentSpan(
                kind="section",
                start=heading.start(),
                end=end,
                label=label,
                page_number=page_number,
            )
        )
    return tuple(spans)
