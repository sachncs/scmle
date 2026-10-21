"""Document data contracts and dependency-light ingestion.

The document slice deliberately keeps parsing small and explicit. UTF-8 text
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

from scml.contracts import data as data_contracts
from scml.contracts import errors


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
        """Initializes a structured document validation failure."""
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
        """Initializes a structured optional-dependency failure."""
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
    """Count supported sensitive-value patterns without returning values."""
    return {
        name: len(pattern.findall(content))
        for name, pattern in sorted(SENSITIVE_PATTERNS.items())
        if pattern.search(content) is not None
    }


def redact_sensitive_text(content: str) -> str:
    """Replace supported sensitive values with type-only placeholders."""
    redacted = content
    for name, pattern in sorted(SENSITIVE_PATTERNS.items()):
        redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)
    return redacted


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentSpan:
    """A character range with simple document provenance."""

    kind: str
    start: int
    end: int
    label: str = ""
    page_number: int | None = None

    def __post_init__(self) -> None:
        """Validates the span and its provenance kind."""
        if self.kind not in {"page", "section"}:
            raise ValueError("kind must be 'page' or 'section'")
        if self.start < 0 or self.end <= self.start:
            raise ValueError("span must have a positive character range")
        if self.kind == "page" and (
            self.page_number is None or self.page_number < 1
        ):
            raise ValueError("page spans require a positive page_number")


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentElement:
    """One extracted layout, OCR, or table element with source offsets."""

    kind: str
    start: int
    end: int
    page_number: int
    confidence: float | None = None
    table_rows: int | None = None
    table_columns: int | None = None

    def __post_init__(self) -> None:
        """Validate element type, offsets, page, confidence, and table shape."""
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
    """Caller-owned extraction provenance for OCR, layout, and tables."""

    source: str
    provider: str
    provider_version: str
    elements: tuple[DocumentElement, ...]
    ocr_confidence: float | None = None

    def __post_init__(self) -> None:
        """Validate the extraction identity and optional OCR confidence."""
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

    @property
    def table_count(self) -> int:
        """Return the number of extracted table elements."""
        return sum(element.kind == "table" for element in self.elements)

    def to_dict(self) -> dict[str, object]:
        """Return JSON-native extraction evidence."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentProfile:
    """Bounded lexical profile for one document."""

    document_id: str
    character_count: int
    token_count: int
    language: str
    truncated: bool

    def __post_init__(self) -> None:
        """Validates profile counts and identity."""
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
        """Return whether no unresolved finding blocks the next lifecycle."""
        return not self.unresolved_blocking_findings

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native evidence for a dataset card or run artifact."""
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
            "release_ready": self.release_ready,
            "unresolved_blocking_findings": [
                finding.finding_id
                for finding in self.unresolved_blocking_findings
            ],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentSample:
    """Immutable decoded document content and its identity."""

    document_id: str
    content: str
    format: DocumentFormat = DocumentFormat.TEXT
    source_uri: str = ""
    encoding: str = "utf-8"
    label: str | int | float | bool | None = None
    provenance: tuple[DocumentSpan, ...] = ()
    layout: DocumentLayout | None = None
    content_hash: str = dataclasses.field(init=False)
    size_bytes: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates content and derives immutable identity fields."""
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
        content_bytes = self.content.encode("utf-8")
        object.__setattr__(
            self,
            "content_hash",
            hashlib.sha256(content_bytes).hexdigest(),
        )
        object.__setattr__(self, "size_bytes", len(content_bytes))

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
            A deterministic bounded profile.

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
    """Leakage-aware document-level split."""

    train: tuple[DocumentSample, ...]
    validation: tuple[DocumentSample, ...]
    test: tuple[DocumentSample, ...]

    def document_ids(self) -> dict[str, tuple[str, ...]]:
        """Returns sorted document IDs assigned to each split."""
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
    """Immutable collection of documents with identity and split operations."""

    samples: tuple[DocumentSample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates the collection and derives order-independent identity."""
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
        """Returns deterministic groups of samples with identical content."""
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
        """Profiles every sample using the same bounded limits."""
        return tuple(
            sample.profile(max_characters, max_tokens)
            for sample in self.samples
        )

    def curate(
        self,
        max_characters: int = 100_000,
        max_tokens: int = 20_000,
    ) -> DocumentCurationReport:
        """Build bounded, reviewable curation evidence for the corpus.

        Duplicate content is reported separately from conflicting labels. A
        duplicate is often expected for replicated sources, while identical
        content with different labels is a blocking risk for supervised work.
        The method does not silently remove samples or infer a modelling task.
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
            """Return a stable human-readable label bucket."""
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
            """Create a document-specific finding with one stable rule."""
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


class DocumentReader:
    """Reads UTF-8 text, Markdown, and optionally injected PDF pages."""

    def __init__(
        self,
        max_bytes: int = 10_000_000,
        pdf_reader: PdfPageReader | None = None,
    ) -> None:
        """Initializes a bounded document reader.

        Args:
            max_bytes: Maximum encoded source size accepted by the reader.
            pdf_reader: Optional callable supplied by a PDF extra. It receives
                PDF bytes and returns page text in page order.

        Raises:
            ValueError: If ``max_bytes`` is not positive.
        """
        if max_bytes < 1:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self.pdf_reader = pdf_reader

    def format_for(
        self,
        source: str | pathlib.Path,
        document_format: DocumentFormat | str | None = None,
    ) -> DocumentFormat:
        """Resolves a format from an explicit value or a path suffix."""
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
            document_format: Explicit format, required only when bytes are not
                a PDF or text path with a recognizable suffix.
            label: Optional task label carried without interpretation.
            layout: Optional caller-owned OCR, layout, and table evidence.

        Returns:
            An immutable decoded document sample.

        Raises:
            DocumentDependencyError: If PDF support was not supplied.
            DocumentValidationError: If bytes, size, format, or content is
                invalid.
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
        return DocumentSample(
            document_id=resolved_id,
            content=content,
            format=resolved_format,
            source_uri=source_uri,
            label=label,
            layout=layout,
        )

    def decode_utf8(self, raw_bytes: bytes, document_id: str) -> str:
        """Decodes strict UTF-8 and reports malformed byte offsets."""
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
        """Reads PDF pages through the optional injected parser."""
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

    def default_document_id(
        self, raw_bytes: bytes, path: pathlib.Path | None
    ) -> str:
        """Returns a deterministic fallback identity for an unlabelled source.

        The identity is stable for the same path name or source bytes.
        """
        if path is not None:
            return path.name
        return f"document-{hashlib.sha256(raw_bytes).hexdigest()[:16]}"


def language_heuristic(content: str) -> str:
    """Returns a conservative language code using Unicode script ratios."""
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
    """Builds page and Markdown-heading spans from character offsets."""
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
