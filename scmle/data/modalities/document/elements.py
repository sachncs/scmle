"""Document formats, errors, provenance spans, layout and profiles."""

import dataclasses
import enum
import math
import re
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records


class Format(enum.StrEnum):
    """Document formats understood by :class:`Reader`."""

    TEXT = "text"
    MARKDOWN = "markdown"
    PDF = "pdf"


class DocumentError(errors.DataError):
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


class DependencyError(errors.DataError):
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


LOCAL = 64
"""Longest local part of an address (RFC 5321, 4.5.3.1.1)."""

DOMAIN = 253
"""Longest domain name before its last label: 255 octets (RFC 5321,
4.5.3.1.2) less the dot and two-letter shortest top-level label."""

LABEL = 63
"""Longest label of a domain name (RFC 1035, 2.3.4)."""

SENSITIVE_PATTERNS: dict[str, re.Pattern[str]] = {
    # Each quantifier is bounded by the standard's length limit, so a start
    # position costs at most LOCAL + DOMAIN x LABEL steps and the scan is
    # linear in the text (MM-1): an unbounded `+` re-scanned a long dotted
    # run without `@` from every position (80 KB took 18 s).
    "email": re.compile(
        rf"\b[A-Z0-9._%+-]{{1,{LOCAL}}}@[A-Z0-9.-]{{1,{DOMAIN}}}"
        rf"\.[A-Z]{{2,{LABEL}}}\b",
        re.IGNORECASE,
    ),
    "phone": re.compile(
        r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(\d{3}\)|\d{3})"
        r"[\s.-]\d{3}[\s.-]\d{4}(?!\d)"
    ),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def pii(content: str) -> dict[str, int]:
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


@dataclasses.dataclass(frozen=True, slots=True)
class Span:
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
            raise errors.InputError("kind must be 'page' or 'section'")
        if self.start < 0 or self.end <= self.start:
            raise errors.InputError("span must have a positive character range")
        if self.kind == "page" and (
            self.page_number is None or self.page_number < 1
        ):
            raise errors.InputError("page spans require a positive page_number")


@dataclasses.dataclass(frozen=True, slots=True)
class Citation:
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

    KEYS: ClassVar[records.Migration] = records.nest(
        {
            "document_id": "document",
            "source_uri": "uri",
            "page_number": "page",
        },
        groups={"digests": {"source_digest": "source"}},
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates the source identity and character locator.

        Raises:
            ValueError: If document_id, source_uri or locator is blank,
                source_digest is not a lowercase SHA-256 digest, the range is
                empty or starts below zero, or page_number is below one.
        """
        if not self.document_id.strip():
            raise errors.InputError("document_id must not be empty")
        if not self.source_uri.strip():
            raise errors.InputError("source_uri must not be empty")
        if not re.fullmatch(r"[0-9a-f]{64}", self.source_digest):
            raise errors.InputError(
                "source_digest must be a lowercase SHA-256 digest"
            )
        if self.start < 0 or self.end <= self.start:
            raise errors.InputError(
                "citation must have a positive character range"
            )
        if not self.locator.strip():
            raise errors.InputError("locator must not be empty")
        if self.page_number is not None and self.page_number < 1:
            raise errors.InputError("page_number must be positive")

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native citation evidence.

        Returns:
            A dict of the citation fields: `document`, `uri`,
            `digests.source`, `start`, `end`, `locator` and `page`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Element:
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
            raise errors.InputError("unsupported document element kind")
        if self.start < 0 or self.end <= self.start:
            raise errors.InputError(
                "document element must have a positive range"
            )
        if self.page_number < 1:
            raise errors.InputError(
                "document elements require a positive page_number"
            )
        if self.confidence is not None and (
            isinstance(self.confidence, bool)
            or not math.isfinite(self.confidence)
            or not 0.0 <= self.confidence <= 1.0
        ):
            raise errors.InputError(
                "document element confidence must be in [0, 1]"
            )
        if self.kind == "table" and (
            self.table_rows is None or self.table_columns is None
        ):
            raise errors.InputError(
                "table elements require row and column counts"
            )
        if self.table_rows is not None and self.table_rows < 1:
            raise errors.InputError("table_rows must be positive")
        if self.table_columns is not None and self.table_columns < 1:
            raise errors.InputError("table_columns must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class Layout:
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
    elements: tuple[Element, ...]
    ocr_confidence: float | None = None
    source_digest: str | None = None

    KEYS: ClassVar[records.Migration] = records.nest(
        {"provider_version": "version", "ocr_confidence": "confidence"},
        groups={"digests": {"source_digest": "source"}},
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def __post_init__(self) -> None:
        """Validates the extraction identity and optional OCR confidence.

        Raises:
            ValueError: If source, provider or provider_version is blank, the
                OCR confidence is a bool, not finite or outside [0, 1], or
                source_digest is not a lowercase SHA-256 digest.
            TypeError: If elements is not a tuple of Element.
        """
        for field_name, value in (
            ("source", self.source),
            ("provider", self.provider),
            ("provider_version", self.provider_version),
        ):
            if not value.strip():
                raise errors.InputError(
                    f"document layout {field_name} must not be empty"
                )
        if not isinstance(self.elements, tuple):
            raise errors.KindError("document layout elements must be a tuple")
        if not all(isinstance(element, Element) for element in self.elements):
            raise errors.KindError("elements must contain Element values")
        if self.ocr_confidence is not None and (
            isinstance(self.ocr_confidence, bool)
            or not math.isfinite(self.ocr_confidence)
            or not 0.0 <= self.ocr_confidence <= 1.0
        ):
            raise errors.InputError("OCR confidence must be in [0, 1]")
        if self.source_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.source_digest
        ):
            raise errors.InputError(
                "source_digest must be a lowercase SHA-256 digest"
            )

    @property
    def table_count(self) -> int:
        """The number of extracted table elements."""
        return sum(element.kind == "table" for element in self.elements)

    def to_dict(self) -> dict[str, object]:
        """Returns JSON-native extraction evidence.

        Returns:
            A dict of the layout fields, with elements as nested dicts, the
            provider version as `version`, the OCR confidence as
            `confidence` and the source digest under `digests.source`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True, slots=True)
class Profile:
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
            raise errors.InputError("document_id must not be empty")
        if self.character_count < 0 or self.token_count < 0:
            raise errors.InputError("profile counts must not be negative")
        if not self.language.strip():
            raise errors.InputError("language must not be empty")
