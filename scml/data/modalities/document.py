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
import pathlib
import random
import re
from typing import Any

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
class DocumentSample:
    """Immutable decoded document content and its identity."""

    document_id: str
    content: str
    format: DocumentFormat = DocumentFormat.TEXT
    source_uri: str = ""
    encoding: str = "utf-8"
    label: str | int | float | bool | None = None
    provenance: tuple[DocumentSpan, ...] = ()
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
    ) -> DocumentSample:
        """Loads one document and validates its source bytes and content.

        Args:
            source: A local path or already-acquired bytes.
            document_id: Stable document identity. Bytes receive a digest-based
                identity when omitted; paths receive their file name.
            document_format: Explicit format, required only when bytes are not
                a PDF or text path with a recognizable suffix.
            label: Optional task label carried without interpretation.

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
