"""Bounded reading of text, Markdown and injected PDF documents."""

from collections.abc import Callable
from collections.abc import Sequence
import hashlib
import pathlib

from scmle.contracts import errors
from scmle.data.modalities.document.elements import DependencyError
from scmle.data.modalities.document.elements import DocumentError
from scmle.data.modalities.document.elements import Format
from scmle.data.modalities.document.elements import Layout
from scmle.data.modalities.document.sample import Sample

DocumentSource = bytes | str | pathlib.Path
PdfPageReader = Callable[[bytes], Sequence[str]]
LayoutReader = Callable[[bytes, str, Format], Layout | None]


def slurp(path: pathlib.Path, max_bytes: int) -> bytes:
    """Reads a document file within a size limit.

    Args:
        path: The document file.
        max_bytes: Largest file size accepted.

    Returns:
        The file's bytes.

    Raises:
        DocumentError: If the file is larger than ``max_bytes``.
    """
    source_size = path.stat().st_size
    if source_size > max_bytes:
        raise DocumentError(
            "document exceeds the configured size limit",
            "size_limit",
            {
                "actual_bytes": source_size,
                "max_bytes": max_bytes,
                "source": str(path),
            },
        )
    return path.read_bytes()


class Reader:
    """Reads UTF-8 text, Markdown, and optionally injected PDF pages.

    Attributes:
        limit: Largest encoded source size accepted.
        pdf: Optional callable that returns page text from PDF bytes.
        reader: Optional callable that returns layout evidence.
    """

    def __init__(
        self,
        limit: int = 10_000_000,
        pdf: PdfPageReader | None = None,
        reader: LayoutReader | None = None,
    ) -> None:
        """Initializes a bounded document reader.

        Args:
            limit: Maximum encoded source size accepted by the reader.
            pdf: Optional callable supplied by a PDF extra. It receives
                PDF bytes and returns page text in page order.
            reader: Optional caller-owned extractor. It receives raw
                bytes, the stable document ID, and the resolved format, and
                returns layout/OCR/table provenance or ``None``.

        Raises:
            ValueError: If ``limit`` is not positive.
        """
        if limit < 1:
            raise errors.InputError("limit must be positive")
        self.limit = limit
        self.pdf = pdf
        self.reader = reader

    def sniff(
        self,
        source: str | pathlib.Path,
        document_format: Format | str | None = None,
    ) -> Format:
        """Resolves a format from an explicit value or a path suffix.

        Args:
            source: Path whose suffix is used when no format is given.
            document_format: Explicit format, as the enum or its value.

        Returns:
            The explicit format when given; otherwise markdown for ``.md``
            and ``.markdown``, pdf for ``.pdf`` and text for ``.txt``,
            ``.text`` or no suffix.

        Raises:
            DocumentError: If no format is given and the suffix is
                not recognised. An explicit value that is not a known format
                raises ValueError from the enum.
        """
        if document_format is not None:
            return Format(document_format)
        suffix = pathlib.Path(source).suffix.lower()
        if suffix in {".md", ".markdown"}:
            return Format.MARKDOWN
        if suffix == ".pdf":
            return Format.PDF
        if suffix in {".txt", ".text", ""}:
            return Format.TEXT
        raise DocumentError(
            f"unsupported document suffix '{suffix}'",
            "unsupported_format",
            {"suffix": suffix},
        )

    def load(
        self,
        source: DocumentSource,
        document_id: str | None = None,
        document_format: Format | str | None = None,
        label: str | float | bool | None = None,
        layout: Layout | None = None,
    ) -> Sample:
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
            reader raises DependencyError from `extract`.

        Raises:
            DocumentError: If a path is not an existing file, the
                size exceeds ``limit``, the suffix is not recognised, or
                the bytes or content are invalid.
        """
        path: pathlib.Path | None = None
        if isinstance(source, bytes):
            raw_bytes = source
            resolved_format = Format(document_format or Format.TEXT)
            source_uri = "memory://"
        else:
            path = pathlib.Path(source)
            if not path.exists() or not path.is_file():
                raise DocumentError(
                    f"document not found: {path}",
                    "source_not_found",
                    {"source": str(path)},
                )
            resolved_format = self.sniff(path, document_format)
            raw_bytes = slurp(path, self.limit)
            source_uri = path.resolve().as_uri()

        if len(raw_bytes) > self.limit:
            raise DocumentError(
                "document exceeds the configured size limit",
                "size_limit",
                {"actual_bytes": len(raw_bytes), "max_bytes": self.limit},
            )
        resolved_id = document_id or self.fingerprint(raw_bytes, path)
        if resolved_format is Format.PDF:
            content = self.extract(raw_bytes)
        else:
            content = self.unicode(raw_bytes, resolved_id)
        resolved_layout = layout
        if resolved_layout is None and self.reader is not None:
            resolved_layout = self.overlay(
                raw_bytes, resolved_id, resolved_format
            )
        return Sample(
            identifier=resolved_id,
            content=content,
            format=resolved_format,
            uri=source_uri,
            digest=hashlib.sha256(raw_bytes).hexdigest(),
            label=label,
            layout=resolved_layout,
        )

    def unicode(self, raw_bytes: bytes, document_id: str) -> str:
        """Decodes strict UTF-8 and reports the first malformed byte range.

        Args:
            raw_bytes: The encoded source.
            document_id: Identity recorded in the error context.

        Returns:
            The decoded text.

        Raises:
            DocumentError: If the bytes are not valid UTF-8; the
                context holds the start and end offsets of the first
                malformed byte run.
        """
        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise DocumentError(
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

    def extract(self, raw_bytes: bytes) -> str:
        """Reads PDF pages through the optional injected parser.

        Args:
            raw_bytes: The PDF file.

        Returns:
            The page texts joined by form feeds.

        Raises:
            DependencyError: If no PDF reader was supplied or the
                reader fails with ImportError.
            DocumentError: If the reader fails with any other
                exception, returns no pages, or returns a page that is not
                text.
        """
        if self.pdf is None:
            raise DependencyError(
                "PDF input requires the optional document parser extra",
                dependency="pypdf",
                extra="documents",
                details={"format": Format.PDF.value},
            )
        try:
            pages = tuple(self.pdf(raw_bytes))
        except ImportError as error:
            raise DependencyError(
                "PDF input requires the optional document parser extra",
                dependency="pypdf",
                extra="documents",
                details={"format": Format.PDF.value},
            ) from error
        except Exception as error:
            raise DocumentError(
                "PDF parser could not decode the source",
                "pdf_parse_error",
                {"format": Format.PDF.value},
            ) from error
        if not pages or any(not isinstance(page, str) for page in pages):
            raise DocumentError(
                "PDF parser returned no valid page text",
                "pdf_empty_or_invalid",
                {"page_count": len(pages)},
            )
        return "\f".join(pages)

    def overlay(
        self,
        raw_bytes: bytes,
        document_id: str,
        document_format: Format,
    ) -> Layout | None:
        """Reads optional layout, OCR, or table evidence through a callback.

        Args:
            raw_bytes: The raw source.
            document_id: Stable document identity passed to the callback.
            document_format: Resolved format passed to the callback.

        Returns:
            The layout the callback returns, or None when no layout reader
            is set or the callback returns None.

        Raises:
            DependencyError: If the callback fails with ImportError.
            DocumentError: If the callback fails with any other
                exception or returns something other than a Layout
                or None.
        """
        if self.reader is None:
            return None
        try:
            layout = self.reader(raw_bytes, document_id, document_format)
        except ImportError as error:
            raise DependencyError(
                "document layout extraction requires its optional parser extra",
                dependency="document-layout-parser",
                extra="documents",
                details={"format": document_format.value},
            ) from error
        except Exception as error:
            raise DocumentError(
                "document layout extractor could not decode the source",
                "layout_parse_error",
                {"document_id": document_id, "format": document_format.value},
            ) from error
        if layout is not None and not isinstance(layout, Layout):
            raise DocumentError(
                "document layout extractor returned an invalid contract",
                "layout_invalid_result",
                {"document_id": document_id},
            )
        return layout

    def fingerprint(self, raw_bytes: bytes, path: pathlib.Path | None) -> str:
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
