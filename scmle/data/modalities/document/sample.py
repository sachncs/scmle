"""Decoded document samples and their document-level split."""

import dataclasses
import hashlib
import re
from typing import Any

from scmle.contracts import errors
from scmle.data.modalities.document.elements import Citation
from scmle.data.modalities.document.elements import DocumentError
from scmle.data.modalities.document.elements import Format
from scmle.data.modalities.document.elements import Layout
from scmle.data.modalities.document.elements import Profile
from scmle.data.modalities.document.elements import Span
from scmle.data.modalities.document.text import detect
from scmle.data.modalities.document.text import segment


@dataclasses.dataclass(frozen=True, slots=True)
class Sample:
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
    format: Format = Format.TEXT
    source_uri: str = ""
    encoding: str = "utf-8"
    source_digest: str | None = None
    label: str | int | float | bool | None = None
    provenance: tuple[Span, ...] = ()
    layout: Layout | None = None
    content_hash: str = dataclasses.field(init=False)
    size_bytes: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates content and derives immutable identity fields.

        Raises:
            DocumentError: If the document id or content is blank,
                the encoding is not UTF-8, the source digest is not a
                lowercase SHA-256 digest, a layout element extends past the
                content, or the layout's source digest differs from the
                sample's.
        """
        if not self.document_id.strip():
            raise DocumentError(
                "document_id must not be empty",
                "empty_document_id",
            )
        if not self.content.strip():
            raise DocumentError(
                "document content must not be empty",
                "empty_content",
                {"document_id": self.document_id},
            )
        if self.encoding.lower().replace("-", "") != "utf8":
            raise DocumentError(
                "document encoding must be UTF-8",
                "unsupported_encoding",
                {"encoding": self.encoding},
            )
        if self.source_digest is not None and not re.fullmatch(
            r"[0-9a-f]{64}", self.source_digest
        ):
            raise DocumentError(
                "source_digest must be a lowercase SHA-256 digest",
                "invalid_source_digest",
                {"document_id": self.document_id},
            )
        declared_format: Any = self.format  # a plain string is accepted
        if not isinstance(declared_format, Format):
            object.__setattr__(self, "format", Format(declared_format))
        normalized_provenance = tuple(self.provenance)
        if not normalized_provenance:
            normalized_provenance = segment(self.content)
        object.__setattr__(self, "provenance", normalized_provenance)
        if self.layout is not None and any(
            element.end > len(self.content) for element in self.layout.elements
        ):
            raise DocumentError(
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
            raise DocumentError(
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

    def citations(self) -> tuple[Citation, ...]:
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
            Citation(
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
    ) -> Profile:
        """Builds a bounded lexical profile without external dependencies.

        Args:
            max_characters: Maximum characters inspected.
            max_tokens: Maximum tokens retained in the profile.

        Returns:
            A deterministic bounded profile. Tokens are word characters
            runs; the language comes from `detect` over the
            inspected characters.

        Raises:
            ValueError: If either bound is not positive.
        """
        if max_characters < 1 or max_tokens < 1:
            raise errors.InputError("profiling bounds must be positive")
        sample = self.content[:max_characters]
        tokens = re.findall(r"\b\w+\b", sample, flags=re.UNICODE)
        truncated = (
            len(self.content) > max_characters or len(tokens) > max_tokens
        )
        return Profile(
            document_id=self.document_id,
            character_count=len(sample),
            token_count=min(len(tokens), max_tokens),
            language=detect(sample),
            truncated=truncated,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Split:
    """Leakage-aware document-level split.

    Attributes:
        train: Samples assigned to training.
        validation: Samples assigned to validation.
        test: Samples assigned to testing.
    """

    train: tuple[Sample, ...]
    validation: tuple[Sample, ...]
    test: tuple[Sample, ...]

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


def bucketize(label: str | float | bool | None) -> str:
    """Returns a stable human-readable label bucket.

    Args:
        label: A sample label, possibly None.

    Returns:
        ``unlabeled`` for None, otherwise ``<type name>:<label>``.
    """
    if label is None:
        return "unlabeled"
    return f"{type(label).__name__}:{label}"
