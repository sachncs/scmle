"""The document and text preparers."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import document
from scmle.models import documents
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import scoring

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.linear_model import LogisticRegression


class Document(curators.Preparer):
    """Adapter for UTF-8, Markdown, and caller-decoded document samples.

    Attributes:
        dataset: Immutable document dataset the adapter wraps.
    """

    modality = modalities.Modality.DOCUMENT

    def __init__(self, dataset: document.Dataset) -> None:
        """Binds an immutable document dataset.

        Args:
            dataset: Documents the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def digest(self) -> str:
        """The document corpus identity."""
        return self.dataset.digest

    @property
    def samples(self) -> tuple[document.Sample, ...]:
        """The document samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> document.Curation:
        """Builds document profiling and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> containers.Partitions:
        """Splits by document identity to prevent chunk leakage.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the sorted document IDs
            of each under `groups`.
        """
        result = self.dataset.split(seed=seed)
        return containers.Partitions(
            result.train,
            result.validation,
            result.test,
            {
                "train": tuple(
                    sorted({sample.identifier for sample in result.train})
                ),
                "validation": tuple(
                    sorted({sample.identifier for sample in result.validation})
                ),
                "test": tuple(
                    sorted({sample.identifier for sample in result.test})
                ),
            },
        )

    def features(self, samples: Sequence[document.Sample]) -> containers.Batch:
        """Prepares bounded lexical features without using labels.

        Args:
            samples: Documents to featurize.

        Returns:
            Batch with one row per document and two columns, "character_count"
            and "token_count".
        """
        profiles = document.Dataset(tuple(samples)).profile()
        rows = numpy.asarray(
            [
                (profile.character_count, profile.token_count)
                for profile in profiles
            ],
            dtype=numpy.float64,
        )
        return containers.Batch(
            tuple(self.sample(sample) for sample in samples),
            ("character_count", "token_count"),
            rows,
        )

    def sample(self, sample: document.Sample) -> str:
        """Gives a stable identity for one document sample or chunk.

        Args:
            sample: Document sample.

        Returns:
            The document ID and content hash joined by a colon.
        """
        return f"{sample.identifier}:{sample.content_hash}"

    def label(self, sample: document.Sample) -> object | None:
        """Gives the document label.

        Args:
            sample: Document sample.

        Returns:
            The sample's label, or None if it has none.
        """
        return sample.label

    def elucidate(
        self,
        sample: Any,
        model: LogisticRegression,
        feature_names: Sequence[str],  # noqa: ARG002 -- overrides Preparer.elucidate
        coefficients: numpy.ndarray,  # noqa: ARG002 -- overrides Preparer.elucidate
    ) -> dict[str, object]:
        """Builds page and section provenance for a document explanation.

        Args:
            sample: Document sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the document's source identity, citations, provenance
            spans, source-span occlusion scores and limitations.

        Raises:
            TypeError: If `sample` is not a `Sample`.
        """
        if not isinstance(sample, document.Sample):
            raise errors.KindError("sample must be a Sample")
        raw_occlusion = self.mask(sample, model)
        return {
            "status": "available",
            "method": "document_span_provenance",
            "document_id": sample.identifier,
            "source_uri": sample.uri,
            "source_digest": sample.digest,
            "format": sample.format.value,
            "content_hash": sample.content_hash,
            "citations": [
                citation.to_dict() for citation in sample.citations()
            ],
            "spans": [
                {
                    "kind": span.kind,
                    "start": span.start,
                    "end": span.end,
                    "label": span.label,
                    "page_number": span.page_number,
                }
                for span in sample.provenance
            ],
            "raw_input_occlusion": raw_occlusion,
            "limitations": [
                (
                    "Span provenance and occlusion identify source context; "
                    "they are not OCR confidence or citation grounding."
                ),
                "The lifecycle model uses bounded lexical counts.",
                "No causal claim is made.",
            ],
        }

    def mask(
        self,
        sample: document.Sample,
        model: LogisticRegression,
    ) -> dict[str, object]:
        """Scores source-character masks while preserving span offsets.

        Args:
            sample: Document sample to explain.
            model: Fitted classifier.

        Returns:
            Mapping with one entry per provenance span holding the change in the
            model's decision score when the span's characters are replaced by
            spaces (full score minus masked score), plus the baseline and
            limitations.
        """
        original_features = self.features((sample,)).values
        selected_class, original_score = scoring.anchor(
            model, original_features
        )
        spans: list[dict[str, object]] = []
        citations = sample.citations()
        for provenance in sample.provenance:
            characters = list(sample.content)
            characters[provenance.start : provenance.end] = " " * (
                provenance.end - provenance.start
            )
            masked_content = "".join(characters)
            if not masked_content.strip():
                masked_content = "."
            masked_sample = document.Sample(
                identifier=sample.identifier,
                content=masked_content,
                format=sample.format,
                uri=sample.uri,
                encoding=sample.encoding,
                digest=sample.digest,
                label=sample.label,
                provenance=sample.provenance,
            )
            masked_features = self.features((masked_sample,)).values
            masked_score = scoring.score(model, masked_features, selected_class)
            spans.append(
                {
                    "kind": provenance.kind,
                    "start": provenance.start,
                    "end": provenance.end,
                    "page_number": provenance.page_number,
                    "citation": next(
                        (
                            citation.to_dict()
                            for citation in citations
                            if citation.start == provenance.start
                            and citation.end == provenance.end
                        ),
                        None,
                    ),
                    "score_delta_from_full": original_score - masked_score,
                    "masked_content_hash": masked_sample.content_hash,
                }
            )
        return {
            "status": "available",
            "method": "raw_document_span_occlusion",
            "spans": spans,
            "citations": [citation.to_dict() for citation in citations],
            "baseline": "space replacement in source character spans",
            "feature_space_only": False,
            "limitations": [
                (
                    "The model consumes bounded character and token counts "
                    "after masking."
                ),
                (
                    "Span masks do not establish citation grounding, OCR "
                    "confidence, or semantic causality."
                ),
            ],
        }

    def monitor(self, observed: curators.Preparer) -> dict[str, object]:
        """Builds the document profile drift report.

        Args:
            observed: Adapter over the newly observed documents.

        Returns:
            The drift report as a JSON-native mapping.

        Raises:
            TypeError: If `observed` is not a `Document`.
        """
        if not isinstance(observed, Document):
            raise errors.KindError(
                "observed adapter must be a DocumentPreparer"
            )
        current = self.dataset.curate()
        report = documents.Monitor(current).observe(observed.dataset)
        return report.to_dict()

    def remarks(self) -> tuple[str, ...]:
        """Describes the bounded document representation.

        Returns:
            Plain statements of what the representation does not cover.
        """
        return (
            (
                "The lifecycle baseline uses character and token counts, not "
                "OCR, layout, or semantic embeddings."
            ),
            (
                "Term-level document explanations require the separate "
                "Classifier path."
            ),
        )


class Plaintext(Document):
    """Document adapter with an explicit text modality declaration."""

    modality = modalities.Modality.TEXT
