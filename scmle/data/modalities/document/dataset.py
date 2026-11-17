"""The immutable document dataset."""

from collections import defaultdict
import dataclasses
import hashlib

from scmle.contracts import errors
from scmle.data.modalities.curation import allocate
from scmle.data.modalities.curation import proportions
from scmle.data.modalities.document.elements import DocumentError
from scmle.data.modalities.document.elements import Profile
from scmle.data.modalities.document.report import Curation
from scmle.data.modalities.document.review import layering
from scmle.data.modalities.document.review import Review
from scmle.data.modalities.document.sample import bucketize
from scmle.data.modalities.document.sample import Sample
from scmle.data.modalities.document.sample import Split


@dataclasses.dataclass(frozen=True, slots=True)
class Dataset(Review):
    """Immutable collection of documents with identity and split operations.

    Attributes:
        samples: The samples, in caller order.
        content_hash: SHA-256 hex digest of the sorted document ids and
            content hashes, so independent of sample order; set on
            construction.
    """

    samples: tuple[Sample, ...]
    content_hash: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates the collection and derives order-independent identity.

        Raises:
            DocumentError: If there are no samples.
            TypeError: If an element is not a Sample.
        """
        normalized_samples = tuple(self.samples)
        if not normalized_samples:
            raise DocumentError(
                "document dataset must contain at least one sample",
                "empty_dataset",
            )
        if any(not isinstance(sample, Sample) for sample in normalized_samples):
            raise errors.KindError("samples must contain only Sample values")
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
    ) -> tuple[Profile, ...]:
        """Profiles every sample using the same bounded limits.

        Args:
            max_characters: Maximum characters inspected per sample.
            max_tokens: Maximum tokens retained per sample.

        Returns:
            One profile per sample, in sample order. A bound that is not
            positive raises ValueError from `Sample.profile`.
        """
        return tuple(
            sample.profile(max_characters, max_tokens)
            for sample in self.samples
        )

    def curate(
        self,
        max_characters: int = 100_000,
        max_tokens: int = 20_000,
    ) -> Curation:
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
            raise errors.InputError("curation bounds must be positive")
        profiles = self.profile(max_characters, max_tokens)
        duplicate_groups = self.duplicate_groups()
        language_counts: dict[str, int] = defaultdict(int)
        format_counts: dict[str, int] = defaultdict(int)
        label_counts: dict[str, int] = defaultdict(int)
        for sample, profile in zip(self.samples, profiles, strict=True):
            language_counts[profile.language] += 1
            format_counts[sample.format.value] += 1
            label_counts[bucketize(sample.label)] += 1
        layouts = [
            sample.layout
            for sample in self.samples
            if sample.layout is not None
        ]
        sensitive_counts, sensitive_document_ids = self.sensitivity()
        quality_findings = self.inspect(
            profiles, duplicate_groups, layouts, max_characters, max_tokens
        )
        risk_findings = self.assess(
            duplicate_groups,
            label_counts,
            sensitive_counts,
            sensitive_document_ids,
        )
        traceable_samples = tuple(
            sample
            for sample in self.samples
            if sample.source_uri.strip() and sample.source_digest is not None
        )
        return Curation(
            dataset_hash=self.content_hash,
            sample_count=len(self.samples),
            document_count=len({sample.document_id for sample in self.samples}),
            profiles=profiles,
            duplicate_groups=duplicate_groups,
            language_counts=dict(sorted(language_counts.items())),
            format_counts=dict(sorted(format_counts.items())),
            label_counts=dict(sorted(label_counts.items())),
            layout_coverage=len(layouts) / len(self.samples),
            **layering(layouts),
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
    ) -> Split:
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
        proportions(validation_fraction, test_fraction)

        grouped: defaultdict[str, list[Sample]] = defaultdict(list)
        for sample in self.samples:
            grouped[sample.document_id].append(sample)
        validation_ids, test_ids = allocate(
            grouped, validation_fraction, test_fraction, seed
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
        return Split(train=train, validation=validation, test=test)
