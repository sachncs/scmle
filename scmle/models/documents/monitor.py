"""Corpus-profile monitoring against a reference snapshot."""

from scmle.contracts import errors
from scmle.data.modalities import document
from scmle.models.documents.records import Drift


class Monitor:
    """Monitors bounded corpus-profile signals against a reference snapshot.

    Attributes:
        reference: Curation report of the reference corpus.
        threshold: Level above which a signal counts as shifted.
    """

    def __init__(
        self,
        reference: document.Curation,
        threshold: float = 0.25,
    ) -> None:
        """Initializes a monitor with an explicit normalized drift threshold.

        Args:
            reference: Curation report of the reference corpus.
            threshold: Level above which a signal counts as shifted; in
                (0, 1].

        Raises:
            ValueError: If `threshold` is outside (0, 1].
        """
        if not 0 < threshold <= 1:
            raise errors.InputError("threshold must be in (0, 1]")
        self.reference = reference
        self.threshold = threshold

    def observe(
        self,
        observed: document.Dataset,
    ) -> Drift:
        """Compares observed document profiles with the reference corpus.

        Args:
            observed: Documents to compare against the reference.

        Returns:
            Drift report with the language distance, the relative change in
            mean character count and the duplicate rate.
        """
        current = observed.curate(
            self.reference.max_characters,
            self.reference.max_tokens,
        )
        languages = set(self.reference.language_counts) | set(
            current.language_counts
        )
        reference_total = max(self.reference.sample_count, 1)
        current_total = max(current.sample_count, 1)
        language_distance = (
            sum(
                abs(
                    self.reference.language_counts.get(language, 0)
                    / reference_total
                    - current.language_counts.get(language, 0) / current_total
                )
                for language in languages
            )
            / 2
        )
        reference_mean = self.measure(self.reference)
        current_mean = self.measure(current)
        duplicate_rate = (
            sum(len(group) for group in current.duplicate_groups)
            / current.sample_count
        )
        return Drift(
            reference_hash=self.reference.dataset_hash,
            observed_hash=current.dataset_hash,
            language_distance=language_distance,
            mean_character_delta=(current_mean - reference_mean)
            / max(reference_mean, 1.0),
            duplicate_rate=duplicate_rate,
            threshold=self.threshold,
        )

    def measure(
        self,
        report: document.Curation,
    ) -> float:
        """Computes the mean bounded character count from a curation report.

        Args:
            report: Curation report whose document profiles are averaged.

        Returns:
            The mean character count, or 0.0 if the report has no profiles.
        """
        if not report.profiles:
            return 0.0
        return sum(
            profile.character_count for profile in report.profiles
        ) / len(report.profiles)
