"""The configuration and the evidence records of the document model."""

import dataclasses
from typing import Any

from scmle.contracts import errors
from scmle.data.modalities import document

NGRAM_BOUND_COUNT = 2


MIN_CLASS_COUNT = 2


BINARY_CLASS_COUNT = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Bounded configuration for the reproducible document baseline.

    Attributes:
        max_features: Maximum vocabulary size; at least 1.
        ngram_range: Smallest and largest n-gram length, 1 <= low <= high.
        min_df: Minimum number of documents a term must appear in; at least 1.
        regularization: Inverse regularization strength of the logistic
            regression; must be positive.
        max_iter: Maximum solver iterations; at least 1.
        random_state: Seed for the estimator.
    """

    __module__ = "scmle.models.documents"

    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    regularization: float = 1.0
    max_iter: int = 500
    random_state: int = 0

    def __post_init__(self) -> None:
        """Validates resource and estimator bounds before fitting.

        Raises:
            ValueError: If `max_features`, `min_df` or `max_iter` is below 1,
                `ngram_range` is not two ordered positive bounds, or
                `regularization` is not positive.
        """
        if self.max_features < 1:
            raise errors.InputError("max_features must be positive")
        if len(self.ngram_range) != NGRAM_BOUND_COUNT or not (
            1 <= self.ngram_range[0] <= self.ngram_range[1]
        ):
            raise errors.InputError(
                "ngram_range must contain positive ordered bounds"
            )
        if self.min_df < 1:
            raise errors.InputError("min_df must be positive")
        if self.regularization <= 0:
            raise errors.InputError("regularization must be positive")
        if self.max_iter < 1:
            raise errors.InputError("max_iter must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Converts the configuration to JSON-compatible values.

        Returns:
            Mapping of field name to value, with `ngram_range` as a list.
        """
        return {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "regularization": self.regularization,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Evaluation:
    """Evaluation evidence for a held-out document split.

    Attributes:
        sample_count: Number of documents evaluated.
        accuracy: Share of documents classified correctly.
        macro_f1: Unweighted mean of the per-class harmonic mean of
            precision and recall.
        log_loss: Mean negative log probability of the true class.
        labels: Class labels, as text, in the classifier's class order.
    """

    sample_count: int
    accuracy: float
    macro_f1: float
    log_loss: float
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Converts the evaluation to JSON-compatible values.

        Returns:
            Mapping of field name to value.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Explanation:
    """Global or local term-weight explanation evidence.

    Attributes:
        model_digest: Digest of the model state that was explained.
        input_hash: Content hash of the explained document, or None for a
            global explanation.
        method: Name of the explanation method.
        terms: Pairs of term and its score, strongest first.
        limitations: Plain statements of what the explanation does not show.
        citations: Source locations of the explained document, if any.
    """

    model_digest: str
    input_hash: str | None
    method: str
    terms: tuple[tuple[str, float], ...]
    limitations: tuple[str, ...]
    citations: tuple[document.Citation, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Converts the explanation to JSON-compatible values.

        Returns:
            Mapping of field name to value, with `terms` as lists of term
            and score and `citations` as dictionaries.
        """
        return {
            "model_digest": self.model_digest,
            "input_hash": self.input_hash,
            "method": self.method,
            "terms": [list(term) for term in self.terms],
            "limitations": list(self.limitations),
            "citations": [citation.to_dict() for citation in self.citations],
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Package:
    """Identity of an atomically written trusted-model package.

    Attributes:
        path: Directory of the package.
        digest: SHA-256 digest of the model payload file.
        schema_version: Version of the package layout.
    """

    path: str
    digest: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Converts the package identity to JSON-compatible values.

        Returns:
            Mapping of field name to value.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Drift:
    """Profile drift evidence comparing a reference and observed corpus.

    Attributes:
        reference_hash: Dataset hash of the reference corpus.
        observed_hash: Dataset hash of the observed corpus.
        language_distance: Total variation distance between the language
            shares of the two corpora.
        mean_character_delta: Change in mean character count relative to the
            reference mean.
        duplicate_rate: Share of observed documents in duplicate groups.
        threshold: Level above which a signal counts as shifted.
    """

    reference_hash: str
    observed_hash: str
    language_distance: float
    mean_character_delta: float
    duplicate_rate: float
    threshold: float

    @property
    def shifted(self) -> bool:
        """Whether any monitored profile signal exceeds the threshold."""
        return (
            self.language_distance > self.threshold
            or abs(self.mean_character_delta) > self.threshold
            or self.duplicate_rate > self.threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the drift report to JSON-compatible values.

        Returns:
            Mapping of field name to value, plus a `shifted` flag.
        """
        return {
            **dataclasses.asdict(self),
            "shifted": self.shifted,
        }
