"""The configuration and the evidence records of the image model."""

import dataclasses
from typing import Any

from scmle.contracts import errors
from scmle.data.modalities import image

PATCH_DIMENSION_COUNT = 2


MIN_CLASS_COUNT = 2


BINARY_CLASS_COUNT = 2


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Bounded configuration for the deterministic pixel baseline.

    Attributes:
        target_size: Width and height images are resized to.
        mode: Pillow color mode of the pixels ("L", "RGB" or "RGBA").
        normalize: Whether pixel values are scaled to [0, 1].
        regularization: Inverse regularization strength of the logistic
            regression; must be positive.
        max_iter: Maximum solver iterations; must be at least 1.
        random_state: Seed for the estimator.
    """

    __module__ = "scmle.models.images"

    target_size: tuple[int, int] = (16, 16)
    mode: str = "RGB"
    normalize: bool = True
    regularization: float = 1.0
    max_iter: int = 500
    random_state: int = 0

    def __post_init__(self) -> None:
        """Validates feature and estimator bounds before fitting.

        Raises:
            ValueError: If `target_size` or `mode` is unsupported,
                `regularization` is not positive, or `max_iter` is below 1.
        """
        image.configure(self.target_size, self.mode)
        if self.regularization <= 0:
            raise errors.InputError("regularization must be positive")
        if self.max_iter < 1:
            raise errors.InputError("max_iter must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Converts the configuration to JSON-compatible values.

        Returns:
            Mapping of field name to value, with `target_size` as a list.
        """
        return {
            "target_size": list(self.target_size),
            "mode": self.mode,
            "normalize": self.normalize,
            "regularization": self.regularization,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Evaluation:
    """Evaluation evidence for a held-out image dataset.

    Attributes:
        sample_count: Number of images evaluated.
        accuracy: Share of images classified correctly.
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
    """Global or local signed pixel-contribution evidence.

    Attributes:
        model_digest: Digest of the model state that was explained.
        input_hash: Content hash of the explained image, or None for a
            global explanation.
        method: Name of the explanation method.
        terms: Pairs of feature or region name and its score, strongest
            first.
        limitations: Plain statements of what the explanation does not show.
    """

    model_digest: str
    input_hash: str | None
    method: str
    terms: tuple[tuple[str, float], ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Converts the explanation to JSON-compatible values.

        Returns:
            Mapping of field name to value, with `terms` as lists of name
            and score.
        """
        return {
            "model_digest": self.model_digest,
            "input_hash": self.input_hash,
            "method": self.method,
            "terms": [list(term) for term in self.terms],
            "limitations": list(self.limitations),
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
    """Profile drift evidence comparing reference and observed images.

    Attributes:
        reference_hash: Dataset hash of the reference images.
        observed_hash: Dataset hash of the observed images.
        mean_pixel_distance: Mean absolute difference of the per-channel
            mean pixel values, divided by 255.
        duplicate_rate: Share of observed images that are duplicates.
        threshold: Level above which a signal counts as shifted.
    """

    reference_hash: str
    observed_hash: str
    mean_pixel_distance: float
    duplicate_rate: float
    threshold: float

    @property
    def shifted(self) -> bool:
        """Whether the pixel distance or duplicate rate passes the threshold."""
        return (
            self.mean_pixel_distance > self.threshold
            or self.duplicate_rate > self.threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Converts the drift report to JSON-compatible values.

        Returns:
            Mapping of field name to value, plus a `shifted` flag.
        """
        return {**dataclasses.asdict(self), "shifted": self.shifted}
