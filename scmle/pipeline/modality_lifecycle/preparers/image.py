"""The decoded image preparer."""

from __future__ import annotations

from typing import Any
from typing import TYPE_CHECKING

import numpy

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.data.modalities import curation
from scmle.data.modalities import image
from scmle.models import images
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import scoring

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sklearn.linear_model import LogisticRegression


class ImagePreparer(curators.Preparer):
    """Adapter for bounded, decoded image samples.

    Attributes:
        dataset: Immutable image dataset the adapter wraps.
    """

    modality = modalities.Modality.IMAGE

    def __init__(self, dataset: image.Dataset) -> None:
        """Binds an immutable image dataset.

        Args:
            dataset: Images the lifecycle runs on.
        """
        self.dataset = dataset

    @property
    def digest(self) -> str:
        """The image corpus identity."""
        return self.dataset.digest

    @property
    def samples(self) -> tuple[image.Sample, ...]:
        """The image samples in the dataset."""
        return self.dataset.samples

    def curate(self) -> curation.Curation:
        """Builds image dimension, encoding, duplicate, and label evidence.

        Returns:
            The curation report for the dataset.
        """
        return self.dataset.curate()

    def split(self, seed: int) -> containers.Partitions:
        """Splits by capture group to prevent identity leakage.

        Args:
            seed: Random seed for the split.

        Returns:
            Train, validation and test partitions, with the dataset's group IDs
            under `groups`.
        """
        result = self.dataset.split(seed=seed)
        return containers.Partitions(
            result.train,
            result.validation,
            result.test,
            result.cohorts(),
        )

    def features(self, samples: Sequence[image.Sample]) -> containers.Batch:
        """Prepares deterministic resized-pixel features.

        Args:
            samples: Images to featurize.

        Returns:
            Batch with one row per image and 768 columns, one per pixel of a
            16 x 16 RGB image, named "pixel[row,column,channel]".
        """
        dataset = image.Dataset(tuple(samples))
        values = dataset.flatten()
        width, height = (16, 16)
        channels = 3
        names = tuple(
            f"pixel[{row},{column},{channel}]"
            for row in range(height)
            for column in range(width)
            for channel in range(channels)
        )
        return containers.Batch(
            tuple(self.sample_id(sample) for sample in samples), names, values
        )

    def sample_id(self, sample: image.Sample) -> str:
        """Gives a source-and-content identity for one image sample.

        Args:
            sample: Image sample.

        Returns:
            The image source and content hash joined by a colon.
        """
        return f"{sample.source}:{sample.content_hash}"

    def label(self, sample: image.Sample) -> object | None:
        """Gives the image label.

        Args:
            sample: Image sample.

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
        """Builds bounded occlusion evidence on transformed RGB pixels.

        Args:
            sample: Image sample to explain.
            model: Fitted classifier.
            feature_names: Names of the model's features (unused).
            coefficients: Linear coefficients (unused).

        Returns:
            Mapping with the ten 4 x 4 pixel regions whose occlusion changes the
            decision score most (score minus occluded score), the source-pixel
            occlusion evidence and limitations.

        Raises:
            TypeError: If `sample` is not an `Sample`.
            ValueError: If the features are not 16 x 16 RGB.
        """
        if not isinstance(sample, image.Sample):
            raise errors.KindError("sample must be an Sample")
        features = self.features((sample,)).values
        if features.shape[1] != 16 * 16 * 3:
            raise errors.InputError(
                "image lifecycle expects 16x16 RGB features"
            )
        selected_class, original_score = scoring.anchor(model, features)
        terms: list[tuple[str, float]] = []
        width = 16
        height = 16
        channels = 3
        for row in range(0, height, 4):
            for column in range(0, width, 4):
                end_row = min(row + 4, height)
                end_column = min(column + 4, width)
                indexes = [
                    (pixel_row * width + pixel_column) * channels + channel
                    for pixel_row in range(row, end_row)
                    for pixel_column in range(column, end_column)
                    for channel in range(channels)
                ]
                occluded = features.copy()
                occluded[0, indexes] = 0.0
                occluded_score = scoring.score(model, occluded, selected_class)
                terms.append(
                    (
                        f"region[{row}:{end_row},{column}:{end_column}]",
                        original_score - occluded_score,
                    )
                )
        selected = scoring.rank(terms)
        raw_terms = self.occlude(sample, model, selected_class, original_score)
        return {
            "status": "available",
            "method": "transformed_pixel_region_occlusion",
            "terms": [[name, float(value)] for name, value in selected],
            "baseline": "zero-valued transformed RGB pixels",
            "feature_space_only": True,
            "raw_input_occlusion": raw_terms,
            "limitations": [
                "Occlusion occurs after resizing and RGB conversion.",
                (
                    "Raw input occlusion is bounded pixel evidence, not object "
                    "localization or validated saliency."
                ),
                "Score changes are associational and not causal.",
            ],
        }

    def occlude(
        self,
        sample: image.Sample,
        model: LogisticRegression,
        selected_class: int,
        original_score: float,
    ) -> dict[str, object]:
        """Scores bounded source-pixel masks before the resize transform.

        The model still consumes the declared 16x16 RGB representation. The
        difference is that each mask is applied to the validated source
        pixels first, so the evidence records a source-space perturbation and
        the exact image dimensions used to construct it.

        Args:
            sample: Image sample to explain.
            model: Fitted classifier.
            selected_class: Index of the class whose score is tracked.
            original_score: Decision score of the unmasked image for that class.

        Returns:
            Mapping with the ten source regions (a 4 x 4 grid over the source
            image) whose masking changes the score most (original minus masked),
            the source dimensions as [width, height], the baseline and
            limitations.

        Raises:
            ValueError: If the decoded image does not have three channels.
        """
        source_pixels = sample.decode("RGB")
        source_height, source_width, source_channels = source_pixels.shape
        if source_channels != constants.RGB_CHANNEL_COUNT:
            raise errors.InputError(
                "decoded RGB image must have three channels"
            )
        image_library = image.pillow("raw image explanation")
        row_step = max(1, int(numpy.ceil(source_height / 4)))
        column_step = max(1, int(numpy.ceil(source_width / 4)))
        terms: list[tuple[str, float]] = []
        for row in range(0, source_height, row_step):
            for column in range(0, source_width, column_step):
                end_row = min(row + row_step, source_height)
                end_column = min(column + column_step, source_width)
                masked_pixels = source_pixels.copy()
                masked_pixels[row:end_row, column:end_column, :] = 0
                resized = image_library.fromarray(masked_pixels).resize(
                    (16, 16), image_library.Resampling.BILINEAR
                )
                features = (
                    numpy.asarray(resized, dtype=numpy.float32).reshape(1, -1)
                    / 255.0
                )
                masked_score = scoring.score(model, features, selected_class)
                terms.append(
                    (
                        f"source_region[{row}:{end_row},{column}:{end_column}]",
                        original_score - masked_score,
                    )
                )
        selected = scoring.rank(terms)
        return {
            "method": "raw_pixel_region_occlusion",
            "terms": [[name, float(value)] for name, value in selected],
            "source_dimensions": [source_width, source_height],
            "baseline": "zero-valued decoded RGB source pixels",
            "feature_space_only": False,
            "limitations": [
                "Source regions are pixel masks, not semantic objects.",
                "The model still consumes the resized RGB representation.",
                "Score changes are associational and not causal.",
            ],
        }

    def monitor(self, observed: curators.Preparer) -> dict[str, object]:
        """Builds the mean-pixel and duplicate drift evidence.

        Args:
            observed: Adapter over the newly observed images.

        Returns:
            The drift report as a JSON-native mapping.

        Raises:
            TypeError: If `observed` is not an `ImagePreparer`.
        """
        if not isinstance(observed, ImagePreparer):
            raise errors.KindError("observed adapter must be an ImagePreparer")
        report = images.Monitor(self.dataset.curate()).observe(observed.dataset)
        return report.to_dict()

    def remarks(self) -> tuple[str, ...]:
        """Describes image lifecycle limits.

        Returns:
            Plain statements of what the representation does not claim.
        """
        return (
            (
                "The baseline uses resized RGB pixels and does not claim "
                "localization, saliency validation, or object detection."
            ),
        )
