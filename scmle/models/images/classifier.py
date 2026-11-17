"""The image classifier and its explanations."""

import numpy

from scmle.contracts import errors
from scmle.data.modalities import image
from scmle.models.images.archive import Archive
from scmle.models.images.records import BINARY_CLASS_COUNT
from scmle.models.images.records import Explanation
from scmle.models.images.records import PATCH_DIMENSION_COUNT


class Classifier(Archive):
    """Pixel-baseline classifier with evidence and safe package boundaries.

    Attributes:
        config: Feature and estimator configuration.
        estimator: Fitted logistic regression, or None before fitting.
        feature_names: Names of the flattened pixel features, empty before
            fitting.
        curation_report: Curation report of the training images, or None
            before fitting.
    """

    __module__ = "scmle.models.images"

    def overview(self, top_k: int = 10) -> Explanation:
        """Ranks pixels by their mean absolute linear coefficient.

        Args:
            top_k: Number of pixel features to keep.

        Returns:
            Global explanation whose terms are (feature name, mean absolute
            coefficient) pairs, largest first.

        Raises:
            ValueError: If `top_k` is below 1.
            RuntimeError: If the classifier has not been fitted.
        """
        if top_k < 1:
            raise errors.InputError("top_k must be positive")
        self.preflight()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        strengths = numpy.mean(numpy.abs(estimator.coef_), axis=0)
        selected = sorted(
            zip(self.feature_names, strengths, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )[:top_k]
        return Explanation(
            model_digest=self.model_digest,
            input_hash=None,
            method="linear_global_pixel_weight",
            terms=tuple((name, float(value)) for name, value in selected),
            limitations=(
                "Pixel weights are associational, not causal.",
                (
                    "Resize, color mode, and corpus composition constrain the "
                    "explanation."
                ),
            ),
        )

    def spotlight(
        self,
        sample: image.Sample,
        top_k: int = 10,
    ) -> Explanation:
        """Computes signed pixel contributions for one image.

        Args:
            sample: Image to explain.
            top_k: Number of pixel features to keep.

        Returns:
            Local explanation whose terms are (feature name, coefficient
            times pixel value) pairs for the predicted class, largest
            absolute value first; positive favors the predicted class.

        Raises:
            ValueError: If `top_k` is below 1.
            RuntimeError: If the classifier has not been fitted.
        """
        if top_k < 1:
            raise errors.InputError("top_k must be positive")
        self.preflight()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        matrix = self.transform(image.Dataset((sample,)))
        predicted_label = estimator.predict(matrix)[0]
        class_index = int(
            numpy.flatnonzero(estimator.classes_ == predicted_label)[0]
        )
        coefficients = estimator.coef_[0]
        if len(estimator.classes_) == BINARY_CLASS_COUNT and class_index == 0:
            coefficients = -coefficients
        elif len(estimator.classes_) > BINARY_CLASS_COUNT:
            coefficients = estimator.coef_[class_index]
        contributions = sorted(
            zip(
                self.feature_names,
                matrix[0] * coefficients,
                strict=True,
            ),
            key=lambda item: (-abs(float(item[1])), item[0]),
        )[:top_k]
        return Explanation(
            model_digest=self.model_digest,
            input_hash=sample.content_hash,
            method="linear_local_pixel_contribution",
            terms=tuple((name, float(value)) for name, value in contributions),
            limitations=(
                "Contributions explain this fitted linear approximation only.",
                "Positive contribution favors the predicted class.",
            ),
        )

    def occlude(
        self,
        sample: image.Sample,
        top_k: int = 10,
        patch_size: tuple[int, int] = (4, 4),
    ) -> Explanation:
        """Measures predicted-class score changes under regional occlusion.

        Occlusion is performed after the declared resize and color transform.
        A patch is replaced by the zero-valued feature baseline, which is black
        for the supported image representations. A positive term means that
        the unoccluded patch supports the predicted class relative to that
        baseline.

        Args:
            sample: Image to explain.
            top_k: Maximum number of regions to retain.
            patch_size: Region width and height in transformed pixels.

        Returns:
            Deterministic local regional perturbation evidence.

        Raises:
            ValueError: If the requested output or patch dimensions are
                invalid.
        """
        if top_k < 1:
            raise errors.InputError("top_k must be positive")
        if len(patch_size) != PATCH_DIMENSION_COUNT or any(
            size < 1 for size in patch_size
        ):
            raise errors.InputError(
                "patch_size must contain positive dimensions"
            )
        self.preflight()
        features = self.transform(image.Dataset((sample,)))
        original_score = self.confidence(features)
        width, height = self.config.target_size
        channels = 1 if self.config.mode == "L" else len(self.config.mode)
        patch_width, patch_height = patch_size
        terms: list[tuple[str, float]] = []
        for row in range(0, height, patch_height):
            for column in range(0, width, patch_width):
                end_row = min(row + patch_height, height)
                end_column = min(column + patch_width, width)
                indexes = [
                    (pixel_row * width + pixel_column) * channels + channel
                    for pixel_row in range(row, end_row)
                    for pixel_column in range(column, end_column)
                    for channel in range(channels)
                ]
                occluded = features.copy()
                occluded[0, indexes] = 0.0
                delta = original_score - self.confidence(occluded)
                terms.append(
                    (
                        f"region[{row}:{end_row},{column}:{end_column}]",
                        float(delta),
                    )
                )
        selected = sorted(
            terms,
            key=lambda item: (-abs(item[1]), item[0]),
        )[:top_k]
        return Explanation(
            model_digest=self.model_digest,
            input_hash=sample.content_hash,
            method="occlusion_pixel_region",
            terms=tuple(selected),
            limitations=(
                "Occlusion replaces transformed pixels with a black baseline.",
                "Regions are not raw-image saliency or object localization.",
                "The score change is associational, not causal.",
            ),
        )

    def confidence(self, features: numpy.ndarray) -> float:
        """Computes the decision score of the predicted class.

        Args:
            features: Feature matrix whose first row is scored.

        Returns:
            The decision-function value for the class predicted for the
            first row; larger means more support for that class.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.preflight()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        predicted = estimator.predict(features)[0]
        class_index = int(numpy.flatnonzero(estimator.classes_ == predicted)[0])
        decision = numpy.asarray(estimator.decision_function(features))
        if decision.ndim == 1:
            raw_score = float(decision[0])
            return raw_score if class_index == 1 else -raw_score
        return float(decision[0, class_index])
