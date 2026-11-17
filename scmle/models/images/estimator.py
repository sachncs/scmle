"""The fitted pixel and logistic regression estimator."""

import hashlib
import json
from typing import Any
from typing import cast
from typing import Self

import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scmle.contracts import errors
from scmle.data.modalities import curation
from scmle.data.modalities import image
from scmle.models.images.records import Config
from scmle.models.images.records import Evaluation
from scmle.models.images.records import MIN_CLASS_COUNT


class Estimator:
    """Resized pixels and a logistic regression, fitted and evaluated.

    Attributes:
        config: Feature and estimator configuration.
        estimator: Fitted logistic regression, or None before fitting.
        feature_names: Names of the flattened pixel features, empty before
            fitting.
        curation_report: Curation report of the training images, or None
            before fitting.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initializes an unfitted classifier.

        Args:
            config: Configuration to use; defaults to `Config()`.
        """
        self.config = config or Config()
        self.estimator: LogisticRegression | None = None
        self.feature_names: tuple[str, ...] = ()
        self.curation_report: curation.Curation | None = None

    @property
    def fitted(self) -> bool:
        """Whether the classifier has been fitted."""
        return self.estimator is not None

    @property
    def classes(self) -> tuple[Any, ...]:
        """The fitted class labels in estimator order.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.preflight()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        return tuple(estimator.classes_.tolist())

    @property
    def model_digest(self) -> str:
        """The SHA-256 digest of the configuration, classes and coefficients.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.preflight()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        state = {
            "config": self.config.to_dict(),
            "feature_names": self.feature_names,
            "classes": estimator.classes_.tolist(),
            "coef": estimator.coef_.tolist(),
            "intercept": cast("numpy.ndarray", estimator.intercept_).tolist(),
        }
        encoded = json.dumps(
            state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fit(self, training_data: image.Dataset) -> Self:
        """Fits the baseline on labelled, deduplicated images.

        Args:
            training_data: Curated training images with at least two labels.

        Returns:
            The fitted classifier.

        Raises:
            ValueError: If labels are missing, duplicate content is present,
                curation is blocking, or fewer than two classes are available.
        """
        report = training_data.curate()
        if report.unresolved_blocking_findings:
            blocking_ids = tuple(
                finding.finding_id
                for finding in report.unresolved_blocking_findings
            )
            detail = (
                "duplicate content has conflicting labels"
                if "risk.image-conflicting-duplicate-labels" in blocking_ids
                else "curation policy requires review"
            )
            raise errors.InputError(
                "image curation has unresolved blocking findings ("
                + detail
                + "): "
                + ", ".join(blocking_ids)
            )
        if report.profile["duplicate_count"]:
            raise errors.InputError(
                "training images contain duplicate content; deduplicate "
                "before fitting"
            )
        labels = tuple(sample.label for sample in training_data.samples)
        if any(label is None for label in labels):
            raise errors.InputError("every training image must have a label")
        if len(set(labels)) < MIN_CLASS_COUNT:
            raise errors.InputError("training images must contain two classes")
        features = training_data.flatten(
            self.config.target_size,
            self.config.mode,
            self.config.normalize,
        )
        if features.shape[1] == 0:
            raise errors.InputError("training images contain no usable pixels")
        estimator = LogisticRegression(
            C=self.config.regularization,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            solver="lbfgs",
        )
        estimator.fit(features, labels)
        self.estimator = estimator
        self.feature_names = self.make_feature_names()
        self.curation_report = report
        return self

    def predict(self, dataset: image.Dataset) -> tuple[Any, ...]:
        """Predicts labels for validated images.

        Args:
            dataset: Images to classify.

        Returns:
            Predicted labels, one per image in order; empty if the dataset
            has no samples.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.preflight()
        if not dataset.samples:
            return ()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        return tuple(
            cast(
                "numpy.ndarray", estimator.predict(self.transform(dataset))
            ).tolist()
        )

    def predict_proba(
        self,
        dataset: image.Dataset,
    ) -> tuple[tuple[float, ...], ...]:
        """Predicts class probabilities for validated images.

        Args:
            dataset: Images to classify.

        Returns:
            One tuple of probabilities per image, in `classes` order; empty
            if the dataset has no samples.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.preflight()
        if not dataset.samples:
            return ()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        probabilities = estimator.predict_proba(self.transform(dataset))
        return tuple(tuple(row) for row in probabilities)

    def evaluate(self, dataset: image.Dataset) -> Evaluation:
        """Evaluates labelled images without refitting or changing features.

        Args:
            dataset: Labelled images to score.

        Returns:
            Evidence with the sample count, accuracy, macro-averaged F-score,
            log loss and the class labels.

        Raises:
            ValueError: If the dataset is empty, any image has no label, or a
                label is not one of the classifier's classes.
            RuntimeError: If the classifier has not been fitted.
        """
        if not dataset.samples:
            raise errors.InputError("evaluation images must not be empty")
        labels = tuple(sample.label for sample in dataset.samples)
        if any(label is None for label in labels):
            raise errors.InputError("every evaluation image must have a label")
        actual = list(labels)
        predicted = self.predict(dataset)
        probabilities = self.predict_proba(dataset)
        return Evaluation(
            sample_count=len(dataset.samples),
            accuracy=float(accuracy_score(actual, predicted)),
            macro_f1=float(f1_score(actual, predicted, average="macro")),
            log_loss=float(
                log_loss(actual, probabilities, labels=list(self.classes))
            ),
            labels=tuple(str(value) for value in self.classes),
        )

    def transform(self, dataset: image.Dataset) -> numpy.ndarray:
        """Prepares pixels with the fitted configuration.

        Args:
            dataset: Images to convert.

        Returns:
            Array with one row of flattened pixels per image.
        """
        self.preflight()
        return dataset.flatten(
            self.config.target_size,
            self.config.mode,
            self.config.normalize,
        )

    def make_feature_names(self) -> tuple[str, ...]:
        """Builds the names of the flattened pixel features.

        Returns:
            Names of the form "pixel[row,column,channel]", ordered by row,
            then column, then channel.
        """
        width, height = self.config.target_size
        channels = 1 if self.config.mode == "L" else len(self.config.mode)
        names: list[str] = []
        for row in range(height):
            for column in range(width):
                names.extend(
                    f"pixel[{row},{column},{channel}]"
                    for channel in range(channels)
                )
        return tuple(names)

    def preflight(self) -> None:
        """Checks that the classifier has been fitted.

        Raises:
            RuntimeError: If it has not been fitted.
        """
        if not self.fitted:
            raise errors.OrderError("image classifier has not been fitted")
