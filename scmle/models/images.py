"""A bounded, explainable baseline lifecycle for labelled images.

The classifier deliberately uses deterministic resized pixels rather than
claiming a production computer-vision model. It is an experimental library
path that makes its curation, feature, evaluation, explanation, packaging,
and drift boundaries explicit.
"""

import dataclasses
import hashlib
import json
import os
import pathlib
import shutil
import tempfile
from typing import Any
from typing import cast
from typing import Self

import joblib
import numpy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scmle import durability
from scmle.data.modalities import curation
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
        image.validate_feature_options(self.target_size, self.mode)
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")

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


class Classifier:
    """Pixel-baseline classifier with evidence and safe package boundaries.

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
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        return tuple(estimator.classes_.tolist())

    @property
    def model_digest(self) -> str:
        """The SHA-256 digest of the configuration, classes and coefficients.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
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
            raise ValueError(
                "image curation has unresolved blocking findings ("
                + detail
                + "): "
                + ", ".join(blocking_ids)
            )
        if report.profile["duplicate_count"]:
            raise ValueError(
                "training images contain duplicate content; deduplicate "
                "before fitting"
            )
        labels = tuple(sample.label for sample in training_data.samples)
        if any(label is None for label in labels):
            raise ValueError("every training image must have a label")
        if len(set(labels)) < MIN_CLASS_COUNT:
            raise ValueError("training images must contain two classes")
        features = training_data.prepare_tabular_features(
            self.config.target_size,
            self.config.mode,
            self.config.normalize,
        )
        if features.shape[1] == 0:
            raise ValueError("training images contain no usable pixels")
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
        self.require_fitted()
        if not dataset.samples:
            return ()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
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
        self.require_fitted()
        if not dataset.samples:
            return ()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
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
            raise ValueError("evaluation images must not be empty")
        labels = tuple(sample.label for sample in dataset.samples)
        if any(label is None for label in labels):
            raise ValueError("every evaluation image must have a label")
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

    def global_explanation(self, top_k: int = 10) -> Explanation:
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
            raise ValueError("top_k must be positive")
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
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

    def local_explanation(
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
            raise ValueError("top_k must be positive")
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
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

    def occlusion_explanation(
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
            raise ValueError("top_k must be positive")
        if len(patch_size) != PATCH_DIMENSION_COUNT or any(
            size < 1 for size in patch_size
        ):
            raise ValueError("patch_size must contain positive dimensions")
        self.require_fitted()
        features = self.transform(image.Dataset((sample,)))
        original_score = self.predicted_class_score(features)
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
                delta = original_score - self.predicted_class_score(occluded)
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

    def predicted_class_score(self, features: numpy.ndarray) -> float:
        """Computes the decision score of the predicted class.

        Args:
            features: Feature matrix whose first row is scored.

        Returns:
            The decision-function value for the class predicted for the
            first row; larger means more support for that class.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        predicted = estimator.predict(features)[0]
        class_index = int(numpy.flatnonzero(estimator.classes_ == predicted)[0])
        decision = numpy.asarray(estimator.decision_function(features))
        if decision.ndim == 1:
            raw_score = float(decision[0])
            return raw_score if class_index == 1 else -raw_score
        return float(decision[0, class_index])

    def save(self, path: str | pathlib.Path) -> Package:
        """Atomically saves this fitted classifier as a verified package.

        Args:
            path: Directory to create; it must not exist yet.

        Returns:
            The package path and the digest of its model payload.

        Raises:
            FileExistsError: If `path` already exists.
        """
        self.require_fitted()
        target = pathlib.Path(path).expanduser()
        if target.exists():
            raise FileExistsError(f"model package already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_directory: pathlib.Path | None = None
        try:
            temporary_directory = pathlib.Path(
                tempfile.mkdtemp(dir=target.parent, prefix=f".{target.name}.")
            )
            model_path = temporary_directory / "model.joblib"
            joblib.dump(self, model_path, compress=3)
            durability.sync_file(model_path)
            manifest = {
                "schema_version": 1,
                "model_path": "model.joblib",
                "artifact_digest": file_digest(model_path),
                "model_digest": self.model_digest,
                "config": self.config.to_dict(),
            }
            manifest_path = temporary_directory / "manifest.json"
            with manifest_path.open(
                "w", encoding="utf-8", newline="\n"
            ) as stream:
                json.dump(
                    manifest,
                    stream,
                    ensure_ascii=True,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            durability.sync_directory(temporary_directory)
            pathlib.Path(temporary_directory).replace(target)
            durability.sync_directory(target.parent)
        except Exception:
            if temporary_directory is not None:
                shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        return Package(str(target), str(manifest["artifact_digest"]))

    @classmethod
    def load(cls, path: str | pathlib.Path, trusted: bool = False) -> Self:
        """Loads a package only after an explicit caller trust decision.

        Args:
            path: Package directory written by `save`.
            trusted: Must be True; loading unpickles the payload, so only
                packages from trusted sources may be loaded.

        Returns:
            The fitted classifier stored in the package.

        Raises:
            PermissionError: If `trusted` is not True.
            FileNotFoundError: If `path` is not a directory.
            ValueError: If the manifest or payload is missing, the version
                is unsupported, or a digest does not match.
            TypeError: If the payload is not an `Classifier`.
        """
        if not trusted:
            raise PermissionError(
                "loading an image model package requires trusted=True"
            )
        target = pathlib.Path(path).expanduser()
        if not target.is_dir():
            raise FileNotFoundError(target)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("image model package has no manifest")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported image model package version")
        model_path = target / str(manifest.get("model_path", ""))
        if not model_path.is_file():
            raise ValueError("image model package has no model payload")
        if file_digest(model_path) != manifest.get("artifact_digest"):
            raise ValueError("image model payload digest does not match")
        loaded = joblib.load(model_path)
        if not isinstance(loaded, cls):
            raise TypeError("package does not contain an Classifier")
        loaded.require_fitted()
        if loaded.model_digest != manifest.get("model_digest"):
            raise ValueError("image model digest does not match manifest")
        return loaded

    def transform(self, dataset: image.Dataset) -> numpy.ndarray:
        """Prepares pixels with the fitted configuration.

        Args:
            dataset: Images to convert.

        Returns:
            Array with one row of flattened pixels per image.
        """
        self.require_fitted()
        return dataset.prepare_tabular_features(
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

    def require_fitted(self) -> None:
        """Checks that the classifier has been fitted.

        Raises:
            RuntimeError: If it has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("image classifier has not been fitted")


class Monitor:
    """Monitors bounded image-profile signals against a reference snapshot.

    Attributes:
        reference: Curation report of the reference images.
        threshold: Level above which a signal counts as shifted.
    """

    def __init__(
        self,
        reference: curation.Curation,
        threshold: float = 0.25,
    ) -> None:
        """Initializes a monitor with an explicit normalized threshold.

        Args:
            reference: Curation report of the reference images.
            threshold: Level above which a signal counts as shifted; in
                (0, 1].

        Raises:
            ValueError: If `threshold` is outside (0, 1].
        """
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.reference = reference
        self.threshold = threshold

    def observe(self, observed: image.Dataset) -> Drift:
        """Compares observed mean pixels and duplicate rate with reference.

        Args:
            observed: Images to compare against the reference.

        Returns:
            Drift report with the mean pixel distance and duplicate rate.
        """
        current = observed.curate()
        reference_mean = numpy.asarray(
            self.reference.profile.get("mean_pixel", (0.0, 0.0, 0.0)),
            dtype=float,
        )
        current_mean = numpy.asarray(
            current.profile.get("mean_pixel", (0.0, 0.0, 0.0)),
            dtype=float,
        )
        distance = float(
            numpy.mean(numpy.abs(current_mean - reference_mean)) / 255.0
        )
        duplicate_rate = current.profile["duplicate_count"] / max(
            current.sample_count, 1
        )
        return Drift(
            reference_hash=self.reference.dataset_hash,
            observed_hash=current.dataset_hash,
            mean_pixel_distance=distance,
            duplicate_rate=float(duplicate_rate),
            threshold=self.threshold,
        )


def file_digest(path: pathlib.Path) -> str:
    """Computes a SHA-256 digest for a package artifact.

    Args:
        path: File to hash.

    Returns:
        The hexadecimal digest of the file contents.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


__all__ = [
    "Classifier",
    "Config",
    "Drift",
    "Evaluation",
    "Explanation",
    "Monitor",
    "Package",
    "file_digest",
]
