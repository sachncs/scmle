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
from typing import Self

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scmle.data.modalities import image


@dataclasses.dataclass(frozen=True, slots=True)
class ImageModelConfig:
    """Bounded configuration for the deterministic pixel baseline."""

    target_size: tuple[int, int] = (16, 16)
    mode: str = "RGB"
    normalize: bool = True
    regularization: float = 1.0
    max_iter: int = 500
    random_state: int = 0

    def __post_init__(self) -> None:
        """Validate feature and estimator bounds before fitting."""
        image.validate_feature_options(self.target_size, self.mode)
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible configuration evidence."""
        return {
            "target_size": list(self.target_size),
            "mode": self.mode,
            "normalize": self.normalize,
            "regularization": self.regularization,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ImageEvaluation:
    """Evaluation evidence for a held-out image dataset."""

    sample_count: int
    accuracy: float
    macro_f1: float
    log_loss: float
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evaluation evidence."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ImageExplanation:
    """Global or local signed pixel-contribution evidence."""

    model_digest: str
    input_hash: str | None
    method: str
    terms: tuple[tuple[str, float], ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible explanation evidence."""
        return {
            "model_digest": self.model_digest,
            "input_hash": self.input_hash,
            "method": self.method,
            "terms": [list(term) for term in self.terms],
            "limitations": list(self.limitations),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class ImageModelArtifact:
    """Identity of an atomically written trusted-model package."""

    path: str
    digest: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible package identity."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ImageDriftReport:
    """Profile drift evidence comparing reference and observed images."""

    reference_hash: str
    observed_hash: str
    mean_pixel_distance: float
    duplicate_rate: float
    threshold: float

    @property
    def shifted(self) -> bool:
        """Return whether a monitored image-profile signal shifted."""
        return (
            self.mean_pixel_distance > self.threshold
            or self.duplicate_rate > self.threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible monitoring evidence."""
        return {**dataclasses.asdict(self), "shifted": self.shifted}


class ImageClassifier:
    """Pixel-baseline classifier with evidence and safe package boundaries."""

    def __init__(self, config: ImageModelConfig | None = None) -> None:
        """Initialize an unfitted classifier."""
        self.config = config or ImageModelConfig()
        self.estimator: LogisticRegression | None = None
        self.feature_names: tuple[str, ...] = ()
        self.curation_report: image.ModalityCurationReport | None = None

    @property
    def fitted(self) -> bool:
        """Return whether feature preparation and modelling are complete."""
        return self.estimator is not None

    @property
    def classes(self) -> tuple[Any, ...]:
        """Return fitted class labels in estimator order."""
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        return tuple(estimator.classes_.tolist())

    @property
    def model_digest(self) -> str:
        """Return a deterministic digest of the fitted model state."""
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        state = {
            "config": self.config.to_dict(),
            "feature_names": self.feature_names,
            "classes": estimator.classes_.tolist(),
            "coef": estimator.coef_.tolist(),
            "intercept": estimator.intercept_.tolist(),
        }
        encoded = json.dumps(
            state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fit(self, training_data: image.ImageDataset) -> Self:
        """Fit the baseline on labelled, deduplicated images.

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
        if len(set(labels)) < 2:
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

    def predict(self, dataset: image.ImageDataset) -> tuple[Any, ...]:
        """Predict labels for validated images."""
        self.require_fitted()
        if not dataset.samples:
            return ()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        return tuple(estimator.predict(self.transform(dataset)).tolist())

    def predict_proba(
        self,
        dataset: image.ImageDataset,
    ) -> tuple[tuple[float, ...], ...]:
        """Return class probabilities for validated images."""
        self.require_fitted()
        if not dataset.samples:
            return ()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        probabilities = estimator.predict_proba(self.transform(dataset))
        return tuple(tuple(row) for row in probabilities)

    def evaluate(self, dataset: image.ImageDataset) -> ImageEvaluation:
        """Evaluate labelled images without refitting or changing features."""
        if not dataset.samples:
            raise ValueError("evaluation images must not be empty")
        labels = tuple(sample.label for sample in dataset.samples)
        if any(label is None for label in labels):
            raise ValueError("every evaluation image must have a label")
        actual = list(labels)
        predicted = self.predict(dataset)
        probabilities = self.predict_proba(dataset)
        return ImageEvaluation(
            sample_count=len(dataset.samples),
            accuracy=float(accuracy_score(actual, predicted)),
            macro_f1=float(f1_score(actual, predicted, average="macro")),
            log_loss=float(
                log_loss(actual, probabilities, labels=list(self.classes))
            ),
            labels=tuple(str(value) for value in self.classes),
        )

    def global_explanation(self, top_k: int = 10) -> ImageExplanation:
        """Return global pixel importance from linear coefficients."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        strengths = np.mean(np.abs(estimator.coef_), axis=0)
        selected = sorted(
            zip(self.feature_names, strengths, strict=True),
            key=lambda item: (-float(item[1]), item[0]),
        )[:top_k]
        return ImageExplanation(
            model_digest=self.model_digest,
            input_hash=None,
            method="linear_global_pixel_weight",
            terms=tuple((name, float(value)) for name, value in selected),
            limitations=(
                "Pixel weights are associational, not causal.",
                "Resize, color mode, and corpus composition constrain the "
                "explanation.",
            ),
        )

    def local_explanation(
        self,
        sample: image.ImageSample,
        top_k: int = 10,
    ) -> ImageExplanation:
        """Return local signed pixel contributions for one image."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        matrix = self.transform(image.ImageDataset((sample,)))
        predicted_label = estimator.predict(matrix)[0]
        class_index = int(
            np.flatnonzero(estimator.classes_ == predicted_label)[0]
        )
        coefficients = estimator.coef_[0]
        if len(estimator.classes_) == 2 and class_index == 0:
            coefficients = -coefficients
        elif len(estimator.classes_) > 2:
            coefficients = estimator.coef_[class_index]
        contributions = sorted(
            zip(
                self.feature_names,
                matrix[0] * coefficients,
                strict=True,
            ),
            key=lambda item: (-abs(float(item[1])), item[0]),
        )[:top_k]
        return ImageExplanation(
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
        sample: image.ImageSample,
        top_k: int = 10,
        patch_size: tuple[int, int] = (4, 4),
    ) -> ImageExplanation:
        """Measure predicted-class score changes under regional occlusion.

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
            RuntimeError: If the classifier has not been fitted.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if len(patch_size) != 2 or any(size < 1 for size in patch_size):
            raise ValueError("patch_size must contain positive dimensions")
        self.require_fitted()
        features = self.transform(image.ImageDataset((sample,)))
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
        return ImageExplanation(
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

    def predicted_class_score(self, features: np.ndarray) -> float:
        """Return the decision score for the class predicted by the model."""
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        predicted = estimator.predict(features)[0]
        class_index = int(np.flatnonzero(estimator.classes_ == predicted)[0])
        decision = np.asarray(estimator.decision_function(features))
        if decision.ndim == 1:
            raw_score = float(decision[0])
            return raw_score if class_index == 1 else -raw_score
        return float(decision[0, class_index])

    def save(self, path: str | pathlib.Path) -> ImageModelArtifact:
        """Atomically save this fitted classifier as a verified package."""
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
            with model_path.open("rb") as stream:
                os.fsync(stream.fileno())
            manifest = {
                "schema_version": 1,
                "model_path": "model.joblib",
                "artifact_digest": file_digest(model_path),
                "model_digest": self.model_digest,
                "config": self.config.to_dict(),
            }
            manifest_path = temporary_directory / "manifest.json"
            with manifest_path.open("w", encoding="utf-8") as stream:
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
            flush_directory(temporary_directory)
            os.replace(temporary_directory, target)
            flush_directory(target.parent)
        except Exception:
            if temporary_directory is not None:
                shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        return ImageModelArtifact(str(target), str(manifest["artifact_digest"]))

    @classmethod
    def load(cls, path: str | pathlib.Path, trusted: bool = False) -> Self:
        """Load a package only after an explicit caller trust decision."""
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
            raise TypeError("package does not contain an ImageClassifier")
        loaded.require_fitted()
        if loaded.model_digest != manifest.get("model_digest"):
            raise ValueError("image model digest does not match manifest")
        return loaded

    def transform(self, dataset: image.ImageDataset) -> np.ndarray:
        """Prepare pixels with the fitted configuration."""
        self.require_fitted()
        return dataset.prepare_tabular_features(
            self.config.target_size,
            self.config.mode,
            self.config.normalize,
        )

    def make_feature_names(self) -> tuple[str, ...]:
        """Return deterministic channel-major names for flattened pixels."""
        width, height = self.config.target_size
        channels = 1 if self.config.mode == "L" else len(self.config.mode)
        names = []
        for row in range(height):
            for column in range(width):
                for channel in range(channels):
                    names.append(f"pixel[{row},{column},{channel}]")
        return tuple(names)

    def require_fitted(self) -> None:
        """Raise when an operation needs a fitted classifier."""
        if not self.fitted:
            raise RuntimeError("image classifier has not been fitted")


class ImageMonitor:
    """Monitor bounded image-profile signals against a reference snapshot."""

    def __init__(
        self,
        reference: image.ModalityCurationReport,
        threshold: float = 0.25,
    ) -> None:
        """Initialize a monitor with an explicit normalized threshold."""
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.reference = reference
        self.threshold = threshold

    def observe(self, observed: image.ImageDataset) -> ImageDriftReport:
        """Compare observed mean pixels and duplicate rate with reference."""
        current = observed.curate()
        reference_mean = np.asarray(
            self.reference.profile.get("mean_pixel", (0.0, 0.0, 0.0)),
            dtype=float,
        )
        current_mean = np.asarray(
            current.profile.get("mean_pixel", (0.0, 0.0, 0.0)),
            dtype=float,
        )
        distance = float(np.mean(np.abs(current_mean - reference_mean)) / 255.0)
        duplicate_rate = current.profile["duplicate_count"] / max(
            current.sample_count, 1
        )
        return ImageDriftReport(
            reference_hash=self.reference.dataset_hash,
            observed_hash=current.dataset_hash,
            mean_pixel_distance=distance,
            duplicate_rate=float(duplicate_rate),
            threshold=self.threshold,
        )


def file_digest(path: pathlib.Path) -> str:
    """Return a SHA-256 digest for a package artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def flush_directory(path: pathlib.Path) -> None:
    """Flush a directory entry after an atomic package replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "ImageClassifier",
    "ImageDriftReport",
    "ImageEvaluation",
    "ImageExplanation",
    "ImageModelArtifact",
    "ImageModelConfig",
    "ImageMonitor",
    "file_digest",
    "flush_directory",
]
