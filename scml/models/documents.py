"""A bounded, explainable baseline lifecycle for labelled documents.

This module is intentionally a document-specific capability slice. It uses
the validated document contracts, keeps feature preparation and modelling in
one immutable configuration, and refuses to imply that a serialized model is
safe to load without an explicit trust decision.
"""

from collections.abc import Sequence
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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scml.data.modalities import document


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentModelConfig:
    """Bounded configuration for the reproducible document baseline."""

    max_features: int = 10_000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int = 1
    regularization: float = 1.0
    max_iter: int = 500
    random_state: int = 0

    def __post_init__(self) -> None:
        """Validate resource and estimator bounds before fitting."""
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if len(self.ngram_range) != 2 or not (
            1 <= self.ngram_range[0] <= self.ngram_range[1]
        ):
            raise ValueError("ngram_range must contain positive ordered bounds")
        if self.min_df < 1:
            raise ValueError("min_df must be positive")
        if self.regularization <= 0:
            raise ValueError("regularization must be positive")
        if self.max_iter < 1:
            raise ValueError("max_iter must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible configuration evidence."""
        return {
            "max_features": self.max_features,
            "ngram_range": list(self.ngram_range),
            "min_df": self.min_df,
            "regularization": self.regularization,
            "max_iter": self.max_iter,
            "random_state": self.random_state,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentEvaluation:
    """Evaluation evidence for a held-out document split."""

    sample_count: int
    accuracy: float
    macro_f1: float
    log_loss: float
    labels: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evaluation evidence."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentExplanation:
    """Global or local term-weight explanation evidence."""

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
class DocumentModelArtifact:
    """Identity of an atomically written trusted-model package."""

    path: str
    digest: str
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible package identity."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class DocumentDriftReport:
    """Profile drift evidence comparing a reference and observed corpus."""

    reference_hash: str
    observed_hash: str
    language_distance: float
    mean_character_delta: float
    duplicate_rate: float
    threshold: float

    @property
    def shifted(self) -> bool:
        """Return whether any monitored profile signal exceeds its bound."""
        return (
            self.language_distance > self.threshold
            or abs(self.mean_character_delta) > self.threshold
            or self.duplicate_rate > self.threshold
        )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible monitoring evidence."""
        return {
            **dataclasses.asdict(self),
            "shifted": self.shifted,
        }


class DocumentClassifier:
    """TF-IDF document classifier with evidence and safe package boundaries."""

    def __init__(self, config: DocumentModelConfig | None = None) -> None:
        """Initialize an unfitted classifier."""
        self.config = config or DocumentModelConfig()
        self.vectorizer: TfidfVectorizer | None = None
        self.estimator: LogisticRegression | None = None
        self.curation_report: document.DocumentCurationReport | None = None

    @property
    def fitted(self) -> bool:
        """Return whether feature preparation and modelling are complete."""
        return self.vectorizer is not None and self.estimator is not None

    @property
    def model_digest(self) -> str:
        """Return a deterministic digest of the fitted model state."""
        self.require_fitted()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise RuntimeError("classifier is not fitted")
        state = {
            "config": self.config.to_dict(),
            "vocabulary": sorted(
                (str(term), int(index))
                for term, index in vectorizer.vocabulary_.items()
            ),
            "classes": estimator.classes_.tolist(),
            "coef": estimator.coef_.tolist(),
            "intercept": estimator.intercept_.tolist(),
        }
        encoded = json.dumps(
            state, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fit(
        self,
        training_samples: Sequence[document.DocumentSample],
    ) -> Self:
        """Fit the baseline on labelled, curated training documents.

        Args:
            training_samples: Samples from the training partition only.

        Returns:
            The fitted classifier.

        Raises:
            ValueError: If labels are missing, conflicting duplicate content
                is present, or fewer than two classes are available.
        """
        values = tuple(training_samples)
        if not values:
            raise ValueError("training_samples must not be empty")
        dataset = document.DocumentDataset(values)
        report = dataset.curate()
        if report.unresolved_blocking_findings:
            raise ValueError(
                "document curation has unresolved blocking findings: "
                + ", ".join(
                    finding.finding_id
                    for finding in report.unresolved_blocking_findings
                )
            )
        if any(sample.label is None for sample in values):
            raise ValueError("every training document must have a label")
        labels = [sample.label for sample in values]
        if len(set(labels)) < 2:
            raise ValueError("training documents must contain two classes")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            token_pattern=r"(?u)\b\w\w+\b",
        )
        matrix = vectorizer.fit_transform([sample.content for sample in values])
        if matrix.shape[1] == 0:
            raise ValueError("training documents contain no usable tokens")
        estimator = LogisticRegression(
            C=self.config.regularization,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            solver="lbfgs",
        )
        estimator.fit(matrix, labels)
        self.vectorizer = vectorizer
        self.estimator = estimator
        self.curation_report = report
        return self

    def predict(
        self,
        samples: Sequence[document.DocumentSample],
    ) -> tuple[Any, ...]:
        """Predict labels for validated documents."""
        self.require_fitted()
        if not samples:
            return ()
        matrix = self.transform(samples)
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        return tuple(estimator.predict(matrix).tolist())

    def predict_proba(
        self,
        samples: Sequence[document.DocumentSample],
    ) -> tuple[tuple[float, ...], ...]:
        """Return class probabilities for validated documents."""
        self.require_fitted()
        if not samples:
            return ()
        matrix = self.transform(samples)
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        return tuple(tuple(row) for row in estimator.predict_proba(matrix))

    def evaluate(
        self,
        samples: Sequence[document.DocumentSample],
    ) -> DocumentEvaluation:
        """Evaluate labelled documents without refitting or touching training.

        The evaluator never changes the fitted vocabulary or estimator.
        """
        values = tuple(samples)
        if not values:
            raise ValueError("evaluation samples must not be empty")
        if any(sample.label is None for sample in values):
            raise ValueError("every evaluation document must have a label")
        labels = tuple(str(value) for value in self.classes)
        actual = [sample.label for sample in values]
        predicted = self.predict(values)
        probabilities = self.predict_proba(values)
        return DocumentEvaluation(
            sample_count=len(values),
            accuracy=float(accuracy_score(actual, predicted)),
            macro_f1=float(f1_score(actual, predicted, average="macro")),
            log_loss=float(
                log_loss(actual, probabilities, labels=list(self.classes))
            ),
            labels=labels,
        )

    def global_explanation(self, top_k: int = 10) -> DocumentExplanation:
        """Return stable global term importance from linear coefficients."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.require_fitted()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise RuntimeError("classifier is not fitted")
        terms = vectorizer.get_feature_names_out()
        strengths = np.mean(np.abs(estimator.coef_), axis=0)
        selected = sorted(
            (
                (str(terms[index]), float(strengths[index]))
                for index in range(len(terms))
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
        return DocumentExplanation(
            model_digest=self.model_digest,
            input_hash=None,
            method="linear_global_term_weight",
            terms=tuple(selected),
            limitations=(
                "Term weights are associational, not causal.",
                "Tokenization and corpus vocabulary constrain the explanation.",
            ),
        )

    def local_explanation(
        self,
        sample: document.DocumentSample,
        top_k: int = 10,
    ) -> DocumentExplanation:
        """Return local signed term contributions for one document."""
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.require_fitted()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise RuntimeError("classifier is not fitted")
        matrix = self.transform((sample,))
        predicted_label = estimator.predict(matrix)[0]
        class_index = int(
            np.flatnonzero(estimator.classes_ == predicted_label)[0]
        )
        if len(estimator.classes_) == 2:
            coefficients = estimator.coef_[0]
            if class_index == 0:
                coefficients = -coefficients
        else:
            coefficients = estimator.coef_[class_index]
        row = matrix.toarray()[0]
        terms = vectorizer.get_feature_names_out()
        contributions = sorted(
            (
                (str(terms[index]), float(row[index] * coefficients[index]))
                for index in np.flatnonzero(row)
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:top_k]
        return DocumentExplanation(
            model_digest=self.model_digest,
            input_hash=sample.content_hash,
            method="linear_local_term_contribution",
            terms=tuple(contributions),
            limitations=(
                "Contributions explain this fitted linear approximation only.",
                "A positive term contribution favors the predicted class.",
            ),
        )

    @property
    def classes(self) -> tuple[Any, ...]:
        """Return the fitted class labels in estimator order."""
        self.require_fitted()
        estimator = self.estimator
        if estimator is None:
            raise RuntimeError("classifier is not fitted")
        return tuple(estimator.classes_.tolist())

    def save(self, path: str | pathlib.Path) -> DocumentModelArtifact:
        """Atomically save this fitted classifier as a verified package."""
        self.require_fitted()
        target = pathlib.Path(path).expanduser()
        if target.exists():
            raise FileExistsError(f"model package already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_directory: pathlib.Path | None = None
        try:
            temporary_directory = pathlib.Path(
                tempfile.mkdtemp(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                )
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
            sync_directory(temporary_directory)
            os.replace(temporary_directory, target)
            sync_directory(target.parent)
        except Exception:
            if temporary_directory is not None:
                shutil.rmtree(temporary_directory, ignore_errors=True)
            raise
        return DocumentModelArtifact(
            str(target), str(manifest["artifact_digest"])
        )

    @classmethod
    def load(
        cls,
        path: str | pathlib.Path,
        trusted: bool = False,
    ) -> Self:
        """Load a package only after an explicit caller trust decision."""
        if not trusted:
            raise PermissionError(
                "loading a document model package requires trusted=True"
            )
        target = pathlib.Path(path).expanduser()
        if not target.is_dir():
            raise FileNotFoundError(target)
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError("document model package has no manifest")
        with manifest_path.open(encoding="utf-8") as stream:
            manifest = json.load(stream)
        if manifest.get("schema_version") != 1:
            raise ValueError("unsupported document model package version")
        model_path = target / str(manifest.get("model_path", ""))
        if not model_path.is_file():
            raise ValueError("document model package has no model payload")
        if file_digest(model_path) != manifest.get("artifact_digest"):
            raise ValueError("document model payload digest does not match")
        loaded = joblib.load(model_path)
        if not isinstance(loaded, cls):
            raise TypeError("package does not contain a DocumentClassifier")
        loaded.require_fitted()
        if loaded.model_digest != manifest.get("model_digest"):
            raise ValueError("document model digest does not match manifest")
        return loaded

    def transform(self, samples: Sequence[document.DocumentSample]) -> Any:
        """Transform documents with the fitted training vocabulary."""
        self.require_fitted()
        vectorizer = self.vectorizer
        if vectorizer is None:
            raise RuntimeError("classifier is not fitted")
        return vectorizer.transform([sample.content for sample in samples])

    def require_fitted(self) -> None:
        """Raise when an operation needs a fitted classifier."""
        if not self.fitted:
            raise RuntimeError("document classifier has not been fitted")


class DocumentMonitor:
    """Monitor bounded corpus-profile signals against a reference snapshot."""

    def __init__(
        self,
        reference: document.DocumentCurationReport,
        threshold: float = 0.25,
    ) -> None:
        """Initialize a monitor with an explicit normalized drift threshold."""
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        self.reference = reference
        self.threshold = threshold

    def observe(
        self,
        observed: document.DocumentDataset,
    ) -> DocumentDriftReport:
        """Compare observed document profiles with the reference corpus."""
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
        reference_mean = self.mean_characters(self.reference)
        current_mean = self.mean_characters(current)
        duplicate_rate = (
            sum(len(group) for group in current.duplicate_groups)
            / current.sample_count
        )
        return DocumentDriftReport(
            reference_hash=self.reference.dataset_hash,
            observed_hash=current.dataset_hash,
            language_distance=language_distance,
            mean_character_delta=(current_mean - reference_mean)
            / max(reference_mean, 1.0),
            duplicate_rate=duplicate_rate,
            threshold=self.threshold,
        )

    def mean_characters(
        self,
        report: document.DocumentCurationReport,
    ) -> float:
        """Return the mean bounded character count from a curation report."""
        if not report.profiles:
            return 0.0
        return sum(
            profile.character_count for profile in report.profiles
        ) / len(report.profiles)


def file_digest(path: pathlib.Path) -> str:
    """Return a SHA-256 digest for a package artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sync_directory(path: pathlib.Path) -> None:
    """Flush a directory entry after an atomic package replacement."""
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


__all__ = [
    "DocumentClassifier",
    "DocumentDriftReport",
    "DocumentEvaluation",
    "DocumentExplanation",
    "DocumentModelArtifact",
    "DocumentModelConfig",
    "DocumentMonitor",
    "file_digest",
    "sync_directory",
]
