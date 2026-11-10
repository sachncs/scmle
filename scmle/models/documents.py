"""A bounded, explainable baseline lifecycle for labelled documents.

This module covers documents only. It uses the validated document contracts,
keeps feature preparation and modelling in one immutable configuration, and
refuses to imply that a serialized model is safe to load without an explicit
trust decision.
"""

from collections.abc import Sequence
import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import tempfile
from typing import Any
from typing import Self

import joblib
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scmle import durability
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
            raise ValueError("max_features must be positive")
        if len(self.ngram_range) != NGRAM_BOUND_COUNT or not (
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


class Classifier:
    """TF-IDF document classifier with evidence and safe package boundaries.

    Attributes:
        config: Feature and estimator configuration.
        vectorizer: Fitted TF-IDF vectorizer, or None before fitting.
        estimator: Fitted logistic regression, or None before fitting.
        curation_report: Curation report of the training documents, or None
            before fitting.
    """

    def __init__(self, config: Config | None = None) -> None:
        """Initializes an unfitted classifier.

        Args:
            config: Configuration to use; defaults to
                `Config()`.
        """
        self.config = config or Config()
        self.vectorizer: TfidfVectorizer | None = None
        self.estimator: LogisticRegression | None = None
        self.curation_report: document.Curation | None = None

    @property
    def fitted(self) -> bool:
        """Whether the vectorizer and the estimator are both fitted."""
        return self.vectorizer is not None and self.estimator is not None

    @property
    def model_digest(self) -> str:
        """The SHA-256 digest of the configuration, vocabulary and coefficients.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
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
        training_samples: Sequence[document.Sample],
    ) -> Self:
        """Fits the baseline on labelled, curated training documents.

        Args:
            training_samples: Samples from the training partition only.

        Returns:
            The fitted classifier.

        Raises:
            ValueError: If there are no samples, curation has unresolved
                blocking findings, a label is missing, fewer than two
                classes are present, or no usable tokens remain.
        """
        values = tuple(training_samples)
        if not values:
            raise ValueError("training_samples must not be empty")
        dataset = document.Dataset(values)
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
        if len(set(labels)) < MIN_CLASS_COUNT:
            raise ValueError("training documents must contain two classes")

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            token_pattern=r"(?u)\b\w\w+\b",  # noqa: S106 -- regex, not a secret  # nosec B106
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
        samples: Sequence[document.Sample],
    ) -> tuple[Any, ...]:
        """Predicts labels for validated documents.

        Args:
            samples: Documents to classify.

        Returns:
            Predicted labels, one per document in order; empty if there are
            no samples.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
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
        samples: Sequence[document.Sample],
    ) -> tuple[tuple[float, ...], ...]:
        """Predicts class probabilities for validated documents.

        Args:
            samples: Documents to classify.

        Returns:
            One tuple of probabilities per document, in `classes` order;
            empty if there are no samples.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
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
        samples: Sequence[document.Sample],
    ) -> Evaluation:
        """Evaluates labelled documents without refitting or touching training.

        The evaluator never changes the fitted vocabulary or estimator.

        Args:
            samples: Labelled documents to score.

        Returns:
            Accuracy, macro-averaged precision/recall balance and log loss
            over the documents.

        Raises:
            ValueError: If there are no samples or any document has no label.
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
        return Evaluation(
            sample_count=len(values),
            accuracy=float(accuracy_score(actual, predicted)),
            macro_f1=float(f1_score(actual, predicted, average="macro")),
            log_loss=float(
                log_loss(actual, probabilities, labels=list(self.classes))
            ),
            labels=labels,
        )

    def global_explanation(self, top_k: int = 10) -> Explanation:
        """Ranks terms by their mean absolute linear coefficient.

        Args:
            top_k: Number of terms to keep.

        Returns:
            Global explanation whose terms are (term, mean absolute
            coefficient) pairs, largest first, ties broken alphabetically.

        Raises:
            ValueError: If `top_k` is below 1.
            RuntimeError: If the classifier has not been fitted.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        self.require_fitted()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise RuntimeError("classifier is not fitted")
        terms = vectorizer.get_feature_names_out()
        strengths = numpy.mean(numpy.abs(estimator.coef_), axis=0)
        selected = sorted(
            (
                (str(terms[index]), float(strengths[index]))
                for index in range(len(terms))
            ),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
        return Explanation(
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
        sample: document.Sample,
        top_k: int = 10,
    ) -> Explanation:
        """Computes signed term contributions for one document.

        Args:
            sample: Document to explain.
            top_k: Number of terms to keep.

        Returns:
            Local explanation whose terms are (term, TF-IDF value times
            coefficient) pairs for the predicted class, largest absolute
            value first; positive favors the predicted class. It carries
            the document's citations.

        Raises:
            ValueError: If `top_k` is below 1.
            RuntimeError: If the classifier has not been fitted.
        """
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
            numpy.flatnonzero(estimator.classes_ == predicted_label)[0]
        )
        if len(estimator.classes_) == BINARY_CLASS_COUNT:
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
                for index in numpy.flatnonzero(row)
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:top_k]
        return Explanation(
            model_digest=self.model_digest,
            input_hash=sample.content_hash,
            method="linear_local_term_contribution",
            terms=tuple(contributions),
            limitations=(
                "Contributions explain this fitted linear approximation only.",
                "A positive term contribution favors the predicted class.",
            ),
            citations=sample.citations(),
        )

    def term_ablation_explanation(
        self,
        sample: document.Sample,
        top_k: int = 10,
        max_terms: int = 100,
    ) -> Explanation:
        """Measures predicted-class score changes when terms are removed.

        Candidate terms are the terms present in the document, ranked by
        absolute local linear contribution and cut to `max_terms`. Each is
        removed from the original text on its own (every whole-word,
        case-insensitive match). The score reported for a term is the original
        predicted-class score minus the ablated score; positive values mean
        that the term supported the predicted class relative to its absence.
        The `top_k` terms with the largest absolute score change are returned,
        ties broken alphabetically.

        Args:
            sample: Document to explain.
            top_k: Maximum number of ablations to return; at least 1.
            max_terms: Maximum number of candidate terms to evaluate; at least
                1.

        Returns:
            Deterministic raw-text perturbation evidence whose `terms` hold
            the selected (term, score change) pairs.

        Raises:
            ValueError: If `top_k` or `max_terms` is below 1.
            RuntimeError: If the classifier has not been fitted.
        """
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_terms < 1:
            raise ValueError("max_terms must be positive")
        self.require_fitted()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise RuntimeError("classifier is not fitted")
        matrix = self.transform((sample,))
        terms = vectorizer.get_feature_names_out()
        row = matrix.toarray()[0]
        predicted = estimator.predict(matrix)[0]
        class_index = int(numpy.flatnonzero(estimator.classes_ == predicted)[0])
        coefficients = estimator.coef_[0]
        if len(estimator.classes_) == BINARY_CLASS_COUNT and class_index == 0:
            coefficients = -coefficients
        elif len(estimator.classes_) > BINARY_CLASS_COUNT:
            coefficients = estimator.coef_[class_index]
        candidates = sorted(
            (
                (str(terms[index]), float(row[index] * coefficients[index]))
                for index in numpy.flatnonzero(row)
            ),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:max_terms]
        original_score = self.predicted_class_score(matrix)
        ablations: list[tuple[str, float]] = []
        for candidate in candidates:
            term = candidate[0]
            pattern = rf"(?iu)\b{re.escape(term)}\b"
            ablated_content = re.sub(pattern, " ", sample.content)
            ablated_matrix = vectorizer.transform([ablated_content or " "])
            delta = original_score - self.predicted_class_score(ablated_matrix)
            ablations.append((term, float(delta)))
        selected = sorted(
            ablations,
            key=lambda item: (-abs(item[1]), item[0]),
        )[:top_k]
        return Explanation(
            model_digest=self.model_digest,
            input_hash=sample.content_hash,
            method="term_ablation",
            terms=tuple(selected),
            limitations=(
                "Ablation removes lexical matches from the original text.",
                "Candidate evaluation is bounded by max_terms.",
                "The score change is associational, not causal.",
            ),
            citations=sample.citations(),
        )

    def predicted_class_score(self, matrix: Any) -> float:
        """Computes the decision score of the predicted class.

        Args:
            matrix: Feature matrix whose first row is scored.

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
        predicted = estimator.predict(matrix)[0]
        class_index = int(numpy.flatnonzero(estimator.classes_ == predicted)[0])
        decision = numpy.asarray(estimator.decision_function(matrix))
        if decision.ndim == 1:
            raw_score = float(decision[0])
            return raw_score if class_index == 1 else -raw_score
        return float(decision[0, class_index])

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
                tempfile.mkdtemp(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                )
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
    def load(
        cls,
        path: str | pathlib.Path,
        trusted: bool = False,
    ) -> Self:
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
            TypeError: If the payload is not a `Classifier`.
        """
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
            raise TypeError("package does not contain a Classifier")
        loaded.require_fitted()
        if loaded.model_digest != manifest.get("model_digest"):
            raise ValueError("document model digest does not match manifest")
        return loaded

    def transform(self, samples: Sequence[document.Sample]) -> Any:
        """Transforms documents with the fitted training vocabulary.

        Args:
            samples: Documents to convert.

        Returns:
            Sparse TF-IDF matrix with one row per document.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.require_fitted()
        vectorizer = self.vectorizer
        if vectorizer is None:
            raise RuntimeError("classifier is not fitted")
        return vectorizer.transform([sample.content for sample in samples])

    def require_fitted(self) -> None:
        """Checks that the classifier has been fitted.

        Raises:
            RuntimeError: If it has not been fitted.
        """
        if not self.fitted:
            raise RuntimeError("document classifier has not been fitted")


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
            raise ValueError("threshold must be in (0, 1]")
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
        reference_mean = self.mean_characters(self.reference)
        current_mean = self.mean_characters(current)
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

    def mean_characters(
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
