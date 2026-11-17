"""The fitted TF-IDF and logistic regression estimator."""

from collections.abc import Sequence
import hashlib
import json
from typing import Any
from typing import cast
from typing import Self
from typing import TYPE_CHECKING

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss

from scmle.contracts import errors
from scmle.data.modalities import document
from scmle.models.documents.records import Config
from scmle.models.documents.records import Evaluation
from scmle.models.documents.records import MIN_CLASS_COUNT

if TYPE_CHECKING:
    import numpy


class Estimator:
    """TF-IDF features and a logistic regression, fitted and evaluated.

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
        self.preflight()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise errors.OrderError("classifier is not fitted")
        state = {
            "config": self.config.to_dict(),
            "vocabulary": sorted(
                (str(term), int(index))
                for term, index in vectorizer.vocabulary_.items()
            ),
            "classes": estimator.classes_.tolist(),
            "coef": estimator.coef_.tolist(),
            "intercept": cast("numpy.ndarray", estimator.intercept_).tolist(),
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
            raise errors.InputError("training_samples must not be empty")
        dataset = document.Dataset(values)
        report = dataset.curate()
        if report.unresolved_blocking_findings:
            raise errors.InputError(
                "document curation has unresolved blocking findings: "
                + ", ".join(
                    finding.finding_id
                    for finding in report.unresolved_blocking_findings
                )
            )
        if any(sample.label is None for sample in values):
            raise errors.InputError("every training document must have a label")
        labels = [sample.label for sample in values]
        if len(set(labels)) < MIN_CLASS_COUNT:
            raise errors.InputError(
                "training documents must contain two classes"
            )

        vectorizer = TfidfVectorizer(
            lowercase=True,
            max_features=self.config.max_features,
            ngram_range=self.config.ngram_range,
            min_df=self.config.min_df,
            token_pattern=r"(?u)\b\w\w+\b",  # noqa: S106 -- regex, not a secret  # nosec B106
        )
        matrix: Any = vectorizer.fit_transform(
            [sample.content for sample in values]
        )
        if matrix.shape[1] == 0:
            raise errors.InputError(
                "training documents contain no usable tokens"
            )
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
        self.preflight()
        if not samples:
            return ()
        matrix = self.transform(samples)
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        return tuple(cast("numpy.ndarray", estimator.predict(matrix)).tolist())

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
        self.preflight()
        if not samples:
            return ()
        matrix = self.transform(samples)
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
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
            raise errors.InputError("evaluation samples must not be empty")
        if any(sample.label is None for sample in values):
            raise errors.InputError(
                "every evaluation document must have a label"
            )
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

    def transform(self, samples: Sequence[document.Sample]) -> Any:
        """Transforms documents with the fitted training vocabulary.

        Args:
            samples: Documents to convert.

        Returns:
            Sparse TF-IDF matrix with one row per document.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.preflight()
        vectorizer = self.vectorizer
        if vectorizer is None:
            raise errors.OrderError("classifier is not fitted")
        return vectorizer.transform([sample.content for sample in samples])

    def preflight(self) -> None:
        """Checks that the classifier has been fitted.

        Raises:
            RuntimeError: If it has not been fitted.
        """
        if not self.fitted:
            raise errors.OrderError("document classifier has not been fitted")
