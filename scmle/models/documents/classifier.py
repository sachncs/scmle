"""The document classifier and its explanations."""

import re
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.data.modalities import document
from scmle.models.documents.archive import Archive
from scmle.models.documents.records import BINARY_CLASS_COUNT
from scmle.models.documents.records import Explanation


class Classifier(Archive):
    """TF-IDF document classifier with evidence and safe package boundaries.

    Attributes:
        config: Feature and estimator configuration.
        vectorizer: Fitted TF-IDF vectorizer, or None before fitting.
        estimator: Fitted logistic regression, or None before fitting.
        curation_report: Curation report of the training documents, or None
            before fitting.
    """

    __module__ = "scmle.models.documents"

    def overview(self, top_k: int = 10) -> Explanation:
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
            raise errors.InputError("top_k must be positive")
        self.preflight()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise errors.OrderError("classifier is not fitted")
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

    def spotlight(
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
            raise errors.InputError("top_k must be positive")
        self.preflight()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise errors.OrderError("classifier is not fitted")
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

    def ablate(
        self,
        sample: document.Sample,
        top_k: int = 10,
        max_terms: int = 100,
    ) -> Explanation:
        """Measures predicted-class score changes when terms are removed.

        Candidate terms are the terms present in the document, ranked by
        absolute local linear contribution and cut to `max_terms`. Each is
        removed from the original text on its own (every whole-word,
        case-insensitive match). A term's score change is the predicted
        class's score minus the ablated text's score for that same class,
        even when the removal flips the prediction; positive values mean
        the term supported the predicted class relative to its absence. The
        `top_k` largest absolute changes are returned, ties alphabetically.

        Args:
            sample: Document to explain.
            top_k: Maximum number of ablations to return; at least 1.
            max_terms: Most candidate terms to evaluate; at least 1.

        Returns:
            Deterministic raw-text perturbation evidence whose `terms` hold
            the selected (term, score change) pairs.

        Raises:
            ValueError: If `top_k` or `max_terms` is below 1.
            RuntimeError: If the classifier has not been fitted.
        """
        if top_k < 1:
            raise errors.InputError("top_k must be positive")
        if max_terms < 1:
            raise errors.InputError("max_terms must be positive")
        self.preflight()
        vectorizer = self.vectorizer
        estimator = self.estimator
        if vectorizer is None or estimator is None:
            raise errors.OrderError("classifier is not fitted")
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
        original_score = self.confidence(matrix, class_index)
        ablations: list[tuple[str, float]] = []
        for term in (candidate[0] for candidate in candidates):
            pattern = rf"(?iu)\b{re.escape(term)}\b"
            text = re.sub(pattern, " ", sample.content)
            ablated = vectorizer.transform([text or " "])
            delta = original_score - self.confidence(ablated, class_index)
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

    def confidence(self, matrix: Any, index: int | None = None) -> float:
        """Computes the decision score of the predicted or a given class.

        Args:
            matrix: Feature matrix whose first row is scored.
            index: Position in `classes_` of the class to score; None
                scores the class predicted for the first row.

        Returns:
            The decision-function value for that class on the first row;
            larger means more support for it.

        Raises:
            RuntimeError: If the classifier has not been fitted.
        """
        self.preflight()
        estimator = self.estimator
        if estimator is None:
            raise errors.OrderError("classifier is not fitted")
        if index is None:
            predicted = estimator.predict(matrix)[0]
            index = int(numpy.flatnonzero(estimator.classes_ == predicted)[0])
        decision = numpy.asarray(estimator.decision_function(matrix))
        if decision.ndim == 1:
            raw_score = float(decision[0])
            return raw_score if index == 1 else -raw_score
        return float(decision[0, index])
