"""Fitted feature transformers (scikit-learn compatible)."""

from typing import Any
from typing import Self

import numpy as np
import pandas as pd
from sklearn import base
from sklearn.feature_extraction import text as sk_text

DEFAULT_SKEW_THRESHOLD = 1.0


class SkewedLogTransformer(base.TransformerMixin, base.BaseEstimator):
    """Applies `log1p` to strongly right-skewed, non-negative columns.

    Which columns qualify is learned from the training data only (fit), so the
    transform is fold-local and identical at inference.

    Attributes:
        threshold: Minimum skewness for a column to be transformed.
        columns_: Indices of the columns selected during fit.
    """

    def __init__(self, threshold: float = DEFAULT_SKEW_THRESHOLD) -> None:
        """Initializes the transformer.

        Args:
            threshold: Minimum skewness for a column to be transformed.
        """
        self.threshold = threshold

    def fit(self, features: Any, target: Any = None) -> Self:
        """Selects non-negative columns whose skewness exceeds the threshold.

        Args:
            features: Numeric feature matrix.
            target: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted transformer.
        """
        frame = pd.DataFrame(np.asarray(features, dtype=float))
        skew = frame.skew().fillna(0.0).to_numpy()
        non_negative = (frame.min() >= 0).to_numpy()
        self.columns_ = np.flatnonzero(non_negative & (skew > self.threshold))
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, features: Any) -> np.ndarray:
        """Log-transforms the selected columns (negatives clip to zero).

        Args:
            features: Numeric feature matrix with the fitted column layout.

        Returns:
            A float copy of the input with `log1p` applied to the columns
            chosen in fit.
        """
        values = np.array(features, dtype=float, copy=True)
        if self.columns_.size:
            values[:, self.columns_] = np.log1p(
                np.clip(values[:, self.columns_], 0.0, None)
            )
        return values

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Returns the unchanged feature names.

        Args:
            input_features: Names of the input columns.

        Returns:
            The input names as an object array.
        """
        return np.asarray(input_features, dtype=object)


class TextVectorizer(base.TransformerMixin, base.BaseEstimator):
    """TF-IDF terms of a text column, with the vocabulary learned per fold.

    Attributes:
        max_terms: Largest vocabulary kept (by corpus frequency).
        vectorizer_: The fitted scikit-learn vectorizer.
    """

    def __init__(self, max_terms: int = 200) -> None:
        """Stores the vocabulary size.

        Args:
            max_terms: Largest vocabulary kept (by corpus frequency).
        """
        self.max_terms = max_terms

    def fit(self, texts: Any, target: Any = None) -> Self:
        """Learns the vocabulary and term weights from the training texts.

        Unigrams and bigrams occurring in at least two texts are used; if
        that leaves no term, unigrams of any frequency are used instead.

        Args:
            texts: The training texts.
            target: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted transformer.
        """
        self.vectorizer_ = sk_text.TfidfVectorizer(
            max_features=self.max_terms,
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            strip_accents="unicode",
        )
        try:
            self.vectorizer_.fit(np.asarray(texts, dtype=str))
        except ValueError:  # Empty vocabulary: no term occurs twice.
            self.vectorizer_ = sk_text.TfidfVectorizer(
                max_features=self.max_terms, strip_accents="unicode"
            ).fit(np.asarray(texts, dtype=str))
        return self

    def transform(self, texts: Any) -> Any:
        """Computes the TF-IDF matrix.

        Args:
            texts: The texts to transform.

        Returns:
            A sparse matrix with one row per text and one column per term.
        """
        return self.vectorizer_.transform(np.asarray(texts, dtype=str))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Terms of the vocabulary.

        Args:
            input_features: Ignored; the names do not depend on the input.

        Returns:
            The vocabulary terms, in column order.
        """
        return self.vectorizer_.get_feature_names_out()


class AsinhTarget(base.TransformerMixin, base.BaseEstimator):
    """Inverse hyperbolic sine of the target, scaled by its spread.

    Behaves like a log for large values and like the identity near zero, is
    defined for negative values, and inverts exactly everywhere (unlike a
    power transform, whose inverse can leave its domain for extreme
    predictions).

    Attributes:
        scale_: Spread of the training target, set by fit.
    """

    LIMIT = 50.0

    def fit(self, target: Any, unused: Any = None) -> Self:
        """Learns the scale from the training target.

        Args:
            target: The training target values.
            unused: Ignored; present for scikit-learn compatibility.

        Returns:
            The fitted transformer; the scale is 1 if the target is constant.
        """
        spread = float(np.std(np.asarray(target, dtype=float)))
        self.scale_ = spread if spread > 0 else 1.0
        return self

    def transform(self, target: Any) -> np.ndarray:
        """Compresses the tails with the scaled inverse hyperbolic sine.

        Args:
            target: Target values in original units.

        Returns:
            `arcsinh(target / scale_)` as a float array of the same shape.
        """
        return np.arcsinh(np.asarray(target, dtype=float) / self.scale_)

    def inverse_transform(self, target: Any) -> np.ndarray:
        """Maps predictions back to the original units.

        Args:
            target: Values on the transformed scale; they are clipped to
                `[-LIMIT, LIMIT]` first so the result stays finite.

        Returns:
            `sinh(target) * scale_` as a float array of the same shape.
        """
        clipped = np.clip(
            np.asarray(target, dtype=float), -self.LIMIT, self.LIMIT
        )
        return np.sinh(clipped) * self.scale_
