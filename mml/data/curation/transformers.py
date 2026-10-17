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
        """Selects non-negative columns whose skewness exceeds the threshold."""
        frame = pd.DataFrame(np.asarray(features, dtype=float))
        skew = frame.skew().fillna(0.0).to_numpy()
        non_negative = (frame.min() >= 0).to_numpy()
        self.columns_ = np.flatnonzero(non_negative & (skew > self.threshold))
        self.n_features_in_ = frame.shape[1]
        return self

    def transform(self, features: Any) -> np.ndarray:
        """Log-transforms the selected columns (negatives clip to zero)."""
        values = np.array(features, dtype=float, copy=True)
        if self.columns_.size:
            values[:, self.columns_] = np.log1p(
                np.clip(values[:, self.columns_], 0.0, None)
            )
        return values

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Returns the unchanged feature names."""
        return np.asarray(input_features, dtype=object)


class TextVectorizer(base.TransformerMixin, base.BaseEstimator):
    """TF-IDF terms of a text column, with the vocabulary learned per fold.

    Attributes:
        max_terms: Largest vocabulary kept (by corpus frequency).
        vectorizer_: The fitted scikit-learn vectorizer.
    """

    def __init__(self, max_terms: int = 200) -> None:
        """Stores the vocabulary size."""
        self.max_terms = max_terms

    def fit(self, texts: Any, target: Any = None) -> Self:
        """Learns the vocabulary and term weights from the training texts."""
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
        """Returns the TF-IDF matrix (sparse)."""
        return self.vectorizer_.transform(np.asarray(texts, dtype=str))

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Terms of the vocabulary."""
        return self.vectorizer_.get_feature_names_out()
