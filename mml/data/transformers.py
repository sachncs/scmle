"""Fitted feature transformers (scikit-learn compatible)."""

from typing import Any
from typing import Self

import numpy as np
import pandas as pd
from sklearn import base

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
