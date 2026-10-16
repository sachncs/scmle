"""Automatic feature construction and selection (fold-local transformers).

Both are fitted on the training fold only. They exist so the search can test,
on untouched validation folds, whether combining features or dropping
redundant ones helps, instead of assuming it does.
"""

from itertools import combinations
from typing import Any
from typing import Self

import numpy as np
import pandas as pd
from scipy import stats
from sklearn import base
from sklearn import linear_model

PREFIX = "auto__"
DEFAULT_TOP_COLUMNS = 5
EPSILON = 1e-9
MIN_ROWS = 20
DEFAULT_ALPHA = 0.02
DEFAULT_STRENGTH = 0.1
ZERO = 1e-8


class FeatureConstructor(base.TransformerMixin, base.BaseEstimator):
    """Appends pairwise differences and ratios of the most informative columns.

    The columns that are combined are the numeric ones most associated with
    the target (absolute Spearman correlation) in the training fold. Original
    columns are kept; the new ones are named `auto__a__minus__b` and
    `auto__a__over__b`, so a later step can select them by prefix.

    Attributes:
        columns: The numeric columns eligible for combination.
        top: How many of them (by association) are combined.
        pairs_: The (left, right) column pairs chosen during fit.
    """

    def __init__(
        self,
        columns: tuple[str, ...] = (),
        top: int = DEFAULT_TOP_COLUMNS,
    ) -> None:
        """Stores the eligible columns and how many to combine."""
        self.columns = columns
        self.top = top

    def fit(self, features: pd.DataFrame, target: Any = None) -> Self:
        """Chooses the column pairs from the training fold."""
        self.pairs_: list[tuple[str, str]] = []
        usable = [c for c in self.columns if c in features.columns]
        if target is not None and len(features) >= MIN_ROWS and usable:
            outcome = pd.Series(np.asarray(target)).astype("category").cat.codes
            scored = []
            for column in usable:
                values = pd.to_numeric(features[column]).reset_index(drop=True)
                if values.nunique() < 2:
                    continue
                rho = stats.spearmanr(values, outcome, nan_policy="omit")[0]
                scored.append((abs(float(np.nan_to_num(rho))), column))
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = [column for __, column in scored[: self.top]]
            self.pairs_ = list(combinations(sorted(chosen), 2))
        self.feature_names_in_ = np.asarray(features.columns, dtype=object)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Returns the input with the constructed columns appended."""
        built = {}
        for left, right in self.pairs_:
            a = pd.to_numeric(features[left]).to_numpy(dtype=float)
            b = pd.to_numeric(features[right]).to_numpy(dtype=float)
            built[f"{PREFIX}{left}__minus__{right}"] = a - b
            safe = np.where(np.abs(b) < EPSILON, np.nan, b)
            built[f"{PREFIX}{left}__over__{right}"] = a / safe
        extra = pd.DataFrame(built, index=features.index)
        return pd.concat([features, extra], axis=1)

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the output columns."""
        names = list(
            self.feature_names_in_ if input_features is None else input_features
        )
        for left, right in self.pairs_:
            names += [
                f"{PREFIX}{left}__minus__{right}",
                f"{PREFIX}{left}__over__{right}",
            ]
        return np.asarray(names, dtype=object)


class SparseSelector(base.TransformerMixin, base.BaseEstimator):
    """Keeps the columns an L1-penalised linear model gives weight.

    The matrix should be scaled so the penalty treats columns alike. At
    least one column always survives: when the penalty zeroes everything, the
    column with the largest weight before shrinkage is kept.

    Attributes:
        classification: Whether the target is a class label.
        strength: Inverse penalty for classification (larger keeps more).
        alpha: Penalty for regression (smaller keeps more).
        support_: Boolean mask of kept columns, set by fit.
    """

    def __init__(
        self,
        classification: bool = True,
        strength: float = DEFAULT_STRENGTH,
        alpha: float = DEFAULT_ALPHA,
        random_state: int = 0,
    ) -> None:
        """Stores the penalty settings."""
        self.classification = classification
        self.strength = strength
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the sparse model and records which columns it keeps."""
        model: Any
        if self.classification:
            model = linear_model.LogisticRegression(
                penalty="l1",
                solver="liblinear",
                C=self.strength,
                random_state=self.random_state,
            )
        else:
            model = linear_model.Lasso(
                alpha=self.alpha, random_state=self.random_state
            )
        model.fit(features, target)
        weight = np.abs(np.atleast_2d(model.coef_)).max(axis=0)
        self.support_ = weight > ZERO
        if not self.support_.any():
            fallback = linear_model.Ridge(alpha=1.0).fit(
                features,
                np.asarray(
                    pd.Series(np.asarray(target)).astype("category").cat.codes
                )
                if self.classification
                else target,
            )
            self.support_[int(np.argmax(np.abs(fallback.coef_)))] = True
        self.n_features_in_ = len(self.support_)
        return self

    def transform(self, features: Any) -> Any:
        """Returns only the kept columns."""
        return features[:, self.support_]

    def get_feature_names_out(self, input_features: Any = None) -> np.ndarray:
        """Names of the kept columns."""
        names = np.asarray(input_features, dtype=object)
        return names[self.support_]
