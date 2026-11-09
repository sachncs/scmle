"""Automatic feature construction and selection (fold-local transformers).

Both are fitted on the training fold only. They exist so the search can test,
on untouched validation folds, whether combining features or dropping
redundant ones helps, instead of assuming it does.
"""

from itertools import combinations
from typing import Any
from typing import Self

import numpy
import pandas
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
BINARY_CLASSES = 2
MIN_DISTINCT_VALUES = 2


class FeatureConstructor(base.TransformerMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information, so its base classes are Any
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
        """Stores the eligible columns and how many to combine.

        Args:
            columns: The numeric columns eligible for combination.
            top: How many of them (by association) are combined.
        """
        self.columns = columns
        self.top = top

    def fit(self, features: pandas.DataFrame, target: Any = None) -> Self:
        """Chooses the column pairs from the training fold.

        No pairs are chosen when there is no target, fewer than `MIN_ROWS`
        rows or no eligible column present. Constant columns are skipped.

        Args:
            features: Training features.
            target: Training target; class labels are coded as categories.

        Returns:
            The fitted transformer.
        """
        self.pairs_: list[tuple[str, str]] = []
        usable = [c for c in self.columns if c in features.columns]
        if target is not None and len(features) >= MIN_ROWS and usable:
            outcome = (
                pandas.Series(numpy.asarray(target))
                .astype("category")
                .cat.codes
            )
            scored = []
            for column in usable:
                values = pandas.to_numeric(features[column]).reset_index(
                    drop=True
                )
                if values.nunique() < MIN_DISTINCT_VALUES:
                    continue
                rho = stats.spearmanr(values, outcome, nan_policy="omit")[0]
                scored.append((abs(float(numpy.nan_to_num(rho))), column))
            scored.sort(key=lambda item: (-item[0], item[1]))
            chosen = [entry[1] for entry in scored[: self.top]]
            self.pairs_ = list(combinations(sorted(chosen), 2))
        self.feature_names_in_ = numpy.asarray(features.columns, dtype=object)
        return self

    def transform(self, features: pandas.DataFrame) -> pandas.DataFrame:
        """Returns the input with the constructed columns appended.

        A ratio is missing (NaN) where the denominator is below `EPSILON`
        in absolute value.

        Args:
            features: Features containing the columns chosen in fit.

        Returns:
            The input columns followed by a difference and a ratio column
            for each chosen pair, with the input's index.
        """
        built = {}
        for left, right in self.pairs_:
            a = pandas.to_numeric(features[left]).to_numpy(dtype=float)
            b = pandas.to_numeric(features[right]).to_numpy(dtype=float)
            built[f"{PREFIX}{left}__minus__{right}"] = a - b
            safe = numpy.where(numpy.abs(b) < EPSILON, numpy.nan, b)
            built[f"{PREFIX}{left}__over__{right}"] = a / safe
        extra = pandas.DataFrame(built, index=features.index)
        return pandas.concat([features, extra], axis=1)

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> numpy.ndarray:
        """Names of the output columns.

        Args:
            input_features: Names of the input columns; the names seen in fit
                are used if None.

        Returns:
            The input names followed by the constructed names, as an object
            array.
        """
        names = list(
            self.feature_names_in_ if input_features is None else input_features
        )
        for left, right in self.pairs_:
            names += [
                f"{PREFIX}{left}__minus__{right}",
                f"{PREFIX}{left}__over__{right}",
            ]
        return numpy.asarray(names, dtype=object)


class SparseSelector(base.TransformerMixin, base.BaseEstimator):  # type: ignore[misc]  # scikit-learn ships no type information, so its base classes are Any
    """Keeps the columns an L1-penalised linear model gives weight.

    The matrix should be scaled so the penalty treats columns alike. At
    least one column always survives: when the penalty zeroes everything, the
    column with the largest absolute coefficient of a ridge fit with
    `alpha=1` (class labels coded as integers) is kept.

    Attributes:
        classification: Whether the target is a class label.
        strength: Inverse penalty for classification (larger keeps more).
        alpha: Penalty for regression (smaller keeps more).
        random_state: Seed for the sparse model.
        support_: Boolean mask of kept columns, set by fit.
        n_features_in_: Number of columns seen in fit.
    """

    def __init__(
        self,
        classification: bool = True,
        strength: float = DEFAULT_STRENGTH,
        alpha: float = DEFAULT_ALPHA,
        random_state: int = 0,
    ) -> None:
        """Stores the penalty settings.

        Args:
            classification: Whether the target is a class label.
            strength: Inverse penalty for classification (larger keeps more).
            alpha: Penalty for regression (smaller keeps more).
            random_state: Seed for the sparse model.
        """
        self.classification = classification
        self.strength = strength
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, features: Any, target: Any) -> Self:
        """Fits the sparse model and records which columns it keeps.

        Args:
            features: Scaled feature matrix.
            target: Training target; class labels when `classification`.

        Returns:
            The fitted selector, with at least one column kept.
        """
        model: Any
        if self.classification:
            # liblinear is fast but cannot fit three or more classes.
            multiclass = len(numpy.unique(target)) > BINARY_CLASSES
            model = linear_model.LogisticRegression(
                l1_ratio=1.0,
                solver="saga" if multiclass else "liblinear",
                C=self.strength,
                random_state=self.random_state,
            )
        else:
            model = linear_model.Lasso(
                alpha=self.alpha, random_state=self.random_state
            )
        model.fit(features, target)
        weight = numpy.abs(numpy.atleast_2d(model.coef_)).max(axis=0)
        self.support_ = weight > ZERO
        if not self.support_.any():
            fallback = linear_model.Ridge(alpha=1.0).fit(
                features,
                numpy.asarray(
                    pandas.Series(numpy.asarray(target))
                    .astype("category")
                    .cat.codes
                )
                if self.classification
                else target,
            )
            self.support_[int(numpy.argmax(numpy.abs(fallback.coef_)))] = True
        self.n_features_in_ = len(self.support_)
        return self

    def transform(self, features: Any) -> Any:
        """Returns only the kept columns.

        Args:
            features: Feature matrix with the fitted column layout.

        Returns:
            The input restricted to the columns kept in fit.
        """
        return features[:, self.support_]

    def get_feature_names_out(
        self, input_features: Any = None
    ) -> numpy.ndarray:
        """Names of the kept columns.

        Args:
            input_features: Names of the input columns.

        Returns:
            The names of the columns kept in fit, as an object array.
        """
        names = numpy.asarray(input_features, dtype=object)
        kept: numpy.ndarray = names[self.support_]
        return kept
