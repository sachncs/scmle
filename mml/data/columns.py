"""Column kinds.

Each `ColumnKind` subclass encapsulates everything kind-specific: detection,
statistics, exclusion rules, the preprocessing transformer and the
target-association heuristic used for leakage hypotheses. Profiling and
preprocessing therefore iterate over kinds instead of branching on dtype.
"""

import abc
from typing import Any
from typing import ClassVar

import numpy as np
import pandas as pd
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing

from mml.contracts import data as data_contracts
from mml.contracts import tasks
from mml.data import transformers

MISSING_TOKEN = "__missing__"  # nosec B105 - category placeholder, not a secret
ID_UNIQUE_RATIO = 0.95
MAX_DIRECT_VALUES = 20
MIN_ASSOCIATION_ROWS = 10
MIN_MISSING_RATE = 0.05
MIN_PRESENT_ROWS = 20
OUTLIER_IQR = 1.5


def entropy(series: pd.Series) -> float:
    """Shannon entropy (bits) of a column's value distribution."""
    proportions = series.dropna().value_counts(normalize=True).to_numpy()
    if proportions.size == 0:
        return 0.0
    return float(-(proportions * np.log2(proportions)).sum())


def outlier_rate(series: pd.Series) -> float:
    """Share of values outside Tukey's 1.5 x IQR fences."""
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < 4:
        return 0.0
    q1, q3 = np.percentile(values, [25, 75])
    spread = q3 - q1
    if spread == 0:
        return 0.0
    outside = (values < q1 - OUTLIER_IQR * spread) | (
        values > q3 + OUTLIER_IQR * spread
    )
    return float(outside.mean())


def to_finite(series: pd.Series) -> pd.Series:
    """Coerces to float, mapping unparseable and infinite values to NaN."""
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.replace([np.inf, -np.inf], np.nan)


def to_float(values: Any) -> np.ndarray:
    """Coerces a 2-D block to float; unparseable and infinite become NaN."""
    frame = pd.DataFrame(values).apply(to_finite)
    return frame.to_numpy()


def to_object(values: Any) -> np.ndarray:
    """Casts a 2-D block to object dtype with NaN for missing entries."""
    frame = pd.DataFrame(values).astype(object)
    return frame.where(frame.notna(), np.nan).to_numpy()


def purity_association(key: pd.Series, target: pd.Series) -> float:
    """Measures how well `key` groups determine the target class.

    Args:
        key: Grouping of rows (category values or numeric bins).
        target: Class labels.

    Returns:
        A score in [0, 1]; 1 means every group maps to a single class,
        0 means no improvement over predicting the majority class.
    """
    majority = target.value_counts().iloc[0]
    if len(target) == majority:
        return 0.0
    group_majority = target.groupby(key, observed=True).agg(
        lambda labels: labels.value_counts().iloc[0]
    )
    return float((group_majority.sum() - majority) / (len(target) - majority))


class ColumnKind(abc.ABC):
    """Behaviour shared by a family of columns.

    Attributes:
        name: Kind name; matches `FeaturePlan` fields for feature kinds.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def matches(self, series: pd.Series) -> bool:
        """Whether `series` belongs to this kind."""

    def statistics(self, series: pd.Series) -> dict[str, Any]:
        """Returns descriptive statistics for the column."""
        return {
            "kind": self.name,
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
        }

    def exclusion_reason(self, series: pd.Series) -> str | None:
        """Returns why the column cannot be a feature, if it cannot."""
        return None

    def transformer(
        self, options: data_contracts.PreprocessingOptions
    ) -> Any | None:
        """Builds the preprocessing transformer, or None for non-features.

        Args:
            options: Preprocessing switches.
        """
        return None

    def association(
        self, series: pd.Series, target: pd.Series, task: tasks.TaskType
    ) -> float:
        """Returns a [0, 1] single-feature association with the target."""
        return 0.0

    def missingness_association(
        self, series: pd.Series, target: pd.Series, task: tasks.TaskType
    ) -> float:
        """Measures how strongly *being missing* tells the target apart.

        Values recorded only for some outcomes (a discharge code, a recovered
        body, a lifeboat) are a classic leakage pattern that value-based
        associations cannot see, because the informative rows are the
        missing ones.

        Args:
            series: Feature values.
            target: Labels or values.
            task: Resolved task type.

        Returns:
            A score in [0, 1]; 1.0 when the feature is recorded for only one
            outcome class, otherwise the gain of the missingness indicator.
        """
        indicator = series.isna()
        rate = float(indicator.mean())
        present = int((~indicator).sum())
        if rate < MIN_MISSING_RATE or present < MIN_PRESENT_ROWS:
            return 0.0
        if not task.is_classification:
            corr = indicator.astype(float).corr(
                target.astype(float), method="spearman"
            )
            return float(abs(corr)) if pd.notna(corr) else 0.0
        gain = purity_association(indicator, target)
        one_sided = (
            target[~indicator].nunique() == 1
            and target[indicator].nunique() > 1
        )
        return max(gain, 1.0 if one_sided else 0.0)

    def usable_rows(
        self, series: pd.Series, target: pd.Series
    ) -> tuple[pd.Series, pd.Series] | None:
        """Aligns non-missing rows; None when too few rows to judge."""
        mask = series.notna() & target.notna()
        if mask.sum() < MIN_ASSOCIATION_ROWS or series[mask].nunique() < 2:
            return None
        return series[mask], target[mask]


class NumericKind(ColumnKind):
    """Numeric and boolean columns."""

    name = "numeric"

    def matches(self, series: pd.Series) -> bool:
        """True for numeric and boolean dtypes."""
        return pd.api.types.is_numeric_dtype(
            series
        ) or pd.api.types.is_bool_dtype(series)

    def statistics(self, series: pd.Series) -> dict[str, Any]:
        """Adds moments and range for non-boolean columns."""
        stats = super().statistics(series)
        if not pd.api.types.is_bool_dtype(series):
            finite = to_finite(series)
            names = ["skew", "mean", "std", "min", "max"]
            moments = finite.agg(names).to_dict()
            stats |= {
                str(k): float(np.nan_to_num(v)) for k, v in moments.items()
            }
            stats["outlier_rate"] = outlier_rate(finite)
        return stats

    def transformer(
        self, options: data_contracts.PreprocessingOptions
    ) -> pipeline.Pipeline:
        """Casts, imputes, then optionally log-transforms, scales, crosses."""
        steps = [
            (
                "cast",
                preprocessing.FunctionTransformer(
                    to_float, feature_names_out="one-to-one"
                ),
            ),
            (
                "impute",
                impute.SimpleImputer(
                    strategy="median", keep_empty_features=True
                ),
            ),
        ]
        if options.log_skewed:
            steps.append(("log", transformers.SkewedLogTransformer()))
        if options.scale:
            steps.append(("scale", preprocessing.StandardScaler()))
        if options.interactions:
            steps.append(
                (
                    "interactions",
                    preprocessing.PolynomialFeatures(
                        degree=2, interaction_only=True, include_bias=False
                    ),
                )
            )
        return pipeline.Pipeline(steps)

    def association(
        self, series: pd.Series, target: pd.Series, task: tasks.TaskType
    ) -> float:
        """Spearman for regression; binned class purity for classification."""
        rows = self.usable_rows(series, target)
        if rows is None:
            return 0.0
        values, labels = rows
        if not task.is_classification:
            corr = values.astype(float).corr(
                labels.astype(float), method="spearman"
            )
            return float(abs(corr)) if pd.notna(corr) else 0.0
        if values.nunique() <= MAX_DIRECT_VALUES:
            key = values
        else:
            key = pd.qcut(values.astype(float), 20, duplicates="drop")
        return purity_association(key, labels)


class DatetimeKind(ColumnKind):
    """Datetime columns: usable for splitting only in v0.1."""

    name = "datetime"

    def matches(self, series: pd.Series) -> bool:
        """True for datetime dtypes."""
        return pd.api.types.is_datetime64_any_dtype(series)

    def exclusion_reason(self, series: pd.Series) -> str | None:
        """Datetimes are not yet engineered into features."""
        return (
            "datetime (unsupported in v0.1; set as time_column or "
            "engineer features)"
        )


class CategoricalKind(ColumnKind):
    """Everything else: strings, categories, objects."""

    name = "categorical"

    def matches(self, series: pd.Series) -> bool:
        """Fallback kind; matches any column."""
        return True

    def statistics(self, series: pd.Series) -> dict[str, Any]:
        """Adds the most frequent values."""
        stats = super().statistics(series)
        top = series.value_counts().head(5)
        stats["top"] = {str(k): int(v) for k, v in top.items()}
        stats["entropy_bits"] = entropy(series)
        return stats

    def exclusion_reason(self, series: pd.Series) -> str | None:
        """Near-unique text columns are treated as identifiers."""
        if series.nunique(dropna=True) / max(len(series), 1) > ID_UNIQUE_RATIO:
            return "id-like (near-unique text)"
        return None

    def transformer(
        self, options: data_contracts.PreprocessingOptions
    ) -> pipeline.Pipeline:
        """Casts to object, imputes a token, one-hot encodes."""
        return pipeline.Pipeline(
            [
                (
                    "cast",
                    preprocessing.FunctionTransformer(
                        to_object, feature_names_out="one-to-one"
                    ),
                ),
                (
                    "impute",
                    impute.SimpleImputer(
                        strategy="constant", fill_value=MISSING_TOKEN
                    ),
                ),
                (
                    "onehot",
                    preprocessing.OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=0.005,
                        max_categories=50,
                        sparse_output=False,
                    ),
                ),
            ]
        )

    def association(
        self, series: pd.Series, target: pd.Series, task: tasks.TaskType
    ) -> float:
        """Class purity of category values (classification only)."""
        rows = self.usable_rows(series, target)
        if rows is None or not task.is_classification:
            return 0.0
        values, labels = rows
        if values.nunique() > len(values) / 2:
            return 0.0  # Near-unique keys are trivially pure; see id rule.
        return purity_association(values, labels)


KINDS: tuple[ColumnKind, ...] = (
    NumericKind(),
    DatetimeKind(),
    CategoricalKind(),
)


def kind_of(series: pd.Series) -> ColumnKind:
    """Returns the first kind that matches `series`."""
    return next(kind for kind in KINDS if kind.matches(series))
