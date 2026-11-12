"""Column kinds.

Each `ColumnKind` subclass encapsulates everything kind-specific: detection,
statistics, exclusion rules, the preprocessing transformer and the
target-association heuristic used for leakage hypotheses. Profiling and
preprocessing therefore iterate over kinds instead of branching on dtype.
"""

import abc
import importlib
from typing import Any
from typing import ClassVar

import numpy
import pandas
from sklearn import impute
from sklearn import pipeline
from sklearn import preprocessing
import sklearn.tree

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.curation import embedding
from scmle.data.curation import transformers

MISSING_TOKEN = "__missing__"  # nosec B105  # noqa: S105 -- a category placeholder, not a secret
ID_UNIQUE_RATIO = 0.95
MIN_TEXT_WORDS = 4
MIN_IQR_VALUES = 4
MIN_DISTINCT_VALUES = 2
MIN_TEXT_ROWS = 30
TEXT_UNIQUE_RATIO = 0.5
TEXT_SAMPLE = 2000
TEXT_TERMS = 200
MAX_DIRECT_VALUES = 20
TREE_LEAVES = 20
MIN_ASSOCIATION_ROWS = 10
MIN_MISSING_RATE = 0.05
MIN_PRESENT_ROWS = 20
OUTLIER_IQR = 1.5


def entropy(series: pandas.Series) -> float:
    """Computes the Shannon entropy of a column's value distribution.

    Args:
        series: Column values; missing entries are ignored.

    Returns:
        The entropy in bits, 0.0 when the column has no non-missing values.
    """
    proportions = series.dropna().value_counts(normalize=True).to_numpy()
    if proportions.size == 0:
        return 0.0
    return float(-(proportions * numpy.log2(proportions)).sum())


def outlier_rate(series: pandas.Series) -> float:
    """Computes the share of values outside Tukey's 1.5 x IQR fences.

    Args:
        series: Numeric column; missing entries are ignored.

    Returns:
        The fraction of non-missing values below Q1 - 1.5 x IQR or above
        Q3 + 1.5 x IQR, in [0, 1]. 0.0 when fewer than four values remain
        or the interquartile range is zero.
    """
    values = series.dropna().to_numpy(dtype=float)
    if len(values) < MIN_IQR_VALUES:
        return 0.0
    q1, q3 = numpy.percentile(values, [25, 75])
    spread = q3 - q1
    if spread == 0:
        return 0.0
    outside = (values < q1 - OUTLIER_IQR * spread) | (
        values > q3 + OUTLIER_IQR * spread
    )
    return float(outside.mean())


def to_finite(series: pandas.Series) -> pandas.Series:
    """Coerces a column to float, mapping unparseable and infinite to NaN.

    Args:
        series: Column values.

    Returns:
        A float series with the same index in which unparseable and
        infinite entries are NaN.
    """
    numeric = pandas.to_numeric(series, errors="coerce").astype(float)
    return numeric.replace([numpy.inf, -numpy.inf], numpy.nan)


def to_float(values: Any) -> numpy.ndarray:
    """Coerces a 2-D block to float; unparseable and infinite become NaN.

    Args:
        values: Array-like block convertible to a DataFrame.

    Returns:
        A writable float array of the same shape as `values`.
    """
    frame = pandas.DataFrame(values).apply(to_finite)
    return numpy.array(frame.to_numpy(), dtype=float, copy=True)


def experimental_impute() -> Any:
    """Builds the iterative imputer (scikit-learn gates it behind an import).

    Returns:
        An unfitted `IterativeImputer` with five iterations, a fixed random
        state and empty features kept.
    """
    # Imported for its side effect (it enables the experimental imputer) and
    # kept lazy so it runs only when the imputer is requested.
    importlib.import_module("sklearn.experimental.enable_iterative_imputer")

    return impute.IterativeImputer(
        max_iter=5,
        keep_empty_features=True,
        random_state=0,
        sample_posterior=False,
    )


def to_text(values: Any) -> numpy.ndarray:
    """Casts a one-column block to a 1-D array of strings ("" if missing).

    Args:
        values: Array-like block whose first column is used.

    Returns:
        A 1-D array of strings, one per row, with missing values as "".
    """
    frame = pandas.DataFrame(values)
    return frame.iloc[:, 0].fillna("").astype(str).to_numpy()


def to_object(values: Any) -> numpy.ndarray:
    """Casts a 2-D block to object dtype with NaN for missing entries.

    Args:
        values: Array-like block convertible to a DataFrame.

    Returns:
        An object array of the same shape as `values`, missing entries NaN.
    """
    frame = pandas.DataFrame(values).astype(object)
    return frame.where(frame.notna(), numpy.nan).to_numpy()


def purity_association(key: pandas.Series, target: pandas.Series) -> float:
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


def threshold_association(
    values: pandas.Series, labels: pandas.Series
) -> float:
    """Class-separation gain of a small decision tree on one numeric feature.

    Fixed-width or quantile bins blur a sharp threshold (the bin that
    straddles it is impure), so a feature that separates the classes
    perfectly could score below the leakage cut-off. A depth-limited tree
    places its splits where the classes change, so a threshold rule scores
    1.0 while noise stays low (leaves must hold several rows).

    Args:
        values: Numeric feature values.
        labels: Class labels.

    Returns:
        In-sample accuracy gain over predicting the majority class, in [0, 1].
    """
    majority = labels.value_counts(normalize=True).iloc[0]
    if majority >= 1.0:
        return 0.0
    tree = sklearn.tree.DecisionTreeClassifier(
        max_leaf_nodes=TREE_LEAVES,
        min_samples_leaf=max(3, len(values) // 100),
        random_state=0,
    ).fit(values.to_numpy().reshape(-1, 1), labels)
    accuracy = float(tree.score(values.to_numpy().reshape(-1, 1), labels))
    return max(0.0, (accuracy - majority) / (1.0 - majority))


class ColumnKind(abc.ABC):
    """Behaviour shared by a family of columns.

    Attributes:
        name: Kind name; matches `FeaturePlan` fields for feature kinds.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def matches(self, series: pandas.Series) -> bool:
        """Reports whether `series` belongs to this kind.

        Args:
            series: Column values.

        Returns:
            True when the column is of this kind.
        """

    def statistics(self, series: pandas.Series) -> dict[str, Any]:
        """Returns descriptive statistics for the column.

        Args:
            series: Column values.

        Returns:
            A dict with the keys `kind`, `dtype`, `missing_rate` (fraction
            of missing entries) and `n_unique` (distinct non-missing values).
        """
        return {
            "kind": self.name,
            "dtype": str(series.dtype),
            "missing_rate": float(series.isna().mean()),
            "n_unique": int(series.nunique(dropna=True)),
        }

    def exclusion_reason(self, series: pandas.Series) -> str | None:  # noqa: ARG002 -- ColumnKind hook; every kind shares this signature
        """Returns why the column cannot be a feature, if it cannot.

        Args:
            series: Column values.

        Returns:
            A human-readable reason, or None when the column may be used.
        """
        return None

    def transformer(self, options: data.Preprocessing) -> Any | None:  # noqa: ARG002 -- ColumnKind hook; every kind shares this signature
        """Builds the preprocessing transformer, or None for non-features.

        Args:
            options: Preprocessing switches.

        Returns:
            An unfitted transformer for this kind, or None when the kind
            produces no features.
        """
        return None

    def association(
        self,
        series: pandas.Series,  # noqa: ARG002 -- ColumnKind hook; every kind shares this signature
        target: pandas.Series,  # noqa: ARG002 -- ColumnKind hook; every kind shares this signature
        task: tasks.Task,  # noqa: ARG002 -- ColumnKind hook; every kind shares this signature
    ) -> float:
        """Returns a [0, 1] single-feature association with the target.

        Args:
            series: Feature values.
            target: Labels or values, aligned with `series`.
            task: Resolved task type.

        Returns:
            The association score in [0, 1]; the base implementation
            returns 0.0.
        """
        return 0.0

    def missingness_association(
        self, series: pandas.Series, target: pandas.Series, task: tasks.Task
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
            A score in [0, 1]. 0.0 when fewer than `MIN_MISSING_RATE` of the
            values are missing or fewer than `MIN_PRESENT_ROWS` are present.
            For a regression task, the absolute Spearman correlation of the
            missingness indicator with the target. For classification, the
            gain of the indicator, or 1.0 if the recorded rows hold a single
            class while the missing rows hold several.
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
            return 0.0 if numpy.isnan(corr) else float(abs(corr))
        gain = purity_association(indicator, target)
        one_sided = (
            target[~indicator].nunique() == 1  # noqa: PD101 -- counts distinct non-missing labels; an equality check would treat missing labels as different
            and target[indicator].nunique() > 1  # noqa: PD101 -- counts distinct non-missing labels; an equality check would treat missing labels as different
        )
        return max(gain, 1.0 if one_sided else 0.0)

    def usable_rows(
        self, series: pandas.Series, target: pandas.Series
    ) -> tuple[pandas.Series, pandas.Series] | None:
        """Aligns non-missing rows; None when too few rows to judge.

        Args:
            series: Feature values.
            target: Labels or values, aligned with `series`.

        Returns:
            The feature and target restricted to rows where both are
            present, or None when fewer than `MIN_ASSOCIATION_ROWS` such
            rows remain or the feature has fewer than two distinct values.
        """
        mask = series.notna() & target.notna()
        if (
            mask.sum() < MIN_ASSOCIATION_ROWS
            or series[mask].nunique() < MIN_DISTINCT_VALUES
        ):
            return None
        return series[mask], target[mask]


class NumericKind(ColumnKind):
    """Numeric and boolean columns."""

    name = "numeric"

    def matches(self, series: pandas.Series) -> bool:
        """True for numeric and boolean dtypes.

        Args:
            series: Column values.

        Returns:
            True when the dtype is numeric or boolean.
        """
        return pandas.api.types.is_numeric_dtype(
            series
        ) or pandas.api.types.is_bool_dtype(series)

    def statistics(self, series: pandas.Series) -> dict[str, Any]:
        """Adds moments and range for non-boolean columns.

        Args:
            series: Column values.

        Returns:
            The base statistics plus, for non-boolean columns, `skew`,
            `mean`, `std`, `min`, `max` (NaN replaced by 0.0) and
            `outlier_rate`.
        """
        stats = super().statistics(series)
        if not pandas.api.types.is_bool_dtype(series):
            finite = to_finite(series)
            names = ["skew", "mean", "std", "min", "max"]
            moments = finite.agg(names).to_dict()
            stats |= {
                str(k): float(numpy.nan_to_num(v)) for k, v in moments.items()
            }
            stats["outlier_rate"] = outlier_rate(finite)
        return stats

    @staticmethod
    def imputer(strategy: str) -> Any:
        """The imputer for a missing-value strategy.

        Args:
            strategy: One of `median`, `indicator` (median plus missing
                indicator columns) or `iterative`.

        Returns:
            An unfitted scikit-learn imputer.

        Raises:
            ConfigError: For an unknown strategy.
        """
        if strategy == "median":
            return impute.SimpleImputer(
                strategy="median", keep_empty_features=True
            )
        if strategy == "indicator":
            return impute.SimpleImputer(
                strategy="median", keep_empty_features=True, add_indicator=True
            )
        if strategy == "iterative":
            return experimental_impute()
        raise errors.ConfigError(f"unknown missing-value strategy '{strategy}'")

    def transformer(self, options: data.Preprocessing) -> pipeline.Pipeline:
        """Casts, imputes, then optionally log-transforms, scales, crosses.

        Args:
            options: Preprocessing switches (`impute`, `missing`,
                `log_skewed`, `scale`, `interactions`).

        Returns:
            An unfitted pipeline that casts to float and then applies each
            enabled step in that order. An unknown `options.missing`
            strategy raises `ConfigError` (from `imputer`) when imputation
            is enabled.
        """
        steps: list[tuple[str, Any]] = [
            (
                "cast",
                preprocessing.FunctionTransformer(
                    to_float, feature_names_out="one-to-one"
                ),
            ),
        ]
        if options.impute:
            steps.append(("impute", self.imputer(options.missing)))
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
        self, series: pandas.Series, target: pandas.Series, task: tasks.Task
    ) -> float:
        """Spearman for regression; class purity for classification.

        Classification uses direct group purity for at most
        `MAX_DIRECT_VALUES` distinct values and a decision-tree gain
        otherwise.

        Args:
            series: Feature values.
            target: Labels or values, aligned with `series`.
            task: Resolved task type.

        Returns:
            The association score in [0, 1]; 0.0 when too few usable rows.
        """
        rows = self.usable_rows(series, target)
        if rows is None:
            return 0.0
        values, labels = rows
        if not task.is_classification:
            corr = values.astype(float).corr(
                labels.astype(float), method="spearman"
            )
            return 0.0 if numpy.isnan(corr) else float(abs(corr))
        if values.nunique() <= MAX_DIRECT_VALUES:
            return purity_association(values, labels)
        return threshold_association(values.astype(float), labels)


class DatetimeKind(ColumnKind):
    """Datetime columns: expanded into calendar and trend features."""

    name = "datetime"

    def matches(self, series: pandas.Series) -> bool:
        """True for datetime dtypes.

        Args:
            series: Column values.

        Returns:
            True when the dtype is a datetime type.
        """
        return pandas.api.types.is_datetime64_any_dtype(series)

    def exclusion_reason(self, series: pandas.Series) -> str | None:  # noqa: ARG002 -- ColumnKind hook; every kind shares this signature
        """Datetimes are kept: the profiler expands them into features.

        Args:
            series: Column values.

        Returns:
            Always None.
        """
        return None


class TextKind(ColumnKind):
    """Free text: sentences or descriptions, encoded as TF-IDF terms."""

    name = "text"

    def matches(self, series: pandas.Series) -> bool:
        """Strings with several words per value and mostly unique values.

        Args:
            series: Column values.

        Returns:
            True for string or object columns whose first `TEXT_SAMPLE`
            non-missing values number at least `MIN_TEXT_ROWS`, average at
            least `MIN_TEXT_WORDS` words and are more than
            `TEXT_UNIQUE_RATIO` unique.
        """
        if not (
            pandas.api.types.is_object_dtype(series)
            or pandas.api.types.is_string_dtype(series)
        ):
            return False
        values = series.dropna().astype(str).head(TEXT_SAMPLE)
        if len(values) < MIN_TEXT_ROWS:
            return False
        words = float(numpy.mean([len(v.split()) for v in values]))
        unique = values.nunique() / len(values)
        return bool(words >= MIN_TEXT_WORDS and unique > TEXT_UNIQUE_RATIO)

    def statistics(self, series: pandas.Series) -> dict[str, Any]:
        """Adds the typical number of words per value.

        Args:
            series: Column values.

        Returns:
            The base statistics plus `mean_words`, the mean word count of
            non-missing values (0.0 if there are none).
        """
        stats = super().statistics(series)
        counts = [len(v.split()) for v in series.dropna().astype(str)]
        stats["mean_words"] = float(numpy.mean(counts)) if counts else 0.0
        return stats

    def transformer(self, options: data.Preprocessing) -> pipeline.Pipeline:
        """Casts to text, then TF-IDF terms (fold-local) or embeddings.

        Args:
            options: Preprocessing switches (`text_encoding`,
                `embedding_model`).

        Returns:
            An unfitted pipeline that casts to strings and then encodes at
            most `TEXT_TERMS` terms or embedding dimensions.
        """
        encoder: Any = (
            embedding.TextEmbedder(options.embedding_model, TEXT_TERMS)
            if options.text_encoding == "embedding"
            else transformers.TextVectorizer(max_terms=TEXT_TERMS)
        )
        return pipeline.Pipeline(
            [
                (
                    "text",
                    preprocessing.FunctionTransformer(
                        to_text, feature_names_out="one-to-one"
                    ),
                ),
                (
                    "tfidf",
                    encoder,
                ),
            ]
        )


class CategoricalKind(ColumnKind):
    """Everything else: strings, categories, objects."""

    name = "categorical"

    def matches(self, series: pandas.Series) -> bool:  # noqa: ARG002 -- ColumnKind hook; every kind shares this signature
        """Fallback kind; matches any column.

        Args:
            series: Column values.

        Returns:
            Always True.
        """
        return True

    def statistics(self, series: pandas.Series) -> dict[str, Any]:
        """Adds the most frequent values.

        Args:
            series: Column values.

        Returns:
            The base statistics plus `top` (the five most frequent values
            mapped to their counts) and `entropy_bits`.
        """
        stats = super().statistics(series)
        top = series.value_counts().head(5)
        stats["top"] = {str(k): int(v) for k, v in top.items()}
        stats["entropy_bits"] = entropy(series)
        return stats

    def exclusion_reason(self, series: pandas.Series) -> str | None:
        """Near-unique text columns are treated as identifiers.

        Args:
            series: Column values.

        Returns:
            A reason when more than `ID_UNIQUE_RATIO` of the rows are
            distinct, otherwise None.
        """
        if series.nunique(dropna=True) / max(len(series), 1) > ID_UNIQUE_RATIO:
            return "id-like (near-unique text)"
        return None

    def transformer(self, options: data.Preprocessing) -> pipeline.Pipeline:
        """Casts to object, imputes a token, one-hot encodes.

        Args:
            options: Preprocessing switches; `sparse` selects sparse output.

        Returns:
            An unfitted pipeline ending in a one-hot encoder that groups
            rare levels and keeps at most 50 categories.
        """
        return self.encoded_with(
            preprocessing.OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                min_frequency=0.005,
                max_categories=50,
                sparse_output=options.sparse,
            )
        )

    def encoded_with(self, encoder: Any) -> pipeline.Pipeline:
        """Casts to object, imputes a missing token, then applies `encoder`.

        Args:
            encoder: Final scikit-learn encoder step.

        Returns:
            An unfitted pipeline of cast, constant imputation with
            `MISSING_TOKEN`, and `encoder`.
        """
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
                ("encode", encoder),
            ]
        )

    def high_cardinality_transformer(
        self, encoding: str, seed: int = 0
    ) -> pipeline.Pipeline | None:
        """Encoder for high-cardinality columns, or None for plain one-hot.

        Args:
            encoding: `target` (smoothed, cross-fitted target encoding, so
                the encoding of a training row never sees its own label) or
                `ordinal` (integer codes, unseen levels -1, for models that
                treat the codes as categories natively).
            seed: Random seed for the cross-fitting folds.

        Returns:
            An unfitted pipeline for `target` or `ordinal`, or None for any
            other encoding.
        """
        if encoding == "target":
            return self.encoded_with(
                preprocessing.TargetEncoder(
                    smooth="auto",
                    cv=5,
                    random_state=seed,  # pyright: ignore[reportArgumentType]  # scikit-learn infers this parameter as str from its default
                )
            )
        if encoding == "ordinal":
            return self.encoded_with(
                preprocessing.OrdinalEncoder(
                    handle_unknown="use_encoded_value", unknown_value=-1
                )
            )
        return None

    def association(
        self, series: pandas.Series, target: pandas.Series, task: tasks.Task
    ) -> float:
        """Class purity of category values (classification only).

        Args:
            series: Feature values.
            target: Labels, aligned with `series`.
            task: Resolved task type.

        Returns:
            The purity score in [0, 1]; 0.0 for a non-classification task,
            too few usable rows, or when more than half the usable rows
            hold distinct values (near-unique keys).
        """
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
    TextKind(),
    CategoricalKind(),
)


def kind_of(series: pandas.Series) -> ColumnKind:
    """Returns the first kind that matches `series`.

    Args:
        series: Column values.

    Returns:
        The first entry of `KINDS` whose `matches` accepts the column.
    """
    return next(kind for kind in KINDS if kind.matches(series))
