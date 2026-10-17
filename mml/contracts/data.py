"""Dataset identity, feature plan and profile contracts."""

from collections.abc import Mapping
import dataclasses
from typing import Any

HIGH_CARDINALITY_LEVELS = 20


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetManifest:
    """Immutable identity of an ingested dataset.

    Attributes:
        dataset_id: Short identifier derived from the content hash.
        schema_hash: Hash of column names and dtypes.
        data_hash: Hash of the row contents.
        row_count: Number of rows.
        column_count: Number of columns.
        target: Name of the target column.
        uri: Where the data was read from.
    """

    dataset_id: str
    schema_hash: str
    data_hash: str
    row_count: int
    column_count: int
    target: str
    uri: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class FeaturePlan:
    """Which columns become features and why the others were dropped.

    Attributes:
        numeric: Numeric feature columns.
        categorical: Categorical feature columns.
        text: Free-text columns (encoded as TF-IDF terms; also listed in
            `extra_inputs` so inference requires them).
        dropped: Mapping of dropped column to the reason.
        derived: Row-wise feature-graph nodes computed inside the pipeline.
        series: Ordered (time-series) feature-graph nodes.
        extra_inputs: Raw columns only the feature graph reads.
        high_cardinality: Categorical features with many distinct levels
            (`HIGH_CARDINALITY_LEVELS` or more), which one-hot encoding
            handles poorly.
        group_column: Entity column, so inner cross-validation keeps entities
            together.
        time_column: Ordering column, so inner cross-validation keeps time
            order.
    """

    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    dropped: Mapping[str, str] = dataclasses.field(default_factory=dict)
    text: tuple[str, ...] = ()
    derived: tuple[Any, ...] = ()
    series: tuple[Any, ...] = ()
    extra_inputs: tuple[str, ...] = ()
    high_cardinality: tuple[str, ...] = ()
    group_column: str | None = None
    time_column: str | None = None

    @property
    def features(self) -> tuple[str, ...]:
        """All feature columns the model consumes, numeric first."""
        return self.numeric + self.categorical

    @property
    def derived_names(self) -> frozenset[str]:
        """Columns computed by the feature graph (row-wise or ordered)."""
        return frozenset(n.name for n in (*self.derived, *self.series))

    @property
    def input_columns(self) -> tuple[str, ...]:
        """Raw columns the fitted pipeline reads (plus series features).

        These exist in the frame a pipeline is fitted on: model features that
        are not computed inside the pipeline, graph-only inputs, and the
        ordered (series) features computed at ingestion.
        """
        rowwise = frozenset(n.name for n in self.derived)
        columns = [c for c in self.features if c not in rowwise]
        return tuple(dict.fromkeys([*columns, *self.extra_inputs]))

    def columns_for(self, kind_name: str) -> tuple[str, ...]:
        """Returns the feature columns of one kind.

        Args:
            kind_name: `numeric` or `categorical`.

        Returns:
            Matching columns, empty if the kind carries no features.
        """
        return getattr(self, kind_name, ())

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "numeric": list(self.numeric),
            "categorical": list(self.categorical),
            "text": list(self.text),
            "dropped": dict(self.dropped),
            "derived": [n.model_dump(mode="json") for n in self.derived],
            "series": [n.model_dump(mode="json") for n in self.series],
            "extra_inputs": list(self.extra_inputs),
            "high_cardinality": list(self.high_cardinality),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class DatasetProfile:
    """Statistical profile and findings for a dataset.

    Attributes:
        manifest: Identity of the profiled dataset.
        column_statistics: Per-column statistics.
        quality_findings: Data-quality observations.
        risk_findings: Leakage / imbalance / shift hypotheses (not verdicts).
        target_summary: Target distribution summary.
        feature_plan: Resulting feature selection.
        dataset_statistics: Dataset-level fingerprint grouped by dimension
            (shape, types, missingness, target, features, relationships,
            structure, quality, operational, risk).
    """

    manifest: DatasetManifest
    column_statistics: Mapping[str, Any]
    quality_findings: tuple[str, ...]
    risk_findings: tuple[str, ...]
    target_summary: Mapping[str, Any]
    feature_plan: FeaturePlan
    dataset_statistics: Mapping[str, Any] = dataclasses.field(
        default_factory=dict
    )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return {
            "manifest": self.manifest.to_dict(),
            "columns": dict(self.column_statistics),
            "quality_findings": list(self.quality_findings),
            "risk_findings": list(self.risk_findings),
            "target": dict(self.target_summary),
            "feature_plan": self.feature_plan.to_dict(),
            "dataset": dict(self.dataset_statistics),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class PreprocessingOptions:
    """Switches that shape the preprocessing transformers.

    Attributes:
        scale: Standardize numeric features.
        log_skewed: Log-transform strongly right-skewed numeric features.
        interactions: Add pairwise numeric interaction features.
        impute: Impute missing numeric values (off for models that handle
            missing values natively).
        high_cardinality: How high-cardinality categorical features are
            encoded: `onehot`, `target` (regularized, cross-fitted target
            encoding) or `ordinal` (integer codes for native categorical
            handling).
        sparse: Emit sparse one-hot output (for sparse-capable models).
        construct: Append differences and ratios of the numeric features
            most associated with the target.
        select: Keep only the columns an L1-penalised linear model uses.
        missing: How missing numeric values are filled: `median`,
            `indicator` (median plus a flag column per feature that has
            missing values) or `iterative` (each feature predicted from the
            others).
    """

    scale: bool = False
    log_skewed: bool = False
    interactions: bool = False
    impute: bool = True
    high_cardinality: str = "onehot"
    sparse: bool = False
    construct: bool = False
    select: bool = False
    missing: str = "median"
