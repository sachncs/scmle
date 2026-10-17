"""Typed, versioned project specification (pydantic)."""

import enum
import hashlib
import json
from typing import Any
from typing import Literal
from typing import Self

import pydantic

CONFIG_VERSION = 1
FAST_BUDGET_SECONDS = 300
FAST_SEARCH_MODELS = 1
FAST_SEARCH_CONFIGS = 3
GRAPH_VERSION = 1
DELAYED_SUFFIX = "__available"


class TaskType(enum.StrEnum):
    """Supervised learning task kinds."""

    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"

    @property
    def is_classification(self) -> bool:
        """Whether the task predicts discrete classes."""
        return self is not TaskType.REGRESSION


TaskRequest = Literal[
    "auto",
    "classification",
    "regression",
    "binary_classification",
    "multiclass_classification",
]
"""Public task requests; `classification` is resolved from the target."""


class FrozenModel(pydantic.BaseModel):
    """Immutable pydantic base that rejects unknown fields."""

    model_config = pydantic.ConfigDict(extra="forbid", frozen=True)


class FeatureNode(FrozenModel):
    """One derived feature: `name = op(inputs; params)`.

    Attributes:
        name: Name of the new column; must be unique and not clash with data.
        op: Registered operation name (for example `log1p`, `ratio`, `bin`,
            `datepart`, `lag`, `rolling_mean`).
        inputs: Source columns or earlier nodes.
        params: Operation parameters.
    """

    name: str
    op: str
    inputs: list[str]
    params: dict[str, Any] = pydantic.Field(default_factory=dict)


class FeatureGraphSpec(FrozenModel):
    """An ordered set of derived features.

    Nodes are listed in dependency order: a node may only use raw columns and
    nodes that appear before it, which makes the graph acyclic by
    construction. The spec is part of the project configuration, so changing
    it changes the configuration hash.

    Attributes:
        version: Graph schema version.
        nodes: The derived features, in dependency order.
    """

    version: int = GRAPH_VERSION
    nodes: list[FeatureNode] = pydantic.Field(default_factory=list)

    @pydantic.model_validator(mode="after")
    def check_structure(self) -> Self:
        """Validates unique names and dependency order.

        Raises:
            ValueError: On duplicate names or a node using a later node.
        """
        seen: set[str] = set()
        names = [node.name for node in self.nodes]
        if len(names) != len(set(names)):
            raise ValueError("feature node names must be unique")
        for node in self.nodes:
            if not node.inputs:
                raise ValueError(f"node '{node.name}' has no inputs")
            later = set(names) - seen - {node.name}
            used = later & set(node.inputs)
            if used or node.name in node.inputs:
                culprits = sorted(used | ({node.name} & set(node.inputs)))
                raise ValueError(
                    f"node '{node.name}' depends on a node defined later or "
                    f"on itself: {culprits}"
                )
            seen.add(node.name)
        return self

    def fingerprint(self) -> str:
        """Content hash identifying this exact graph."""
        payload = json.dumps(
            [node.model_dump(mode="json") for node in self.nodes],
            sort_keys=True,
        )
        return hashlib.sha256(
            f"v{self.version}:{payload}".encode()
        ).hexdigest()[:16]


class DataSpec(FrozenModel):
    """Where the data lives and what its columns mean.

    Attributes:
        uri: Where the data lives.
        target: Target column.
        id_columns: Identifier columns (never features).
        group_column: Entity column for grouped and series data.
        time_column: Column ordering the rows in time.
        time_as_feature: Also give the time column to the model as a
            feature (a numeric stamp stays numeric; a date becomes date
            parts). Off by default because a model that keys on absolute
            time must extrapolate into the future.
        excluded_columns: Columns never used.
        unavailable_at_prediction: Columns only known after the outcome.
        feature_delays: For columns whose value becomes known only some
            rows after the row it describes (for example a monthly report
            published a quarter later), the number of rows of delay. The
            model then sees the value from that many rows earlier; the raw
            column is excluded. Needs `time_column`.
        monotonic_constraints: Numeric feature to +1 or -1: the prediction
            must never fall (+1) or never rise (-1) as the feature grows,
            other features held fixed. Only models that can enforce this
            are planned (binary classification and regression).
        unlabeled: What to do with rows whose target is missing: `block`
            (refuse, the default), `ignore` (set them aside and report how
            many), or `pseudo_label` (classification only: label the rows the
            model is confident about and train on them, but only if that
            wins in cross-validation).
        pseudo_label_confidence: Smallest top-class probability trusted
            when pseudo-labeling.
        weight_column: Numeric column of non-negative per-row weights (for
            example survey weights or recency). Rows count in training by
            their weight; the column is never a feature and is not needed
            at prediction time. Test metrics are also reported weighted.
        sensitive_columns: Columns describing groups of people (for example
            sex or age band). They are not hidden from the model unless also
            excluded; the fairness report compares outcomes across them.
    """

    uri: str
    target: str
    id_columns: list[str] = pydantic.Field(default_factory=list)
    group_column: str | None = None
    time_column: str | None = None
    time_as_feature: bool = False
    excluded_columns: list[str] = pydantic.Field(default_factory=list)
    unavailable_at_prediction: list[str] = pydantic.Field(default_factory=list)
    feature_delays: dict[str, int] = pydantic.Field(default_factory=dict)
    sensitive_columns: list[str] = pydantic.Field(default_factory=list)
    monotonic_constraints: dict[str, int] = pydantic.Field(default_factory=dict)
    weight_column: str | None = None
    unlabeled: Literal["block", "ignore", "pseudo_label"] = "block"
    pseudo_label_confidence: float = pydantic.Field(default=0.9, gt=0.5, lt=1.0)


class ValidationSpec(FrozenModel):
    """How data is partitioned for model selection and final testing.

    Attributes:
        strategy: Validation protocol. `temporal` uses expanding training
            windows; `rolling` slides a fixed-size training window forward
            (see `window_rows` and `horizon_rows`).
        test_fraction: Share of rows locked away for the final test.
        folds: Number of validation folds.
        seed: Random seed.
        embargo_rows: Rows dropped between training and validation windows
            so serial dependence cannot leak across the boundary.
        horizon_rows: Rolling only: rows in each validation window.
        window_rows: Rolling only: rows in each training window (None keeps
            every earlier row, an expanding window).
    """

    strategy: Literal[
        "auto", "stratified", "random", "group", "temporal", "rolling"
    ] = "auto"
    test_fraction: float = pydantic.Field(default=0.20, gt=0, lt=1)
    folds: int = pydantic.Field(default=5, ge=2)
    seed: int = 42
    embargo_rows: int = pydantic.Field(default=0, ge=0)
    horizon_rows: int | None = pydantic.Field(default=None, ge=1)
    window_rows: int | None = pydantic.Field(default=None, ge=1)


class ResourceSpec(FrozenModel):
    """Compute limits for a run."""

    time_budget_seconds: int = pydantic.Field(default=1800, gt=0)
    memory_limit_gb: float = pydantic.Field(default=8.0, gt=0)
    cpu_limit: int | None = pydantic.Field(default=None, ge=1)
    workers: int = pydantic.Field(default=1, ge=1)
    executor: Literal["process", "ray"] = "process"
    gpu: Literal["auto", "disabled", "required"] = "auto"
    enforcement: Literal["hard", "cooperative"] = "hard"


class SearchSpec(FrozenModel):
    """Hyperparameter search settings (disabled when `top_models` is 0)."""

    top_models: int = pydantic.Field(default=2, ge=0)
    configs_per_model: int = pydantic.Field(default=6, ge=1)
    optimizer: Literal["random", "optuna"] = "random"
    prune_fraction: float = pydantic.Field(default=0.5, ge=0, lt=1)


class StrategySpec(FrozenModel):
    """Which data strategies to evaluate (imbalance and feature variants)."""

    names: list[str] = pydantic.Field(
        default_factory=lambda: [
            "class_weight",
            "random_oversample",
            "random_undersample",
            "smote",
            "borderline_smote",
            "adasyn",
            "calibrated",
            "log_skewed",
            "interactions",
            "construct_features",
            "select_features",
            "missing_indicators",
            "iterative_imputation",
            "target_transform",
            "target_encoding",
            "rare_target_weighting",
        ]
    )
    top_models: int = pydantic.Field(default=2, ge=0)


class EnsembleSpec(FrozenModel):
    """Stacking ensemble settings (disabled when `size` < 2)."""

    size: int = pydantic.Field(default=3, ge=0)


class SelectionSpec(FrozenModel):
    """How the winning model is chosen and which constraints it must meet.

    Attributes:
        maximize: Metric to optimize (the project's `metric`; setting both
            to different values is an error).
        max_model_mb: Largest allowed serialized size.
        max_p95_latency_ms: Largest allowed p95 single-row latency.
        latency_safety_factor: Multiplier applied to the p95 latency measured
            on this machine before it is compared with the limit, to account
            for slower serving hardware.
        max_candidates_checked: How many ranked candidates are measured.
        tie_breaker: How candidates with indistinguishable scores are
            ordered: `simpler_model` (cheaper model family, default
            strategy) or `lower_variance` (steadier across folds).
        tie_tolerance: Absolute score difference regarded as a tie.
        uncertainty: `one_standard_error` also treats every candidate within
            one standard error of the best mean score as tied, so noise is
            not mistaken for a real difference. `bayesian` compares paired
            fold scores and treats a candidate as tied with the best unless
            the best is better with at least `bayesian_confidence`
            probability (beyond a region of practical equivalence `rope`).
        bayesian_confidence: Probability the best candidate must reach to be
            declared better than another.
        rope: Half-width, in metric units, of the region in which two
            candidates count as equivalent.
    """

    maximize: str | None = None
    max_model_mb: float | None = pydantic.Field(default=None, gt=0)
    max_p95_latency_ms: float | None = pydantic.Field(default=None, gt=0)
    latency_safety_factor: float = pydantic.Field(default=1.0, ge=1.0)
    max_candidates_checked: int = pydantic.Field(default=10, ge=1)
    tie_breaker: Literal["simpler_model", "lower_variance"] = "simpler_model"
    tie_tolerance: float = pydantic.Field(default=0.0, ge=0.0)
    uncertainty: Literal["none", "one_standard_error", "bayesian"] = "none"
    bayesian_confidence: float = pydantic.Field(default=0.95, gt=0.5, lt=1.0)
    rope: float = pydantic.Field(default=0.0, ge=0.0)


class ModelSpec(FrozenModel):
    """Which model families may be considered.

    Attributes:
        exclude_models: Model names that must not run.
        exclude_families: Model families that must not run (for example
            `deep_learning`, `boosting`).
    """

    exclude_models: list[str] = pydantic.Field(default_factory=list)
    exclude_families: list[str] = pydantic.Field(default_factory=list)


class DecisionSpec(FrozenModel):
    """How binary probabilities become decisions (threshold optimization)."""

    threshold_objective: Literal[
        "none", "f1", "accuracy", "balanced_accuracy", "cost"
    ] = "none"
    false_positive_cost: float = pydantic.Field(default=1.0, gt=0)
    false_negative_cost: float = pydantic.Field(default=1.0, gt=0)


class ProjectSpec(FrozenModel):
    """The explicit, versioned specification of an ML project."""

    version: int = CONFIG_VERSION
    data: DataSpec
    task: TaskRequest = "auto"
    metric: str | None = None
    validation: ValidationSpec = pydantic.Field(default_factory=ValidationSpec)
    resources: ResourceSpec = pydantic.Field(default_factory=ResourceSpec)
    search: SearchSpec = pydantic.Field(default_factory=SearchSpec)
    strategies: StrategySpec = pydantic.Field(default_factory=StrategySpec)
    models: ModelSpec = pydantic.Field(default_factory=ModelSpec)
    feature_graph: FeatureGraphSpec = pydantic.Field(
        default_factory=FeatureGraphSpec
    )
    ensemble: EnsembleSpec = pydantic.Field(default_factory=EnsembleSpec)
    selection: SelectionSpec = pydantic.Field(default_factory=SelectionSpec)
    decision: DecisionSpec = pydantic.Field(default_factory=DecisionSpec)
    leakage_policy: Literal["warn", "block"] = "warn"

    @pydantic.model_validator(mode="before")
    @classmethod
    def expand_feature_delays(cls, values: Any) -> Any:
        """Turns `data.feature_delays` into lagged nodes and exclusions.

        The expansion is idempotent, so a saved configuration (which already
        contains the nodes) loads unchanged.
        """
        if not isinstance(values, dict):
            return values
        data = values.get("data")
        data_record = (
            data.model_dump()
            if isinstance(data, DataSpec)
            else dict(data or {})
        )
        delays = data_record.get("feature_delays") or {}
        if not delays:
            return values
        graph = values.get("feature_graph")
        graph_record = (
            graph.model_dump()
            if isinstance(graph, FeatureGraphSpec)
            else dict(graph or {})
        )
        nodes = list(graph_record.get("nodes", []))
        present = {node["name"] for node in nodes}
        for column, delay in delays.items():
            name = f"{column}{DELAYED_SUFFIX}"
            if name not in present:
                nodes.append(
                    {
                        "name": name,
                        "op": "lag",
                        "inputs": [column],
                        "params": {"k": delay},
                    }
                )
        excluded = list(data_record.get("excluded_columns", []))
        excluded += [c for c in delays if c not in excluded]
        return {
            **values,
            "data": {**data_record, "excluded_columns": excluded},
            "feature_graph": {**graph_record, "nodes": nodes},
        }

    @pydantic.model_validator(mode="after")
    def check_cross_field_rules(self) -> Self:
        """Validates constraints that span several fields.

        Returns:
            The validated specification.

        Raises:
            ValueError: If the combination of fields is inconsistent.
        """
        data, validation = self.data, self.validation
        if validation.strategy == "group" and not data.group_column:
            raise ValueError("strategy 'group' requires data.group_column")
        if validation.strategy in ("temporal", "rolling") and (
            not data.time_column
        ):
            raise ValueError(
                f"strategy '{validation.strategy}' requires data.time_column"
            )
        if (
            validation.horizon_rows or validation.window_rows
        ) and validation.strategy not in ("rolling", "auto"):
            raise ValueError(
                "horizon_rows and window_rows apply to the rolling strategy"
            )
        if data.feature_delays and not data.time_column:
            raise ValueError("data.feature_delays requires data.time_column")
        if any(delay < 1 for delay in data.feature_delays.values()):
            raise ValueError("feature delays must be at least 1 row")
        stale = [
            node.name
            for node in self.feature_graph.nodes
            if set(node.inputs) & set(data.feature_delays)
            and not node.name.endswith(DELAYED_SUFFIX)
        ]
        if stale:
            raise ValueError(
                f"feature nodes {stale} read a delayed column directly; "
                f"read '<column>{DELAYED_SUFFIX}' instead"
            )
        if validation.strategy == "stratified" and self.task == "regression":
            raise ValueError(
                "stratified validation is undefined for regression"
            )
        for name in ("group_column", "time_column"):
            if getattr(data, name) == data.target:
                raise ValueError(f"data.{name} must differ from the target")
        if data.target in data.id_columns:
            raise ValueError("the target cannot be an id column")
        if data.target in data.excluded_columns:
            raise ValueError("the target cannot be excluded")
        if any(
            sign not in (-1, 1) for sign in data.monotonic_constraints.values()
        ):
            raise ValueError("monotonic constraints must be +1 or -1")
        if data.target in data.monotonic_constraints:
            raise ValueError("the target cannot have a monotonic constraint")
        if data.target in data.sensitive_columns:
            raise ValueError("the target cannot be a sensitive column")
        if data.target in data.unavailable_at_prediction:
            raise ValueError("the target cannot be unavailable at prediction")
        wanted = self.selection.maximize
        if wanted and self.metric and wanted != self.metric:
            raise ValueError(
                f"selection.maximize '{wanted}' conflicts with metric "
                f"'{self.metric}'"
            )
        if validation.embargo_rows and not data.time_column:
            raise ValueError(
                "validation.embargo_rows requires data.time_column"
            )
        return self

    @property
    def primary_metric(self) -> str | None:
        """The metric to optimize: `metric`, else `selection.maximize`."""
        return self.metric or self.selection.maximize

    @property
    def fast_budget(self) -> bool:
        """Whether the time budget is short enough to shrink the search."""
        return self.resources.time_budget_seconds < FAST_BUDGET_SECONDS

    def effective_search(self) -> SearchSpec:
        """The search settings in force: reduced under a short budget."""
        if not self.fast_budget:
            return self.search
        return self.search.model_copy(
            update={
                "top_models": min(self.search.top_models, FAST_SEARCH_MODELS),
                "configs_per_model": min(
                    self.search.configs_per_model, FAST_SEARCH_CONFIGS
                ),
            }
        )

    def effective_strategies(self) -> StrategySpec:
        """The strategy settings in force: reduced under a short budget."""
        if not self.fast_budget:
            return self.strategies
        return self.strategies.model_copy(
            update={
                "top_models": min(
                    self.strategies.top_models, FAST_SEARCH_MODELS
                )
            }
        )

    def canonical_json(self) -> str:
        """Returns a key-sorted, whitespace-free JSON rendering."""
        return json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        )

    def config_hash(self) -> str:
        """Returns a short deterministic hash of the canonical configuration."""
        digest = hashlib.sha256(self.canonical_json().encode()).hexdigest()
        return digest[:16]
