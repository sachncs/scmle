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
HORIZON_SEPARATOR = "__h"
DEFAULT_LAGS = (1, 2, 3, 7)
DEFAULT_WINDOWS = (3, 7)


class TaskType(enum.StrEnum):
    """Supervised learning task kinds."""

    BINARY = "binary_classification"
    MULTICLASS = "multiclass_classification"
    REGRESSION = "regression"
    SURVIVAL = "survival"
    ORDINAL = "ordinal"
    RANKING = "ranking"
    MULTILABEL = "multilabel"
    MULTIOUTPUT = "multioutput"
    FORECAST = "forecast"
    UPLIFT = "uplift"

    @property
    def is_classification(self) -> bool:
        """Whether the task predicts discrete classes."""
        return self in (TaskType.BINARY, TaskType.MULTICLASS)


FAIRNESS_WEIGHT = "fairness_weight"
"""Weight column the reweighing mitigation adds to development rows."""

STANDARD_TASKS = frozenset(
    {TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION}
)
"""Tasks scikit-learn style estimators handle; survival needs its own models."""

TaskRequest = Literal[
    "auto",
    "classification",
    "regression",
    "survival",
    "ordinal",
    "ranking",
    "multilabel",
    "multioutput",
    "forecast",
    "uplift",
    "binary_classification",
    "multiclass_classification",
]
"""Public task requests; `classification` is resolved from the target."""

DatasetModality = Literal[
    "tabular",
    "text",
    "document",
    "image",
    "audio",
    "video",
    "graph",
    "geospatial",
    "raster",
]
"""Dataset families accepted by the configuration contract.

The modality is an explicit declaration, not an inference from file suffixes.
Each declared modality must still cross the capability registry before a run
can train a model.
"""


def horizon_names(value: str, horizon: int) -> list[str]:
    """Returns the future-value target names `<value>__h1 .. <value>__h<H>`.

    Args:
        value: The series value column.
        horizon: The number of future steps.

    Returns:
        One name for each step from 1 to `horizon`, in step order.
    """
    return [f"{value}{HORIZON_SEPARATOR}{k}" for k in range(1, horizon + 1)]


def series_node(
    name: str, op: str, source: str, **params: Any
) -> dict[str, Any]:
    """Returns a feature-node record over one source column.

    Args:
        name: The node's name.
        op: The operation.
        source: The source column.
        **params: The operation's parameters.

    Returns:
        The node record, with the source as its only input.
    """
    return {"name": name, "op": op, "inputs": [source], "params": params}


def forecast_nodes(record: dict[str, Any], value: str) -> list[dict[str, Any]]:
    """Returns lag, rolling and calendar nodes for a forecasting data record.

    Args:
        record: The data record (a dict of `DataSpec` fields), giving the
            lags, windows, seasonal period, time column and calendar parts;
            an empty lags or windows list means the defaults.
        value: The series value column.

    Returns:
        Lag nodes (plus one for the seasonal period if it is not already a
        lag), a rolling mean and a rolling standard deviation node for each
        window, and one date-part node per calendar part when a time column
        is set.
    """
    period = record.get("seasonal_period")
    lags = list(record.get("forecast_lags") or DEFAULT_LAGS)
    if period and period not in lags:
        lags.append(int(period))
    windows = list(record.get("forecast_windows") or DEFAULT_WINDOWS)
    nodes = [series_node(f"{value}__lag{k}", "lag", value, k=k) for k in lags]
    for w in windows:
        nodes.append(
            series_node(f"{value}__mean{w}", "rolling_mean", value, window=w)
        )
        nodes.append(
            series_node(f"{value}__std{w}", "rolling_std", value, window=w)
        )
    clock = record.get("time_column")
    if clock:
        nodes += [
            series_node(f"{clock}__{part}", "datepart", clock, part=part)
            for part in record.get("forecast_calendar") or []
        ]
    return nodes


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

        Returns:
            The specification, unchanged.

        Raises:
            ValueError: On duplicate names, a node without inputs, or a node
                using a later node or itself.
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
        """Returns a content hash identifying this exact graph.

        Returns:
            The first 16 hex characters of a SHA-256 over the version and the
            nodes.
        """
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
        modality: Dataset family declared by the caller. Only `tabular` has
            a supported execution path; every declared modality must still
            pass the capability registry before a run can train a model.
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
        ordinal_order: For ordinal tasks, the target's labels from lowest to
            highest (for example `["low", "medium", "high"]`). Every label in
            the data must be listed and every listed label must occur.
        treatment_column: For uplift tasks, the 0/1 column saying whether a
            row received the treatment; the target is the outcome. Never a
            feature; not needed at prediction time.
        memory_budget_gb: Largest in-memory size of the dataset. A file
            bigger than this is read in batches: the locked rows are chosen
            by hashing row positions, development uses a bounded random
            sample, and the locked rows are scored by streaming them all.
            Needs a Parquet or CSV file and random or stratified
            validation.
        training: With a streamed file, `sample` trains on the development
            sample; `stream` also passes the whole development partition
            through the final model in batches (incremental models only).
        horizon: For forecasting, how many steps ahead to predict. The
            target column is then the series' value, and the framework adds
            one target per step ahead (`<value>__h1 .. <value>__h<horizon>`)
            and lag / rolling features of the value.
        value_column: The series' value column (set from `target`).
        seasonal_period: Rows per season (for example 7 for daily data with
            a weekly cycle); adds a seasonal lag and a seasonal baseline.
        forecast_lags: Lags (in rows) of the value to use as features;
            defaults to 1, 2, 3 and 7.
        forecast_windows: Rolling-window sizes for mean and spread features;
            defaults to 3 and 7.
        forecast_calendar: Date parts of the time column to add as features
            (for example `["dayofweek", "month"]`; needs a datetime column).
        targets: For multi-label, multi-output and forecast tasks, every
            target column (at least two distinct, the first equal to
            `target`). Binary 0/1 columns for `multilabel`, numeric ones for
            `multioutput`; filled in for `forecast` from `horizon`.
        text_encoding: How free-text columns become features: `tfidf`
            (default) or `embedding` (a pretrained sentence-embedding model
            run in the PyTorch worker; falls back to TF-IDF if unavailable).
        embedding_model: Sentence-embedding model for `text_encoding`.
        query_column: For ranking tasks, the column identifying the query
            (or session) each row belongs to; the target is the relevance
            grade. Never a feature; not needed at prediction time.
        event_column: For survival tasks, the 0/1 column saying whether the
            event was observed (1) or the row was censored (0); the target
            is then the time to the event or to censoring. Never a feature.
        reference_uri: A file with a recent sample of the population the model
            will serve (same feature columns). The development rows are
            compared with it and a gap is reported as a data-quality finding.
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
        baseline_column: With a charter that has a cost model and a binary
            task, the 0/1 decisions of the current process. Never a feature.
            A rule is frozen on the exploration rows and the claim that it
            beats the current process is registered in the ledger and tested
            once on the confirmation set (`baseline_report.json`).
        baseline_subject: What the baseline claim is about. `frozen_rule`
            (default) is the logistic rule frozen on the exploration rows;
            `selected_model` defers the opening of the confirmation set until
            the model selection is frozen and tests that model instead.
        ingest_plan: Path of an ingest plan (JSON) that names a database and
            the as-of views to synthesize features from. Needs a charter with
            an `ingest` block (charter format 7 or later); the table then
            needs the ingest plan's key and moment columns. The synthesized
            columns are added
            before the split.
        logged_action_column: With a charter decision block, the 0/1 action
            the current process logged. Never a feature.
        propensity_column: With a charter decision block and
            `logged_action_column`, the recorded probability that action 1
            was taken. Never a feature. It lets
            the run estimate the frozen rule's off-policy value on the
            locked rows (`policy_value.json`); without it that file says
            `not_assessable`.
        framing_draft: Path of the `framing_draft.json` that `scmle frame`
            wrote for the charter. A claim that depends on an LLM-filled field
            of an unsigned charter stays `unverified`.
    """

    uri: str
    target: str
    modality: DatasetModality = "tabular"
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
    event_column: str | None = None
    memory_budget_gb: float | None = pydantic.Field(default=None, gt=0)
    training: Literal["sample", "stream"] = "sample"
    treatment_column: str | None = None
    horizon: int | None = pydantic.Field(default=None, ge=1)
    value_column: str | None = None
    seasonal_period: int | None = pydantic.Field(default=None, ge=2)
    forecast_lags: list[int] = pydantic.Field(default_factory=list)
    forecast_windows: list[int] = pydantic.Field(default_factory=list)
    forecast_calendar: list[str] = pydantic.Field(default_factory=list)
    targets: list[str] = pydantic.Field(default_factory=list)
    text_encoding: Literal["tfidf", "embedding"] = "tfidf"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    query_column: str | None = None
    ordinal_order: list[str | int] = pydantic.Field(default_factory=list)
    reference_uri: str | None = None
    unlabeled: Literal["block", "ignore", "pseudo_label"] = "block"
    pseudo_label_confidence: float = pydantic.Field(default=0.9, gt=0.5, lt=1.0)
    baseline_column: str | None = None
    baseline_subject: Literal["frozen_rule", "selected_model"] = "frozen_rule"
    framing_draft: str | None = None
    propensity_column: str | None = None
    logged_action_column: str | None = None
    ingest_plan: str | None = None


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
        "auto",
        "stratified",
        "stratified_events",
        "multilabel_stratified",
        "treatment_stratified",
        "random",
        "group",
        "temporal",
        "rolling",
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
    cluster_address: str | None = None
    gpu: Literal["auto", "disabled", "required"] = "auto"
    enforcement: Literal["hard", "cooperative"] = "hard"


class SearchSpec(FrozenModel):
    """Hyperparameter search settings (disabled when `top_models` is 0)."""

    top_models: int = pydantic.Field(default=2, ge=0)
    configs_per_model: int = pydantic.Field(default=6, ge=1)
    optimizer: Literal["random", "grid", "optuna"] = "random"
    prune_fraction: float = pydantic.Field(default=0.5, ge=0, lt=1)
    pruning: Literal["halving", "hyperband"] = "halving"
    eta: int = pydantic.Field(default=3, ge=2)


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
        bias_variance_replicates: Bootstrap refits of the selected model used
            to split its error into bias and variance (0 turns it off;
            regression and classification only).
        sensitivity_repeats: Repeats for the sensitivity report on the selected
            model: spread of the score over training seeds, bootstrap
            resamples and splits, and its decay under added input noise
            (0 turns it off; regression and classification only).
        pareto: Measure the top candidates' size and latency, report the
            score / latency / size Pareto front, and pick, among candidates
            within `pareto_tolerance` of the best score, the fastest.
        pareto_tolerance: Score gap (in the metric's units) that is accepted
            for a faster or smaller model.
        nested_cv: Also estimate how optimistic the winning score is by
            repeating the configuration search inside each outer fold (random
            and stratified validation only; costs several extra fits).
        weighted: With a `data.weight_column`, score validation folds with
            the weights so selection agrees with the weighted test metrics
            (metrics that cannot take weights stay unweighted).
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
    weighted: bool = True
    nested_cv: bool = False
    bias_variance_replicates: int = pydantic.Field(default=0, ge=0)
    sensitivity_repeats: int = pydantic.Field(default=0, ge=0)
    pareto: bool = False
    pareto_tolerance: float = pydantic.Field(default=0.0, ge=0.0)


class ModelSpec(FrozenModel):
    """Which model families may be considered.

    Attributes:
        exclude_models: Model names that must not run.
        exclude_families: Model families that must not run (for example
            `deep_learning`, `boosting`).
        experimental_models: Explicitly opt-in model names that are not part
            of automatic beta search. Their experimental capability status is
            still preserved in the execution plan and release evidence. Built-in
            experimental names currently include torch_ft_transformer and
            torch_jepa; both require the deeplearning extra and remain
            tabular-only.
    """

    exclude_models: list[str] = pydantic.Field(default_factory=list)
    exclude_families: list[str] = pydantic.Field(default_factory=list)
    experimental_models: list[str] = pydantic.Field(default_factory=list)


class DecisionSpec(FrozenModel):
    """How binary probabilities become decisions (threshold optimization)."""

    threshold_objective: Literal[
        "none", "f1", "accuracy", "balanced_accuracy", "cost"
    ] = "none"
    false_positive_cost: float = pydantic.Field(default=1.0, gt=0)
    false_negative_cost: float = pydantic.Field(default=1.0, gt=0)


class FairnessSpec(FrozenModel):
    """Optional reduction of outcome gaps between groups (binary tasks).

    Attributes:
        mitigation: `none` only reports gaps; `group_thresholds` gives each
            group of a sensitive column its own probability threshold when
            that closes the gap without costing too much accuracy. The
            decision and its evidence are always recorded. `reweighing`
            instead weights the development rows (never the locked ones) so
            group and label are independent in training, for binary and
            multiclass tasks; it cannot be combined with
            `data.weight_column`.
        criterion: Gap to close: `demographic_parity` (equal selection
            rates) or `equal_opportunity` (equal true-positive rates).
        column: Sensitive column to mitigate on; defaults to the first of
            `data.sensitive_columns`.
        trigger_gap: Smallest gap worth acting on.
        min_reduction: Share of the gap the thresholds must remove.
        max_accuracy_cost: Most accuracy that may be given up.
    """

    mitigation: Literal["none", "group_thresholds", "reweighing"] = "none"
    criterion: Literal["demographic_parity", "equal_opportunity"] = (
        "demographic_parity"
    )
    column: str | None = None
    trigger_gap: float = pydantic.Field(default=0.05, ge=0.0, lt=1.0)
    min_reduction: float = pydantic.Field(default=0.5, gt=0.0, le=1.0)
    max_accuracy_cost: float = pydantic.Field(default=0.03, ge=0.0, lt=1.0)


class ProblemSpec(FrozenModel):
    """What the model is for, so the result can be judged against it.

    Attributes:
        decision: The decision the predictions will inform.
        population: Who or what the predictions are about.
        success_criteria: Metric name to the score that must be reached on
            the locked rows (a floor for higher-is-better metrics, a
            ceiling for lower-is-better ones).
    """

    decision: str = ""
    population: str = ""
    success_criteria: dict[str, float] = pydantic.Field(default_factory=dict)


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
    fairness: FairnessSpec = pydantic.Field(default_factory=FairnessSpec)
    problem: ProblemSpec = pydantic.Field(default_factory=ProblemSpec)
    leakage_policy: Literal["warn", "block"] = "warn"
    # Path of an analysis charter. When set, the run splits the development
    # rows into exploration and sealed confirmation sets.
    charter_path: str | None = None
    # Choices for the claim ledger of a charter run. `ledger_rule`
    # is the online FDR rule for exploration: e-LOND is valid under arbitrary
    # dependence; LORD++ and SAFFRON assume independent p-values.
    # `confirmation_rule` adjusts the survivors tested once on `C`.
    ledger_rule: Literal["e-lond", "lord++", "saffron"] = "e-lond"
    confirmation_rule: Literal["bh", "by", "holm", "bonferroni"] = "by"
    confirmation_failure_policy: Literal["unconfirmed"] = "unconfirmed"
    # JSON file of per-column attestations that a strong predictor flagged by
    # the leakage screen is legitimate (charter runs only; see
    # `scmle.data.admissibility.attestations`).
    leakage_attestations: str | None = None

    @pydantic.model_validator(mode="before")
    @classmethod
    def expand_feature_delays(cls, values: Any) -> Any:
        """Turns `data.feature_delays` into lagged nodes and exclusions.

        The expansion is idempotent, so a saved configuration (which already
        contains the nodes) loads unchanged.

        Args:
            values: The raw specification.

        Returns:
            The specification with a lagged node for each delayed column and
            those columns excluded; unchanged if it is not a dict or has no
            delays.
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

    @pydantic.model_validator(mode="before")
    @classmethod
    def expand_forecast(cls, values: Any) -> Any:
        """Turns a forecasting request into targets and series features.

        `data.target` is the series value; the expansion points the target at
        the first future-value column, lists them all as targets and adds the
        lag / rolling / calendar feature nodes and records the value column.
        It is idempotent, so a saved configuration loads unchanged.

        Args:
            values: The raw specification.

        Returns:
            The expanded specification; unchanged if the task is not
            `forecast`, no horizon is set or `value_column` is already set
            (already expanded).
        """
        if not isinstance(values, dict) or values.get("task") != "forecast":
            return values
        data = values.get("data")
        record = (
            data.model_dump()
            if isinstance(data, DataSpec)
            else dict(data or {})
        )
        if record.get("value_column") or not record.get("horizon"):
            return values
        value, horizon = record["target"], int(record["horizon"])
        names = horizon_names(value, horizon)
        graph = values.get("feature_graph")
        graph_record = (
            graph.model_dump()
            if isinstance(graph, FeatureGraphSpec)
            else dict(graph or {})
        )
        known = list(graph_record.get("nodes", []))
        present = {node["name"] for node in known}
        added = [
            node
            for node in forecast_nodes(record, value)
            if node["name"] not in present
        ]
        return {
            **values,
            "data": {
                **record,
                "value_column": value,
                "target": names[0],
                "targets": names,
            },
            "feature_graph": {**graph_record, "nodes": known + added},
        }

    @pydantic.model_validator(mode="after")
    def check_cross_field_rules(self) -> Self:
        """Validates constraints that span several fields.

        Returns:
            The specification, unchanged.

        Raises:
            ValueError: If the combination of fields is inconsistent, for
                example a validation strategy without the column it needs, or
                a task without (or, for other tasks, with) its task-specific
                data columns.
        """
        data, validation = self.data, self.validation
        if validation.strategy == "group" and not (
            data.group_column or data.query_column
        ):
            raise ValueError(
                "strategy 'group' requires data.group_column (or "
                "data.query_column for ranking)"
            )
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
        if validation.strategy == "stratified" and self.task in (
            "regression",
            "survival",
            "ordinal",
            "ranking",
            "multilabel",
            "multioutput",
            "forecast",
            "uplift",
        ):
            raise ValueError(
                f"stratified validation is undefined for {self.task}"
            )
        if (self.task == "survival") != (data.event_column is not None):
            raise ValueError(
                "survival tasks need data.event_column, and only survival "
                "tasks may set it"
            )
        if (self.task == "ordinal") != bool(data.ordinal_order):
            raise ValueError(
                "ordinal tasks need data.ordinal_order, and only ordinal "
                "tasks may set it"
            )
        if len(set(data.ordinal_order)) != len(data.ordinal_order):
            raise ValueError("data.ordinal_order must not repeat a label")
        if self.task == "ordinal" and len(data.ordinal_order) < 3:
            raise ValueError("ordinal tasks need at least three levels")
        if data.memory_budget_gb is not None and (
            data.time_column
            or data.group_column
            or data.query_column
            or data.horizon
            or data.feature_delays
            or self.feature_graph.nodes
            or data.unlabeled != "block"
        ):
            raise ValueError(
                "streaming a file larger than memory supports random or "
                "stratified validation only (no time, group, query, "
                "feature graph, forecast or unlabeled-row options)"
            )
        if data.training == "stream" and data.memory_budget_gb is None:
            raise ValueError("training='stream' needs data.memory_budget_gb")
        if (
            self.fairness.mitigation == "reweighing"
            and data.weight_column not in (None, FAIRNESS_WEIGHT)
        ):
            raise ValueError(
                "fairness reweighing cannot be combined with data.weight_column"
            )
        if self.fairness.mitigation != "none":
            chosen = self.fairness.column or next(
                iter(data.sensitive_columns), None
            )
            if chosen is None or chosen not in data.sensitive_columns:
                raise ValueError(
                    "fairness mitigation needs a column listed in "
                    "data.sensitive_columns"
                )
        if self.task == "forecast":
            if not (data.horizon and data.time_column):
                raise ValueError(
                    "forecast tasks need data.horizon and data.time_column"
                )
        elif data.horizon is not None:
            raise ValueError("data.horizon applies to forecast tasks only")
        multi = self.task in ("multilabel", "multioutput", "forecast")
        if multi != bool(data.targets):
            raise ValueError(
                "multilabel and multioutput tasks need data.targets, and "
                "only they may set it"
            )
        if multi:
            if len(set(data.targets)) != len(data.targets) or (
                len(data.targets) < 2
            ):
                raise ValueError(
                    "data.targets needs at least two distinct columns"
                )
            if data.targets[0] != data.target:
                raise ValueError("data.targets must start with data.target")
        if (self.task == "uplift") != (data.treatment_column is not None):
            raise ValueError(
                "uplift tasks need data.treatment_column, and only uplift "
                "tasks may set it"
            )
        if (self.task == "ranking") != (data.query_column is not None):
            raise ValueError(
                "ranking tasks need data.query_column, and only ranking "
                "tasks may set it"
            )
        if data.event_column == data.target:
            raise ValueError("data.event_column must differ from the target")
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
        """Returns the search settings in force, reduced under a short budget.

        Returns:
            The configured search settings; when the time budget is under 300
            seconds, `top_models` is capped at 1 and `configs_per_model` at 3.
        """
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
        """Returns the strategy settings in force, reduced under a short budget.

        Returns:
            The configured strategy settings; when the time budget is under
            300 seconds, `top_models` is capped at 1.
        """
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
