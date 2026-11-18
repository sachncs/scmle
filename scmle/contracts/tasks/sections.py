"""The sections of a project configuration other than the feature graph.

Each section is a `records.Spelled` part: its fields keep their names in
code while a stored configuration carries single words. A section's
`vocabulary` maps each renamed field to its stored key, where a dotted key
(`rolling.horizon`) nests the field under a shared object; `spelling` is
the step built from it. A configuration in the old spelling still loads.
"""

from collections.abc import Mapping
from typing import Any
from typing import ClassVar
from typing import Literal

import pydantic

from scmle.contracts import records
from scmle.contracts.tasks import kinds


def spell(vocabulary: Mapping[str, str]) -> records.Migration:
    """Builds the spelling step of a section from its vocabulary.

    Args:
        vocabulary: Field name to stored key; a dotted key nests the field
            under the object named before the dot.

    Returns:
        The step from the field-name layout to the stored layout.
    """
    flat = {old: new for old, new in vocabulary.items() if "." not in new}
    groups: dict[str, dict[str, str]] = {}
    for old, new in vocabulary.items():
        if "." in new:
            name, leaf = new.split(".")
            groups.setdefault(name, {})[old] = leaf
    return records.nest(flat, groups=groups)


class Section(records.Spelled):
    """A configuration section stored with single-word keys.

    Attributes:
        vocabulary: Field name to stored key for each renamed field.
    """

    vocabulary: ClassVar[dict[str, str]] = {}

    @pydantic.model_validator(mode="before")
    @classmethod
    def migrate(cls, raw: Any) -> Any:
        """Reads either spelling; a field name wins over its stored key.

        Code that edits a dumped section writes the field name next to the
        stored key it dumped (`resources.time_budget_seconds` beside
        `budget`); the edit is the value meant, so the stored key is
        dropped before the section moves back to its field names.

        Args:
            raw: The section, keyword arguments, or a value pydantic will
                reject.

        Returns:
            The layout keyed by field name, or `raw` unchanged when it is not
            a mapping.
        """
        if not isinstance(raw, Mapping):
            return raw
        layout = dict(raw)
        for old, new in cls.vocabulary.items():
            if old not in layout:
                continue
            name, dot, leaf = new.partition(".")
            if not dot:
                layout.pop(name, None)
            elif isinstance(layout.get(name), Mapping):
                group = {k: v for k, v in layout[name].items() if k != leaf}
                layout[name] = group
                if not group:
                    layout.pop(name)
        return super().migrate(layout)  # type: ignore[operator]


class Dataset(Section):
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

    vocabulary: ClassVar[dict[str, str]] = {
        "id_columns": "identifiers",
        "group_column": "group",
        "time_column": "time",
        "time_as_feature": "temporal",
        "excluded_columns": "excluded",
        "unavailable_at_prediction": "unavailable",
        "feature_delays": "delays",
        "sensitive_columns": "sensitive",
        "monotonic_constraints": "monotonic",
        "weight_column": "weight",
        "event_column": "event",
        "memory_budget_gb": "memory",
        "treatment_column": "treatment",
        "value_column": "value",
        "seasonal_period": "period",
        "forecast_lags": "forecast.lags",
        "forecast_windows": "forecast.windows",
        "forecast_calendar": "forecast.calendar",
        "text_encoding": "encoding",
        "embedding_model": "embedder",
        "query_column": "query",
        "ordinal_order": "order",
        "reference_uri": "reference",
        "pseudo_label_confidence": "confidence",
        "baseline_column": "baseline",
        "baseline_subject": "comparator",
        "framing_draft": "draft",
        "propensity_column": "propensity",
        "logged_action_column": "action",
        "ingest_plan": "ingest",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

    uri: str
    target: str
    modality: kinds.DatasetModality = "tabular"
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


class Validation(Section):
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

    vocabulary: ClassVar[dict[str, str]] = {
        "test_fraction": "test",
        "embargo_rows": "embargo",
        "horizon_rows": "rolling.horizon",
        "window_rows": "rolling.window",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

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


class Resources(Section):
    """Compute limits for a run (time in seconds, memory in GB)."""

    vocabulary: ClassVar[dict[str, str]] = {
        "time_budget_seconds": "budget",
        "memory_limit_gb": "limits.memory",
        "cpu_limit": "limits.cpu",
        "cluster_address": "cluster",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

    time_budget_seconds: int = pydantic.Field(default=1800, gt=0)
    memory_limit_gb: float = pydantic.Field(default=8.0, gt=0)
    cpu_limit: int | None = pydantic.Field(default=None, ge=1)
    workers: int = pydantic.Field(default=1, ge=1)
    executor: Literal["process", "ray"] = "process"
    cluster_address: str | None = None
    gpu: Literal["auto", "disabled", "required"] = "auto"
    enforcement: Literal["hard", "cooperative"] = "hard"


class Search(Section):
    """Hyperparameter search settings (disabled when `top_models` is 0)."""

    vocabulary: ClassVar[dict[str, str]] = {
        "top_models": "leaders",
        "configs_per_model": "configurations",
        "prune_fraction": "discard",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

    top_models: int = pydantic.Field(default=2, ge=0)
    configs_per_model: int = pydantic.Field(default=6, ge=1)
    optimizer: Literal["random", "grid", "optuna"] = "random"
    prune_fraction: float = pydantic.Field(default=0.5, ge=0, lt=1)
    pruning: Literal["halving", "hyperband"] = "halving"
    eta: int = pydantic.Field(default=3, ge=2)


class Strategies(Section):
    """Which data strategies to evaluate (imbalance and feature variants)."""

    vocabulary: ClassVar[dict[str, str]] = {"top_models": "leaders"}
    spelling: ClassVar[records.Migration] = spell(vocabulary)

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


class Stacking(records.Schema):
    """Stacking ensemble settings (disabled when `size` < 2)."""

    size: int = pydantic.Field(default=3, ge=0)


class Selection(Section):
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

    vocabulary: ClassVar[dict[str, str]] = {
        "max_model_mb": "size",
        "max_p95_latency_ms": "latency",
        "latency_safety_factor": "headroom",
        "max_candidates_checked": "depth",
        "tie_breaker": "tiebreaker",
        "tie_tolerance": "tolerance",
        "bayesian_confidence": "credibility",
        "nested_cv": "nested",
        "bias_variance_replicates": "replicates",
        "sensitivity_repeats": "repeats",
        "pareto_tolerance": "slack",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

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


class Candidates(Section):
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

    vocabulary: ClassVar[dict[str, str]] = {
        "exclude_models": "exclude.models",
        "exclude_families": "exclude.families",
        "experimental_models": "experimental",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

    exclude_models: list[str] = pydantic.Field(default_factory=list)
    exclude_families: list[str] = pydantic.Field(default_factory=list)
    experimental_models: list[str] = pydantic.Field(default_factory=list)


class Threshold(Section):
    """How binary probabilities become decisions (threshold optimization)."""

    vocabulary: ClassVar[dict[str, str]] = {
        "threshold_objective": "objective",
        "false_positive_cost": "costs.fp",
        "false_negative_cost": "costs.fn",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

    threshold_objective: Literal[
        "none", "f1", "accuracy", "balanced_accuracy", "cost"
    ] = "none"
    false_positive_cost: float = pydantic.Field(default=1.0, gt=0)
    false_negative_cost: float = pydantic.Field(default=1.0, gt=0)


class Parity(Section):
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

    vocabulary: ClassVar[dict[str, str]] = {
        "trigger_gap": "trigger",
        "min_reduction": "reduction",
        "max_accuracy_cost": "cost",
    }
    spelling: ClassVar[records.Migration] = spell(vocabulary)

    mitigation: Literal["none", "group_thresholds", "reweighing"] = "none"
    criterion: Literal["demographic_parity", "equal_opportunity"] = (
        "demographic_parity"
    )
    column: str | None = None
    trigger_gap: float = pydantic.Field(default=0.05, ge=0.0, lt=1.0)
    min_reduction: float = pydantic.Field(default=0.5, gt=0.0, le=1.0)
    max_accuracy_cost: float = pydantic.Field(default=0.03, ge=0.0, lt=1.0)


class Problem(Section):
    """What the model is for, so the result can be judged against it.

    Attributes:
        decision: The decision the predictions will inform.
        population: Who or what the predictions are about.
        success_criteria: Metric name to the score that must be reached on
            the locked rows (a floor for higher-is-better metrics, a
            ceiling for lower-is-better ones).
    """

    vocabulary: ClassVar[dict[str, str]] = {"success_criteria": "criteria"}
    spelling: ClassVar[records.Migration] = spell(vocabulary)

    decision: str = ""
    population: str = ""
    success_criteria: dict[str, float] = pydantic.Field(default_factory=dict)
