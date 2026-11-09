"""Candidate pipelines: preprocessing and estimator as one unit.

Everything is a single scikit-learn `Pipeline`, so cross-validation clones
and refits the whole thing per fold (fold-local fitting) and inference reuses
the identical fitted graph.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any

from sklearn import compose
from sklearn import pipeline

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.data.curation import selection
from scmle.data.validation import columns
from scmle.features import feature_graph
from scmle.models import weighting
from scmle.models import wrappers
from scmle.models.forecasting import estimators
import scmle.models.multitarget.wrapper
import scmle.models.ranking.wrapper
import scmle.models.survival.wrapper
import scmle.models.uplift.wrapper

SPARSE_THRESHOLD = 0.3


def build_preprocessor(
    plan: data.FeaturePlan,
    options: data.PreprocessingOptions,
) -> compose.ColumnTransformer:
    """Builds the column transformer for a feature plan.

    Args:
        plan: Feature columns grouped by kind.
        options: Preprocessing switches.

    Returns:
        An unfitted transformer that drops every non-feature column.
    """
    parts = []
    special = split_high_cardinality(plan, options)
    if special:
        encoder = columns.CategoricalKind().high_cardinality_transformer(
            options.high_cardinality
        )
        parts.append(("high_cardinality", encoder, list(special)))
    for kind in columns.KINDS:
        selected = [c for c in plan.columns_for(kind.name) if c not in special]
        if not selected:
            continue
        kind_options = options
        if kind.name == "numeric" and options.missing == "indicator":
            flagged, selected = split_flagged(plan, selected)
            if flagged:
                parts.append(
                    ("numeric_flagged", kind.transformer(options), flagged)
                )
            kind_options = dataclasses.replace(options, missing="median")
        transformer = kind.transformer(kind_options)
        if selected and transformer is not None:
            parts.append((kind.name, transformer, selected))
    if options.construct:
        parts.append(
            (
                "constructed",
                columns.NumericKind().transformer(options),
                compose.make_column_selector(pattern=f"^{selection.PREFIX}"),
            )
        )
    return compose.ColumnTransformer(
        parts,
        remainder="drop",
        verbose_feature_names_out=False,
        sparse_threshold=SPARSE_THRESHOLD if options.sparse else 0.0,
    )


def split_flagged(
    plan: data.FeaturePlan, selected: list[str]
) -> tuple[list[str], list[str]]:
    """Numeric columns whose gaps are informative, and the rest.

    Without a missingness report every column gets indicators.

    Args:
        plan: Feature plan; its `informative_missing` lists the flagged
            columns, or is None when no missingness report exists.
        selected: The numeric columns to split.

    Returns:
        A pair `(flagged, rest)`: the columns whose gaps are informative
        (all of them if there is no report) and the remaining columns, each
        in the order of `selected`.
    """
    if plan.informative_missing is None:
        return list(selected), []
    flagged = [c for c in selected if c in plan.informative_missing]
    return flagged, [c for c in selected if c not in flagged]


def monotonic_signs(plan: data.FeaturePlan, offset: int) -> tuple[int, ...]:
    """Monotonic signs for the leading columns of the encoded matrix.

    The encoded matrix starts with the specially encoded high-cardinality
    columns (`offset` of them, one column each), then the numeric features
    in plan order; later columns are unconstrained. The function only builds
    the sign vector from that assumed layout and does not inspect the fitted
    matrix, so it does not follow a reordering of the numeric columns or the
    removal of columns after encoding.

    Args:
        plan: Feature plan with the numeric columns and their monotonic
            signs.
        offset: Number of specially encoded high-cardinality columns.

    Returns:
        `offset` zeros followed by the sign of each numeric column (+1, -1
        or 0 where none is declared).
    """
    return (0,) * offset + tuple(
        plan.monotonic.get(column, 0) for column in plan.numeric
    )


def monotonic_layout_problem(
    plan: data.FeaturePlan, select: bool, missing: str
) -> str | None:
    """Why the monotonic signs cannot follow the encoded numeric columns.

    Args:
        plan: Feature plan with the numeric columns and their monotonic
            signs.
        select: Whether the sparse column selector follows the encoder.
        missing: The numeric missing-value strategy.

    Returns:
        A reason when the encoded matrix would not hold the numeric columns
        in plan order, or None when the sign vector lines up.
    """
    if not plan.monotonic:
        return None
    if select:
        return "feature selection drops columns the signs are indexed by"
    if missing == "indicator":
        flagged, rest = split_flagged(plan, list(plan.numeric))
        if flagged and rest:
            return (
                "missing indicators move the flagged numeric columns ahead "
                "of the others"
            )
    return None


def split_high_cardinality(
    plan: data.FeaturePlan,
    options: data.PreprocessingOptions,
) -> tuple[str, ...]:
    """Columns that get a special encoder instead of one-hot encoding.

    They are placed first in the transformed matrix, so their positions are
    `0..len-1`, which native-categorical estimators can be told about.

    Args:
        plan: Feature plan with the categorical and high-cardinality columns.
        options: Preprocessing switches; with `onehot` encoding no column
            is special.

    Returns:
        The high-cardinality categorical columns, in plan order; empty with
        `onehot` encoding.
    """
    if options.high_cardinality == "onehot":
        return ()
    return tuple(c for c in plan.categorical if c in plan.high_cardinality)


def build_front_end(
    plan: data.FeaturePlan,
    options: data.PreprocessingOptions,
) -> Any:
    """Builds the preprocessing front end, with the feature graph if any.

    Args:
        plan: Feature columns by kind (and row-wise graph nodes).
        options: Preprocessing switches.

    Returns:
        The column transformer, or a pipeline of graph transformer followed
        by the column transformer when features are derived.
    """
    preprocessor = build_preprocessor(plan, options)
    if not plan.derived:
        return preprocessor
    return pipeline.Pipeline(
        [
            ("graph", feature_graph.FeatureGraphTransformer(plan.derived)),
            ("prep", preprocessor),
        ]
    )


def default_options(
    plan: data.FeaturePlan, adapter: models.ModelAdapter
) -> data.PreprocessingOptions:
    """Preprocessing switches implied by an adapter's capabilities.

    Scaling for models that benefit from it, no imputation for models that
    handle missing values, sparse one-hot output for sparse-capable models on
    high-cardinality data, and integer-coded native categoricals where the
    adapter supports them.

    Args:
        plan: Feature plan, used for its high-cardinality columns and text
            settings.
        adapter: Model family whose capabilities decide the switches.

    Returns:
        The preprocessing options for the adapter.
    """
    caps = adapter.capabilities
    native = bool(
        plan.high_cardinality
        and caps.native_categorical
        and adapter.categorical_params([0])
    )
    return data.PreprocessingOptions(
        scale=caps.needs_scaling,
        impute=not caps.handles_missing,
        high_cardinality="ordinal" if native else "onehot",
        sparse=caps.sparse_input and bool(plan.high_cardinality),
        text_encoding=plan.text_encoding,
        embedding_model=plan.embedding_model,
    )


def build_pipeline(
    plan: data.FeaturePlan,
    adapter: models.ModelAdapter,
    task: tasks.TaskType,
    seed: int,
    params: Mapping[str, Any] | None = None,
    options: data.PreprocessingOptions | None = None,
) -> Any:
    """Builds a complete, independently evaluable candidate pipeline.

    Args:
        plan: Feature columns grouped by kind.
        adapter: Model family to place after preprocessing.
        task: Resolved task type.
        seed: Random seed forwarded to the estimator.
        params: Hyperparameter overrides for the estimator.
        options: Preprocessing switches; by default `default_options`.

    Returns:
        An unfitted pipeline of preprocessing followed by the estimator.
    """
    if plan.ordinal_levels:
        params = {**(params or {}), "levels": plan.ordinal_levels}
    if plan.treatment_column:
        params = {**(params or {}), "classify": plan.binary_outcome}
    assembled: Any
    if task is tasks.TaskType.FORECAST and adapter.name == "dummy":
        assembled = estimators.LastValueForecaster(
            None, plan.target_columns, plan.value_column or ""
        )
    else:
        assembled = adapter.assemble(plan, task, seed, params)
    if assembled is not None:
        if plan.weight_column:
            return weighting.weighted(assembled, plan.weight_column, seed)
        return assembled
    params = dict(params or {})
    if options is None:
        options = default_options(plan, adapter)
    special = split_high_cardinality(plan, options)
    if options.high_cardinality == "ordinal" and special:
        params |= adapter.categorical_params(list(range(len(special))))
    steps: list[tuple[str, Any]] = []
    if plan.derived:
        steps.append(
            ("graph", feature_graph.FeatureGraphTransformer(plan.derived))
        )
    if options.construct:
        steps.append(
            ("construct", selection.FeatureConstructor(tuple(plan.numeric)))
        )
    steps.append(("prep", build_preprocessor(plan, options)))
    if options.select:
        steps.append(
            (
                "select",
                selection.SparseSelector(
                    classification=task.is_classification, random_state=seed
                ),
            )
        )
    estimator = adapter.build(task, seed, params)
    problem = monotonic_layout_problem(plan, options.select, options.missing)
    if problem:
        raise errors.ConfigError(
            f"monotonic constraints cannot apply: {problem}"
        )
    if plan.monotonic:
        estimator = wrappers.monotone(
            estimator,
            adapter.monotonic_param or "",
            monotonic_signs(plan, len(special)),
        )
    steps.append(("model", estimator))
    built: Any = pipeline.Pipeline(steps)
    if plan.event_column:
        built = scmle.models.survival.wrapper.EventAwareEstimator(
            built, plan.event_column
        )
    if task is tasks.TaskType.FORECAST:
        built = estimators.ForecastEstimator(
            built, plan.target_columns, plan.value_column or ""
        )
    elif plan.target_columns:
        kind = (
            scmle.models.multitarget.wrapper.MultiLabelEstimator
            if task is tasks.TaskType.MULTILABEL
            else scmle.models.multitarget.wrapper.MultiOutputEstimator
        )
        built = kind(built, plan.target_columns)
    if plan.treatment_column:
        built = scmle.models.uplift.wrapper.TreatmentAwareEstimator(
            built, plan.treatment_column
        )
    if plan.query_column:
        built = scmle.models.ranking.wrapper.QueryAwareEstimator(
            built, plan.query_column
        )
    if plan.weight_column:
        return weighting.weighted(built, plan.weight_column, seed)
    return built
