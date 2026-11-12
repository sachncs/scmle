"""Candidate pipelines: preprocessing and estimator as one unit.

Everything is a single scikit-learn `Pipeline`, so cross-validation clones
and refits the whole thing per fold (fold-local fitting) and inference reuses
the identical fitted graph.
"""

from collections.abc import Mapping
import dataclasses
from typing import Any
from typing import cast

from sklearn import base
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
    options: data.Preprocessing,
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
    options: data.Preprocessing,
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
    options: data.Preprocessing,
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
    plan: data.FeaturePlan, adapter: models.Model
) -> data.Preprocessing:
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
    return data.Preprocessing(
        scale=caps.needs_scaling,
        impute=not caps.handles_missing,
        high_cardinality="ordinal" if native else "onehot",
        sparse=caps.sparse_input and bool(plan.high_cardinality),
        text_encoding=plan.text_encoding,
        embedding_model=plan.embedding_model,
    )


def adapter_params(
    plan: data.FeaturePlan, params: Mapping[str, Any] | None
) -> Mapping[str, Any] | None:
    """Adds the plan's ordinal levels and uplift flag to the overrides.

    Args:
        plan: Feature columns grouped by kind.
        params: Hyperparameter overrides for the estimator.

    Returns:
        `params` with `levels` set when the plan has ordinal levels and
        `classify` set when the plan has a treatment column; the argument
        itself when neither applies.
    """
    if plan.ordinal_levels:
        params = {**(params or {}), "levels": plan.ordinal_levels}
    if plan.treatment_column:
        params = {**(params or {}), "classify": plan.binary_outcome}
    return params


def adapter_assembly(
    plan: data.FeaturePlan,
    adapter: models.Model,
    task: tasks.Task,
    seed: int,
    params: Mapping[str, Any] | None,
) -> Any:
    """Asks the model family for a pipeline it assembles itself.

    Args:
        plan: Feature columns grouped by kind.
        adapter: Model family.
        task: Resolved task type.
        seed: Random seed forwarded to the estimator.
        params: Hyperparameter overrides for the estimator.

    Returns:
        The assembled pipeline, or None when the family leaves the
        preprocessing to this module. The dummy family on a forecast task
        is a last-value forecaster.
    """
    if task is tasks.Task.FORECAST and adapter.name == "dummy":
        return estimators.LastValueForecaster(
            None, plan.target_columns, plan.value_column or ""
        )
    return adapter.assemble(plan, task, seed, params)


def with_weights(model: Any, plan: data.FeaturePlan, seed: int) -> Any:
    """Wraps a model so that it reads the plan's weight column.

    Args:
        model: The pipeline or estimator.
        plan: Feature columns grouped by kind.
        seed: Random seed forwarded to the wrapper.

    Returns:
        The weighted model, or `model` itself when the plan has no weight
        column.
    """
    if plan.weight_column:
        return weighting.weighted(model, plan.weight_column, seed)
    return model


def preprocessing_steps(
    plan: data.FeaturePlan,
    options: data.Preprocessing,
    task: tasks.Task,
    seed: int,
) -> list[tuple[str, Any]]:
    """Lists the steps that come before the estimator.

    Args:
        plan: Feature columns grouped by kind.
        options: Preprocessing switches.
        task: Resolved task type.
        seed: Random seed forwarded to the selector.

    Returns:
        The graph, construction, preprocessing and selection steps that the
        plan and the options call for, in that order.
    """
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
    return steps


def wrap_estimator(built: Any, plan: data.FeaturePlan, task: tasks.Task) -> Any:
    """Wraps a pipeline in the estimators the plan's columns call for.

    Args:
        built: The pipeline.
        plan: Feature columns grouped by kind.
        task: Resolved task type.

    Returns:
        The pipeline wrapped for survival events, forecasting or several
        targets, then for treatment and query columns, when the plan has
        them.
    """
    if plan.event_column:
        built = scmle.models.survival.wrapper.EventAwareEstimator(
            built, plan.event_column
        )
    if task is tasks.Task.FORECAST:
        built = estimators.ForecastEstimator(
            built, plan.target_columns, plan.value_column or ""
        )
    elif plan.target_columns:
        kind = (
            scmle.models.multitarget.wrapper.MultiLabelEstimator
            if task is tasks.Task.MULTILABEL
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
    return built


def build_pipeline(
    plan: data.FeaturePlan,
    adapter: models.Model,
    task: tasks.Task,
    seed: int,
    params: Mapping[str, Any] | None = None,
    options: data.Preprocessing | None = None,
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

    Raises:
        ConfigError: If the plan's monotonic constraints cannot apply to the
            layout the preprocessing produces.
    """
    params = adapter_params(plan, params)
    assembled = adapter_assembly(plan, adapter, task, seed, params)
    if assembled is not None:
        return with_weights(assembled, plan, seed)
    params = dict(params or {})
    if options is None:
        options = default_options(plan, adapter)
    special = split_high_cardinality(plan, options)
    if options.high_cardinality == "ordinal" and special:
        params |= adapter.categorical_params(list(range(len(special))))
    steps = preprocessing_steps(plan, options, task, seed)
    estimator = adapter.build(task, seed, params)
    problem = monotonic_layout_problem(plan, options.select, options.missing)
    if problem:
        raise errors.ConfigError(
            f"monotonic constraints cannot apply: {problem}"
        )
    if plan.monotonic:
        kind = (
            wrappers.MonotoneClassifier
            if base.is_classifier(estimator)
            else wrappers.MonotoneRegressor
        )
        estimator = kind(
            cast("base.BaseEstimator", estimator),
            adapter.monotonic_param or "",
            monotonic_signs(plan, len(special)),
        )
    steps.append(("model", estimator))
    built = wrap_estimator(pipeline.Pipeline(steps), plan, task)
    return with_weights(built, plan, seed)
