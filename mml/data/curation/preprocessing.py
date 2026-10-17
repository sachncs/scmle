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

from mml.contracts import data as data_contracts
from mml.contracts import models
from mml.contracts import tasks
from mml.data.curation import selection
from mml.data.validation import columns
from mml.features import feature_graph
from mml.models import weighting
from mml.models import wrappers
from mml.models.ranking import wrapper as ranking_wrapper
from mml.models.survival import wrapper as survival_wrapper

SPARSE_THRESHOLD = 0.3


def build_preprocessor(
    plan: data_contracts.FeaturePlan,
    options: data_contracts.PreprocessingOptions,
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
    plan: data_contracts.FeaturePlan, selected: list[str]
) -> tuple[list[str], list[str]]:
    """Numeric columns whose gaps are informative, and the rest.

    Without a missingness report every column gets indicators, as before.
    """
    if plan.informative_missing is None:
        return list(selected), []
    flagged = [c for c in selected if c in plan.informative_missing]
    return flagged, [c for c in selected if c not in flagged]


def monotonic_signs(
    plan: data_contracts.FeaturePlan, offset: int
) -> tuple[int, ...]:
    """Monotonic signs for the leading columns of the encoded matrix.

    The encoded matrix starts with the specially encoded high-cardinality
    columns (`offset` of them, one column each), then the numeric features
    in plan order; later columns are unconstrained.
    """
    return (0,) * offset + tuple(
        plan.monotonic.get(column, 0) for column in plan.numeric
    )


def split_high_cardinality(
    plan: data_contracts.FeaturePlan,
    options: data_contracts.PreprocessingOptions,
) -> tuple[str, ...]:
    """Columns that get a special encoder instead of one-hot encoding.

    They are placed first in the transformed matrix, so their positions are
    `0..len-1`, which native-categorical estimators can be told about.
    """
    if options.high_cardinality == "onehot":
        return ()
    return tuple(c for c in plan.categorical if c in plan.high_cardinality)


def build_front_end(
    plan: data_contracts.FeaturePlan,
    options: data_contracts.PreprocessingOptions,
) -> Any:
    """Builds the preprocessing front end, with the feature graph if any.

    Args:
        plan: Feature columns by kind (and row-wise graph nodes).
        options: Preprocessing switches.

    Returns:
        The column transformer, or a pipeline of graph transformer followed
        by the column transformer when the plan derives features.
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
    plan: data_contracts.FeaturePlan, adapter: models.ModelAdapter
) -> data_contracts.PreprocessingOptions:
    """Preprocessing switches implied by an adapter's capabilities.

    Scaling for models that benefit from it, no imputation for models that
    handle missing values, sparse one-hot output for sparse-capable models on
    high-cardinality data, and integer-coded native categoricals where the
    adapter supports them.
    """
    caps = adapter.capabilities
    native = bool(
        plan.high_cardinality
        and caps.native_categorical
        and adapter.categorical_params([0])
    )
    return data_contracts.PreprocessingOptions(
        scale=caps.needs_scaling,
        impute=not caps.handles_missing,
        high_cardinality="ordinal" if native else "onehot",
        sparse=caps.sparse_input and bool(plan.high_cardinality),
    )


def build_pipeline(
    plan: data_contracts.FeaturePlan,
    adapter: models.ModelAdapter,
    task: tasks.TaskType,
    seed: int,
    params: Mapping[str, Any] | None = None,
    options: data_contracts.PreprocessingOptions | None = None,
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
    if plan.monotonic:
        estimator = wrappers.monotone(
            estimator,
            adapter.monotonic_param or "",
            monotonic_signs(plan, len(special)),
        )
    steps.append(("model", estimator))
    built: Any = pipeline.Pipeline(steps)
    if plan.event_column:
        built = survival_wrapper.EventAwareEstimator(built, plan.event_column)
    if plan.query_column:
        built = ranking_wrapper.QueryAwareEstimator(built, plan.query_column)
    if plan.weight_column:
        return weighting.weighted(built, plan.weight_column, seed)
    return built
