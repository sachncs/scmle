"""Candidate pipelines: preprocessing and estimator as one unit.

Everything is a single scikit-learn `Pipeline`, so cross-validation clones
and refits the whole thing per fold (fold-local fitting) and inference reuses
the identical fitted graph.
"""

from collections.abc import Mapping
from typing import Any

from sklearn import compose
from sklearn import pipeline

from mml.contracts import data as data_contracts
from mml.contracts import models
from mml.contracts import tasks
from mml.data import columns


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
    for kind in columns.KINDS:
        selected = plan.columns_for(kind.name)
        transformer = kind.transformer(options)
        if selected and transformer is not None:
            parts.append((kind.name, transformer, list(selected)))
    return compose.ColumnTransformer(
        parts, remainder="drop", verbose_feature_names_out=False
    )


def build_pipeline(
    plan: data_contracts.FeaturePlan,
    adapter: models.ModelAdapter,
    task: tasks.TaskType,
    seed: int,
    params: Mapping[str, Any] | None = None,
    options: data_contracts.PreprocessingOptions | None = None,
) -> pipeline.Pipeline:
    """Builds a complete, independently evaluable candidate pipeline.

    Args:
        plan: Feature columns grouped by kind.
        adapter: Model family to place after preprocessing.
        task: Resolved task type.
        seed: Random seed forwarded to the estimator.
        params: Hyperparameter overrides for the estimator.
        options: Preprocessing switches; by default only scaling, and only
            for adapters that benefit from it.

    Returns:
        An unfitted pipeline of preprocessing followed by the estimator.
    """
    if options is None:
        options = data_contracts.PreprocessingOptions(
            scale=adapter.capabilities.needs_scaling
        )
    preprocessor = build_preprocessor(plan, options)
    return pipeline.Pipeline(
        [("prep", preprocessor), ("model", adapter.build(task, seed, params))]
    )
