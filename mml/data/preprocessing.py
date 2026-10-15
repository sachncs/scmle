"""Candidate pipelines: preprocessing and estimator as one unit.

Everything is a single scikit-learn `Pipeline`, so cross-validation clones
and refits the whole thing per fold (fold-local fitting) and inference reuses
the identical fitted graph.
"""

from sklearn import compose
from sklearn import pipeline

from mml.contracts import data as data_contracts
from mml.contracts import models
from mml.contracts import tasks
from mml.data import columns


def build_preprocessor(
    plan: data_contracts.FeaturePlan, scale: bool
) -> compose.ColumnTransformer:
    """Builds the column transformer for a feature plan.

    Args:
        plan: Feature columns grouped by kind.
        scale: Whether numeric features should be standardized.

    Returns:
        An unfitted transformer that drops every non-feature column.
    """
    parts = []
    for kind in columns.KINDS:
        selected = plan.columns_for(kind.name)
        transformer = kind.transformer(scale)
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
) -> pipeline.Pipeline:
    """Builds a complete, independently evaluable candidate pipeline.

    Args:
        plan: Feature columns grouped by kind.
        adapter: Model family to place after preprocessing.
        task: Resolved task type.
        seed: Random seed forwarded to the estimator.

    Returns:
        An unfitted pipeline of preprocessing followed by the estimator.
    """
    preprocessor = build_preprocessor(
        plan, scale=adapter.capabilities.needs_scaling
    )
    return pipeline.Pipeline(
        [("prep", preprocessor), ("model", adapter.build(task, seed))]
    )
