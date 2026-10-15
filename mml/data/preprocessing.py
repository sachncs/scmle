"""Candidate pipelines: preprocessing + estimator as one unit.

Everything is a single sklearn Pipeline, so cross-validation clones and refits the whole
thing per fold (fold-local fitting) and inference reuses the identical fitted graph.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mml.contracts.data import FeaturePlan
from mml.contracts.models import ModelSpec
from mml.contracts.tasks import TaskType

MISSING_TOKEN = "__missing__"


def _to_float(X):
    return pd.DataFrame(X).apply(pd.to_numeric, errors="coerce").astype(float).to_numpy()


def _to_object(X):
    return pd.DataFrame(X).astype(object).where(lambda d: d.notna(), np.nan).to_numpy()


def build_preprocessor(plan: FeaturePlan, scale: bool) -> ColumnTransformer:
    transformers = []
    if plan.numeric:
        steps = [
            ("cast", FunctionTransformer(_to_float, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ]
        if scale:
            steps.append(("scale", StandardScaler()))
        transformers.append(("num", Pipeline(steps), list(plan.numeric)))
    if plan.categorical:
        steps = [
            ("cast", FunctionTransformer(_to_object, feature_names_out="one-to-one")),
            ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_TOKEN)),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.005,
                    max_categories=50,
                    sparse_output=False,
                ),
            ),
        ]
        transformers.append(("cat", Pipeline(steps), list(plan.categorical)))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def build_pipeline(plan: FeaturePlan, model: ModelSpec, task: TaskType, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("prep", build_preprocessor(plan, scale=model.capabilities.needs_scaling)),
            ("model", model.factory(task, seed)),
        ]
    )
