"""Independent plain-scikit-learn reference on the framework's exact split.

The reference knows nothing about magicml's planning or search: it fits a
default HistGradientBoosting model on the same development rows and scores it
on the same locked test rows, so the comparison is apples to apples.
"""

import json
import pathlib

import pandas as pd
from sklearn import compose
from sklearn import ensemble
from sklearn import pipeline
from sklearn import preprocessing

from mml.contracts import tasks
from mml.evaluation import metrics as mml_metrics


def reference_score(
    frame: pd.DataFrame,
    target: str,
    run_dir: pathlib.Path,
    task: str,
    metric_name: str,
    classes: bool,
) -> float:
    """Scores default HistGradientBoosting on the run's own split.

    Args:
        frame: The full dataset.
        target: Target column.
        run_dir: Completed run directory (for its validation manifest).
        task: Resolved task type value.
        metric_name: Metric to report (natural units).
        classes: Whether the task is classification.

    Returns:
        The reference test score for `metric_name`.
    """
    manifest = json.loads((run_dir / "validation_manifest.json").read_text())
    features = frame.drop(columns=[target])
    categorical = list(features.select_dtypes(exclude="number").columns)
    encoder = preprocessing.OrdinalEncoder(
        handle_unknown="use_encoded_value", unknown_value=-1
    )
    prep = pipeline.make_pipeline(
        preprocessing.FunctionTransformer(lambda x: x.astype(str)), encoder
    )
    transformer = compose.ColumnTransformer(
        [("cat", prep, categorical)], remainder="passthrough"
    )
    model_class = (
        ensemble.HistGradientBoostingClassifier
        if classes
        else ensemble.HistGradientBoostingRegressor
    )
    estimator = pipeline.make_pipeline(transformer, model_class(random_state=0))
    dev, test = manifest["train_ids"], manifest["test_ids"]
    estimator.fit(features.iloc[dev], frame[target].iloc[dev])
    metric = mml_metrics.CATALOG.metrics[metric_name]
    positive = mml_metrics.positive_label(frame[target], tasks.TaskType(task))
    return metric.score(
        estimator, features.iloc[test], frame[target].iloc[test], positive
    )
