"""Metric catalog with explicit orientation and a fixed scoring protocol.

Scores are always reported in natural units (log loss is positive);
`greater_is_better` tells rankers which direction wins.
"""

from typing import Any

import pandas as pd
from sklearn import metrics as sk_metrics

from mml.contracts import errors
from mml.contracts import metrics as metric_contracts
from mml.contracts import models
from mml.contracts import tasks

BINARY = tasks.TaskType.BINARY
MULTICLASS = tasks.TaskType.MULTICLASS
REGRESSION = tasks.TaskType.REGRESSION
CLASSIFICATION = frozenset({BINARY, MULTICLASS})


def positive_label(target: pd.Series, task: tasks.TaskType) -> Any:
    """Returns the positive class of a binary target (None otherwise).

    Follows scikit-learn's convention: the second class in sorted order,
    which is `classes_[1]` of every fitted classifier.

    Args:
        target: Labels.
        task: Resolved task type.
    """
    if task is not BINARY:
        return None
    return sorted(target.dropna().unique())[1]


DEFAULT_METRICS = (
    metric_contracts.Metric("roc_auc", "roc_auc", True, frozenset({BINARY})),
    metric_contracts.Metric(
        "roc_auc_ovr", "roc_auc_ovr", True, frozenset({MULTICLASS})
    ),
    metric_contracts.PositiveLabelMetric(
        "pr_auc",
        "average_precision",
        True,
        frozenset({BINARY}),
        sk_metrics.average_precision_score,
        ("decision_function", "predict_proba"),
    ),
    metric_contracts.Metric("log_loss", "neg_log_loss", False, CLASSIFICATION),
    metric_contracts.PositiveLabelMetric(
        "f1",
        "f1",
        True,
        frozenset({BINARY}),
        sk_metrics.f1_score,
        ("predict",),
    ),
    metric_contracts.Metric("f1_macro", "f1_macro", True, CLASSIFICATION),
    metric_contracts.Metric(
        "balanced_accuracy", "balanced_accuracy", True, CLASSIFICATION
    ),
    metric_contracts.Metric("accuracy", "accuracy", True, CLASSIFICATION),
    metric_contracts.Metric(
        "mae", "neg_mean_absolute_error", False, frozenset({REGRESSION})
    ),
    metric_contracts.Metric(
        "rmse", "neg_root_mean_squared_error", False, frozenset({REGRESSION})
    ),
    metric_contracts.Metric("r2", "r2", True, frozenset({REGRESSION})),
)


class MetricCatalog:
    """Resolves metric names and applies the report protocol."""

    default_metric = {
        BINARY: "roc_auc",
        MULTICLASS: "log_loss",
        REGRESSION: "rmse",
    }
    report_names = {
        BINARY: (
            "roc_auc",
            "pr_auc",
            "log_loss",
            "f1",
            "balanced_accuracy",
            "accuracy",
        ),
        MULTICLASS: ("log_loss", "f1_macro", "balanced_accuracy", "accuracy"),
        REGRESSION: ("mae", "rmse", "r2"),
    }

    def __init__(
        self, metrics: tuple[metric_contracts.Metric, ...] = DEFAULT_METRICS
    ) -> None:
        """Initializes the catalog.

        Args:
            metrics: Available metrics.
        """
        self.metrics = {metric.name: metric for metric in metrics}

    def resolve(
        self, name: str | None, task: tasks.TaskType
    ) -> metric_contracts.Metric:
        """Finds the metric to optimize.

        Args:
            name: Requested name, or None for the task default.
            task: Resolved task type.

        Returns:
            The metric (`roc_auc` maps to its one-vs-rest variant for
            multiclass tasks).

        Raises:
            ConfigError: If the name is unknown or invalid for the task.
        """
        name = name or self.default_metric[task]
        if name == "roc_auc" and task is MULTICLASS:
            name = "roc_auc_ovr"
        metric = self.metrics.get(name)
        if metric is None:
            raise errors.ConfigError(
                f"unknown metric '{name}'; available: {sorted(self.metrics)}"
            )
        if task not in metric.tasks:
            raise errors.ConfigError(
                f"metric '{name}' is not defined for {task.value}"
            )
        return metric

    def evaluate(
        self,
        estimator: models.Estimator,
        features: pd.DataFrame,
        target: pd.Series,
        task: tasks.TaskType,
    ) -> dict[str, float]:
        """Applies every report metric for the task.

        Args:
            estimator: Fitted estimator.
            features: Evaluation features.
            target: Evaluation labels.
            task: Resolved task type.

        Returns:
            Metric name to natural-unit score; NaN when a metric cannot be
            computed (for example a class absent from the labels).
        """
        results: dict[str, float] = {}
        classes = getattr(estimator, "classes_", None)
        positive = (
            classes[1] if task is BINARY and classes is not None else None
        )
        for name in self.report_names[task]:
            try:
                results[name] = self.metrics[name].score(
                    estimator, features, target, positive
                )
            except ValueError:
                results[name] = float("nan")
        return results


CATALOG = MetricCatalog()
