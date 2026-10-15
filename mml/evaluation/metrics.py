"""Metric catalog with explicit orientation and a fixed scoring protocol.

Scores are always reported in natural units (log loss is positive);
`greater_is_better` tells rankers which direction wins.
"""

import dataclasses

import pandas as pd
from sklearn import metrics as sk_metrics

from mml.contracts import errors
from mml.contracts import models
from mml.contracts import tasks

BINARY = tasks.TaskType.BINARY
MULTICLASS = tasks.TaskType.MULTICLASS
REGRESSION = tasks.TaskType.REGRESSION
CLASSIFICATION = frozenset({BINARY, MULTICLASS})


@dataclasses.dataclass(frozen=True, slots=True)
class Metric:
    """A scoring rule.

    Attributes:
        name: Public metric name.
        scorer_name: scikit-learn scorer (sign-adjusted so larger is better).
        greater_is_better: Whether larger natural values are better.
        tasks: Tasks the metric is defined for.
    """

    name: str
    scorer_name: str
    greater_is_better: bool
    tasks: frozenset[tasks.TaskType]

    def to_natural(self, scorer_value: float) -> float:
        """Converts a sign-adjusted scorer value to natural units."""
        return scorer_value if self.greater_is_better else -scorer_value

    def score(
        self,
        estimator: models.Estimator,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> float:
        """Scores a fitted estimator in natural units."""
        scorer = sk_metrics.get_scorer(self.scorer_name)
        return self.to_natural(float(scorer(estimator, features, target)))


DEFAULT_METRICS = (
    Metric("roc_auc", "roc_auc", True, frozenset({BINARY})),
    Metric("roc_auc_ovr", "roc_auc_ovr", True, frozenset({MULTICLASS})),
    Metric("pr_auc", "average_precision", True, frozenset({BINARY})),
    Metric("log_loss", "neg_log_loss", False, CLASSIFICATION),
    Metric("f1", "f1", True, frozenset({BINARY})),
    Metric("f1_macro", "f1_macro", True, CLASSIFICATION),
    Metric("balanced_accuracy", "balanced_accuracy", True, CLASSIFICATION),
    Metric("accuracy", "accuracy", True, CLASSIFICATION),
    Metric("mae", "neg_mean_absolute_error", False, frozenset({REGRESSION})),
    Metric(
        "rmse", "neg_root_mean_squared_error", False, frozenset({REGRESSION})
    ),
    Metric("r2", "r2", True, frozenset({REGRESSION})),
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

    def __init__(self, metrics: tuple[Metric, ...] = DEFAULT_METRICS) -> None:
        """Initializes the catalog.

        Args:
            metrics: Available metrics.
        """
        self.metrics = {metric.name: metric for metric in metrics}

    def resolve(self, name: str | None, task: tasks.TaskType) -> Metric:
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
        for name in self.report_names[task]:
            try:
                results[name] = self.metrics[name].score(
                    estimator, features, target
                )
            except ValueError:
                results[name] = float("nan")
        return results


CATALOG = MetricCatalog()
