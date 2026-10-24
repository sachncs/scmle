"""Metric catalog with explicit orientation and a fixed scoring protocol.

Scores are always reported in natural units (log loss is positive);
`greater_is_better` tells rankers which direction wins.
"""

from collections.abc import Callable
from typing import Any

import pandas as pd
from sklearn import metrics as sk_metrics

from scmle.contracts import errors
from scmle.contracts import metrics as metric_contracts
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.evaluation import concordance
from scmle.evaluation import forecast
from scmle.evaluation import multitarget
from scmle.evaluation import ordinal
from scmle.evaluation import ranking
from scmle.evaluation import survival
from scmle.evaluation import uplift

BINARY = tasks.TaskType.BINARY
MULTICLASS = tasks.TaskType.MULTICLASS
REGRESSION = tasks.TaskType.REGRESSION
SURVIVAL = tasks.TaskType.SURVIVAL
ORDINAL = tasks.TaskType.ORDINAL
RANKING = tasks.TaskType.RANKING
MULTILABEL = tasks.TaskType.MULTILABEL
MULTIOUTPUT = tasks.TaskType.MULTIOUTPUT
FORECAST = tasks.TaskType.FORECAST
UPLIFT = tasks.TaskType.UPLIFT
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
    metric_contracts.FunctionMetric(
        "qwk",
        "",
        True,
        frozenset({ORDINAL}),
        ordinal.quadratic_weighted_kappa,
        "predict",
    ),
    metric_contracts.FunctionMetric(
        "mace",
        "",
        False,
        frozenset({ORDINAL}),
        ordinal.mean_absolute_class_error,
        "predict",
    ),
    metric_contracts.Metric(
        "accuracy_levels", "accuracy", True, frozenset({ORDINAL})
    ),
    multitarget.MultiTargetMetric(
        "f1_macro_labels", "", True, frozenset({MULTILABEL}), "f1_macro_labels"
    ),
    multitarget.MultiTargetMetric(
        "f1_micro_labels", "", True, frozenset({MULTILABEL}), "f1_micro_labels"
    ),
    multitarget.MultiTargetMetric(
        "hamming", "", False, frozenset({MULTILABEL}), "hamming"
    ),
    multitarget.MultiTargetMetric(
        "subset_accuracy", "", True, frozenset({MULTILABEL}), "subset_accuracy"
    ),
    multitarget.MultiTargetMetric(
        "r2_mean", "", True, frozenset({MULTIOUTPUT}), "r2_mean"
    ),
    multitarget.MultiTargetMetric(
        "mae_scaled", "", False, frozenset({MULTIOUTPUT}), "mae_scaled"
    ),
    forecast.ForecastMetric("mase", "", False, frozenset({FORECAST}), "mase"),
    forecast.ForecastMetric(
        "forecast_mae", "", False, frozenset({FORECAST}), "mae"
    ),
    forecast.ForecastMetric(
        "forecast_smape", "", False, frozenset({FORECAST}), "smape"
    ),
    uplift.UpliftMetric("qini", "", True, frozenset({UPLIFT}), "qini"),
    uplift.UpliftMetric(
        "uplift_at_30", "", True, frozenset({UPLIFT}), "uplift_at_30"
    ),
    ranking.RankingMetric("ndcg", "", True, frozenset({RANKING}), "ndcg"),
    ranking.RankingMetric("map", "", True, frozenset({RANKING}), "map"),
    ranking.RankingMetric("mrr", "", True, frozenset({RANKING}), "mrr"),
    ranking.RankingMetric(
        "precision_at_10",
        "",
        True,
        frozenset({RANKING}),
        "precision_at_10",
    ),
    ranking.RankingMetric(
        "recall_at_10", "", True, frozenset({RANKING}), "recall_at_10"
    ),
    concordance.ConcordanceMetric("c_index", "", True, frozenset({SURVIVAL})),
    survival.SurvivalMetric("ibs", "", False, frozenset({SURVIVAL}), "ibs"),
    survival.SurvivalMetric(
        "td_auc", "", True, frozenset({SURVIVAL}), "td_auc"
    ),
)


class MetricCatalog:
    """Resolves metric names and applies the report protocol."""

    default_metric = {
        BINARY: "roc_auc",
        MULTICLASS: "log_loss",
        REGRESSION: "rmse",
        SURVIVAL: "c_index",
        ORDINAL: "qwk",
        RANKING: "ndcg",
        MULTILABEL: "f1_macro_labels",
        MULTIOUTPUT: "r2_mean",
        FORECAST: "mase",
        UPLIFT: "qini",
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
        SURVIVAL: ("c_index", "ibs", "td_auc"),
        ORDINAL: ("qwk", "mace", "accuracy_levels"),
        RANKING: ("ndcg", "map", "mrr", "precision_at_10", "recall_at_10"),
        MULTILABEL: (
            "f1_macro_labels",
            "f1_micro_labels",
            "hamming",
            "subset_accuracy",
        ),
        MULTIOUTPUT: ("r2_mean", "mae_scaled"),
        FORECAST: ("mase", "forecast_mae", "forecast_smape"),
        UPLIFT: ("qini", "uplift_at_30"),
    }

    def __init__(
        self, metrics: tuple[metric_contracts.Metric, ...] = DEFAULT_METRICS
    ) -> None:
        """Initializes the catalog.

        Args:
            metrics: Available metrics.
        """
        self.metrics = {metric.name: metric for metric in metrics}
        self.report_names = {
            task: tuple(names)
            for task, names in type(self).report_names.items()
        }

    def register(
        self, metric: metric_contracts.Metric, replace: bool = False
    ) -> None:
        """Adds a user-defined metric, reported alongside the standard ones.

        Args:
            metric: The metric.
            replace: Allow overriding an existing metric of the same name.

        Raises:
            ConfigError: If the name is taken and `replace` is False.
        """
        if metric.name in self.metrics and not replace:
            raise errors.ConfigError(f"metric '{metric.name}' already exists")
        self.metrics[metric.name] = metric
        for task in metric.tasks:
            names = self.report_names[task]
            if metric.name not in names:
                self.report_names[task] = (*names, metric.name)

    def unregister(self, name: str) -> None:
        """Removes a user-defined metric (no-op if it is not registered)."""
        self.metrics.pop(name, None)
        self.report_names = {
            task: tuple(n for n in names if n != name)
            for task, names in self.report_names.items()
        }

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
        sample_weight: Any = None,
    ) -> dict[str, float]:
        """Applies every report metric for the task.

        Args:
            estimator: Fitted estimator.
            features: Evaluation features.
            target: Evaluation labels.
            task: Resolved task type.
            sample_weight: Optional per-row weights applied to every score.

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
                    estimator, features, target, positive, sample_weight
                )
            except (ValueError, TypeError):
                results[name] = float("nan")
        return results


CATALOG = MetricCatalog()


def register_metric(
    name: str,
    score_function: Callable[..., float],
    greater_is_better: bool,
    task_types: tuple[tasks.TaskType, ...] = tuple(tasks.STANDARD_TASKS),
    response_method: str = "predict",
    replace: bool = False,
) -> metric_contracts.FunctionMetric:
    """Registers a user-defined metric with the default catalog.

    Args:
        name: Public metric name, usable as `MLProject(metric=name)`.
        score_function: Module-level function `(y_true, y_pred) -> float`.
        greater_is_better: Orientation of the score.
        task_types: Tasks the metric is defined for.
        response_method: `predict`, `predict_proba` or `decision_function`.
        replace: Allow overriding an existing metric of the same name.

    Returns:
        The registered metric.
    """
    metric = metric_contracts.FunctionMetric(
        name,
        "",
        greater_is_better,
        frozenset(task_types),
        score_function,
        response_method,
    )
    CATALOG.register(metric, replace)
    return metric
