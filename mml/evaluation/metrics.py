"""Metric registry with explicit orientation.

`value` is always reported in natural units (e.g. log loss is positive); `greater_is_better`
tells rankers which direction wins. sklearn scorers are used for fixed, consistent protocol.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import get_scorer

from mml.contracts.errors import ConfigError
from mml.contracts.tasks import TaskType


@dataclass(frozen=True)
class Metric:
    name: str
    scorer: str  # sklearn scoring name (sign-adjusted so larger is better)
    greater_is_better: bool
    tasks: frozenset[TaskType]

    def to_natural(self, scorer_value: float) -> float:
        return scorer_value if self.greater_is_better else -scorer_value


_B, _M, _R = TaskType.BINARY, TaskType.MULTICLASS, TaskType.REGRESSION
_CLF = frozenset({_B, _M})

METRICS: dict[str, Metric] = {
    m.name: m
    for m in [
        Metric("roc_auc", "roc_auc", True, frozenset({_B})),
        Metric("roc_auc_ovr", "roc_auc_ovr", True, frozenset({_M})),
        Metric("pr_auc", "average_precision", True, frozenset({_B})),
        Metric("log_loss", "neg_log_loss", False, _CLF),
        Metric("f1", "f1", True, frozenset({_B})),
        Metric("f1_macro", "f1_macro", True, _CLF),
        Metric("balanced_accuracy", "balanced_accuracy", True, _CLF),
        Metric("accuracy", "accuracy", True, _CLF),
        Metric("mae", "neg_mean_absolute_error", False, frozenset({_R})),
        Metric("rmse", "neg_root_mean_squared_error", False, frozenset({_R})),
        Metric("r2", "r2", True, frozenset({_R})),
    ]
}

DEFAULT_METRIC = {_B: "roc_auc", _M: "log_loss", _R: "rmse"}
REPORT_METRICS = {
    _B: ["roc_auc", "pr_auc", "log_loss", "f1", "balanced_accuracy", "accuracy"],
    _M: ["log_loss", "f1_macro", "balanced_accuracy", "accuracy"],
    _R: ["mae", "rmse", "r2"],
}


def resolve_metric(name: str | None, task: TaskType) -> Metric:
    name = name or DEFAULT_METRIC[task]
    if name == "roc_auc" and task is _M:
        name = "roc_auc_ovr"
    m = METRICS.get(name)
    if m is None:
        raise ConfigError(f"unknown metric '{name}'; available: {sorted(METRICS)}")
    if task not in m.tasks:
        raise ConfigError(f"metric '{name}' is not defined for {task.value}")
    return m


def score_estimator(estimator, X: pd.DataFrame, y: pd.Series, task: TaskType) -> dict[str, float]:
    """Fixed scoring protocol: all report metrics for the task, natural units."""
    out: dict[str, float] = {}
    for name in REPORT_METRICS[task]:
        m = METRICS[name]
        scorer = get_scorer(m.scorer)
        try:
            out[name] = float(m.to_natural(scorer(estimator, X, y)))
        except Exception:  # e.g. a class absent from the test split
            out[name] = float("nan")
    return out
