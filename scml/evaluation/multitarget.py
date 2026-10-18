"""Metrics over several targets, read off a multi-target estimator."""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import metrics as sk_metrics

from scml.contracts import metrics as metric_contracts


def f1_macro_labels(truth: np.ndarray, guess: np.ndarray) -> float:
    """F1 averaged over labels, each label counting equally."""
    return float(
        sk_metrics.f1_score(truth, guess, average="macro", zero_division=0)
    )


def f1_micro_labels(truth: np.ndarray, guess: np.ndarray) -> float:
    """F1 over all label decisions pooled."""
    return float(
        sk_metrics.f1_score(truth, guess, average="micro", zero_division=0)
    )


def hamming(truth: np.ndarray, guess: np.ndarray) -> float:
    """Share of label decisions that are wrong (lower is better)."""
    return float(sk_metrics.hamming_loss(truth, guess))


def subset_accuracy(truth: np.ndarray, guess: np.ndarray) -> float:
    """Share of rows where every label is right."""
    return float((np.asarray(truth) == np.asarray(guess)).all(axis=1).mean())


def r2_mean(truth: np.ndarray, guess: np.ndarray) -> float:
    """R-squared averaged over the target columns."""
    return float(
        sk_metrics.r2_score(truth, guess, multioutput="uniform_average")
    )


def mae_mean_scaled(truth: np.ndarray, guess: np.ndarray) -> float:
    """Mean absolute error per column, in units of that column's spread."""
    truth = np.asarray(truth, dtype=float)
    spread = np.where(truth.std(axis=0) > 0, truth.std(axis=0), 1.0)
    return float((np.abs(truth - np.asarray(guess)) / spread).mean())


FUNCTIONS = {
    "f1_macro_labels": f1_macro_labels,
    "f1_micro_labels": f1_micro_labels,
    "hamming": hamming,
    "subset_accuracy": subset_accuracy,
    "r2_mean": r2_mean,
    "mae_scaled": mae_mean_scaled,
}


@dataclasses.dataclass(frozen=True, slots=True)
class MultiTargetMetric(metric_contracts.Metric):
    """A metric over all target columns.

    Attributes:
        kind: Key into `FUNCTIONS`.
    """

    kind: str = "f1_macro_labels"

    def accepts_weights(self) -> bool:
        """These scores are unweighted."""
        return False

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer `(estimator, features, primary_target)`."""
        function = FUNCTIONS[self.kind]
        sign = 1.0 if self.greater_is_better else -1.0

        def score(
            estimator: Any,
            features: pd.DataFrame,
            target: pd.Series,
            sample_weight: Any = None,
        ) -> float:
            truth = np.column_stack(
                [
                    np.asarray(target),
                    features[list(estimator.target_columns[1:])].to_numpy(),
                ]
            )
            guess = np.asarray(estimator.predict(features))
            return sign * function(truth, guess)

        return score
