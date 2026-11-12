"""Metrics over several targets, read off a multi-target estimator."""

import dataclasses
from typing import Any

import numpy
import pandas
from sklearn import metrics

import scmle.contracts.metrics


def f1_macro_labels(truth: numpy.ndarray, guess: numpy.ndarray) -> float:
    """F-measure averaged over labels, each label counting equally.

    Args:
        truth: Observed 0/1 label matrix, `(rows, labels)`.
        guess: Predicted 0/1 label matrix, same shape.

    Returns:
        The macro-averaged F-measure; undefined cases count as 0.
    """
    return float(
        metrics.f1_score(
            truth,
            guess,
            average="macro",
            zero_division=0,  # pyright: ignore[reportArgumentType]  # scikit-learn infers zero_division as str from its "warn" default
        )
    )


def f1_micro_labels(truth: numpy.ndarray, guess: numpy.ndarray) -> float:
    """F-measure over all label decisions pooled.

    Args:
        truth: Observed 0/1 label matrix, `(rows, labels)`.
        guess: Predicted 0/1 label matrix, same shape.

    Returns:
        The micro-averaged F-measure; undefined cases count as 0.
    """
    return float(
        metrics.f1_score(
            truth,
            guess,
            average="micro",
            zero_division=0,  # pyright: ignore[reportArgumentType]  # scikit-learn infers zero_division as str from its "warn" default
        )
    )


def hamming(truth: numpy.ndarray, guess: numpy.ndarray) -> float:
    """Share of label decisions that are wrong (lower is better).

    Args:
        truth: Observed 0/1 label matrix, `(rows, labels)`.
        guess: Predicted 0/1 label matrix, same shape.

    Returns:
        The Hamming loss.
    """
    return float(metrics.hamming_loss(truth, guess))


def subset_accuracy(truth: numpy.ndarray, guess: numpy.ndarray) -> float:
    """Share of rows where every label is right.

    Args:
        truth: Observed label matrix, `(rows, labels)`.
        guess: Predicted label matrix, same shape.

    Returns:
        The share of rows that match in every column.
    """
    return float(
        (numpy.asarray(truth) == numpy.asarray(guess)).all(axis=1).mean()
    )


def r2_mean(truth: numpy.ndarray, guess: numpy.ndarray) -> float:
    """R-squared averaged over the target columns.

    Args:
        truth: Observed values.
        guess: Predicted values, same shape as `truth`.

    Returns:
        The uniformly averaged R-squared.
    """
    return float(metrics.r2_score(truth, guess, multioutput="uniform_average"))


def mae_mean_scaled(truth: numpy.ndarray, guess: numpy.ndarray) -> float:
    """Mean absolute error with each column scaled by its spread.

    Args:
        truth: Observed values.
        guess: Predicted values, same shape as `truth`.

    Returns:
        The mean over all entries of the absolute error divided by the
        (population) standard deviation of the observed column, which is
        taken as 1 if the column is constant.
    """
    truth = numpy.asarray(truth, dtype=float)
    spread = numpy.where(truth.std(axis=0) > 0, truth.std(axis=0), 1.0)
    return float((numpy.abs(truth - numpy.asarray(guess)) / spread).mean())


FUNCTIONS = {
    "f1_macro_labels": f1_macro_labels,
    "f1_micro_labels": f1_micro_labels,
    "hamming": hamming,
    "subset_accuracy": subset_accuracy,
    "r2_mean": r2_mean,
    "mae_scaled": mae_mean_scaled,
}


@dataclasses.dataclass(frozen=True, slots=True)
class MultiTargetMetric(scmle.contracts.metrics.Metric):
    """A metric over all target columns.

    Attributes:
        kind: Key into `FUNCTIONS`.
    """

    kind: str = "f1_macro_labels"

    def accepts_weights(self) -> bool:
        """These scores are unweighted.

        Returns:
            Always False.
        """
        return False

    def scorer(self, positive: Any = None) -> Any:  # noqa: ARG002 -- metric scorer interface
        """Builds a scorer `(estimator, features, primary_target)`.

        Args:
            positive: Unused; there is no positive class.

        Returns:
            A function that scores the estimator's predictions for all
            target columns (the other columns are read from the features by
            the estimator's `target_columns`). The sign is flipped when
            lower is better, so higher is always better. Sample weights are
            ignored.
        """
        function = FUNCTIONS[self.kind]
        sign = 1.0 if self.greater_is_better else -1.0

        def score(
            estimator: Any,
            features: pandas.DataFrame,
            target: pandas.Series,
            sample_weight: Any = None,  # noqa: ARG001 -- scorer protocol passes sample_weight
        ) -> float:
            truth = numpy.column_stack(
                [
                    numpy.asarray(target),
                    features[list(estimator.target_columns[1:])].to_numpy(),
                ]
            )
            guess = numpy.asarray(estimator.predict(features))
            return sign * function(truth, guess)

        return score
