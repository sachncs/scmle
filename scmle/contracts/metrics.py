"""Metric contracts: scoring rules with explicit orientation."""

from collections.abc import Callable
import dataclasses
import inspect
from typing import Any

import pandas as pd
from sklearn import metrics as sk_metrics

from scmle.contracts import models
from scmle.contracts import tasks


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
        """Converts a sign-adjusted scorer value to natural units.

        Args:
            scorer_value: The value the scorer returned, negated for metrics
                where lower is better.

        Returns:
            The value in the metric's own units and direction.
        """
        return scorer_value if self.greater_is_better else -scorer_value

    def accepts_weights(self) -> bool:
        """Returns whether the metric can be scored with `sample_weight`.

        Returns:
            Always True here: stock scikit-learn scorers accept per-row
            ``sample_weight``. Subclasses override it.
        """
        return True

    def scorer(self, positive: Any = None) -> Any:
        """Returns the scikit-learn scorer named by `scorer_name`.

        Args:
            positive: Positive class label; ignored here, used by subclasses
                that need it.

        Returns:
            The stock scorer from `sklearn.metrics.get_scorer`.
        """
        return sk_metrics.get_scorer(self.scorer_name)

    def score(
        self,
        estimator: models.Estimator,
        features: pd.DataFrame,
        target: pd.Series,
        positive: Any = None,
        sample_weight: Any = None,
    ) -> float:
        """Scores a fitted estimator in natural units.

        Args:
            estimator: Fitted estimator.
            features: Evaluation features.
            target: Evaluation labels.
            positive: Positive class label for binary metrics.
            sample_weight: Optional per-row weights for the score.

        Returns:
            The score in natural units (not sign-adjusted).
        """
        scorer = self.scorer(positive)
        extra = (
            {} if sample_weight is None else {"sample_weight": sample_weight}
        )
        return self.to_natural(
            float(scorer(estimator, features, target, **extra))
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PositiveLabelMetric(Metric):
    """A binary metric defined relative to a positive class.

    scikit-learn's stock scorers hard-code `pos_label=1`, which breaks on
    string or other non-{0, 1} labels; this builds the scorer with the real
    positive class instead.

    Attributes:
        score_function: Function `(y_true, y_pred_or_score, pos_label=...)`.
        response_methods: Estimator methods supplying the prediction.
    """

    score_function: Callable[..., float] = sk_metrics.f1_score
    response_methods: tuple[str, ...] = ("predict",)

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer bound to `positive` (stock scorer if unknown).

        Args:
            positive: The positive class label, or None.

        Returns:
            A scorer that treats `positive` as the positive class, or the
            metric's stock scorer if `positive` is None.
        """
        if positive is None:
            return super().scorer(None)
        return sk_metrics.make_scorer(
            self.score_function,
            greater_is_better=self.greater_is_better,
            response_method=self.response_methods,
            pos_label=positive,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class FunctionMetric(Metric):
    """A metric defined by a user function of the true and predicted values.

    The function is pickled by reference into trial processes, so it must be
    a module-level function that those processes can import.

    Attributes:
        score_function: Function `(y_true, y_pred) -> float`.
        response_method: What the estimator supplies to the function:
            `predict`, `predict_proba` or `decision_function`.
    """

    score_function: Callable[..., float] = sk_metrics.accuracy_score
    response_method: str = "predict"

    def accepts_weights(self) -> bool:
        """Returns whether the user function takes `sample_weight`.

        Returns:
            True if the user function has a ``sample_weight`` parameter.
        """
        return (
            "sample_weight" in inspect.signature(self.score_function).parameters
        )

    def scorer(self, positive: Any = None) -> Any:
        """Builds a scorer that applies the user function.

        Args:
            positive: Not used; the user function decides what is positive.

        Returns:
            A scorer wrapping the user function.
        """
        return sk_metrics.make_scorer(
            self.score_function,
            greater_is_better=self.greater_is_better,
            response_method=self.response_method,
        )
