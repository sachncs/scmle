"""Metric contracts: scoring rules with explicit orientation."""

from collections.abc import Callable
import dataclasses
import inspect
from typing import Any
from typing import cast

import pandas
from sklearn import metrics

from scmle.contracts import models
from scmle.contracts import tasks


@dataclasses.dataclass(frozen=True, slots=True)
class Metric:
    """A scoring rule.

    Attributes:
        name: Public metric name.
        scoring: scikit-learn scorer (sign-adjusted so larger is better).
        maximize: Whether larger natural values are better.
        tasks: Tasks the metric is defined for.
    """

    name: str
    scoring: str
    maximize: bool
    tasks: frozenset[tasks.Task]

    def naturalise(self, scorer_value: float) -> float:
        """Converts a sign-adjusted scorer value to natural units.

        Args:
            scorer_value: The value the scorer returned, negated for metrics
                where lower is better.

        Returns:
            The value in the metric's own units and direction.
        """
        return scorer_value if self.maximize else -scorer_value

    def weightable(self) -> bool:
        """Returns whether the metric can be scored with `sample_weight`.

        Returns:
            Always True here: stock scikit-learn scorers accept per-row
            ``sample_weight``. Subclasses override it.
        """
        return True

    def scorer(
        self,
        positive: Any = None,  # noqa: ARG002 -- shared scorer signature
    ) -> Any:
        """Returns the scikit-learn scorer named by `scoring`.

        Args:
            positive: Positive class label; ignored here, used by subclasses
                that need it.

        Returns:
            The stock scorer from `sklearn.metrics.get_scorer`.
        """
        return metrics.get_scorer(self.scoring)

    def score(
        self,
        estimator: models.Estimator,
        features: pandas.DataFrame,
        target: pandas.Series,
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
        return self.naturalise(
            float(scorer(estimator, features, target, **extra))
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PositiveLabelMetric(Metric):
    """A binary metric defined relative to a positive class.

    scikit-learn's stock scorers hard-code `pos_label=1`, which breaks on
    string or other non-{0, 1} labels; this builds the scorer with the real
    positive class instead.

    Attributes:
        function: Function `(y_true, y_pred_or_score, pos_label=...)`.
        methods: Estimator methods supplying the prediction.
    """

    function: Callable[..., float] = metrics.f1_score
    methods: tuple[str, ...] = ("predict",)

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
        return metrics.make_scorer(
            self.function,
            greater_is_better=self.maximize,
            response_method=cast("Any", self.methods),
            pos_label=positive,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Formula(Metric):
    """A metric defined by a user function of the true and predicted values.

    The function is pickled by reference into trial processes, so it must be
    a module-level function that those processes can import.

    Attributes:
        function: Function `(y_true, y_pred) -> float`.
        method: What the estimator supplies to the function:
            `predict`, `predict_proba` or `decision_function`.
    """

    function: Callable[..., float] = metrics.accuracy_score
    method: str = "predict"

    def weightable(self) -> bool:
        """Returns whether the user function takes `sample_weight`.

        Returns:
            True if the user function has a ``sample_weight`` parameter.
        """
        return "sample_weight" in inspect.signature(self.function).parameters

    def scorer(
        self,
        positive: Any = None,  # noqa: ARG002 -- shared scorer signature
    ) -> Any:
        """Builds a scorer that applies the user function.

        Args:
            positive: Not used; the user function decides what is positive.

        Returns:
            A scorer wrapping the user function.
        """
        return metrics.make_scorer(
            self.function,
            greater_is_better=self.maximize,
            response_method=self.method,
        )
