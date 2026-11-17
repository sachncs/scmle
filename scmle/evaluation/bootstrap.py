"""The bootstrap interval of the primary test metric.

`Uncertainty` resamples the evaluation rows (whole groups, time blocks or
rows) and re-scores cached predictions (`Cache`), so a bootstrap of
thousands of resamples predicts once. Its cost model sizes the resamples to
the time left (`Uncertainty.cost`).
"""

from typing import Any
from typing import cast

import numpy
import pandas
from sklearn import base

from scmle.contracts import errors
from scmle.contracts import models
from scmle.contracts import tasks
from scmle.evaluation import diagnostics

# Share of the planned resamples that must be scorable for an interval.
# Policy: below half, the percentiles come from a selected set of resamples
# (those that kept every class), so the interval is reported as None.
KEPT = 0.5
# Bootstrap seconds per resample, classification: `RESAMPLE_SECONDS +
# RESAMPLE_ROW_SECONDS * t + RESAMPLE_CLASS_ROW_SECONDS * t * (C - 2)` for
# t evaluation rows and C > 2 classes (the cached predictions make it
# independent of the model and of the feature count). Fitted on the 21
# classification runs described at `diagnostics.MARGIN` (t 250..26,018,
# C 2..8); measured over fitted 0.72..1.35.
RESAMPLE_SECONDS = 5.3e-4
RESAMPLE_ROW_SECONDS = 2.0e-6
RESAMPLE_CLASS_ROW_SECONDS = 3.1e-7
# Regression bootstrap seconds per resample: `REGRESSION_RESAMPLE_SECONDS +
# REGRESSION_RESAMPLE_ROW_SECONDS * t`, fitted on the 10 regression runs
# described there (t 303..4,592); measured over fitted 0.92..1.13.
REGRESSION_RESAMPLE_SECONDS = 3.6e-4
REGRESSION_RESAMPLE_ROW_SECONDS = 3.5e-8


class Cache(base.BaseEstimator):  # type: ignore[misc]  # scikit-learn is untyped
    """Replays a fitted model's predictions by row label.

    Lets a bootstrap re-score thousands of resamples without re-predicting:
    the wrapped estimator runs once, and every later call is a lookup on the
    (unique) index of the rows it is handed.

    Attributes:
        predictions: Predicted labels / values by row label.
        probabilities: Class probabilities by row label (classifiers).
        classes_: Class labels of the wrapped model, if any.

    `probabilities` is None unless the wrapped model has both `classes_` and
    `predict_proba`.
    """

    def __init__(self, estimator: Any, features: pandas.DataFrame) -> None:
        """Predicts every row once.

        Args:
            estimator: Fitted model.
            features: Rows with a unique index.
        """
        self.classes_ = getattr(estimator, "classes_", None)
        self.predictions = pandas.Series(
            numpy.asarray(estimator.predict(features)), index=features.index
        )
        self.probabilities = (
            pandas.DataFrame(
                estimator.predict_proba(features), index=features.index
            )
            if self.classes_ is not None and hasattr(estimator, "predict_proba")
            else None
        )

    def predict(self, features: pandas.DataFrame) -> numpy.ndarray:
        """Looks up predictions for the given rows.

        Args:
            features: Rows whose index labels were seen at construction.

        Returns:
            The cached predictions, in the order of `features`.
        """
        return self.predictions.loc[features.index].to_numpy()

    def predict_proba(self, features: pandas.DataFrame) -> numpy.ndarray:
        """Looks up probabilities for the given rows.

        Args:
            features: Rows whose index labels were seen at construction.

        Returns:
            The cached class probabilities, in the order of `features`.

        Raises:
            errors.ArtifactError: If no probabilities were cached, because the
                wrapped model lacks `classes_` or `predict_proba`.
        """
        if self.probabilities is None:
            raise errors.ArtifactError("the model has no probabilities")
        return self.probabilities.loc[features.index].to_numpy()


class Replay(base.ClassifierMixin, Cache):  # type: ignore[misc]  # scikit-learn is untyped
    """Cached predictions that scikit-learn treats as a classifier."""


class RegressorReplay(base.RegressorMixin, Cache):  # type: ignore[misc]  # scikit-learn is untyped
    """Cached predictions that scikit-learn treats as a regressor."""


def redraw(
    rows: int,
    groups: pandas.Series | None,
    temporal: bool,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Draws one bootstrap resample that respects the data's dependence.

    Args:
        rows: Number of evaluation rows.
        groups: Entity labels, if rows are clustered by entity.
        temporal: Whether to draw contiguous blocks; ignored when `groups`
            is given.
        rng: Source of randomness.

    Returns:
        Row positions of the resample: whole groups drawn with replacement
        (its length varies with the group sizes), or blocks of
        `max(5, int(sqrt(rows)))` positions truncated to `rows`, or `rows`
        positions drawn with replacement.
    """
    if groups is not None:
        members = (
            groups.reset_index(drop=True).groupby(groups.to_numpy()).indices
        )
        labels = list(members)
        drawn = rng.integers(0, len(labels), len(labels))
        return numpy.concatenate([members[labels[i]] for i in drawn])
    if temporal:
        length = max(5, int(numpy.sqrt(rows)))
        starts = rng.integers(0, max(rows - length + 1, 1), -(-rows // length))
        blocks = numpy.concatenate(
            [numpy.arange(s, min(s + length, rows)) for s in starts]
        )
        return blocks[:rows]
    return rng.integers(0, rows, rows)


class Uncertainty(diagnostics.Diagnostic):
    """Bootstrap confidence interval for the primary test metric.

    Resampling follows the independent units of observation: whole groups
    for clustered data, contiguous blocks for temporal data, and rows
    otherwise, so intervals are not falsely narrow.
    """

    name = "metric_uncertainty"
    resamples = 2000
    minimum_resamples = 200
    row_budget = 4_000_000
    confidence = 95

    def replicates(self, rows: int) -> int:
        """Chooses how many bootstrap resamples to draw.

        Few resamples make the interval ends noisy and the interval too
        narrow (a simulation of ROC AUC on 200 rows covered the true value in
        91 percent of 400 runs with 200 resamples and 92.8 percent with
        2,000), so small evaluation sets get `resamples` and large ones fewer,
        so that the rows scored stay within `row_budget`.

        Args:
            rows: Number of evaluation rows.

        Returns:
            A count between `minimum_resamples` and `resamples`.
        """
        affordable = self.row_budget // max(rows, 1)
        return min(self.resamples, max(self.minimum_resamples, affordable))

    def cost(self, inputs: diagnostics.DiagnosticInput) -> float:
        """Predicts the seconds of one resample (upper envelope).

        Args:
            inputs: The evaluation data and task.

        Returns:
            `diagnostics.MARGIN` times the fitted per-resample cost for the
            task, the number of evaluation rows and, for classification, the
            classes.
        """
        rows = len(inputs.features)
        if inputs.task is tasks.Task.REGRESSION:
            return diagnostics.MARGIN * (
                REGRESSION_RESAMPLE_SECONDS
                + REGRESSION_RESAMPLE_ROW_SECONDS * rows
            )
        labels = getattr(inputs.estimator, "classes_", None)
        classes = 0 if labels is None else len(labels)
        return diagnostics.MARGIN * (
            RESAMPLE_SECONDS
            + RESAMPLE_ROW_SECONDS * rows
            + RESAMPLE_CLASS_ROW_SECONDS * rows * max(classes - 2, 0)
        )

    def estimate(self, inputs: diagnostics.DiagnosticInput) -> float:
        """Predicts the seconds of the full bootstrap.

        Args:
            inputs: The evaluation data, with `pace` when it was timed.

        Returns:
            Two predictions (labels and probabilities, cached once) plus
            `replicates` resamples.
        """
        return diagnostics.MARGIN * 2 * (inputs.pace or 0.0) + self.replicates(
            len(inputs.features)
        ) * self.cost(inputs)

    def least(self, inputs: diagnostics.DiagnosticInput) -> float:
        """Predicts the seconds of a bootstrap of `minimum_resamples`.

        Args:
            inputs: The evaluation data, with `pace` when it was timed.

        Returns:
            Two predictions plus `minimum_resamples` resamples.
        """
        return diagnostics.MARGIN * 2 * (
            inputs.pace or 0.0
        ) + self.minimum_resamples * self.cost(inputs)

    def compute(self, inputs: diagnostics.DiagnosticInput) -> dict[str, Any]:
        """Bootstraps the primary metric on the evaluation rows.

        The number of resamples is `replicates`, cut to what `cost` says
        fits before the deadline; below `minimum_resamples` (too few for an
        interval that is not too narrow) the bootstrap is skipped, and the
        deadline is checked again before each resample.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The metric name, its point estimate, the percentile interval
            (key `ci_` plus the confidence; None when every resample
            failed or too few ran), the resampling unit, the number of
            resamples `planned`, the number of successful resamples
            (resamples that fail with a `ValueError`, such as one missing a
            class, are skipped and counted in `dropped`), the number of
            evaluation rows and the number of independent units (groups,
            else rows). When the deadline left fewer than
            `minimum_resamples`, the report is `skipped` with the reason.
        """
        features = inputs.features[list(inputs.feature_plan.sourced)]
        features = features.reset_index(drop=True)
        target = inputs.target.reset_index(drop=True)
        replay_class = (
            RegressorReplay if inputs.task is tasks.Task.REGRESSION else Replay
        )
        # Scorers only call predict and predict_proba on the replay.
        replay = cast(
            "models.Estimator", replay_class(inputs.estimator, features)
        )
        metric, positive = inputs.metric, inputs.positive_label
        estimate = metric.score(replay, features, target, positive)
        groups = (
            inputs.groups.reset_index(drop=True)
            if inputs.groups is not None
            else None
        )
        rng = numpy.random.default_rng(inputs.seed)
        scores: list[float] = []
        unit = (
            "groups"
            if groups is not None
            else "blocks"
            if inputs.temporal
            else "rows"
        )
        each = self.cost(inputs)
        planned = self.replicates(len(target))
        if inputs.deadline is not None:
            planned = min(planned, max(int(inputs.left() // each), 0))
        attempted = 0
        if planned >= self.minimum_resamples:
            for _ in range(planned):
                if inputs.expired():
                    break
                rows = redraw(len(target), groups, inputs.temporal, rng)
                attempted += 1
                try:
                    scores.append(
                        metric.score(
                            replay,
                            features.iloc[rows],
                            target.iloc[rows],
                            positive,
                        )
                    )
                except (
                    ValueError
                ):  # handled: counted in `dropped`; too many give no interval
                    continue
        interval = None
        if attempted >= self.minimum_resamples and len(scores) >= (
            attempted * KEPT
        ):
            low, high = numpy.percentile(
                scores,
                [
                    (100 - self.confidence) / 2,
                    100 - (100 - self.confidence) / 2,
                ],
            )
            interval = [float(low), float(high)]
        report = {
            "metric": metric.name,
            "estimate": float(estimate),
            f"ci_{self.confidence}": interval,
            "resampling_unit": unit,
            "planned": planned,
            "resamples": len(scores),
            "dropped": attempted - len(scores),
            "evaluation_rows": len(target),
            "independent_units": int(groups.nunique())
            if groups is not None
            else len(target),
        }
        if attempted < self.minimum_resamples:
            return {
                **report,
                **diagnostics.skipped(
                    f"time for {attempted} of the minimum "
                    f"{self.minimum_resamples} resamples (about {each:.4f} s "
                    f"each) before the deadline"
                ),
            }
        return report
