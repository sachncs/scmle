"""The bootstrap interval of the primary test metric.

`Uncertainty` resamples the evaluation rows (whole groups, time blocks or
rows) and re-scores cached predictions (`Cache`), so a bootstrap of
thousands of resamples predicts once. ROC AUC and RMSE get a studentized
(bootstrap-t) interval with a standard error clustered by unit
(`studentize`); other metrics get the percentile interval. Its cost model
sizes the resamples to the time left (`Uncertainty.cost`).
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


def draw(
    rows: int,
    groups: pandas.Series | None,
    temporal: bool,
    rng: numpy.random.Generator,
) -> tuple[numpy.ndarray, numpy.ndarray]:
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
        positions drawn with replacement. Beside them, the independent unit
        of each position: each drawn copy of a group or block is its own
        unit, and each drawn row is its own unit.
    """
    if groups is not None:
        members = (
            groups.reset_index(drop=True).groupby(groups.to_numpy()).indices
        )
        labels = list(members)
        drawn = rng.integers(0, len(labels), len(labels))
        parts = [members[labels[i]] for i in drawn]
    elif temporal:
        length = max(5, int(numpy.sqrt(rows)))
        starts = rng.integers(0, max(rows - length + 1, 1), -(-rows // length))
        parts = [numpy.arange(s, min(s + length, rows)) for s in starts]
    else:
        return rng.integers(0, rows, rows), numpy.arange(rows)
    positions = numpy.concatenate(parts)
    units = numpy.repeat(numpy.arange(len(parts)), [len(p) for p in parts])
    kept = len(positions) if groups is not None else rows
    return positions[:kept], units[:kept]


def redraw(
    rows: int,
    groups: pandas.Series | None,
    temporal: bool,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Draws the row positions of one resample (`draw` without the units).

    Args:
        rows: Number of evaluation rows.
        groups: Entity labels, if rows are clustered by entity.
        temporal: Whether to draw contiguous blocks.
        rng: Source of randomness.

    Returns:
        The positions `draw` returns.
    """
    return draw(rows, groups, temporal, rng)[0]


def rank(
    values: numpy.ndarray, truth: numpy.ndarray
) -> tuple[float, numpy.ndarray] | None:
    """ROC AUC and each row's share of its sampling error (DeLong).

    Args:
        values: Scores of the positive class.
        truth: Whether each row is positive.

    Returns:
        The area (ties count one half) and each row's influence: for a
        positive row, its placement among the negatives minus the area,
        over the number of positives; likewise for a negative row. None
        when a class is absent.
    """
    positives, negatives = values[truth], values[~truth]
    if not len(positives) or not len(negatives):
        return None
    ordered = numpy.sort(negatives)
    wins = (
        numpy.searchsorted(ordered, positives, "left")
        + numpy.searchsorted(ordered, positives, "right")
    ) / (2 * len(negatives))
    ordered = numpy.sort(positives)
    losses = 1 - (
        numpy.searchsorted(ordered, negatives, "left")
        + numpy.searchsorted(ordered, negatives, "right")
    ) / (2 * len(positives))
    area = float(wins.mean())
    effects = numpy.empty(len(values))
    effects[truth] = (wins - area) / len(positives)
    effects[~truth] = (losses - area) / len(negatives)
    return area, effects


def square(
    values: numpy.ndarray, truth: numpy.ndarray
) -> tuple[float, numpy.ndarray] | None:
    """RMSE and each row's share of its sampling error (delta method).

    Args:
        values: Predictions.
        truth: Observed values.

    Returns:
        The RMSE and each row's influence, `(e^2 - MSE) / (2 n RMSE)`; None
        when the RMSE is zero (no spread to studentize by).
    """
    squares = (truth - values) ** 2
    mean = float(squares.mean())
    if mean <= 0:
        return None
    error = numpy.sqrt(mean)
    return float(error), (squares - mean) / (2 * len(values) * error)


# Metrics with a closed-form influence, so the interval is studentized.
# Measured with benchmarks/evidence/coverage.py (trials 0..299 per family,
# percentile -> studentized): ROC AUC on 60 groups 265 -> 279 of 300, RMSE
# on 300 rows 281 -> 286, ROC AUC on 300 rows 284 -> 281; over 882 trials
# 816 -> 836, 818 -> 827 and 848 -> 845. The percentile interval missed
# because its spread moves with the estimate; dividing each resample by its
# own standard error removes that.
LINEAR = {"roc_auc": rank, "rmse": square}
# The values each studentized metric can take; the interval is clipped to it.
RANGES = {"roc_auc": (0.0, 1.0), "rmse": (0.0, numpy.inf)}


def studentize(
    name: str,
    values: numpy.ndarray,
    truth: numpy.ndarray,
    units: numpy.ndarray,
) -> tuple[float, float] | None:
    """A metric and its standard error clustered by independent unit.

    Args:
        name: Metric name; only those in `LINEAR` are supported.
        values: Positive-class scores (ROC AUC) or predictions (RMSE).
        truth: Positive flags (ROC AUC) or observed values (RMSE).
        units: Independent unit of each row.

    Returns:
        The metric and `sqrt(K / (K - 1) * sum_k (sum of influences in unit
        k)^2)` over K units; None for an unsupported metric, a metric the
        rows cannot score, or fewer than two units.
    """
    found = LINEAR[name](values, truth) if name in LINEAR else None
    labels, codes = numpy.unique(units, return_inverse=True)
    if found is None or len(labels) < 2:  # noqa: PLR2004 -- K - 1 divides
        return None
    count = len(labels)
    sums = numpy.bincount(codes, weights=found[1])
    return found[0], float(numpy.sqrt(count / (count - 1) * (sums**2).sum()))


def divide(difference: float, error: float) -> float:
    """A resample's studentized difference.

    Args:
        difference: Resample metric minus the full-sample metric.
        error: The resample's standard error.

    Returns:
        `difference / error`; with a zero error, 0.0 for no difference and
        a signed infinity otherwise (which leaves the percentile interval).
    """
    if error > 0:
        return difference / error
    return (
        0.0 if difference == 0 else float(numpy.copysign(numpy.inf, difference))
    )


def expose(
    replay: Cache, target: pandas.Series, name: str
) -> tuple[numpy.ndarray, numpy.ndarray] | None:
    """The cached responses and truth a studentized metric reads.

    Args:
        replay: Cached predictions of the evaluation rows.
        target: Their labels, in the same order.
        name: Metric name.

    Returns:
        Predictions and values for `rmse`; positive-class probabilities and
        positive flags for binary `roc_auc`; None otherwise.
    """
    if name == "rmse":
        return replay.predictions.to_numpy(float), target.to_numpy(float)
    classes = replay.classes_
    if name != "roc_auc" or replay.probabilities is None or classes is None:
        return None
    if len(classes) != 2:  # noqa: PLR2004 -- binary only
        return None
    return (
        replay.probabilities.to_numpy()[:, 1],
        (target == classes[1]).to_numpy(),
    )


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

        ROC AUC (binary) and RMSE, resampled by rows or groups, get the
        studentized (bootstrap-t) interval: each resample is scored in
        closed form (`studentize`) together with its standard error, which
        is also cheaper than the scorer. Other metrics, and temporal
        blocks, get the percentile interval of the scorer's values.

        The number of resamples is `replicates`, cut to what `cost` says
        fits before the deadline; below `minimum_resamples` (too few for an
        interval that is not too narrow) the bootstrap is skipped, and the
        deadline is checked again before each resample.

        Args:
            inputs: The fitted candidate and evaluation data.

        Returns:
            The metric name, its point estimate, the interval of `bound`
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
        cache = replay_class(inputs.estimator, features)
        # Scorers only call predict and predict_proba on the replay.
        replay = cast("models.Estimator", cache)
        metric, positive = inputs.metric, inputs.positive_label
        estimate = metric.score(replay, features, target, positive)
        groups = (
            inputs.groups.reset_index(drop=True)
            if inputs.groups is not None
            else None
        )
        rng = numpy.random.default_rng(inputs.seed)
        scores: list[float] = []
        ratios: list[float] = []
        unit = (
            "groups"
            if groups is not None
            else "blocks"
            if inputs.temporal
            else "rows"
        )
        # Blocks overlap and the full sample has no blocks to cluster by, so
        # a temporal bootstrap keeps the percentile interval.
        exposed = (
            None if inputs.temporal else expose(cache, target, metric.name)
        )
        centre = (
            None
            if exposed is None
            else studentize(
                metric.name,
                *exposed,
                numpy.arange(len(target))
                if groups is None
                else groups.to_numpy(),
            )
        )
        each = self.cost(inputs)
        planned = self.replicates(len(target))
        if inputs.deadline is not None:
            planned = min(planned, max(int(inputs.left() // each), 0))
        attempted = 0
        if planned >= self.minimum_resamples:
            for replicate in range(planned):  # noqa: B007
                if inputs.expired():
                    break
                rows, units = draw(len(target), groups, inputs.temporal, rng)
                attempted += 1
                if exposed is not None and centre is not None:
                    found = studentize(
                        metric.name, exposed[0][rows], exposed[1][rows], units
                    )
                    if found is not None:
                        scores.append(found[0])
                        ratios.append(divide(found[0] - centre[0], found[1]))
                    continue
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
            interval = self.bound(scores, ratios, centre, metric.name)
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

    def bound(
        self,
        scores: list[float],
        ratios: list[float],
        centre: tuple[float, float] | None,
        name: str,
    ) -> list[float]:
        """The interval of `confidence` percent from the resamples.

        Args:
            scores: The metric on each kept resample.
            ratios: For a studentized metric, each resample's
                `(score - estimate) / its standard error`; else empty.
            centre: The full-sample metric and its standard error, or None.
            name: Metric name.

        Returns:
            The bootstrap-t interval `estimate - q_high * error, estimate -
            q_low * error` clipped to `RANGES`, when `centre` has a positive
            error and every ratio is finite; otherwise the percentile
            interval of `scores`.
        """
        tail = (100 - self.confidence) / 2
        low, high = numpy.percentile(scores, [tail, 100 - tail])
        if (
            centre is not None
            and centre[1] > 0
            and numpy.isfinite(ratios).all()
        ):
            top, bottom = numpy.percentile(ratios, [100 - tail, tail])
            floor, ceiling = RANGES[name]
            low = max(centre[0] - top * centre[1], floor)
            high = min(centre[0] - bottom * centre[1], ceiling)
        return [float(low), float(high)]
