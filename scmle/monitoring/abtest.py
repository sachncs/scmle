"""A/B test analysis: two models served to different requests.

Unlike champion-challenger (both score the same rows), each request here is
answered by exactly one model, so the two score different rows and the
difference needs an interval from independent resampling of the two arms.
The verdict is `adopt` only when the whole interval favours the treatment,
`keep` when it clearly favours the control, and `inconclusive` otherwise or
when an arm has too few rows to say.
"""

import dataclasses
import math
from typing import Any
from typing import ClassVar

import numpy
import pandas

from scmle.contracts import records
from scmle.contracts import tasks
from scmle.evaluation import metrics
from scmle.evaluation import replay

BOOTSTRAPS = 300
MIN_ROWS = 100
# Share of the planned resamples that must be scorable for an interval.
# Policy: below half, the percentiles come from a different, selected set of
# resamples (those that kept every class), so no interval is reported.
KEPT = 0.5


@dataclasses.dataclass(frozen=True)
class Arm:
    """What one model served and what then happened.

    Attributes:
        predictions: Predicted labels or values for the arm's requests.
        outcomes: True outcomes aligned with the predictions.
        probabilities: Class probabilities, if the model has them.
        classes: Class labels in probability-column order.
    """

    predictions: numpy.ndarray
    outcomes: numpy.ndarray
    probabilities: numpy.ndarray | None = None
    classes: numpy.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class Comparison:
    """Outcome of an A/B comparison.

    Attributes:
        metric: Metric used.
        control_score: Score of the control arm.
        treatment_score: Score of the treatment arm.
        advantage: Treatment minus control in the metric's favour.
        interval: 95% bootstrap interval of the advantage.
        control_rows: Requests the control served.
        treatment_rows: Requests the treatment served.
        decision: `adopt`, `keep` or `inconclusive`.
        dropped: Resamples that could not be scored (a resample missing a
            class) and are not in the interval.
        KEYS: Version-1 keys to the written ones (`scores.control`,
            `scores.treatment`, `rows.control`, `rows.treatment`);
            `downgrade` restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.nest(
        {},
        groups={
            "scores": {
                "control_score": "control",
                "treatment_score": "treatment",
            },
            "rows": {"control_rows": "control", "treatment_rows": "treatment"},
        },
    )

    metric: str
    control_score: float
    treatment_score: float
    advantage: float
    interval: tuple[float, float]
    control_rows: int
    treatment_rows: int
    decision: str
    dropped: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready record.

        Returns:
            A dict of all fields, the arm scores under `scores` and the arm
            sizes under `rows`.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


def tally(
    arm: Arm,
    rows: numpy.ndarray,
    metric: Any,
    task: tasks.Task,
) -> float:
    """Scores a subset of an arm's requests.

    Args:
        arm: The arm.
        rows: Positions of the requests to score.
        metric: The metric, from the catalog.
        task: Resolved task type.

    Returns:
        The metric on those requests, in natural units.
    """
    outcomes = pandas.Series(arm.outcomes[rows])
    classification = task.is_classification
    stand_in = (
        replay.Predictions(
            arm.predictions[rows],
            None if arm.probabilities is None else arm.probabilities[rows],
            arm.classes,
        )
        if classification
        else replay.Replayed(arm.predictions[rows])
    )
    positive = None
    if task is tasks.Task.BINARY and arm.classes is not None:
        positive = arm.classes[1]
    return float(
        metric.score(
            stand_in, pandas.DataFrame(index=outcomes.index), outcomes, positive
        )
    )


def compare_arms(
    task: tasks.Task,
    metric_name: str,
    control: Arm,
    treatment: Arm,
    bootstraps: int = BOOTSTRAPS,
    seed: int = 0,
    min_rows: int = MIN_ROWS,
) -> Comparison:
    """Compares two arms with independent bootstrap resampling.

    Resamples that miss a class are skipped.

    Args:
        task: Resolved task type.
        metric_name: Metric to compare on.
        control: The model currently serving.
        treatment: The model being tried.
        bootstraps: Resamples of each arm.
        seed: Random seed.
        min_rows: Fewest requests per arm for a verdict.

    Returns:
        The result with its 95% interval and decision: `adopt` if the whole
        interval is above zero, `keep` if it is below, otherwise (or if an arm
        has fewer than `min_rows` requests) `inconclusive`. Resamples that
        cannot be scored are counted in `dropped`; when fewer than `KEPT` of
        them remain the interval is NaN and the decision `inconclusive`.
    """
    metric = metrics.CATALOG.resolve(metric_name, task)
    sign = 1.0 if metric.greater_is_better else -1.0
    every = {
        "control": numpy.arange(len(control.outcomes)),
        "treatment": numpy.arange(len(treatment.outcomes)),
    }
    control_score = tally(control, every["control"], metric, task)
    treatment_score = tally(treatment, every["treatment"], metric, task)
    gains = []
    for draw in range(bootstraps):
        rng = numpy.random.default_rng([seed, draw])
        first = rng.integers(0, len(every["control"]), len(every["control"]))
        second = rng.integers(
            0, len(every["treatment"]), len(every["treatment"])
        )
        try:
            gains.append(
                sign
                * (
                    tally(treatment, second, metric, task)
                    - tally(control, first, metric, task)
                )
            )
        except (
            ValueError
        ):  # handled: counted in `dropped`; too many make it inconclusive
            continue
    dropped = bootstraps - len(gains)
    scant = len(gains) < bootstraps * KEPT
    low, high = (
        (math.nan, math.nan) if scant else numpy.percentile(gains, [2.5, 97.5])
    )
    thin = min(len(every["control"]), len(every["treatment"])) < min_rows
    if thin or scant or low <= 0 <= high:
        decision = "inconclusive"
    else:
        decision = "adopt" if low > 0 else "keep"
    return Comparison(
        metric.name,
        control_score,
        treatment_score,
        sign * (treatment_score - control_score),
        (float(low), float(high)),
        len(every["control"]),
        len(every["treatment"]),
        decision,
        dropped,
    )
