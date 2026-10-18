"""A/B test analysis: two models served to different requests.

Unlike champion-challenger (both score the same rows), each request here is
answered by exactly one model, so the two score different rows and the
difference needs an interval from independent resampling of the two arms.
The verdict is `adopt` only when the whole interval favours the treatment,
`keep` when it clearly favours the control, and `inconclusive` otherwise or
when an arm has too few rows to say.
"""

import dataclasses
from typing import Any

import numpy as np
import pandas as pd

from scml.contracts import tasks
from scml.evaluation import metrics
from scml.evaluation import replay

BOOTSTRAPS = 300
MIN_ROWS = 100


@dataclasses.dataclass(frozen=True)
class Arm:
    """What one model served and what then happened.

    Attributes:
        predictions: Predicted labels or values for the arm's requests.
        outcomes: True outcomes aligned with the predictions.
        probabilities: Class probabilities, if the model has them.
        classes: Class labels in probability-column order.
    """

    predictions: np.ndarray
    outcomes: np.ndarray
    probabilities: np.ndarray | None = None
    classes: np.ndarray | None = None


@dataclasses.dataclass(frozen=True)
class ABResult:
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
    """

    metric: str
    control_score: float
    treatment_score: float
    advantage: float
    interval: tuple[float, float]
    control_rows: int
    treatment_rows: int
    decision: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        return dataclasses.asdict(self)


def score_arm(
    arm: Arm,
    rows: np.ndarray,
    metric: Any,
    task: tasks.TaskType,
) -> float:
    """The metric on a subset of an arm's requests (natural units)."""
    outcomes = pd.Series(arm.outcomes[rows])
    classification = task.is_classification
    stand_in = (
        replay.ClassifierPredictions(
            arm.predictions[rows],
            None if arm.probabilities is None else arm.probabilities[rows],
            arm.classes,
        )
        if classification
        else replay.RegressorPredictions(arm.predictions[rows])
    )
    positive = None
    if task is tasks.TaskType.BINARY and arm.classes is not None:
        positive = arm.classes[1]
    return float(
        metric.score(
            stand_in, pd.DataFrame(index=outcomes.index), outcomes, positive
        )
    )


def compare_arms(
    task: tasks.TaskType,
    metric_name: str,
    control: Arm,
    treatment: Arm,
    bootstraps: int = BOOTSTRAPS,
    seed: int = 0,
    min_rows: int = MIN_ROWS,
) -> ABResult:
    """Compares two arms with independent bootstrap resampling.

    Args:
        task: Resolved task type.
        metric_name: Metric to compare on.
        control: The model currently serving.
        treatment: The model being tried.
        bootstraps: Resamples of each arm.
        seed: Random seed.
        min_rows: Fewest requests per arm for a verdict.

    Returns:
        The result with its 95% interval and decision.
    """
    metric = metrics.CATALOG.resolve(metric_name, task)
    sign = 1.0 if metric.greater_is_better else -1.0
    every = {
        "control": np.arange(len(control.outcomes)),
        "treatment": np.arange(len(treatment.outcomes)),
    }
    control_score = score_arm(control, every["control"], metric, task)
    treatment_score = score_arm(treatment, every["treatment"], metric, task)
    gains = []
    for draw in range(bootstraps):
        rng = np.random.default_rng([seed, draw])
        first = rng.integers(0, len(every["control"]), len(every["control"]))
        second = rng.integers(
            0, len(every["treatment"]), len(every["treatment"])
        )
        try:
            gains.append(
                sign
                * (
                    score_arm(treatment, second, metric, task)
                    - score_arm(control, first, metric, task)
                )
            )
        except ValueError:  # A resample missing a class.
            continue
    low, high = np.percentile(gains, [2.5, 97.5])
    thin = min(len(every["control"]), len(every["treatment"])) < min_rows
    if thin:
        decision = "inconclusive"
    elif low > 0:
        decision = "adopt"
    elif high < 0:
        decision = "keep"
    else:
        decision = "inconclusive"
    return ABResult(
        metric.name,
        control_score,
        treatment_score,
        sign * (treatment_score - control_score),
        (float(low), float(high)),
        len(every["control"]),
        len(every["treatment"]),
        decision,
    )
