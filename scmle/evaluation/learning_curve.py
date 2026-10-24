"""Would more data help? A learning curve for the selected candidate.

The candidate is fitted on growing fractions of one cross-validation fold's
training rows and scored on that fold's held-out rows. If the score is still
climbing when the data runs out, collecting more rows is likely to pay; if it
has flattened, effort is better spent on features or on a different model.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
import pandas as pd
from sklearn import base

FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
MIN_POINTS = 3
MAX_CLASSES = 20
MIN_TRAIN_ROWS = 100
CLIMBING_SHARE = 0.25
FLAT_TOLERANCE = 0.002


@dataclasses.dataclass(frozen=True)
class LearningCurve:
    """Scores at growing training sizes.

    Attributes:
        sizes: Training rows used at each point.
        scores: Held-out score at each point, larger is better.
        verdict: `more data likely helps`, `plateau` or `unclear`.
        last_gain: Score gained over the last step.
        total_gain: Score gained from the first point to the last.
    """

    sizes: tuple[int, ...]
    scores: tuple[float, ...]
    verdict: str
    last_gain: float
    total_gain: float

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        record = dataclasses.asdict(self)
        record["sizes"] = list(self.sizes)
        record["scores"] = list(self.scores)
        return record


def judge(scores: Sequence[float]) -> tuple[str, float, float]:
    """Classifies a curve from its scores (larger is better).

    The curve is climbing when the last step still earns a meaningful share
    of what the whole curve earned per step; it is flat when the whole curve
    moves by less than a tiny tolerance, and unclear when more data made the
    score worse.
    """
    total = float(scores[-1] - scores[0])
    last = float(scores[-1] - scores[-2])
    if total < -FLAT_TOLERANCE:
        return "unclear", last, total
    if total <= FLAT_TOLERANCE:
        return "plateau", last, total
    average_step = total / (len(scores) - 1)
    if last > CLIMBING_SHARE * average_step and last > 0:
        return "more data likely helps", last, total
    return "plateau", last, total


def compute(
    candidate: Any,
    features: pd.DataFrame,
    target: pd.Series,
    fit_positions: Sequence[int],
    evaluate_positions: Sequence[int],
    scorer: Callable[..., float],
    greater_is_better: bool,
    should_stop: Callable[[], bool] = lambda: False,
) -> LearningCurve | None:
    """Fits the candidate on growing slices of one fold's training rows.

    Args:
        candidate: Unfitted candidate pipeline.
        features: Development features.
        target: Development labels.
        fit_positions: Training positions of one fold.
        evaluate_positions: Held-out positions of that fold.
        scorer: scikit-learn scorer (sign-adjusted, larger is better).
        greater_is_better: The metric's orientation, to report natural units.
        should_stop: Called before each fit; True abandons the curve.

    Returns:
        The curve, or None when there are too few rows or time ran out.
    """
    fit = np.asarray(fit_positions)
    if len(fit) < MIN_TRAIN_ROWS:
        return None
    held = np.asarray(evaluate_positions)
    sizes, scores = [], []
    distinct = target.iloc[fit].nunique()
    needed = distinct if distinct <= MAX_CLASSES else 2
    for fraction in FRACTIONS:
        if should_stop():
            return None
        used = fit[: max(int(len(fit) * fraction), 2)]
        if target.iloc[used].nunique() < needed:
            continue  # An ordered prefix can miss classes.
        fitted = base.clone(candidate).fit(
            features.iloc[used], target.iloc[used]
        )
        scores.append(
            float(scorer(fitted, features.iloc[held], target.iloc[held]))
        )
        sizes.append(len(used))
    if len(scores) < MIN_POINTS:
        return None
    verdict, last, total = judge(scores)
    sign = 1.0 if greater_is_better else -1.0
    return LearningCurve(
        tuple(sizes),
        tuple(sign * s for s in scores),
        verdict,
        sign * last,
        sign * total,
    )
