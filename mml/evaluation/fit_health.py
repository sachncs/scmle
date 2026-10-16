"""Is a candidate under-trained, over-trained, or unstable?

The verdict compares what the model gained over the trivial baseline on the
training folds with what it kept on the validation folds. Everything is
measured relative to the baseline so the same thresholds work for AUC, log
loss and RMSE alike. Scores must already be oriented so larger is better.
"""

from collections.abc import Sequence
import dataclasses
import math
from typing import Any

import numpy as np

OVERFIT_LOSS_SHARE = 0.25
UNDERFIT_RELATIVE_GAIN = 0.05
UNSTABLE_NOISE_SHARE = 0.5
TINY = 1e-12


@dataclasses.dataclass(frozen=True)
class FitHealth:
    """The assessment of one candidate.

    Attributes:
        verdict: `healthy`, `overfit`, `underfit` or `unstable`.
        train_gain: Mean training-fold gain over the baseline.
        validation_gain: Mean validation-fold gain over the baseline.
        retained: Share of the training gain kept on validation (1 means
            none was lost).
        noise_share: Validation fold-to-fold spread relative to the gain.
        note: What the verdict means and what could be done.
    """

    verdict: str
    train_gain: float
    validation_gain: float
    retained: float
    noise_share: float
    note: str

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-serializable mapping."""
        return dataclasses.asdict(self)


def assess(
    train_scores: Sequence[float],
    validation_scores: Sequence[float],
    baseline_score: float,
) -> FitHealth:
    """Judges a candidate from its per-fold scores.

    Args:
        train_scores: Fold scores on the rows each model was fitted on.
        validation_scores: Fold scores on the held-out rows.
        baseline_score: Score of the trivial constant model on the same
            folds, same orientation.

    Returns:
        The verdict with the numbers behind it.
    """
    train = float(np.mean(train_scores)) - baseline_score
    held_out = float(np.mean(validation_scores)) - baseline_score
    spread = (
        float(np.std(validation_scores, ddof=1))
        if (len(validation_scores) > 1)
        else 0.0
    )
    retained = held_out / train if train > TINY else 1.0
    noise = spread / abs(held_out) if abs(held_out) > TINY else math.inf
    scale = max(abs(baseline_score), TINY)
    if held_out <= UNDERFIT_RELATIVE_GAIN * scale and train <= (
        UNDERFIT_RELATIVE_GAIN * scale * 2
    ):
        verdict, note = (
            "underfit",
            "neither training nor validation clearly beats the baseline; "
            "try more capacity, more training or better features",
        )
    elif train > TINY and retained < 1.0 - OVERFIT_LOSS_SHARE:
        verdict, note = (
            "overfit",
            f"only {max(retained, 0.0):.0%} of the training gain survives on "
            "validation; try stronger regularisation, fewer features, "
            "early stopping or more data",
        )
    elif noise > UNSTABLE_NOISE_SHARE:
        verdict, note = (
            "unstable",
            "the fold-to-fold spread is large relative to the gain, so the "
            "ranking against other candidates is unreliable",
        )
    else:
        verdict, note = "healthy", "training gain carries over to validation"
    return FitHealth(verdict, train, held_out, retained, noise, note)
