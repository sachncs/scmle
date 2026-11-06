"""Differential missingness of outcomes in a randomized rollout (F11).

A rollout whose outcomes go missing more often in one arm cannot be read
from the observed units alone: the observed arms are no longer comparable.
Under randomization, if an outcome is missing independently of the arm, the
units whose outcome is missing are themselves assigned to the challenger
with the recorded probability `q`. So the sample-ratio Bayes factor of
`design.srm_log_bayes_factor`, applied to the units whose outcome is missing,
is a likelihood-ratio martingale under "missingness does not depend on the
arm", and by Ville's inequality `P(sup BF >= 1 / level) <= level` however
often it is read.

What this does and does not do. It detects missingness that depends on the
arm. It does not remove the bias of missingness that is balanced across arms
but depends on the outcome, and it does not correct an estimate: the rollout
rule still analyses the observed units only. A flag is a description with the
label `missingness_not_handled` (the label the F11 plan names); the rollout's
promote rule is unchanged, so a user who sees the flag decides. Its false
alarm rate is the Ville bound, checked by simulation in
`tests/unit/test_decision_missing.py`; no certificate for its power exists,
so it carries no decision.
"""

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from scmle.decision import design

LABEL = "missingness_not_handled"
NOTE = (
    "the check detects missingness that depends on the arm; the rollout "
    "does not handle missing outcomes, so a flag means the observed arms "
    "may not be comparable and a person decides"
)


def differential_missingness(
    arm: ArrayLike,
    observed: ArrayLike,
    assignment_prob: float,
    level: float,
) -> dict[str, Any]:
    """Whether outcomes are missing differently by arm, read at every unit.

    Args:
        arm: 1 for the challenger, 0 for the champion, in unit order.
        observed: Whether the unit's outcome is available.
        assignment_prob: The recorded probability of the challenger.
        level: Level of the anytime-valid test, in (0, 1).

    Returns:
        `differential` (whether the Bayes factor ever reached `1 / level`),
        the log Bayes factor at the end, the number of missing units, the
        unit at which it first crossed (or None), and a description.

    Raises:
        ValueError: On unequal lengths, arms other than 0 and 1, or a
            probability or level outside (0, 1).
    """
    a = np.asarray(arm, dtype=float)
    seen = np.asarray(observed, dtype=bool)
    if a.shape != seen.shape or a.ndim != 1:
        raise ValueError("arm and observed must be one-dimensional and equal")
    if np.any((a != 0.0) & (a != 1.0)):
        raise ValueError("arm must be 0 or 1")
    if not 0.0 < assignment_prob < 1.0 or not 0.0 < level < 1.0:
        raise ValueError("assignment_prob and level must lie in (0, 1)")
    gone = ~seen
    totals = np.cumsum(gone)
    challenger = np.cumsum(a * gone)
    positive = totals > 0
    log_bf = np.full(a.shape, -np.inf)
    if positive.any():
        log_bf[positive] = design.srm_log_bayes_factor(
            challenger[positive], totals[positive], assignment_prob
        )
    crossed = log_bf >= -np.log(level)
    first = int(np.argmax(crossed)) + 1 if crossed.any() else None
    return {
        "status": "described",
        "label": LABEL,
        "differential": bool(crossed.any()),
        "missing_units": int(gone.sum()),
        "log_bayes_factor": float(log_bf[-1])
        if a.size and positive.any()
        else 0.0,
        "first_unit": first,
        "level": level,
        "note": NOTE,
    }
