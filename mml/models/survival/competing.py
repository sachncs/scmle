"""Competing risks: more than one kind of event can end follow-up.

When a patient may die of cause A or cause B, treating B as censoring while
studying A overstates how often A happens (the Kaplan-Meier complement counts
people who could no longer have had A). Two tools are provided:

* `cumulative_incidence`: the nonparametric Aalen-Johansen estimate of the
  probability of each cause by time `t`, with the other causes competing.
* `CauseSpecificCox`: one Cox model per cause (other causes count as
  censoring for that model). Cumulative incidence for a row combines every
  cause's hazard: `CIF_k(t) = sum_{t_j <= t} dH_k(t_j) S(t_j-)`, with
  `S = exp(-sum_k H_k)`.

Event codes are 0 for censored and 1, 2, ... for causes.
"""

from typing import Any
from typing import Self

import numpy as np
import pandas as pd

from mml.models.survival import estimators


def cumulative_incidence(
    duration: Any, cause: Any
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Aalen-Johansen cumulative incidence of each cause.

    Args:
        duration: Follow-up time of each row.
        cause: 0 when censored, otherwise the cause code.

    Returns:
        Cause code to (event times, probability of that cause by each time).
    """
    time = np.asarray(duration, dtype=float)
    code = np.asarray(cause, dtype=int)
    causes = sorted(int(c) for c in set(code) if c != 0)
    moments = np.unique(time[code != 0])
    survival = 1.0
    incidence = dict.fromkeys(causes, 0.0)
    curves: dict[int, list[float]] = {c: [] for c in causes}
    for moment in moments:
        at_risk = float((time >= moment).sum())
        here = time == moment
        total = float((here & (code != 0)).sum())
        for c in causes:
            happened = float((here & (code == c)).sum())
            incidence[c] += survival * happened / at_risk
            curves[c].append(incidence[c])
        survival *= 1.0 - total / at_risk
    return {c: (moments, np.asarray(curves[c])) for c in causes}


class CauseSpecificCox:
    """Cause-specific Cox models with a combined cumulative incidence.

    Attributes:
        penalty: Ridge strength of every cause's model.
        causes_: Cause codes seen in training.
        models_: One fitted `CoxRegressor` per cause.
    """

    def __init__(self, penalty: float = 1.0) -> None:
        """Stores the settings."""
        self.penalty = penalty

    def fit(self, features: Any, duration: Any, cause: Any) -> Self:
        """Fits one Cox model per cause, others counted as censored."""
        code = np.asarray(cause, dtype=int)
        self.causes_ = sorted(int(c) for c in set(code) if c != 0)
        if not self.causes_:
            raise ValueError("no event of any cause was observed")
        self.models_ = {
            c: estimators.CoxRegressor(self.penalty).fit(
                features, duration, code == c
            )
            for c in self.causes_
        }
        return self

    def cumulative_hazard(
        self, features: Any, times: np.ndarray
    ) -> dict[int, np.ndarray]:
        """Cumulative cause-specific hazard, rows by times."""
        result = {}
        for c, model in self.models_.items():
            base = estimators.step_lookup(
                model.moments_, model.baseline_, times, 0.0
            )
            relative = np.exp(model.predict(features) - model.scale_)
            result[c] = np.outer(relative, base)
        return result

    def predict_incidence(
        self, features: Any, times: Any, cause: int
    ) -> np.ndarray:
        """Probability of `cause` by each time for each row (rows by times)."""
        if cause not in self.models_:
            raise ValueError(f"unknown cause {cause}; seen {self.causes_}")
        grid = np.unique(
            np.concatenate([m.moments_ for m in self.models_.values()])
        )
        hazards = self.cumulative_hazard(features, grid)
        total = np.sum(list(hazards.values()), axis=0)
        survival_before = np.exp(
            -np.concatenate(
                [np.zeros((total.shape[0], 1)), total[:, :-1]], axis=1
            )
        )
        step = np.diff(
            np.concatenate(
                [np.zeros((total.shape[0], 1)), hazards[cause]], axis=1
            ),
            axis=1,
        )
        running = np.cumsum(step * survival_before, axis=1)
        wanted = np.asarray(times, dtype=float)
        position = np.searchsorted(grid, wanted, side="right") - 1
        padded = np.concatenate(
            [np.zeros((running.shape[0], 1)), running], axis=1
        )
        return padded[:, position + 1]

    def predict(self, features: Any, horizon: float, cause: int) -> np.ndarray:
        """Risk score: probability of `cause` by `horizon`."""
        return self.predict_incidence(features, [horizon], cause)[:, 0]


def naive_complement(duration: Any, cause: Any, target: int) -> pd.Series:
    """Kaplan-Meier `1 - S` that wrongly treats other causes as censored.

    Provided to show the bias that `cumulative_incidence` removes.
    """
    code = np.asarray(cause, dtype=int)
    times, survival = estimators.kaplan_meier(
        np.asarray(duration, dtype=float), code == target
    )
    return pd.Series(1.0 - survival, index=times)
