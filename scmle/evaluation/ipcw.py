"""Inverse-probability-of-censoring weights that may depend on covariates.

The weights in `scmle.evaluation.survival` come from one Kaplan-Meier curve of
the censoring times, which is right only when censoring does not depend on
the covariates. When it does, the weights are biased and so are the scores
built on them. Here the censoring survival function of row `i` is
`G(t | x_i)`, fitted by a Cox model for the censoring time (the censoring
event plays the role of the event), or by the marginal curve when the charter
declares censoring independent of covariates. Every weight is floored and the
number of floored weights is returned, never hidden.

`truncated_concordance` is the IPCW concordance of Uno et al. (2011),
truncated at a horizon `tau` that the charter must fix: statistics cannot
choose it. The estimator's form here is written from memory of that paper and
is checked by simulation and against the Harrell index when nothing is
censored, not against a reference implementation; it has not been compared
with scikit-survival.
"""

from collections.abc import Callable
import dataclasses

import numpy

from scmle.models.survival import estimators

MIN_CENSORING_PROBABILITY = 0.05


@dataclasses.dataclass(frozen=True)
class CensoringWeights:
    """Inverse-probability-of-censoring weights at each row's own time.

    Attributes:
        weights: `1 / max(G(T_i | x_i), floor)` per row.
        floor: The lower bound applied to `G`.
        n_floored: Rows whose `G` was below the floor.
        covariate_dependent: Whether the censoring model used covariates.
        survival_of: `(time, rows) -> G(time | x_j)` for the given row
            positions, used for the pair weights of the concordance.
    """

    weights: numpy.ndarray
    floor: float
    n_floored: int
    covariate_dependent: bool
    survival_of: Callable[[float, numpy.ndarray], numpy.ndarray] | None = None


def censoring_weights(
    duration: numpy.ndarray,
    event: numpy.ndarray,
    features: numpy.ndarray | None = None,
    floor: float = MIN_CENSORING_PROBABILITY,
    penalty: float = 1e-3,
) -> CensoringWeights:
    """Fits the censoring distribution and returns the weights.

    Args:
        duration: Observed times.
        event: 1 where the event was observed (0 means censored).
        features: Covariates for a covariate-dependent censoring model, or
            None for the marginal Kaplan-Meier curve.
        floor: Lower bound on the censoring survival probability.
        penalty: Ridge strength of the Cox censoring model.

    Returns:
        The weights `1 / max(G, floor)` per row, the floor, the number of
        floored rows, whether covariates were used, and the function giving
        `G` at another time for chosen rows.

    Raises:
        ValueError: If `floor` is not strictly between 0 and 1, or
            `duration`, `event` and `features` differ in length.
    """
    if not 0.0 < floor < 1.0:
        raise ValueError("floor must lie strictly between 0 and 1")
    time = numpy.asarray(duration, dtype=float)
    seen = numpy.asarray(event, dtype=float)
    if len(time) != len(seen):
        raise ValueError("duration and event differ in length")
    censored = 1.0 - seen
    if features is None:
        curve = estimators.kaplan_meier(time, censored)
        raw = estimators.step_lookup(curve[0], curve[1], time, 1.0)

        def marginal_at(moment: float, rows: numpy.ndarray) -> numpy.ndarray:
            level = estimators.step_lookup(
                curve[0], curve[1], numpy.array([moment]), 1.0
            )[0]
            return numpy.full(len(rows), level)

        survival_of = marginal_at
    else:
        matrix = numpy.asarray(features, dtype=float)
        if len(matrix) != len(time):
            raise ValueError("features and duration differ in length")
        model = estimators.CoxRegressor(penalty).fit(matrix, time, censored)
        cumulative = estimators.step_lookup(
            model.moments_, model.baseline_, time, 0.0
        )
        relative = numpy.exp(model.predict(matrix) - model.scale_)
        raw = numpy.exp(-cumulative * relative)

        def cox_at(moment: float, rows: numpy.ndarray) -> numpy.ndarray:
            level = estimators.step_lookup(
                model.moments_, model.baseline_, numpy.array([moment]), 0.0
            )[0]
            survival: numpy.ndarray = numpy.exp(-level * relative[rows])
            return survival

        survival_of = cox_at
    floored = raw < floor
    return CensoringWeights(
        1.0 / numpy.maximum(raw, floor),
        floor,
        int(floored.sum()),
        features is not None,
        survival_of,
    )


def truncated_concordance(
    duration: numpy.ndarray,
    event: numpy.ndarray,
    risk: numpy.ndarray,
    tau: float,
    weights: CensoringWeights,
) -> float:
    """IPCW concordance truncated at `tau` (Uno et al. 2011).

    Pairs `(i, j)` with an observed event of `i` before `tau` and
    `T_i < T_j` are comparable. Row `j` must still be uncensored at `T_i`,
    so the pair weight is `1 / (G(T_i | x_i) G(T_i | x_j))`; with a marginal
    curve that is Uno's `G(T_i)^-2`.

    Args:
        duration: Observed times.
        event: 1 where the event was observed.
        risk: Predicted risk (larger means earlier event).
        tau: Truncation time fixed in the charter.
        weights: Output of `censoring_weights` for the same rows.

    Returns:
        The concordance, NaN when no pair is comparable.
    """
    time = numpy.asarray(duration, dtype=float)
    seen = numpy.asarray(event, dtype=bool) & (time < tau)
    score = numpy.asarray(risk, dtype=float)
    numerator = 0.0
    denominator = 0.0
    for index in numpy.flatnonzero(seen):
        later = numpy.flatnonzero(time > time[index])
        if later.size == 0:
            continue
        if weights.survival_of is None:
            partner = numpy.full(later.size, 1.0 / weights.weights[index])
        else:
            partner = numpy.maximum(
                weights.survival_of(float(time[index]), later), weights.floor
            )
        pair = weights.weights[index] / partner
        ties = score[index] == score[later]
        wins = (score[index] > score[later]) + 0.5 * ties
        denominator += float(pair.sum())
        numerator += float((pair * wins).sum())
    return numerator / denominator if denominator else float("nan")
