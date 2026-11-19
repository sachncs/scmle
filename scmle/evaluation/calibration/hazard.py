"""Calibration of predicted survival curves against Kaplan-Meier."""

from typing import Any

import numpy

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.evaluation.calibration import verdict
from scmle.models.survival import estimators

QUARTILES = (0.25, 0.5, 0.75)
"""Default survival horizons, as quantiles of the observed event times:
early, median and late follow-up, each with a quarter of the events beyond
or before it to estimate from."""


def meier(
    duration: numpy.ndarray, event: numpy.ndarray, moment: float
) -> tuple[float, float]:
    """Kaplan-Meier survival at one time, with Greenwood's variance.

    The point estimate is `estimators.kaplan_meier` read at `moment`.

    Args:
        duration: Follow-up time of each row.
        event: 1 where the event was observed, 0 where censored.
        moment: The time.

    Returns:
        The estimate and its variance `S^2 * sum d / (n (n - d))` over the
        event times up to `moment`; zero variance once the curve reaches 0.
    """
    time = numpy.asarray(duration, dtype=float)
    seen = numpy.asarray(event).astype(bool)
    moments, curve = estimators.kaplan_meier(time, seen)
    estimate = float(
        estimators.stair(moments, curve, numpy.array([moment]), 1.0)[0]
    )
    if estimate == 0.0:
        return 0.0, 0.0
    early = moments[moments <= moment]
    ordered = numpy.sort(time)
    risk = len(ordered) - numpy.searchsorted(ordered, early, side="left")
    deaths = numpy.sort(time[seen])
    counts = numpy.searchsorted(
        deaths, early, side="right"
    ) - numpy.searchsorted(deaths, early, side="left")
    return estimate, estimate**2 * float(
        numpy.sum(counts / (risk * (risk - counts)))
    )


def quartiles(duration: Any, event: Any) -> numpy.ndarray:
    """Default horizons: the `QUARTILES` of the observed event times.

    Not `scmle.evaluation.survival.horizons`, which spreads a grid over the
    10-90% quantiles of every follow-up time.

    Args:
        duration: Follow-up time of each row.
        event: 1 where the event was observed.

    Returns:
        The distinct horizons, ascending.

    Raises:
        InputError: If no event was observed.
    """
    time = verdict.vector(duration, "durations")
    seen = verdict.vector(event, "events", len(time)).astype(bool)
    if not seen.any():
        raise errors.InputError("no event observed: no horizon to test")
    return numpy.unique(numpy.quantile(time[seen], QUARTILES))


def horizon(
    time: numpy.ndarray,
    seen: numpy.ndarray,
    predicted: numpy.ndarray,
    moment: float,
) -> verdict.Check:
    """Tests the predicted event-free probability at one horizon.

    Groups need `EXPECTED` predicted events and `EXPECTED` rows still
    followed at the horizon. A group's variance is Greenwood's, but never
    below the binomial variance `S (1 - S) / n` of an uncensored group,
    which censoring can only raise.

    Args:
        time: Follow-up time of each row.
        seen: 1 where the event was observed.
        predicted: Predicted event-free probability at the horizon.
        moment: The horizon.

    Returns:
        The test of this horizon.
    """

    def enough(members: numpy.ndarray) -> bool:
        events = len(members) * (1 - predicted[members].mean())
        followed = (time[members] >= moment).sum()
        return bool(events >= verdict.EXPECTED and followed >= verdict.EXPECTED)

    found = []
    for members in verdict.partition(
        numpy.argsort(-predicted, kind="stable"), enough
    ):
        estimate, variance = meier(time[members], seen[members], moment)
        mean = float(predicted[members].mean())
        found.append(
            verdict.Group(
                count=len(members),
                predicted=mean,
                observed=estimate,
                variance=max(variance, mean * (1 - mean) / len(members)),
            )
        )
    return verdict.Check(f"horizon {moment:g}", tuple(found))


def survival(
    duration: Any,
    event: Any,
    curve: Any,
    times: Any = None,
    level: float = verdict.LEVEL,
) -> verdict.Claim:
    """Tests predicted event-free probabilities at each horizon.

    Args:
        duration: Follow-up time of each row.
        event: 1 where the event was observed, 0 where censored.
        curve: Predicted event-free probability, one column per horizon.
        times: The horizons of the columns; `quartiles(duration, event)`
            when None, and `curve` must then have been read at those.
        level: Family-wise size of the tests.

    Returns:
        The survival claim, one check per horizon.

    Raises:
        InputError: If the shapes disagree or a probability is outside
            `[0, 1]`.
    """
    time = verdict.vector(duration, "durations")
    seen = verdict.vector(event, "events", len(time))
    moments = (
        quartiles(time, seen)
        if times is None
        else verdict.vector(times, "times")
    )
    matrix = verdict.probable(numpy.asarray(curve, dtype=float), "survival")
    if matrix.shape != (len(time), len(moments)):
        raise errors.InputError(
            "survival needs one row per duration and one column per horizon"
        )
    checks = tuple(
        horizon(time, seen, matrix[:, j], float(moment))
        for j, moment in enumerate(moments)
    )
    return verdict.Claim(
        tasks.Task.SURVIVAL,
        verdict.STATEMENTS[tasks.Task.SURVIVAL],
        checks,
        (),
        level,
    )
