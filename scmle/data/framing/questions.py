"""The short list of questions only a person can answer (feature F8).

A question is raised only if the data and the request cannot fill it, a
tipping-point test shows it changes the verdict, and it has a closed form (a
number in a range, a date, a yes or no). The test varies the unknown over its
plausible range and recomputes the verdict. If the verdict never changes,
nothing is asked and the range is reported as robust; if it flips, the
question carries the tipping value.
"""

from collections.abc import Callable
from collections.abc import Collection
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from scmle.data.framing import baseline

GRID_POINTS = 41
BISECTION_WIDTH = 1e-12


@dataclasses.dataclass(frozen=True, slots=True)
class Unknown:
    """A quantity the request does not fix.

    Attributes:
        name: Slot name, for example `cost_threshold`.
        low: Lowest plausible value.
        high: Highest plausible value.
        form: The closed form of the answer asked for.
        verdict: Maps a value to a verdict label.
    """

    name: str
    low: float
    high: float
    form: str
    verdict: Callable[[float], str]


@dataclasses.dataclass(frozen=True, slots=True)
class HumanQuestion:
    """A question raised for a named person.

    Attributes:
        slot: The unknown.
        form: The closed form of the answer.
        low: Lowest plausible value.
        high: Highest plausible value.
        tipping_value: Value at which the verdict changes.
        verdicts: The verdict at each end of the range.
    """

    slot: str
    form: str
    low: float
    high: float
    tipping_value: float
    verdicts: tuple[str, str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready record."""
        record = dataclasses.asdict(self)
        record["verdicts"] = list(self.verdicts)
        return record


def tipping_point(unknown: Unknown) -> HumanQuestion | None:
    """The question for `unknown`, or None if the verdict never changes."""
    grid = np.linspace(unknown.low, unknown.high, GRID_POINTS)
    labels = [unknown.verdict(float(v)) for v in grid]
    if len(set(labels)) == 1:
        return None
    first = next(i for i in range(1, len(labels)) if labels[i] != labels[i - 1])
    low, high = float(grid[first - 1]), float(grid[first])
    while high - low > BISECTION_WIDTH:
        middle = (low + high) / 2.0
        if unknown.verdict(middle) == labels[first - 1]:
            low = middle
        else:
            high = middle
    return HumanQuestion(
        unknown.name,
        unknown.form,
        unknown.low,
        unknown.high,
        (low + high) / 2.0,
        (labels[0], labels[-1]),
    )


def ask(
    unknowns: Sequence[Unknown], derivable: Collection[str] = ()
) -> tuple[list[HumanQuestion], list[dict[str, Any]]]:
    """Questions to raise, and the unknowns reported as robust.

    Args:
        unknowns: What the request leaves open.
        derivable: Slots the data or the request already fill.

    Returns:
        (questions, robust), where `robust` lists each unknown whose range
        leaves the verdict unchanged, with that verdict.
    """
    questions: list[HumanQuestion] = []
    robust: list[dict[str, Any]] = []
    for unknown in unknowns:
        if unknown.name in derivable:
            continue
        found = tipping_point(unknown)
        if found is None:
            robust.append(
                {
                    "slot": unknown.name,
                    "low": unknown.low,
                    "high": unknown.high,
                    "verdict": unknown.verdict(unknown.low),
                }
            )
        else:
            questions.append(found)
    return questions, robust


def gain_sign_verdict(
    y: ArrayLike, a: ArrayLike, b: ArrayLike
) -> Callable[[float], str]:
    """Verdict `model_better` or `baseline_better` as a function of `t`."""

    def verdict(t: float) -> str:
        gain = float(np.mean(baseline.paired_difference(y, a, b, t)))
        return "model_better" if gain > 0 else "baseline_better"

    return verdict
