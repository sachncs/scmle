"""The estimate record and the interface every identification strategy has."""

from collections.abc import Mapping
import dataclasses
import math
from typing import Any

import numpy
import pandas

from scmle.assurance.charter import causal
from scmle.causal.identification import battery


@dataclasses.dataclass(frozen=True, slots=True)
class Est:
    """A point estimate with its standard error and interval.

    Attributes:
        estimate: The point estimate.
        se: Its standard error (NaN when the strategy gives none).
        low: Lower interval limit.
        high: Upper interval limit.
        n: Rows used.
        detail: Strategy-specific extras for the report.
        df: Degrees of freedom of the t reference of the interval; infinite
            when it is normal. A Wald test of the estimate uses the same
            reference.
    """

    estimate: float
    se: float
    low: float
    high: float
    n: int
    detail: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    df: float = math.inf

    def conclusive(self) -> bool:
        """Tests whether the interval excludes zero.

        Returns:
            True when the lower limit is above zero or the upper limit is
            below it.
        """
        return self.low > 0 or self.high < 0


def waldify(est: Est, name: str, kind: str) -> battery.Test | None:
    """Builds a Wald test of a placebo estimate.

    Args:
        est: The placebo estimate.
        name: Name of the test.
        kind: Kind of the test.

    Returns:
        The test, or None when the standard error is not finite and positive.
    """
    if not numpy.isfinite(est.se) or est.se <= 0:
        return None
    return battery.wald(name, kind, est.estimate, est.se, est.df)


class Estimator:
    """What a strategy needs: design check, estimator, controls, extras.

    Subclasses fix `name` and implement the hooks. `prepare` runs on `X`
    only; every later method reads `C`.

    Attributes:
        spec: The charter's causal block.
        x: Exploration rows.
        seed: Seed of the learners.
        learner: `linear` or `boosted` nuisance learners.
        falsify: Familywise level of the falsification battery.
        design: Design diagnostics for the report.
        refusal: Why the claim must be refused, or None.
        weak: Whether the instrument was found weak.
        unit: Whether there is a single treated unit (nothing to hold
            out).
        notes: Remarks for the report.
    """

    name = ""

    def __init__(
        self,
        spec: causal.Question,
        x: pandas.DataFrame,
        seed: int,
        learner: str,
        falsify: float,
    ) -> None:
        """Stores the settings of the strategy.

        Args:
            spec: The charter's causal block.
            x: Exploration rows.
            seed: Seed of the learners.
            learner: `linear` or `boosted` nuisance learners.
            falsify: Familywise level of the falsification battery.
        """
        self.spec = spec
        self.x = x
        self.seed = seed
        self.learner = learner
        self.falsify = falsify
        self.design: dict[str, Any] = {}
        self.refusal: str | None = None
        self.weak = False
        self.unit = False
        self.notes: list[str] = []

    def prepare(self) -> None:
        """Runs the design diagnostics and freezes quantities, from `X` only.

        The base class does nothing; subclasses fill `design` and may set
        `refusal`.
        """

    def estimate(self, c: pandas.DataFrame, level: float) -> Est:
        """Computes the primary estimate on `C`.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The estimate with its standard error and interval.

        Raises:
            NotImplementedError: Always, in the base class; subclasses
                override it.
        """
        raise NotImplementedError

    def reckon(
        self, c: pandas.DataFrame, level: float, outcome: str | None = None
    ) -> Est:
        """Computes the same estimate on `C` with another outcome column.

        Args:
            c: The confirmation rows.
            level: Confidence level.
            outcome: The outcome column to use, or None for the charter's
                outcome.

        Returns:
            The estimate with its standard error and interval.

        Raises:
            NotImplementedError: Always, in the base class; subclasses
                override it.
        """
        raise NotImplementedError

    def controls(self, c: pandas.DataFrame, level: float) -> list[battery.Test]:  # noqa: ARG002 -- base-class hook; overriding strategies use these arguments
        """Runs the falsification tests on `C` (negative controls and placebos).

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The tests; the base class has none.
        """
        return []

    def sensitivity(self, c: pandas.DataFrame, level: float) -> dict[str, Any]:  # noqa: ARG002 -- base-class hook; overriding strategies use these arguments
        """Assesses sensitivity to unmeasured confounding.

        Args:
            c: The confirmation rows.
            level: Confidence level.

        Returns:
            The sensitivity document; the base class returns `holds` as None
            with a note that the strategy has no sensitivity analysis.
        """
        return {
            "holds": None,
            "note": "no sensitivity analysis for this strategy",
        }
