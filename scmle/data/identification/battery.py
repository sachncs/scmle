"""The falsification battery and stratum replication.

Every test here can only falsify. A battery that finds nothing does not show
the assumptions hold: it shows that no violation of the kind and size it was
certified to detect (`delta_min`, measured in the certificate) was present,
and a confounder that the negative controls do not share passes undetected.

Tests. A negative-control outcome (an outcome the treatment cannot move that
shares the confounding) and a negative-control exposure (a treatment that
cannot move the outcome) are each run through the same estimator; the null
"effect = 0" is tested with a Wald statistic. Placebo tests (placebo cutoffs,
pre-period event-study coefficients) arrive as p-values from their own
modules. All p-values of one battery are tested together by Holm's step-down
at `alpha_falsify`, so the probability that a valid design is flagged is at
most that level when the p-values are valid.

Replication. The estimate is recomputed in each declared stratum; the
claim replicates when every stratum estimate has the sign of the pooled
estimate and Cochran's Q does not reject equality at `alpha_heterogeneity`.
At least two strata are required.
"""

import dataclasses
from typing import Any

import numpy as np
from scipy import stats

from scmle.evaluation import coverage

MIN_STRATA = 2
FALSIFY_WEIGHTS = {"estimate": 4, "falsify": 1}


@dataclasses.dataclass(frozen=True, slots=True)
class Test:
    """One falsification test.

    Attributes:
        name: What was tested (for example `negative_outcome:prior_spend`).
        kind: `negative_outcome`, `negative_exposure` or `placebo`.
        p_value: Two-sided p-value of the null of no effect.
        estimate: The placebo effect, if the test has one.
        se: Its standard error, if any.
    """

    name: str
    kind: str
    p_value: float
    estimate: float | None = None
    se: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of the name, kind, p-value, estimate and standard
            error.
        """
        return dataclasses.asdict(self)


def wald_test(name: str, kind: str, estimate: float, se: float) -> Test:
    """Runs a two-sided Wald test of `estimate = 0`.

    Args:
        name: What is tested.
        kind: `negative_outcome`, `negative_exposure` or `placebo`.
        estimate: The placebo effect.
        se: Its standard error, which must be positive.

    Returns:
        The test, with the normal two-sided p-value of ``estimate / se``.

    Raises:
        ValueError: If `se` is not positive.
    """
    if not se > 0:
        raise ValueError("a Wald test needs a positive standard error")
    p = float(2 * stats.norm.sf(abs(estimate) / se))
    return Test(name, kind, p, estimate, se)


@dataclasses.dataclass(frozen=True)
class BatteryReport:
    """Outcome of the battery.

    Attributes:
        alpha: Familywise level (`alpha_falsify`).
        tests: The tests, in the order run.
        rejected: Names of the tests Holm rejected.
        passed: True when nothing was rejected and at least one test ran.
        ran: Whether any test ran (an empty battery is not a pass).
    """

    alpha: float
    tests: tuple[Test, ...]
    rejected: tuple[str, ...]
    passed: bool
    ran: bool

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of the level, the tests (as records), the rejected
            names as a list, and the `passed` and `ran` flags.
        """
        return {
            "alpha": self.alpha,
            "tests": [t.to_dict() for t in self.tests],
            "rejected": list(self.rejected),
            "passed": self.passed,
            "ran": self.ran,
        }


def run_battery(tests: list[Test], alpha: float) -> BatteryReport:
    """Applies Holm's step-down over all tests at familywise level `alpha`.

    Args:
        tests: The tests of the battery, in the order run.
        alpha: Familywise level, strictly between 0 and 1.

    Returns:
        The report: the names of the rejected tests, `ran` when any test was
        given, and `passed` when tests ran and none was rejected.

    Raises:
        ValueError: If `alpha` is not strictly between 0 and 1.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be within (0, 1)")
    flags = (
        coverage.holm_rejections([t.p_value for t in tests], alpha)
        if tests
        else []
    )
    rejected = tuple(t.name for t, f in zip(tests, flags, strict=True) if f)
    return BatteryReport(
        alpha, tuple(tests), rejected, bool(tests) and not rejected, bool(tests)
    )


@dataclasses.dataclass(frozen=True)
class Replication:
    """Stratum-wise replication of an estimate.

    Attributes:
        strata: Stratum to `(estimate, se)`.
        pooled_sign: Sign of the pooled estimate.
        same_sign: Whether every stratum has that sign.
        q: Cochran's Q.
        p_heterogeneity: Its chi-square p-value.
        alpha: Level of the heterogeneity test.
        passed: Same sign everywhere, no heterogeneity, enough strata.
    """

    strata: dict[str, tuple[float, float]]
    pooled_sign: int
    same_sign: bool
    q: float
    p_heterogeneity: float
    alpha: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record.

        Returns:
            A mapping of every field, with each stratum written as a list
            ``[estimate, se]``.
        """
        record = dataclasses.asdict(self)
        record["strata"] = {k: list(v) for k, v in self.strata.items()}
        return record


def replicate(
    strata: dict[str, tuple[float, float]],
    pooled: float,
    alpha: float = 0.05,
) -> Replication:
    """Checks sign agreement and Cochran's Q across strata.

    Args:
        strata: Stratum name to ``(estimate, se)``.
        pooled: The pooled estimate whose sign the strata must share.
        alpha: Level of the heterogeneity test.

    Returns:
        The replication result; `passed` needs every stratum estimate to
        have the sign of `pooled` (a zero `pooled` counts as positive; a
        stratum estimate of exactly zero never matches) and Cochran's Q
        p-value to be at least `alpha`.

    Raises:
        ValueError: If there are fewer than two strata or a standard error
            is not positive.
    """
    if len(strata) < MIN_STRATA:
        raise ValueError(f"replication needs at least {MIN_STRATA} strata")
    est = np.array([v[0] for v in strata.values()])
    se = np.array([v[1] for v in strata.values()])
    if not (se > 0).all():
        raise ValueError("stratum standard errors must be positive")
    weight = 1.0 / se**2
    center = float((weight * est).sum() / weight.sum())
    q = float((weight * (est - center) ** 2).sum())
    p = float(stats.chi2.sf(q, len(est) - 1))
    sign = 1 if pooled >= 0 else -1
    same = bool((np.sign(est) == sign).all())
    return Replication(
        dict(strata), sign, same, q, p, alpha, same and p >= alpha
    )
