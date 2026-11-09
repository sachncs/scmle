"""Closed-form power and feasibility of the confirmation set.

With `K` the most claims carried into confirmation, each claim is tested at
the Bonferroni level `alpha / K`. That level is conservative for BH, so a
pass here is a pass under the real procedure. For a standardised mean effect
`delta` and a two-sided z test,

    n >= ((z_{1-a/(2K)} + z_{power}) / delta) ** 2,

for a correlation `r` it is `((z_{1-a/(2K)} + z_{power}) / atanh(r)) ** 2 + 3`,
and the minimum detectable effect is the inverse of each formula. The
formulas are auditable by hand and match published tables (Cohen: r = 0.3 at
power 0.8 needs 85 rows).
"""

from collections.abc import Mapping
import dataclasses
import enum
import math
from typing import Any

from scipy import stats

MIN_CONFIRMATION_ROWS = 30
FISHER_Z_OFFSET = 3
DEFAULT_CONFIDENCE = 0.95
CONFIRMATORY = "confirmatory"
UNCONFIRMED = "exploratory_unconfirmed"


class Verdict(enum.StrEnum):
    """Whether the confirmation set can support the promised power."""

    FEASIBLE = "feasible"
    DOWNGRADE = "downgrade"
    INFEASIBLE = "infeasible"


def open_unit(value: float, name: str, low: float = 0.0) -> float:
    """Checks that a number lies strictly inside `(low, 1)`.

    Args:
        value: The candidate; a bool or a non-number is rejected.
        name: Name used in the error message.
        low: Exclusive lower bound.

    Returns:
        `value` as a float.

    Raises:
        ValueError: If `value` is a bool, not an int or float, not finite,
            or not strictly between `low` and 1.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not low < value < 1
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be strictly between {low} and 1")
    return float(value)


def positive_delta(value: float, name: str) -> float:
    """Checks that a number is finite and strictly greater than zero.

    Args:
        value: The candidate; a bool or a non-number is rejected.
        name: Name used in the error message.

    Returns:
        `value` as a float.

    Raises:
        ValueError: If `value` is a bool, not an int or float, not finite,
            or not greater than 0.
    """
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite number greater than 0")
    return float(value)


def check_k(k_max: int) -> int:
    """Checks that the claim count is an integer of at least 1.

    Args:
        k_max: The candidate count; a bool is rejected.

    Returns:
        `k_max` unchanged.

    Raises:
        ValueError: If `k_max` is not an int or is below 1.
    """
    if isinstance(k_max, bool) or not isinstance(k_max, int) or k_max < 1:
        raise ValueError("k_max must be an integer of at least 1")
    return k_max


def z_sum(alpha: float, k_max: int, power: float) -> float:
    """Computes `z_{1-alpha/(2K)} + z_{power}`, shared by both formulas.

    Both quantiles are standard normal. `alpha` and `power` must lie
    strictly in (0, 1) and `k_max` must be an integer of at least 1;
    otherwise the checks in `open_unit` and `check_k` raise `ValueError`.

    Args:
        alpha: Family-wise level of the final claims.
        k_max: Most claims carried into confirmation.
        power: Target power.

    Returns:
        The sum of the two standard normal quantiles.
    """
    open_unit(alpha, "alpha")
    open_unit(power, "power")
    check_k(k_max)
    return float(
        stats.norm.ppf(1 - alpha / (2 * k_max)) + stats.norm.ppf(power)
    )


def n_required_mean(
    delta: float, *, alpha: float, k_max: int, power: float
) -> int:
    """Computes the rows needed to detect a standardised mean effect.

    The result is `ceil((z_sum / delta) ** 2)` (see `z_sum`). `delta` must
    be finite and positive and the other inputs valid for `z_sum`;
    otherwise `ValueError` is raised by those checks.

    Args:
        delta: Standardised mean effect to detect.
        alpha: Family-wise level of the final claims.
        k_max: Most claims carried into confirmation.
        power: Target power.

    Returns:
        The smallest number of rows `n`.
    """
    positive_delta(delta, "delta")
    return math.ceil((z_sum(alpha, k_max, power) / delta) ** 2)


def mde_mean(n: int, *, alpha: float, k_max: int, power: float) -> float:
    """Computes the minimum detectable standardised mean effect.

    The result is `z_sum / sqrt(n)` (see `z_sum`, whose checks raise
    `ValueError` for invalid `alpha`, `k_max` or `power` when `n >= 1`).

    Args:
        n: Rows available.
        alpha: Family-wise level of the final claims.
        k_max: Most claims carried into confirmation.
        power: Target power.

    Returns:
        The smallest standardised effect detectable at `n` rows; NaN when
        `n` is below 1.
    """
    if n < 1:
        return math.nan
    return z_sum(alpha, k_max, power) / math.sqrt(n)


def power_mean(delta: float, n: int, *, alpha: float, k_max: int) -> float:
    """Computes the power to detect an effect at the Bonferroni level.

    The result is `Phi(delta * sqrt(max(n, 0)) - z_{1-alpha/(2K)})`. `delta`
    must be finite and positive, `alpha` strictly in (0, 1) and `k_max` an
    integer of at least 1; otherwise `ValueError` is raised by those checks.

    Args:
        delta: Standardised mean effect.
        n: Rows available; values below 0 count as 0.
        alpha: Family-wise level of the final claims.
        k_max: Most claims carried into confirmation.

    Returns:
        The probability, in [0, 1], of rejecting at level `alpha / K` when
        the true effect is `delta`.
    """
    positive_delta(delta, "delta")
    open_unit(alpha, "alpha")
    check_k(k_max)
    critical = stats.norm.ppf(1 - alpha / (2 * k_max))
    return float(stats.norm.cdf(delta * math.sqrt(max(n, 0)) - critical))


def n_required_correlation(
    r: float, *, alpha: float, k_max: int, power: float
) -> int:
    """Computes the rows needed to detect a correlation (Fisher z).

    The result is `ceil((z_sum / atanh(|r|)) ** 2 + 3)` (see `z_sum`).
    `|r|` must lie strictly in (0, 1); otherwise `ValueError` is raised by
    `open_unit`, as it is for invalid `alpha`, `k_max` or `power`.

    Args:
        r: Correlation to detect; only its absolute value matters.
        alpha: Family-wise level of the final claims.
        k_max: Most claims carried into confirmation.
        power: Target power.

    Returns:
        The smallest number of rows `n`.
    """
    open_unit(abs(r), "r")
    return math.ceil((z_sum(alpha, k_max, power) / math.atanh(abs(r))) ** 2 + 3)


def mde_correlation(n: int, *, alpha: float, k_max: int, power: float) -> float:
    """Computes the minimum detectable correlation.

    The result is `tanh(z_sum / sqrt(n - 3))` (see `z_sum`, whose checks
    raise `ValueError` for invalid `alpha`, `k_max` or `power` when
    `n > 3`).

    Args:
        n: Rows available.
        alpha: Family-wise level of the final claims.
        k_max: Most claims carried into confirmation.
        power: Target power.

    Returns:
        The smallest correlation, in (0, 1), detectable at `n` rows; NaN
        when `n` is 3 or less.
    """
    if n <= FISHER_Z_OFFSET:
        return math.nan
    return math.tanh(
        z_sum(alpha, k_max, power) / math.sqrt(n - FISHER_Z_OFFSET)
    )


def n_required_two_proportions(
    p1: float, p2: float, *, alpha: float, k_max: int, power: float
) -> int:
    """Computes the rows per group to tell two proportions apart.

    Uses the pooled-variance test at the Bonferroni level `alpha / K`.

    Args:
        p1: Proportion in the first group, strictly in (0, 1).
        p2: Proportion in the second group, strictly in (0, 1).
        alpha: Family-wise level of the final claims.
        k_max: Most claims carried into confirmation.
        power: Target power.

    Returns:
        The smallest number of rows in each of the two groups.

    Raises:
        ValueError: If `p1` or `p2` is not strictly in (0, 1), if they are
            equal, or if `alpha`, `power` or `k_max` is invalid.
    """
    open_unit(p1, "p1")
    open_unit(p2, "p2")
    if p1 == p2:
        raise ValueError("p1 and p2 must differ")
    open_unit(alpha, "alpha")
    open_unit(power, "power")
    check_k(k_max)
    pooled = (p1 + p2) / 2
    root = stats.norm.ppf(1 - alpha / (2 * k_max)) * math.sqrt(
        2 * pooled * (1 - pooled)
    ) + stats.norm.ppf(power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    return math.ceil(float(root) ** 2 / (p1 - p2) ** 2)


def wilson_half_width(
    p: float, n: int, confidence: float = DEFAULT_CONFIDENCE
) -> float:
    """Computes the half-width of the Wilson interval for a proportion.

    Args:
        p: Observed proportion, in [0, 1].
        n: Rows the proportion is based on.
        confidence: Two-sided confidence level of the interval.

    Returns:
        Half of the width of the Wilson interval; infinity when `n` is
        below 1.
    """
    if n < 1:
        return math.inf
    z = float(stats.norm.ppf(0.5 + confidence / 2))
    spread = math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return z * spread / (1 + z * z / n)


@dataclasses.dataclass(frozen=True, slots=True)
class Feasibility:
    """What the confirmation set can and cannot support.

    Attributes:
        verdict: Feasible, downgrade (claims capped) or infeasible (no `C`).
        scope: `confirmatory` or `exploratory_unconfirmed`.
        n_confirmation: Rows available in the confirmation set.
        n_required: Rows needed for the promised power.
        mde: Minimum detectable standardised effect at `n_confirmation`.
        power_at_min_effect: Achieved power at the charter's `min_effect`.
        reason: Why the verdict was reached.
    """

    verdict: Verdict
    scope: str
    n_confirmation: int
    n_required: int
    mde: float
    power_at_min_effect: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Serializes the report.

        Returns:
            A JSON-ready dict of the fields, with `verdict` as its string
            value and every non-finite float written as None.
        """
        record = dataclasses.asdict(self)
        record["verdict"] = self.verdict.value
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Feasibility":
        """Rebuilds a report written by `to_dict`.

        Args:
            record: Field names to values, as produced by `to_dict`.

        Returns:
            The report, with a None `mde` or `power_at_min_effect` read back
            as NaN.
        """
        values = dict(record)
        values["verdict"] = Verdict(values["verdict"])
        for key in ("mde", "power_at_min_effect"):
            if values[key] is None:
                values[key] = math.nan
        return cls(**values)


def assess(
    *,
    n_confirmation: int,
    min_effect: float,
    alpha: float,
    k_max: int,
    power: float,
    n_minority: int | None = None,
    max_half_width: float | None = None,
) -> Feasibility:
    """Decides whether the confirmation set supports the promised power.

    The verdict is `INFEASIBLE` below `MIN_CONFIRMATION_ROWS` rows,
    otherwise `DOWNGRADE` when the rows are fewer than the mean-effect
    requirement for `min_effect` or when the worst-case (p = 0.5) Wilson
    half-width at `n_minority` exceeds `max_half_width`, otherwise
    `FEASIBLE`. The scope is `CONFIRMATORY` only for `FEASIBLE`. Invalid
    `min_effect`, `alpha`, `power` or `k_max` raise `ValueError` from the
    checks of `n_required_mean`.

    Args:
        n_confirmation: Rows in the confirmation set.
        min_effect: Smallest standardised effect worth acting on.
        alpha: Level of the final claims.
        k_max: Most claims carried into confirmation.
        power: Target power at `min_effect`.
        n_minority: Minority-class rows in the confirmation set, if any.
        max_half_width: Widest Wilson half-width the decision tolerates at
            `n_minority` (worst case p = 0.5).

    Returns:
        The verdict, its scope and reason, the rows required, the minimum
        detectable effect at `n_confirmation` and the power at `min_effect`.
    """
    required = n_required_mean(
        min_effect, alpha=alpha, k_max=k_max, power=power
    )
    mde = mde_mean(n_confirmation, alpha=alpha, k_max=k_max, power=power)
    achieved = power_mean(min_effect, n_confirmation, alpha=alpha, k_max=k_max)
    if n_confirmation < MIN_CONFIRMATION_ROWS:
        verdict, reason = (
            Verdict.INFEASIBLE,
            (
                f"{n_confirmation} confirmation rows is below the "
                f"{MIN_CONFIRMATION_ROWS} needed for a normal "
                "approximation"
            ),
        )
    elif n_confirmation < required:
        verdict, reason = (
            Verdict.DOWNGRADE,
            (
                f"{n_confirmation} rows < {required} needed for power "
                f"{power} at effect {min_effect}; detectable effect is "
                f"{mde:.3f}"
            ),
        )
    elif (
        n_minority is not None
        and max_half_width is not None
        and wilson_half_width(0.5, n_minority) > max_half_width
    ):
        verdict, reason = (
            Verdict.DOWNGRADE,
            (
                f"minority count {n_minority} gives a Wilson half-width of "
                f"{wilson_half_width(0.5, n_minority):.3f} > {max_half_width}"
            ),
        )
    else:
        verdict, reason = (
            Verdict.FEASIBLE,
            (f"{n_confirmation} rows >= {required} needed for power {power}"),
        )
    scope = CONFIRMATORY if verdict is Verdict.FEASIBLE else UNCONFIRMED
    return Feasibility(
        verdict, scope, n_confirmation, required, mde, achieved, reason
    )
