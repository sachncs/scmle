"""Closed-form power and feasibility of the confirmation set (feature F1).

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
DEFAULT_CONFIDENCE = 0.95
CONFIRMATORY = "confirmatory"
UNCONFIRMED = "exploratory_unconfirmed"


class Verdict(enum.StrEnum):
    """Whether the confirmation set can support the promised power."""

    FEASIBLE = "feasible"
    DOWNGRADE = "downgrade"
    INFEASIBLE = "infeasible"


def open_unit(value: float, name: str, low: float = 0.0) -> float:
    """Requires a finite number strictly inside `(low, 1)`."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not low < value < 1
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} must be strictly between {low} and 1")
    return float(value)


def positive_delta(value: float, name: str) -> float:
    """Requires a finite number strictly greater than zero."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a finite number greater than 0")
    return float(value)


def check_k(k_max: int) -> int:
    """Requires a positive integer claim count."""
    if isinstance(k_max, bool) or not isinstance(k_max, int) or k_max < 1:
        raise ValueError("k_max must be an integer of at least 1")
    return k_max


def z_sum(alpha: float, k_max: int, power: float) -> float:
    """`z_{1-alpha/(2K)} + z_{power}`, the quantity both formulas share."""
    open_unit(alpha, "alpha")
    open_unit(power, "power")
    check_k(k_max)
    return float(
        stats.norm.ppf(1 - alpha / (2 * k_max)) + stats.norm.ppf(power)
    )


def n_required_mean(
    delta: float, *, alpha: float, k_max: int, power: float
) -> int:
    """Smallest `n` that detects a standardised mean effect `delta`."""
    positive_delta(delta, "delta")
    return math.ceil((z_sum(alpha, k_max, power) / delta) ** 2)


def mde_mean(n: int, *, alpha: float, k_max: int, power: float) -> float:
    """Minimum detectable standardised mean effect at `n` rows."""
    if n < 1:
        return math.nan
    return z_sum(alpha, k_max, power) / math.sqrt(n)


def power_mean(delta: float, n: int, *, alpha: float, k_max: int) -> float:
    """Power to detect `delta` with `n` rows at the Bonferroni level."""
    positive_delta(delta, "delta")
    open_unit(alpha, "alpha")
    check_k(k_max)
    critical = stats.norm.ppf(1 - alpha / (2 * k_max))
    return float(stats.norm.cdf(delta * math.sqrt(max(n, 0)) - critical))


def n_required_correlation(
    r: float, *, alpha: float, k_max: int, power: float
) -> int:
    """Smallest `n` that detects a correlation `r` (Fisher z)."""
    open_unit(abs(r), "r")
    return math.ceil((z_sum(alpha, k_max, power) / math.atanh(abs(r))) ** 2 + 3)


def mde_correlation(n: int, *, alpha: float, k_max: int, power: float) -> float:
    """Minimum detectable correlation at `n` rows (needs `n > 3`)."""
    if n <= 3:
        return math.nan
    return math.tanh(z_sum(alpha, k_max, power) / math.sqrt(n - 3))


def n_required_two_proportions(
    p1: float, p2: float, *, alpha: float, k_max: int, power: float
) -> int:
    """Rows per group to tell two proportions apart (pooled-variance test)."""
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
    """Half-width of the Wilson interval for a proportion `p` at `n` rows."""
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
        """JSON-ready record (non-finite numbers become null)."""
        record = dataclasses.asdict(self)
        record["verdict"] = self.verdict.value
        for key, value in record.items():
            if isinstance(value, float) and not math.isfinite(value):
                record[key] = None
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Feasibility":
        """Rebuilds a report written by `to_dict`."""
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
        The verdict with the achieved minimum detectable effect.
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
