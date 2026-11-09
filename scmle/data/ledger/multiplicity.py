"""Multiplicity adjustment for the single confirmation test.

The survivors of exploration are tested once on the confirmation set `C` and
adjusted as one family. Decisions are exact: each p-value is compared, as the
exact binary fraction it is, with an exact threshold, so the decision never
depends on rounding. Adjusted p-values are reported as floats.

Guarantees (what the procedures promise, with the assumptions they need):

``bh``
    Benjamini and Hochberg (1995, J. R. Stat. Soc. B 57:289), step-up at
    ``k * alpha / m``. FDR <= ``alpha * m0 / m`` when the p-values are
    independent or positively regression dependent (PRDS; Benjamini and
    Yekutieli 2001).
``by``
    Benjamini and Yekutieli (2001, Ann. Statist. 29:1165, Theorem 1.3):
    BH at ``alpha / sum_{i<=m} 1/i``. FDR <= ``alpha * m0 / m`` under
    *arbitrary* dependence. This is the default because the tests on `C` all
    read the same rows. The statements for BH and BY were read in the
    `scipy.stats.false_discovery_control` documentation, which cites those
    theorems; the original papers were not re-read.
``holm``
    Holm (1979, Scand. J. Statist. 6:65), step-down at ``alpha / (m - k +
    1)``. Strong FWER control at ``alpha`` under any dependence (confirmed
    from secondary sources, see Goeman and Solari 2010, Ann. Statist.
    38:3782, which proves it for sequentially rejective procedures).
``bonferroni``
    Reject when ``p <= alpha / m``; FWER <= ``alpha`` under any dependence.

All four apply to a family of ``m`` hypotheses fixed *before* the p-values
are seen. Here the family is the survivor set frozen in the ledger, selected
using `X` only; because `C` is independent of `X`, the guarantee holds
conditionally on that selection.
"""

from collections.abc import Sequence
import dataclasses
from fractions import Fraction

from scmle.contracts import qualification
from scmle.data.ledger import budget as budget_module

Rule = budget_module.MultiplicityRule
DEFAULT_RULE = Rule.BY
STATEMENTS = {
    Rule.BH: (
        "FDR <= alpha * m0 / m if the p-values on C are independent or "
        "positively regression dependent"
    ),
    Rule.BY: "FDR <= alpha * m0 / m under arbitrary dependence",
    Rule.HOLM: "FWER <= alpha under arbitrary dependence",
    Rule.BONFERRONI: "FWER <= alpha under arbitrary dependence",
}


@dataclasses.dataclass(frozen=True, slots=True)
class Adjustment:
    """Adjusted p-values and decisions, in the order of the input.

    Attributes:
        rule: The procedure.
        alpha: The level.
        adjusted: Adjusted p-values in [0, 1].
        rejected: Exact decision per hypothesis.
    """

    rule: Rule
    alpha: Fraction
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]


def harmonic_exact(m: int) -> Fraction:
    """Computes the harmonic number as an exact fraction.

    Args:
        m: Number of terms.

    Returns:
        ``sum_{i <= m} 1 / i``, exactly; zero when `m` is zero.
    """
    return sum((Fraction(1, i) for i in range(1, m + 1)), Fraction(0))


def adjust(pvalues: Sequence[float], rule: Rule, alpha: Fraction) -> Adjustment:
    """Adjusts a family of p-values and decides at level `alpha`.

    Args:
        pvalues: One p-value per hypothesis, each in [0, 1].
        rule: `bh`, `by`, `holm` or `bonferroni`.
        alpha: Level in (0, 1).

    Returns:
        The adjusted p-values and exact rejection decisions, each in the
        order of `pvalues`; an empty family gives empty tuples.

    Raises:
        ValueError: On an unknown rule, a level outside (0, 1) or a p-value
            outside [0, 1].
    """
    if not isinstance(rule, Rule):
        raise ValueError("rule must be a MultiplicityRule")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    values = [qualification.probability(p, "p-value") for p in pvalues]
    m = len(values)
    if m == 0:
        return Adjustment(rule, alpha, (), ())
    order = sorted(range(m), key=lambda i: (values[i], i))
    exact = [Fraction(values[i]) for i in order]
    if rule in (Rule.BH, Rule.BY):
        factor = harmonic_exact(m) if rule is Rule.BY else Fraction(1)
        count = max(
            (
                k
                for k in range(1, m + 1)
                if exact[k - 1] <= k * alpha / (m * factor)
            ),
            default=0,
        )
        scale = float(m * factor)
        adjusted_sorted = step_up(
            [min(1.0, values[order[k]] * scale / (k + 1)) for k in range(m)]
        )
    elif rule is Rule.HOLM:
        count = 0
        while count < m and exact[count] <= alpha / (m - count):
            count += 1
        adjusted_sorted = step_down(
            [min(1.0, values[order[k]] * (m - k)) for k in range(m)]
        )
    else:
        count = sum(1 for p in exact if p <= alpha / m)
        adjusted_sorted = [min(1.0, values[order[k]] * m) for k in range(m)]
    adjusted = [0.0] * m
    rejected = [False] * m
    for rank, index in enumerate(order):
        adjusted[index] = adjusted_sorted[rank]
        rejected[index] = rank < count
    return Adjustment(rule, alpha, tuple(adjusted), tuple(rejected))


def step_up(scaled: list[float]) -> list[float]:
    """Takes the running minimum from the largest rank down (BH and BY).

    Args:
        scaled: Scaled p-values sorted by increasing p-value.

    Returns:
        A new list in which each entry is the minimum of itself and every
        later entry.
    """
    out = list(scaled)
    for k in range(len(out) - 2, -1, -1):
        out[k] = min(out[k], out[k + 1])
    return out


def step_down(scaled: list[float]) -> list[float]:
    """Takes the running maximum from the smallest rank up (Holm).

    Args:
        scaled: Scaled p-values sorted by increasing p-value.

    Returns:
        A new list in which each entry is the maximum of itself and every
        earlier entry.
    """
    out = list(scaled)
    for k in range(1, len(out)):
        out[k] = max(out[k], out[k - 1])
    return out
