"""The joint specification-curve test against permuted targets."""

import dataclasses
import itertools
import operator
from typing import Any
from typing import ClassVar

import numpy

from scmle.assurance.ledger import bivariate
from scmle.assurance.ledger import controls
from scmle.assurance.robustness import multiverse
from scmle.assurance.robustness.stability import estimation
from scmle.contracts import qualification
from scmle.contracts import records
from scmle.runtime import randomness


@dataclasses.dataclass(frozen=True, slots=True)
class Curve:
    """Joint test of the whole specification curve against permuted targets.

    Attributes:
        permutations: Number of permuted curves.
        median: Observed median absolute effect.
        significant: Observed share of specifications with p <= 0.05.
        p_median: Permutation p-value of the median absolute effect:
            ``(1 + #{permuted >= observed}) / (permutations + 1)``.
        p_share: The same for the share significant.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `median`, `significant` and
            `pvalues.{median, share}`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {"median_abs_effect": "median", "share_significant": "significant"},
            groups={"pvalues": {"p_median": "median", "p_share": "share"}},
        ),
    )

    permutations: int
    median: float
    significant: float
    p_median: float
    p_share: float

    def to_dict(self) -> dict[str, Any]:
        """Converts the test to a JSON-ready dictionary.

        Returns:
            The fields by name.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )


def summarize(
    x: numpy.ndarray, y: numpy.ndarray, grid: multiverse.Grid
) -> tuple[float, float]:
    """Summarises the specification curve.

    Specifications with fewer than `MIN_PAIRS` pairs, or a constant feature
    or target, are left out.

    Args:
        x: Feature values, NaN where missing.
        y: Target codes, one per row.
        grid: The specifications.

    Returns:
        (median absolute Spearman rho, share of specifications with
        p <= `SIGNIFICANCE`). Both are 0 when no specification is usable.
    """
    rows = len(x)
    positions = numpy.arange(rows)
    effects, flags = [], []
    for spec in grid.specs:
        px, py = estimation.prepare(x, y, positions, rows, spec)
        if (
            len(px) < estimation.MIN_PAIRS
            or numpy.ptp(px) == 0.0
            or numpy.ptp(py) == 0.0
        ):
            continue
        outcome = bivariate.spearman(px, py)
        effects.append(abs(outcome.effect))
        flags.append(outcome.p <= estimation.SIGNIFICANCE)
    if not effects:
        return 0.0, 0.0
    return float(numpy.median(effects)), float(numpy.mean(flags))


def specify(
    x: numpy.ndarray,
    y: numpy.ndarray,
    grid: multiverse.Grid,
    seed: int,
    finding: str,
    *,
    permutations: int,
    strata: numpy.ndarray | None = None,
) -> Curve:
    """Specification-curve test by permuting the target within strata.

    The permutation is exact under exchangeability within strata (global
    when `strata` is None); clustered or temporal data must pass strata.

    Args:
        x: Feature values, NaN where missing.
        y: Target codes, one per row.
        grid: The specifications.
        seed: Run seed.
        finding: Name of the finding, part of the permutation stream's name.
        permutations: Number of permuted curves. A value that is not an
            integer of at least 1 raises `ValueError`.
        strata: Labels within which the target is permuted, or None.

    Returns:
        The observed curve statistics and their permutation p-values, each
        ``(1 + #{permuted >= observed}) / (permutations + 1)``.
    """
    qualification.count(permutations, "permutations", minimum=1)
    median, share = summarize(x, y, grid)
    rng = randomness.spawn(seed, f"f6.curve.{finding}", 0)
    curves = list(
        map(
            operator.call,
            itertools.repeat(
                lambda: summarize(x, controls.permute(y, strata, rng), grid),
                permutations,
            ),
        )
    )
    medians = [curve[0] for curve in curves]
    shares = [curve[1] for curve in curves]
    return Curve(
        permutations,
        median,
        share,
        (1.0 + sum(m >= median - 1e-12 for m in medians))
        / (permutations + 1.0),
        (1.0 + sum(s >= share - 1e-12 for s in shares)) / (permutations + 1.0),
    )
