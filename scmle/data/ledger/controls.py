"""Negative controls: how many discoveries appear when there is nothing real.

The whole exploration is re-run `B` times on data where the finding of
interest has been destroyed, and the number of discoveries in each re-run is
recorded. The gate compares the discoveries found on the real data with the
95th percentile of those counts: if the real count does not exceed it, the
findings do not clearly rise above the false-discovery floor and no claim is
promoted.

Two controls:

``permuted_target_control``
    Permutes the *target* within strata (entity groups, or contiguous time
    blocks) and never globally on structured data, so the dependence between
    rows that share a group or a time block survives while any feature-target
    association inside a stratum is destroyed. A global permutation of grouped
    or drifting data destroys that dependence too, which makes the floor look
    lower than it is: the control then passes findings that are artefacts of
    the structure. `benchmarks.qualification.controls_check` measures this.

``resampling_control``
    For findings that do not use the target (drift, shape, pairwise feature
    structure). Each feature column is shuffled independently (within
    strata), which breaks association between features, or the order of
    contiguous blocks of each column is shuffled, which breaks time drift. It
    shuffles rather than bootstraps: it keeps each column's marginal
    distribution exactly and destroys the structure being controlled, which a
    resample would only do on average.

The percentile is the ``ceil(0.95 * (B + 1))``-th smallest re-run count, so
if the observed count is exchangeable with the re-run counts the gate passes
by chance with probability at most 1/21 for B = 20 (and at most 0.05 in
general), which is what a "95th percentile" gate is meant to give.

Monte Carlo error. With `B` repeats the 95th percentile is itself an
estimate. `Control` reports an approximate distribution-free interval
for it from order statistics and the exceedance p-value
``(1 + #{permuted >= observed}) / (B + 1)``, which is a valid p-value for
"the real count is not larger than a typical null count".
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import math
from typing import Any

import numpy
import pandas
from scipy import stats

from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.ledger import budget

QUANTILE = 0.95
INTERVAL_CONFIDENCE = 0.90
PERMUTED_TARGET = "permuted_target"
RESAMPLING = "resampling"
COUNT_FUNCTION = Callable[[pandas.DataFrame, pandas.Series], int]


def quantile_index(b: int, q: float = QUANTILE) -> int:
    """Returns the 0-based index of the ``ceil(q * (B + 1))``-th smallest count.

    Using ``B + 1`` (the observed count plus the B re-runs) makes the gate a
    Monte Carlo test: if the observed count is exchangeable with the re-run
    counts, ``P(observed > percentile) <= 1 - q``. With B = 20 that is the
    largest of the 20 counts (``1/21 < 0.05``), not the 19th, which would
    allow ``2/21 = 9.5%``.

    Args:
        b: Number of re-run counts.
        q: Quantile level in (0, 1).

    Returns:
        The index into the ascending-sorted counts, clamped to the range
        0 to ``b - 1``.
    """
    return min(b, max(1, math.ceil(q * (b + 1) - 1e-9))) - 1


def percentile(counts: tuple[int, ...], q: float = QUANTILE) -> int:
    """Returns the conservative order-statistic percentile of the counts.

    It is the ``ceil(q * (B + 1))``-th smallest value (see `quantile_index`).

    Args:
        counts: Discovery counts of the re-runs.
        q: Quantile level in (0, 1).

    Returns:
        The selected order statistic of `counts`.

    Raises:
        ValueError: If `counts` is empty.
    """
    if not counts:
        raise ValueError("counts must not be empty")
    return sorted(counts)[quantile_index(len(counts), q)]


def percentile_interval(
    counts: tuple[int, ...],
    q: float = QUANTILE,
    confidence: float = INTERVAL_CONFIDENCE,
) -> tuple[int, int]:
    """Returns an approximate distribution-free interval for the q-quantile.

    Uses the binomial distribution of the number of draws below the quantile
    to pick two order statistics. With few repeats the interval is wide and
    reaches the largest count; that is the honest statement of the error.

    Args:
        counts: Discovery counts of the re-runs.
        q: Quantile level in (0, 1).
        confidence: Nominal coverage of the interval.

    Returns:
        The (lower, upper) bounds, each an element of `counts`.

    Raises:
        ValueError: If `counts` is empty.
    """
    b = len(counts)
    if b == 0:
        raise ValueError("counts must not be empty")
    ordered = sorted(counts)
    tail = (1.0 - confidence) / 2.0
    low = int(stats.binom.ppf(tail, b, q))
    high = int(stats.binom.ppf(1.0 - tail, b, q)) + 1
    return ordered[min(max(low - 1, 0), b - 1)], ordered[
        min(max(high - 1, 0), b - 1)
    ]


def interval_coverage(
    b: int,
    q: float = QUANTILE,
    confidence: float = INTERVAL_CONFIDENCE,
) -> float:
    """Returns the exact coverage of `percentile_interval` for `b` draws.

    For continuous draws the number of them below the true q-quantile is
    binomial, and the interval from the k-th to the h-th smallest draw covers
    the quantile exactly when that number is at least k and at most h - 1. So
    the coverage is a binomial probability, not an approximation. With few
    draws it is below the nominal `confidence`, because the upper bound cannot
    pass the largest draw, which lies above the quantile with probability
    only `1 - q ** b`.

    Args:
        b: Number of re-runs.
        q: Quantile level in (0, 1).
        confidence: Nominal coverage the interval was built for.

    Returns:
        The probability that the interval contains the true q-quantile.

    Raises:
        ValueError: If `b` is not positive.
    """
    if b <= 0:
        raise ValueError("b must be positive")
    tail = (1.0 - confidence) / 2.0
    low = int(stats.binom.ppf(tail, b, q))
    high = int(stats.binom.ppf(1.0 - tail, b, q)) + 1
    smallest = min(max(low - 1, 0), b - 1) + 1
    largest = min(max(high - 1, 0), b - 1) + 1
    return float(stats.binom.cdf(largest - 1, b, q)) - float(
        stats.binom.cdf(smallest - 1, b, q)
    )


def permute_within(
    values: numpy.ndarray,
    strata: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Permutes `values` within each stratum, or globally without strata.

    Args:
        values: The array to permute.
        strata: A label per entry; entries permute only among equal labels.
            None permutes globally, which is only right for rows that are
            exchangeable.
        rng: Random generator.

    Returns:
        A permuted copy.

    Raises:
        ValueError: If `strata` is given and does not have one label per
            entry of `values`.
    """
    out = numpy.array(values, copy=True)
    if strata is None:
        return out[rng.permutation(len(out))]
    labels = numpy.asarray(strata)
    if len(labels) != len(out):
        raise ValueError("strata needs one label per row")
    order = numpy.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    cuts = numpy.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    for block in numpy.split(order, cuts):
        out[block] = values[rng.permutation(block)]
    return out


def block_labels(order: numpy.ndarray, block_size: int) -> numpy.ndarray:
    """Builds labels that cut rows, ordered by `order`, into contiguous blocks.

    Args:
        order: Sort key per row (for example a timestamp or row position).
        block_size: Rows per block, at least 2.

    Returns:
        An integer block label per row; ties in `order` keep row order. A
        `block_size` that is not an integer of at least 2 raises a
        `ValueError` from `qualification.count`.
    """
    qualification.count(block_size, "block_size", minimum=2)
    ranks = numpy.empty(len(order), dtype=int)
    ranks[numpy.argsort(order, kind="stable")] = numpy.arange(len(order))
    return ranks // block_size


@dataclasses.dataclass(frozen=True, slots=True)
class Control:
    """The outcome of a negative control.

    Attributes:
        kind: `permuted_target` or `resampling`.
        scope: `strata` (within groups or blocks) or `global`.
        observed: Discoveries on the real data.
        permuted: Discovery counts of the B re-runs.
        p95: The 95th percentile of the permuted counts (see `percentile`).
        p95_interval: Distribution-free interval for that percentile; its
            exact coverage is `interval_coverage(b)`.
        exceedance_p: ``(1 + #{permuted >= observed}) / (B + 1)``.
        exceeds: Whether ``observed > p95``.
    """

    kind: str
    scope: str
    observed: int
    permuted: tuple[int, ...]
    p95: int
    p95_interval: tuple[int, int]
    exceedance_p: float
    exceeds: bool

    @property
    def b(self) -> int:
        """Number of re-runs."""
        return len(self.permuted)

    @property
    def message(self) -> str:
        """The sentence the report shows."""
        if self.exceeds:
            return (
                f"{self.observed} discoveries exceed the 95th percentile "
                f"({self.p95}) of {self.b} {self.kind} re-runs"
            )
        return (
            "findings do not clearly exceed the false-discovery floor: "
            f"{self.observed} discoveries, 95th percentile of {self.b} "
            f"{self.kind} re-runs is {self.p95}"
        )

    def to_dict(self) -> dict[str, Any]:
        """Builds the JSON-ready record (``negative_control.json``).

        Returns:
            A mapping of every field plus the number of re-runs, the
            interval confidence, the report message and a note on Monte
            Carlo error; lists replace tuples.
        """
        return {
            "kind": self.kind,
            "scope": self.scope,
            "observed": self.observed,
            "permuted": list(self.permuted),
            "b": self.b,
            "p95": self.p95,
            "p95_interval": list(self.p95_interval),
            "p95_interval_confidence": INTERVAL_CONFIDENCE,
            "p95_interval_achieved_confidence": interval_coverage(self.b),
            "exceedance_p": self.exceedance_p,
            "exceeds": self.exceeds,
            "message": self.message,
            "note": (
                "The 95th percentile is estimated from B re-runs and has "
                "Monte Carlo error; the interval is distribution-free "
                "and its confidence is the achieved value, which is "
                "below the nominal one when B is small."
            ),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Control":
        """Rebuilds a result from its JSON record.

        Args:
            record: A mapping with the fields written by `to_dict`.

        Returns:
            The result described by the record. The fields are not
            validated, `b`, `message` and the other derived entries are
            ignored, and a missing key raises `KeyError`.
        """
        counts = tuple(int(c) for c in record["permuted"])
        low, high = record["p95_interval"]
        return cls(
            kind=record["kind"],
            scope=record["scope"],
            observed=int(record["observed"]),
            permuted=counts,
            p95=int(record["p95"]),
            p95_interval=(int(low), int(high)),
            exceedance_p=float(record["exceedance_p"]),
            exceeds=bool(record["exceeds"]),
        )


def summarize(
    kind: str, scope: str, observed: int, counts: tuple[int, ...]
) -> Control:
    """Builds the result for an observed count and the re-run counts.

    Args:
        kind: `permuted_target` or `resampling`.
        scope: `strata`, `blocks` or `global`.
        observed: Discoveries on the real data.
        counts: Discovery counts of the re-runs.

    Returns:
        The control result, with the 95th percentile, its interval and the
        exceedance p-value computed from `counts`.

    Raises:
        ValueError: If there are fewer re-runs than the minimum number of
            permutations, or `observed` is not a non-negative integer.
    """
    qualification.count(observed, "observed")
    if len(counts) < budget.MIN_PERMUTATIONS:
        raise ValueError(
            f"a control needs at least {budget.MIN_PERMUTATIONS} re-runs"
        )
    p95 = percentile(counts)
    as_large = sum(1 for c in counts if c >= observed)
    return Control(
        kind=kind,
        scope=scope,
        observed=observed,
        permuted=counts,
        p95=p95,
        p95_interval=percentile_interval(counts),
        exceedance_p=(1 + as_large) / (len(counts) + 1),
        exceeds=observed > p95,
    )


def permuted_target_control(
    discoveries: COUNT_FUNCTION,
    features: pandas.DataFrame,
    target: pandas.Series,
    *,
    observed: int,
    b: int,
    seed: int,
    strata: numpy.ndarray | None,
    component: str = "ledger.control.permuted_target",
) -> Control:
    """Re-runs the exploration with the target permuted within strata.

    Args:
        discoveries: Runs the *whole* exploration (registration, tests and
            the online rule) on the given data and returns the number of
            rejections. It must be the same function used on the real data.
        features: Exploration features, unchanged in every re-run.
        target: Exploration target.
        observed: Discoveries on the real data.
        b: Number of re-runs, at least 20.
        seed: Run seed; re-run `i` uses stream ``(seed, component, i)``.
        strata: Group label or time-block label per row. Pass None only for
            rows that are exchangeable (no groups, no time order): the result
            is then labelled `global`.
        component: Name of the random stream.

    Returns:
        The control result, of kind `permuted_target` and scope `global`
        (no strata) or `strata`.

    Raises:
        ValueError: If `b` is below the minimum or `strata` has the wrong
            length.
    """
    qualification.count(b, "b", minimum=budget.MIN_PERMUTATIONS)
    if strata is not None and len(strata) != len(target):
        raise ValueError("strata needs one label per row")
    counts = []
    for repeat in range(b):
        rng = randomness.component_rng(seed, component, repeat)
        values = permute_within(target.to_numpy(), strata, rng)
        shuffled = pandas.Series(values, index=target.index, name=target.name)
        counts.append(int(discoveries(features, shuffled)))
    return summarize(
        PERMUTED_TARGET,
        "global" if strata is None else "strata",
        observed,
        tuple(counts),
    )


def shuffle_columns(
    features: pandas.DataFrame,
    rng: numpy.random.Generator,
    strata: numpy.ndarray | None,
) -> pandas.DataFrame:
    """Shuffles every column independently (within strata when given).

    Args:
        features: The columns to shuffle.
        rng: Random generator; advanced by each column's shuffle.
        strata: Label per row within which values are shuffled, or None for
            a global shuffle.

    Returns:
        A new frame with the same index and columns.
    """
    out = {}
    for name in features.columns:
        column = features[name].to_numpy()
        out[name] = permute_within(column, strata, rng)
    return pandas.DataFrame(out, index=features.index)


def shuffle_blocks(
    features: pandas.DataFrame,
    blocks: numpy.ndarray,
    rng: numpy.random.Generator,
) -> pandas.DataFrame:
    """Shuffles the order of contiguous blocks of each column independently.

    Rows are first ordered by block label; a column's values keep their order
    inside each block but the blocks are laid out in a random order, which
    destroys slow drift while keeping short-range structure.

    Args:
        features: The columns to shuffle.
        blocks: Block label per row.
        rng: Random generator; one block order is drawn per column.

    Returns:
        A new frame with the same index and columns.
    """
    labels = numpy.asarray(blocks)
    order = numpy.argsort(labels, kind="stable")
    sorted_labels = labels[order]
    cuts = numpy.flatnonzero(sorted_labels[1:] != sorted_labels[:-1]) + 1
    pieces = numpy.split(order, cuts)
    out = {}
    for name in features.columns:
        column = features[name].to_numpy()
        shuffled = numpy.array(column, copy=True)
        sequence = numpy.concatenate(
            [pieces[i] for i in rng.permutation(len(pieces))]
        )
        shuffled[order] = column[sequence]
        out[name] = shuffled
    return pandas.DataFrame(out, index=features.index)


def resampling_control(
    discoveries: COUNT_FUNCTION,
    features: pandas.DataFrame,
    target: pandas.Series,
    *,
    observed: int,
    b: int,
    seed: int,
    mode: str = "columns",
    strata: numpy.ndarray | None = None,
    blocks: numpy.ndarray | None = None,
    component: str = "ledger.control.resampling",
) -> Control:
    """Re-runs the exploration with the features shuffled, ignoring the target.

    Args:
        discoveries: Runs the whole exploration and returns the rejections.
        features: Exploration features.
        target: Exploration target, passed through unchanged.
        observed: Discoveries on the real data.
        b: Number of re-runs, at least 20.
        seed: Run seed.
        mode: `columns` (independent column shuffles, breaks association
            between features) or `blocks` (shuffles contiguous blocks of each
            column, breaks time drift; needs `blocks`).
        strata: Labels within which `columns` shuffles; None is global.
        blocks: Block label per row for `blocks` mode.
        component: Name of the random stream.

    Returns:
        The control result (kind `resampling`); its scope is `blocks` in
        `blocks` mode, otherwise `strata` or `global`.

    Raises:
        ValueError: On an unknown mode, `blocks` mode without labels, or
            `b` below the minimum.
    """
    qualification.count(b, "b", minimum=budget.MIN_PERMUTATIONS)
    if mode not in ("columns", "blocks"):
        raise ValueError("mode must be 'columns' or 'blocks'")
    if mode == "blocks" and blocks is None:
        raise ValueError("blocks mode needs block labels")
    labels = blocks
    counts = []
    for repeat in range(b):
        rng = randomness.component_rng(seed, component, repeat)
        if labels is not None and mode == "blocks":
            changed = shuffle_blocks(features, labels, rng)
        else:
            changed = shuffle_columns(features, rng, strata)
        counts.append(int(discoveries(changed, target)))
    scope = (
        "blocks"
        if mode == "blocks"
        else ("global" if strata is None else "strata")
    )
    return summarize(RESAMPLING, scope, observed, tuple(counts))
