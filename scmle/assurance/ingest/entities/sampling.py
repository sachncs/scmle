"""Clerical sampling by score band, and the thresholds it sets.

Precision and recall of a match call are estimated from a stratified sample
of labelled pairs, not from the model's own probabilities.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import itertools
import math
from typing import Any

import numpy

from scmle.assurance.ingest.entities.blocking import Pair
from scmle.contracts import qualification
from scmle.runtime import randomness

GROUP_PROBABILITY = 0.5


BANDS = 10


Labeler = Callable[[int, int], bool]


@dataclasses.dataclass(frozen=True, slots=True)
class Band:
    """Clerical-sample result for one score band.

    Attributes:
        low: Lowest weight in the band.
        high: Highest weight in the band.
        pairs: Pairs in the band.
        sampled: Pairs labelled.
        matches: Labelled pairs that were true matches.
    """

    low: float
    high: float
    pairs: int
    sampled: int
    matches: int

    def to_dict(self) -> dict[str, Any]:
        """Serializes the band.

        Returns:
            A JSON-ready mapping of the fields.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Estimate:
    """Precision and recall of the link rule, with intervals.

    Attributes:
        precision: Point estimate and 95 percent interval of the share of
            linked pairs that are true matches.
        recall: Share of true matches among the compared pairs that are
            linked. Matches the blocking never compared are not in it.
        bands: The per-band sample.
        reviewer: Who labelled the sample.
    """

    precision: tuple[float, float, float]
    recall: tuple[float, float, float]
    bands: tuple[Band, ...]
    reviewer: str

    def to_dict(self) -> dict[str, Any]:
        """Serializes the estimate.

        Returns:
            A JSON-ready mapping with precision, recall, bands and the
            reviewer.
        """
        return {
            "precision": list(self.precision),
            "recall": list(self.recall),
            "bands": [b.to_dict() for b in self.bands],
            "reviewer": self.reviewer,
        }


def stratify(order: numpy.ndarray, bands: int) -> list[numpy.ndarray]:
    """Splits ascending-ordered pairs into bands that thin out at the top.

    The top band holds the highest `2^-(bands-1)` share of the pairs, the next
    one twice that, and so on; the bottom band takes the rest. Matches are
    rare and sit at the top, so equal-count bands would hide them in one band.

    Args:
        order: Pair indices sorted by ascending weight.
        bands: Number of bands wanted; fewer come back if the sizes
            collapse.

    Returns:
        The bands from lowest to highest weight; empty bands are dropped.
    """
    total = len(order)
    edges = [0]
    edges.extend(
        total - total // 2**k if k > 0 else total
        for k in range(bands - 1, 0, -1)
    )
    edges.append(total)
    edges = sorted(set(edges))
    return [order[a:b] for a, b in itertools.pairwise(edges) if b > a]


def appraise(
    weights: numpy.ndarray,
    pairs: Sequence[Pair],
    threshold: float,
    labeler: Labeler,
    reviewer: str,
    per_band: int,
    seed: int,
    bands: int = BANDS,
    draws: int = 2000,
) -> Estimate:
    """Estimates precision and recall of `weight >= threshold` by sampling.

    Pairs are cut into `bands` score bands (`stratify`, thinning out
    toward the top); `per_band` pairs are drawn in each (all, if fewer) and
    labelled by `labeler`. Each band's match rate gets a Beta posterior with
    the Jeffreys prior and the intervals come from `draws` seeded joint
    draws, which is an approximation, not exact.

    Args:
        weights: Pair weights, aligned with `pairs`.
        pairs: Record-index pairs.
        threshold: Pairs with weight at or above it count as linked.
        labeler: Gives the true match status of a pair.
        reviewer: Who labelled the sample.
        per_band: Pairs sampled per band.
        seed: Seed of the sample and the draws.
        bands: Number of score bands.
        draws: Number of posterior draws for the intervals.

    Returns:
        Precision and recall, each as (point, lower, upper) with a 95
        percent interval, the per-band sample and the reviewer.
    """
    rng = randomness.component_rng(seed, "ingest.entities.clerical", 0)
    order = numpy.argsort(weights, kind="stable")
    cuts = stratify(order, bands)
    sizes, hits, taken, above = [], [], [], []
    summary = []
    for members in cuts:
        if len(members) == 0:
            continue
        pick = rng.choice(
            len(members), min(per_band, len(members)), replace=False
        )
        labelled = [labeler(*pairs[int(members[k])]) for k in pick]
        sizes.append(len(members))
        hits.append(sum(labelled))
        taken.append(len(labelled))
        above.append(bool(weights[members].min() >= threshold))
        summary.append(
            Band(
                float(weights[members].min()),
                float(weights[members].max()),
                len(members),
                len(labelled),
                sum(labelled),
            )
        )
    return infer((sizes, hits, taken), above, summary, reviewer, draws, rng)


def infer(
    counts: tuple[list[int], list[int], list[int]],
    above: list[bool],
    summary: list[Band],
    reviewer: str,
    draws: int,
    rng: numpy.random.Generator,
) -> Estimate:
    """Carries the per-band labels through to precision and recall.

    Each band's match rate gets a Beta posterior with the Jeffreys prior;
    the intervals come from `draws` joint draws of `rng`.

    Args:
        counts: Per band, the pairs in it, the sampled matches and the
            pairs sampled.
        above: Per band, whether it lies at or above the threshold.
        summary: The bands, for the estimate.
        reviewer: Who labelled the sample.
        draws: Number of posterior draws for the intervals.
        rng: Generator of the draws, after the sample was taken.

    Returns:
        Precision and recall, each as (point, lower, upper) with a 95
        percent interval, the per-band sample and the reviewer.
    """
    sizes, hits, taken = counts
    n = numpy.array(sizes, dtype=float)
    k = numpy.array(hits, dtype=float)
    t = numpy.array(taken, dtype=float)
    mask = numpy.array(above)
    rates = rng.beta(k + 0.5, t - k + 0.5, size=(draws, len(n)))
    matches = rates * n
    linked_true = matches[:, mask].sum(axis=1)
    linked = n[mask].sum()
    total_true = matches.sum(axis=1)
    p = linked_true / linked if linked else numpy.zeros(draws)
    r = numpy.divide(
        linked_true, total_true, out=numpy.zeros(draws), where=total_true > 0
    )
    point_rate = (k + 0.5) / (t + 1.0)
    point_true = point_rate * n
    point_p = point_true[mask].sum() / linked if linked else 0.0
    point_r = (
        point_true[mask].sum() / point_true.sum() if point_true.sum() else 0.0
    )
    lo, hi = numpy.percentile(p, [2.5, 97.5])
    rlo, rhi = numpy.percentile(r, [2.5, 97.5])
    return Estimate(
        (float(point_p), float(lo), float(hi)),
        (float(point_r), float(rlo), float(rhi)),
        tuple(summary),
        reviewer,
    )


def calibrate(
    weights: numpy.ndarray,
    pairs: Sequence[Pair],
    prior: float,
    labeler: Labeler | None,
    reviewer: str,
    per_band: int,
    min_precision: float,
    seed: int,
) -> tuple[float, float, Estimate | None, qualification.Tier]:
    """Sets the match and grouping thresholds of the pair weights.

    With a labeler they come from the clerical sample (`cutoffs`) and the
    result is attested; without one they come from the model's posterior
    match probability and the result is unverified.

    Args:
        weights: Pair weights, aligned with `pairs`.
        pairs: Record-index pairs.
        prior: The model's prior match probability.
        labeler: Gives the true match status of a pair, or None.
        reviewer: Who labels.
        per_band: Pairs sampled per score band.
        min_precision: Target precision of the match call.
        seed: Seed of the sample.

    Returns:
        The match threshold, the grouping threshold (not yet capped at the
        match threshold), the clerical estimate or None, and the tier.
    """
    estimate = None
    tier = qualification.Tier.UNVERIFIED
    if labeler is not None:
        estimate_at = appraise(
            weights, pairs, -math.inf, labeler, reviewer, per_band, seed
        )
        t_match, t_group = cutoffs(estimate_at.bands, min_precision)
        estimate = appraise(
            weights, pairs, t_match, labeler, reviewer, per_band, seed
        )
        tier = qualification.Tier.ATTESTED
    else:
        posterior_odds = (
            2.0 ** numpy.clip(weights, -60.0, 60.0) * prior / (1.0 - prior)
        )
        posterior = posterior_odds / (1.0 + posterior_odds)
        t_match = weakest(weights, posterior >= min_precision)
        t_group = weakest(weights, posterior >= GROUP_PROBABILITY)
    return t_match, t_group, estimate, tier


def weakest(weights: numpy.ndarray, keep: numpy.ndarray) -> float:
    """Finds the smallest weight among selected pairs.

    Args:
        weights: Pair weights.
        keep: Boolean mask of the selected pairs.

    Returns:
        The smallest selected weight, or infinity when none is selected.
    """
    chosen = weights[keep]
    return float(chosen.min()) if len(chosen) else math.inf


def cutoffs(bands: Sequence[Band], min_precision: float) -> tuple[float, float]:
    """Derives the match and group thresholds from the labelled bands.

    Bands are scanned from the top. `t_match` is the lowest band edge such
    that the cumulative sampled precision from the top stays at or above
    `min_precision`; `t_group` the lowest edge whose own sampled match rate
    is at least one half.

    Args:
        bands: Labelled bands, from lowest to highest weight.
        min_precision: Cumulative precision the match call must keep.

    Returns:
        The pair `(t_match, t_group)`; each is infinity if no band
        qualifies.
    """
    t_match = math.inf
    t_group = math.inf
    linked = true = 0.0
    for band in reversed(bands):
        rate = band.matches / band.sampled if band.sampled else 0.0
        linked += band.pairs
        true += rate * band.pairs
        if linked and true / linked >= min_precision:
            t_match = band.low
        if rate >= GROUP_PROBABILITY:
            t_group = band.low
    return t_match, t_group
