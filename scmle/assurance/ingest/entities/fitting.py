"""Comparison levels, the Fellegi-Sunter model fitted by EM, and grouping."""

from collections.abc import Sequence
import dataclasses
import math

import numpy

from scmle.assurance.ingest.entities.blocking import Pair

LEVELS = 3


MISSING = -1


DISAGREE = 0


SIMILAR = 1


AGREE = 2


SIMILAR_AT = 0.88


@dataclasses.dataclass(frozen=True, slots=True)
class Model:
    """Fitted Fellegi-Sunter parameters.

    Attributes:
        prior: Estimated share of match pairs among compared pairs.
        m: `m[field][level]` for levels disagree, similar, agree.
        u: `u[field][level]`.
        iterations: EM iterations run.
    """

    prior: float
    m: numpy.ndarray
    u: numpy.ndarray
    iterations: int

    def weights(self, levels: numpy.ndarray) -> numpy.ndarray:
        """Computes the log2 likelihood-ratio weight of each pair.

        Args:
            levels: Comparison levels, shape (pairs, fields), as `compare`
                returns them.

        Returns:
            One weight per pair; missing fields add 0.
        """
        total = numpy.zeros(len(levels))
        for col in range(levels.shape[1]):
            seen = levels[:, col] != MISSING
            level = numpy.where(seen, levels[:, col], 0)
            ratio = numpy.log2(self.m[col][level] / self.u[col][level])
            total += numpy.where(seen, ratio, 0.0)
        return total


def em(
    levels: numpy.ndarray, iterations: int = 100, tol: float = 1e-8
) -> Model:
    """EM for the two-class conditional-independence model.

    Starting values are fixed (m favours agreement, u favours disagreement),
    so the fit is deterministic. Parameters are floored away from 0 and 1.

    Args:
        levels: Comparison levels, shape (pairs, fields), as `compare`
            returns them.
        iterations: Largest number of EM iterations.
        tol: Stop when no parameter moves by this much or more.

    Returns:
        The fitted model, with the number of iterations run.
    """
    pairs, fields = levels.shape
    m = numpy.tile([0.05, 0.15, 0.80], (fields, 1))
    u = numpy.tile([0.90, 0.07, 0.03], (fields, 1))
    prior = 0.05
    floor = 1e-6
    done = 0
    for step in range(1, iterations + 1):
        done = step
        log_m = numpy.full(pairs, math.log(prior))
        log_u = numpy.full(pairs, math.log(1.0 - prior))
        for col in range(fields):
            seen = levels[:, col] != MISSING
            level = numpy.where(seen, levels[:, col], 0)
            match_rates = m[col][level]
            other_rates = u[col][level]
            log_m += numpy.where(seen, numpy.log(match_rates), 0.0)
            log_u += numpy.where(seen, numpy.log(other_rates), 0.0)
        top = numpy.maximum(log_m, log_u)
        gamma = numpy.exp(log_m - top) / (
            numpy.exp(log_m - top) + numpy.exp(log_u - top)
        )
        new_prior = float(numpy.clip(gamma.mean(), floor, 1 - floor))
        new_m, new_u = m.copy(), u.copy()
        for col in range(fields):
            seen = levels[:, col] != MISSING
            for lv in range(LEVELS):
                hit = seen & (levels[:, col] == lv)
                gm, gu = gamma[seen].sum(), (1 - gamma[seen]).sum()
                new_m[col][lv] = gamma[hit].sum() / gm if gm > 0 else m[col][lv]
                new_u[col][lv] = (
                    (1 - gamma[hit]).sum() / gu if gu > 0 else u[col][lv]
                )
        new_m = numpy.clip(new_m, floor, 1.0)
        new_u = numpy.clip(new_u, floor, 1.0)
        shift = max(
            abs(new_prior - prior),
            float(numpy.abs(new_m - m).max()),
            float(numpy.abs(new_u - u).max()),
        )
        m, u, prior = new_m, new_u, new_prior
        if shift < tol:
            break
    return Model(prior, m, u, done)


def components(count: int, links: Sequence[Pair]) -> list[int]:
    """Groups records into connected components of the links.

    Args:
        count: Number of records.
        links: Linked record-index pairs.

    Returns:
        The group id of each record; the id is the smallest record index in
        its component.
    """
    parent = list(range(count))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in links:
        a, b = find(i), find(j)
        if a != b:
            parent[max(a, b)] = min(a, b)
    return [find(i) for i in range(count)]
