"""Spatial blocks for evaluation: a block side derived from the data (MM-11).

Nearby features share their label more than distant ones, so a split that
puts neighbours on both sides scores the model on rows it has in effect
seen. Blocked cross-validation keeps every square block of side `s` on one
side; `s` must reach at least as far as the label's spatial dependence
(Roberts et al. 2017, "Cross-validation strategies for data with temporal,
spatial, hierarchical, or phylogenetic structure", Ecography 40:913-929).

The side is derived, not chosen: `variogram` estimates the empirical
semivariogram of the label over a capped sample, its practical range is the
first lag whose semivariance reaches `REACH` of the sill, and `snap` rounds
the range up to the 1-2-5 series so a small change in the data leaves the
side unchanged. `label` anchors the grid at the CRS origin, never at the
data's minimum, so adding a feature never moves another feature's block.
"""

from collections.abc import Sequence
import dataclasses
import math

import numpy
import pandas
from scipy.spatial import distance

from scmle.contracts import errors

SAMPLE = 2000
"""Most features the semivariogram is estimated from.

2,000 features give n(n-1)/2 = 1,999,000 pairs; spread over `LAGS` lag
classes that is about 100,000 pairs a class, so the standard error of a
class's mean semivariance is about sqrt(2 / 100,000) = 0.45% of the sill,
an order of magnitude inside the 5% band `REACH` leaves. The pair distances
take 16 MB (8 bytes each)."""

LAGS = 20
"""Lag classes between zero and half the largest pair distance.

The range is resolved to 1/20 of the half extent (2.5% of the extent); the
estimate stops at half the extent, beyond which too few pairs remain for a
stable mean (Journel and Huijbregts 1978, Mining Geostatistics, p. 194)."""

PAIRS = 30
"""Fewest pairs a lag class needs to be read (Journel and Huijbregts 1978:
at least 30 pairs per lag)."""

REACH = 0.95
"""Fraction of the sill that defines the practical range: the lag at which
an exponential semivariogram reaches 95% of its sill (Cressie 1993,
Statistics for Spatial Data, section 2.3.1)."""

SERIES = (1.0, 2.0, 5.0)
"""Mantissas of the 1-2-5 series a block side is rounded up to."""

STEPS = 200
"""Most 1-2-5 steps `design` takes down from the range: from 1e9 CRS units
a step of at least 2x reaches 1e-51, below any coordinate a float tells
apart."""

SECONDS = 0.5
"""Bound on the seconds `variogram` takes on `SAMPLE` features: observed at
0.02 s on a laptop (Apple M-series, one thread), with a 25x margin for
slower and shared machines."""


@dataclasses.dataclass(frozen=True)
class Variogram:
    """The empirical semivariogram of a label and its practical range.

    Attributes:
        lags: Upper edge of each lag class, in CRS units.
        values: Mean semivariance of each class (NaN below `PAIRS` pairs).
        sill: Mean semivariance over every pair.
        reach: The practical range: the first lag edge whose class reaches
            `REACH` of the sill; the first edge when the label has no
            variation or no spatial structure; half the extent when it never
            levels off (a trend across the whole area).
        pairs: Pairs of features the estimate used.
    """

    lags: tuple[float, ...]
    values: tuple[float, ...]
    sill: float
    reach: float
    pairs: int


def variogram(
    points: numpy.ndarray, values: numpy.ndarray, seed: int
) -> Variogram:
    """Estimates the semivariogram of a label and its practical range.

    The sample is drawn from the features sorted by position, so the result
    does not depend on the order of the rows. A categorical label enters as
    its indicator columns, whose semivariance is half the chance two
    features differ.

    Args:
        points: (n, 2) feature centres, in CRS units.
        values: (n, k) label values (one column for a number, one indicator
            column per class); rows with a missing value are left out.
        seed: Seed of the sample draw.

    Returns:
        The semivariogram; with fewer than two usable features, or all at
        one place, every lag and the range are zero.
    """
    usable = numpy.isfinite(values).all(axis=1) & numpy.isfinite(points).all(
        axis=1
    )
    points, values = points[usable], values[usable]
    order = numpy.lexsort((points[:, 1], points[:, 0]))
    points, values = points[order], values[order]
    if len(points) > SAMPLE:
        chosen = numpy.sort(
            numpy.random.default_rng(seed).choice(
                len(points), SAMPLE, replace=False
            )
        )
        points, values = points[chosen], values[chosen]
    if len(points) < 2:  # noqa: PLR2004 -- a pair needs two features
        return Variogram((0.0,) * LAGS, (math.nan,) * LAGS, 0.0, 0.0, 0)
    gaps = distance.pdist(points)
    halves = distance.pdist(values, "sqeuclidean") / 2
    extent = float(gaps.max()) / 2
    if extent == 0:
        return Variogram((0.0,) * LAGS, (math.nan,) * LAGS, 0.0, 0.0, len(gaps))
    edges = numpy.linspace(0.0, extent, LAGS + 1)[1:]
    inside = gaps <= extent
    classes = numpy.minimum(
        numpy.ceil(gaps[inside] / extent * LAGS).astype(int) - 1, LAGS - 1
    ).clip(0)
    counts = numpy.bincount(classes, minlength=LAGS)
    sums = numpy.bincount(classes, weights=halves[inside], minlength=LAGS)
    means = numpy.where(
        counts >= PAIRS, sums / numpy.maximum(counts, 1), numpy.nan
    )
    sill = float(halves.mean())
    return Variogram(
        tuple(float(edge) for edge in edges),
        tuple(float(mean) for mean in means),
        sill,
        reach(edges, means, sill),
        len(gaps),
    )


def reach(edges: numpy.ndarray, means: numpy.ndarray, sill: float) -> float:
    """The practical range of a semivariogram.

    Args:
        edges: Upper edge of each lag class.
        means: Mean semivariance of each class (NaN when too few pairs).
        sill: Mean semivariance over every pair.

    Returns:
        The first edge whose class reaches `REACH` of the sill; the first
        edge when the sill is zero; the last edge when no class reaches it.
    """
    if sill <= 0:
        return float(edges[0])
    reached = numpy.flatnonzero(
        numpy.nan_to_num(means, nan=-1.0) >= REACH * sill
    )
    return float(edges[reached[0]] if len(reached) else edges[-1])


def snap(side: float) -> float:
    """Rounds a length up to the 1-2-5 series (1, 2, 5, 10, 20, ...).

    Args:
        side: A positive length.

    Returns:
        The smallest series value at least `side`.
    """
    power = math.floor(math.log10(side))
    for scale in (power, power + 1):
        for mantissa in SERIES:
            value = mantissa * 10.0**scale
            if value >= side * (1 - 1e-12):
                return float(f"{value:.12g}")
    return float(f"{10.0 ** (power + 1):.12g}")


def shrink(side: float) -> float:
    """The next smaller value of the 1-2-5 series.

    Args:
        side: A value of the series.

    Returns:
        The largest series value below `side`.
    """
    power = math.floor(math.log10(side) + 1e-12)
    mantissa = side / 10.0**power
    lower = [value for value in SERIES if value < mantissa * (1 - 1e-12)]
    if lower:
        return float(f"{lower[-1] * 10.0**power:.12g}")
    return float(f"{SERIES[-1] * 10.0 ** (power - 1):.12g}")


def label(points: numpy.ndarray, side: float) -> numpy.ndarray:
    """The block of each point on a grid anchored at the CRS origin.

    Args:
        points: (n, 2) feature centres, in CRS units.
        side: The block side, in CRS units.

    Returns:
        One "column:row" block name per point.
    """
    cells = numpy.floor(numpy.asarray(points, dtype=numpy.float64) / side)
    return numpy.asarray([f"{int(x)}:{int(y)}" for x, y in cells], dtype=object)


def design(
    points: numpy.ndarray, gram: Variogram, needed: int
) -> tuple[float, numpy.ndarray]:
    """Chooses the block side and names each point's block.

    The side starts at the practical range rounded up to the 1-2-5 series
    and steps down the series only while there are fewer than `needed`
    blocks; a side below `gram.reach` then means the blocks are smaller
    than the range.

    Args:
        points: (n, 2) feature centres, in CRS units.
        gram: The label's semivariogram.
        needed: Fewest blocks the split needs.

    Returns:
        The side and the block of each point.

    Raises:
        BlockedError: If fewer distinct places exist than `needed`.
    """
    finite = points[numpy.isfinite(points).all(axis=1)]
    distinct = len(numpy.unique(finite, axis=0))
    if distinct < needed or gram.reach <= 0:
        raise errors.BlockedError(
            f"spatial blocks need at least {needed} distinct places; the "
            f"data have {distinct}"
        )
    side = snap(gram.reach)
    places = label(points, side)
    steps = 0
    while steps < STEPS:
        if len(set(places)) >= needed:
            return side, places
        side = shrink(side)
        places = label(points, side)
        steps += 1
    raise errors.BlockedError(
        f"spatial blocks found no side giving {needed} blocks"
    )


def encode(
    labels: Sequence[object] | pandas.Series, categorical: bool
) -> numpy.ndarray:
    """A label as the numeric columns its semivariogram is taken over.

    Args:
        labels: The label of each feature.
        categorical: Whether the label names classes; a label that is not
            numeric is taken as classes regardless.

    Returns:
        (n, 1) values for a numeric label, else one indicator column per
        class; a missing label is a row of NaN.
    """
    series = pandas.Series(labels)
    if pandas.api.types.is_numeric_dtype(series) and not categorical:
        return series.to_numpy(dtype=numpy.float64, na_value=numpy.nan)[:, None]
    values = numpy.array(
        pandas.get_dummies(series.astype("string"), dtype=float), copy=True
    )
    values[series.isna().to_numpy()] = numpy.nan
    return values
