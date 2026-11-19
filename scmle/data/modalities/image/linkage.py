"""Near-duplicate links between pictures described by their descriptors.

Two media rows are linked when they are the same picture up to small edits:
their luminance signatures (an average hash of the descriptor thumbnail,
`descriptor.SIDE` squared bits) differ in at most `DISTANCE` bits. The
image and video kinds join linked rows into one cohort, which the shared
splitter keeps on one side of the held-out split and of every fold (MM-10,
A9).

The rows are compared all against all, a block of rows at a time sized so
a block holds at most `CELLS` comparisons: linking is quadratic in time
(20,000 rows take two hundred million 64-bit comparisons) and bounded in
memory.
"""

import math

import numpy
import pandas

from scmle.contracts import errors
from scmle.data.modalities.image import descriptor

BITS = descriptor.SIDE**2
"""Bits of a signature: one per thumbnail cell."""

REFERENCE = 100_000
"""Rows of the largest collection the false-link bound is stated for."""

FALSE = 1 / math.comb(REFERENCE, 2)
"""Largest chance a pair of unrelated rows is linked: at most one false
link is expected among all pairs of `REFERENCE` rows."""


def tail(distance: int) -> float:
    """Chance two unrelated signatures differ in at most `distance` bits.

    Average-hash bits are balanced by construction (each is whether a cell
    is brighter than the mean), so for unrelated pictures the differing
    bits count as Binomial(`BITS`, 1/2).

    Args:
        distance: Differing bits.

    Returns:
        The lower tail of that binomial at `distance`.
    """
    return float(
        sum(math.comb(BITS, bit) for bit in range(distance + 1)) / 2**BITS
    )


DISTANCE = max(bit for bit in range(BITS) if tail(bit) <= FALSE)
"""Most differing bits two linked signatures may have: the largest
distance whose null tail is at most `FALSE` (7 of 64 bits)."""

TOLERANCE = 1 / 16
"""Largest mean absolute thumbnail difference, in [0, 1], at which two flat
pictures are linked: one step of a 4-bit intensity quantisation."""

CELLS = 2**22
"""Most pairwise comparisons held at once: 32 MB of 64-bit values."""


def thumbnails(frame: pandas.DataFrame) -> numpy.ndarray:
    """Reads the RGB thumbnails back from the descriptor columns.

    Args:
        frame: Rows holding the `descriptor.names` thumbnail columns.

    Returns:
        A rows x `BITS` x 3 array.

    Raises:
        DataError: If a thumbnail column is missing.
    """
    names = [name for name in descriptor.names() if name.startswith("thumb_")]
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise errors.DataError(
            "media rows lack the thumbnail descriptor columns the "
            "near-duplicate check reads",
            {"missing": missing[:3], "count": len(missing)},
        )
    values = frame[names].to_numpy(dtype=numpy.float64)
    channels = len(descriptor.CHANNELS)
    return values.reshape(len(frame), channels, BITS).transpose(0, 2, 1)


def signatures(thumbs: numpy.ndarray) -> numpy.ndarray:
    """Average-hashes thumbnails.

    Args:
        thumbs: A rows x `BITS` x 3 array of thumbnails.

    Returns:
        One unsigned 64-bit signature per row: bit `i` is set when cell `i`
        is brighter than the thumbnail's mean.
    """
    gray = thumbs @ descriptor.LUMA
    bits = gray > gray.mean(axis=1, keepdims=True)
    weights = numpy.left_shift(
        numpy.uint64(1), numpy.arange(BITS, dtype=numpy.uint64)
    )
    codes: numpy.ndarray = (bits.astype(numpy.uint64) * weights).sum(
        axis=1, dtype=numpy.uint64
    )
    return codes


def flat(codes: numpy.ndarray) -> numpy.ndarray:
    """Whether each signature is within `DISTANCE` of a constant one.

    Such a signature carries no structure (a blank or single-colour
    picture), so it cannot tell two pictures apart.

    Args:
        codes: Signatures.

    Returns:
        A boolean per signature.
    """
    ones = numpy.bitwise_count(codes).astype(int)
    return (ones <= DISTANCE) | (ones >= BITS - DISTANCE)


def edges(frame: pandas.DataFrame) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Finds the linked pairs of rows.

    Structured pictures are linked by signature distance, flat ones by the
    mean absolute difference of their thumbnails; a flat and a structured
    picture are never linked.

    Args:
        frame: Rows holding the thumbnail columns.

    Returns:
        The left and right row positions of every linked pair.
    """
    thumbs = thumbnails(frame)
    codes = signatures(thumbs)
    blank = flat(codes)
    left: list[numpy.ndarray] = []
    right: list[numpy.ndarray] = []
    step = max(1, CELLS // max(1, len(codes)))
    for start in range(0, len(codes), step):
        block = codes[start : start + step]
        distance = numpy.bitwise_count(block[:, None] ^ codes[None, :])
        near = (distance <= DISTANCE) & ~blank[start : start + step, None]
        near &= ~blank[None, :]
        rows, columns = numpy.nonzero(near)
        left.append(rows + start)
        right.append(columns)
    plain = numpy.flatnonzero(blank)
    step = max(1, CELLS // max(1, len(plain) * thumbs[0].size))
    for start in range(0, len(plain), step):
        block = thumbs[plain[start : start + step]]
        difference = numpy.abs(block[:, None] - thumbs[plain][None, :])
        rows, columns = numpy.nonzero(difference.mean(axis=(2, 3)) <= TOLERANCE)
        left.append(plain[rows + start])
        right.append(plain[columns])
    return numpy.concatenate(left), numpy.concatenate(right)
