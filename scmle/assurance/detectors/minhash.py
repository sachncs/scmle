"""MinHash and locality-sensitive hashing for near-duplicate rows and texts.

A document is a set: a text's character shingles (`shingles`) or a row's
``column=value`` tokens (`tokens`). Two documents are near copies when the
Jaccard similarity of their sets is at least a threshold. MinHash
signatures estimate that similarity; banding them (locality-sensitive
hashing) proposes candidate pairs in near-linear time, and every candidate
is then verified on the exact Jaccard similarity, so a reported pair is
always at or above the threshold (no false positive by construction).

Derivation (every threshold is a variable):

- Threshold. A copy that changes at most a share `edits` of a text's
  characters changes at most ``width`` shingles per edited character, so
  ``|A n B| >= L - w e`` and ``|A u B| <= L + w e`` for ``L`` distinct
  shingles and ``e`` edited characters: the similarity is at least
  ``(1 - w d) / (1 + w d)`` with ``d = e / L`` (`similarity`; ``L`` is about
  the text's length, and the bound is tight). For a row, a token is a
  cell, so ``w = 1`` and ``d`` is the share of cells changed.
- Bands. A pair at similarity ``s`` shares one band of ``r`` signature rows
  with probability ``s^r``, so it is missed by all ``b`` bands with
  probability ``(1 - s^r)^b``. `Bands` takes the largest ``r`` (the fewest
  false candidates) whose ``b = ceil(ln miss / ln(1 - s^r))`` fits in
  `PERMUTATIONS`, so a pair at the threshold is found with probability at
  least ``1 - miss`` and a pair above it with more.

The hashes are seeded and process-independent (BLAKE2b, then universal
hashing modulo the Mersenne prime ``2^31 - 1``), so the same input gives the
same pairs on every platform.
"""

from collections.abc import Sequence
import dataclasses
import hashlib
import math

import numpy
import pandas

from scmle.contracts import errors

WIDTH = 5
EDITS = 0.02
MISS = 0.05
PERMUTATIONS = 128
PRIME = 2**31 - 1
SEED = 0
DECIMALS = 6
MISSING = "<missing>"


def similarity(edits: float = EDITS, width: int = WIDTH) -> float:
    """The lowest Jaccard similarity of a copy with a share of edits.

    Args:
        edits: Largest number of edited characters (or cells) per
            distinct shingle (or cell) of the original that a near copy
            makes, within [0, 1) (policy: two in a hundred, typing and
            formatting noise).
        width: Shingle width in characters (policy: 5, the width Leskovec,
            Rajaraman and Ullman advise for short documents); 1 for a row's
            cells.

    Returns:
        ``(1 - width * edits) / (1 + width * edits)``.

    Raises:
        InputError: If `width` is not a positive integer, or the edits
            leave no similarity (``width * edits`` not below 1).
    """
    if int(width) != width or width < 1:
        raise errors.InputError("width must be a positive integer")
    if not 0.0 <= edits < 1.0 or width * edits >= 1.0:
        raise errors.InputError(
            "edits times width must lie within [0, 1) for a threshold"
        )
    return (1 - width * edits) / (1 + width * edits)


def shingles(text: str, width: int = WIDTH) -> frozenset[str]:
    """Splits a text into its character shingles.

    Case is folded and runs of whitespace become one space first.

    Args:
        text: The document.
        width: Shingle width.

    Returns:
        The set of substrings of `width` characters; a shorter non-empty
        text is its own single shingle, an empty one has none.
    """
    plain = " ".join(str(text).casefold().split())
    if len(plain) <= width:
        return frozenset([plain]) if plain else frozenset()
    return frozenset(
        plain[i : i + width] for i in range(len(plain) - width + 1)
    )


def tokens(frame: pandas.DataFrame) -> list[frozenset[str]]:
    """Turns each row into its set of ``column=value`` cells.

    Floats are rounded to six decimals, as the exact-duplicate grouping
    does, and missing values share one marker.

    Args:
        frame: Table.

    Returns:
        One token set per row, in row order.
    """
    columns = []
    for name in frame.columns:
        values = frame[name]
        if pandas.api.types.is_float_dtype(values):
            values = values.round(DECIMALS)
        text = values.astype(str).where(values.notna(), MISSING)
        columns.append((f"{name}\x1f" + text).to_numpy())
    if not columns:
        return [frozenset()] * len(frame)
    return [frozenset(cells) for cells in zip(*columns, strict=True)]


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    """The exact Jaccard similarity of two sets.

    Args:
        left: One set.
        right: The other.

    Returns:
        ``|left n right| / |left u right|``; 0.0 when both are empty.
    """
    union = len(left | right)
    return len(left & right) / union if union else 0.0


@dataclasses.dataclass(frozen=True, slots=True)
class Bands:
    """The banding of MinHash signatures for a similarity threshold.

    Attributes:
        threshold: Jaccard similarity at and above which a pair is a near
            copy, within (0, 1].
        miss: Largest probability that a pair at the threshold is not
            proposed, within (0, 1) (the plan's ``alpha``).
        cap: Largest signature length (policy: cost per document).
    """

    threshold: float
    miss: float = MISS
    cap: int = PERMUTATIONS

    def __post_init__(self) -> None:
        """Validates the parameters and that some banding fits the cap.

        Raises:
            InputError: If a parameter is outside its range or no banding
                within `cap` reaches the miss probability.
        """
        if not 0.0 < self.threshold <= 1.0:
            raise errors.InputError("threshold must be within (0, 1]")
        if not 0.0 < self.miss < 1.0:
            raise errors.InputError("miss must be within (0, 1)")
        if isinstance(self.cap, bool) or not isinstance(self.cap, int):
            raise errors.InputError("cap must be an integer")
        if self.cap < 1 or self.shape is None:
            raise errors.InputError(
                f"no banding of at most {self.cap} hashes finds a pair at "
                f"{self.threshold:g} with probability {1 - self.miss:g}"
            )

    @property
    def shape(self) -> tuple[int, int] | None:
        """The rows per band and the number of bands, or None if none fit."""
        for rows in range(self.cap, 0, -1):
            hit = self.threshold**rows
            if hit >= 1.0:
                return rows, 1
            if hit <= 0.0:
                continue
            count = math.ceil(math.log(self.miss) / math.log1p(-hit))
            if rows * count <= self.cap:
                return rows, count
        return None

    @property
    def rows(self) -> int:
        """Signature rows per band."""
        return self.layout()[0]

    @property
    def count(self) -> int:
        """Number of bands."""
        return self.layout()[1]

    def layout(self) -> tuple[int, int]:
        """Returns the validated shape.

        Returns:
            The rows per band and the number of bands.

        Raises:
            InputError: If no banding fits (refused on construction).
        """
        shape = self.shape
        if shape is None:
            raise errors.InputError("no banding fits the cap")
        return shape

    def chance(self, value: float) -> float:
        """The probability that a pair at a similarity is proposed.

        Args:
            value: The pair's Jaccard similarity.

        Returns:
            ``1 - (1 - value^r)^b``.
        """
        rows, count = self.layout()
        return 1.0 - (1.0 - value**rows) ** count


def hashes(items: Sequence[str], cache: dict[str, int]) -> numpy.ndarray:
    """Hashes strings to integers below `PRIME`, the same on every platform.

    Args:
        items: The strings.
        cache: Hashes already computed, extended in place.

    Returns:
        One unsigned 64-bit hash per string.
    """
    for item in items:
        if item not in cache:
            digest = hashlib.blake2b(item.encode("utf-8"), digest_size=8)
            cache[item] = int.from_bytes(digest.digest(), "little") % PRIME
    return numpy.fromiter((cache[i] for i in items), dtype=numpy.uint64)


def signatures(
    sets: Sequence[frozenset[str]], length: int, seed: int = SEED
) -> numpy.ndarray:
    """Computes the MinHash signature of every set.

    Args:
        sets: The documents' sets.
        length: Signature length (hash functions).
        seed: Seed of the hash functions.

    Returns:
        A ``(len(sets), length)`` array; an empty set's row is all `PRIME`
        (it matches nothing).
    """
    generator = numpy.random.default_rng(seed)
    slope = generator.integers(1, PRIME, size=length, dtype=numpy.uint64)
    shift = generator.integers(0, PRIME, size=length, dtype=numpy.uint64)
    table = numpy.full((len(sets), length), PRIME, dtype=numpy.uint64)
    cache: dict[str, int] = {}
    for row, items in enumerate(sets):
        if items:
            values = hashes(sorted(items), cache)
            mixed = (slope[:, None] * values[None, :] + shift[:, None]) % PRIME
            table[row] = mixed.min(axis=1)
    return table


def candidates(table: numpy.ndarray, bands: Bands) -> set[tuple[int, int]]:
    """Proposes the pairs that share a band of their signatures.

    Args:
        table: Signatures from `signatures`.
        bands: The banding.

    Returns:
        Pairs ``(i, j)`` with ``i < j``; empty sets are never proposed.
    """
    rows, count = bands.layout()
    found: set[tuple[int, int]] = set()
    filled = numpy.flatnonzero(~numpy.all(table == PRIME, axis=1))
    for band in range(count):
        buckets: dict[bytes, list[int]] = {}
        part = numpy.ascontiguousarray(
            table[:, band * rows : (band + 1) * rows]
        )
        for position in filled:
            key = part[position].tobytes()
            buckets.setdefault(key, []).append(int(position))
        for members in buckets.values():
            found.update(
                (members[i], members[j])
                for i in range(len(members))
                for j in range(i + 1, len(members))
            )
    return found


def links(
    keys: Sequence[frozenset[str]], bands: Bands, seed: int = SEED
) -> list[tuple[int, int, float]]:
    """Finds the near copies among distinct sets.

    Args:
        keys: Distinct sets.
        bands: The banding, with its threshold.
        seed: Seed of the hash functions.

    Returns:
        ``(i, j, similarity)`` for every pair ``i < j`` whose exact
        similarity is at least the threshold, sorted.
    """
    table = signatures(keys, bands.rows * bands.count, seed)
    found = []
    for left, right in candidates(table, bands):
        value = jaccard(keys[left], keys[right])
        if value >= bands.threshold:
            found.append((left, right, value))
    return sorted(found)


def group(
    sets: Sequence[frozenset[str]],
) -> tuple[list[frozenset[str]], numpy.ndarray]:
    """Groups identical sets.

    Args:
        sets: The documents' sets.

    Returns:
        The distinct sets in order of first appearance, and for each
        document the position of its set among them.
    """
    seen: dict[frozenset[str], int] = {}
    labels = numpy.fromiter(
        (seen.setdefault(items, len(seen)) for items in sets),
        dtype=numpy.int64,
        count=len(sets),
    )
    return list(seen), labels


def pairs(
    sets: Sequence[frozenset[str]],
    threshold: float | None = None,
    miss: float = MISS,
    seed: int = SEED,
) -> list[tuple[int, int, float]]:
    """Finds the near copies among documents.

    Identical sets are grouped first, so a value repeated many times costs
    one signature; every proposed pair is verified exactly. Identical
    non-empty documents are pairs at similarity 1.

    Args:
        sets: The documents' sets.
        threshold: Jaccard threshold; `similarity()` by default.
        miss: Largest probability of missing a pair at the threshold.
        seed: Seed of the hash functions.

    Returns:
        ``(i, j, similarity)`` for every pair ``i < j`` of documents whose
        exact similarity is at least the threshold, sorted.
    """
    bands = Bands(similarity() if threshold is None else threshold, miss)
    keys, labels = group(sets)
    members = [
        numpy.flatnonzero(labels == k).tolist() for k in range(len(keys))
    ]
    found = [
        (inside[i], inside[j], 1.0)
        for key, inside in zip(keys, members, strict=True)
        if key
        for i in range(len(inside))
        for j in range(i + 1, len(inside))
    ]
    for left, right, value in links(keys, bands, seed):
        found.extend(
            (min(i, j), max(i, j), value)
            for i in members[left]
            for j in members[right]
        )
    return sorted(found)


def flags(
    sets: Sequence[frozenset[str]],
    threshold: float | None = None,
    miss: float = MISS,
    seed: int = SEED,
) -> numpy.ndarray:
    """Flags every document that nearly copies an earlier one.

    Args:
        sets: The documents' sets.
        threshold: Jaccard threshold; `similarity()` by default.
        miss: Largest probability of missing a pair at the threshold.
        seed: Seed of the hash functions.

    Returns:
        A boolean per document, True when an earlier document is at or
        above the threshold (an identical non-empty one included).
    """
    bands = Bands(similarity() if threshold is None else threshold, miss)
    keys, labels = group(sets)
    first = numpy.full(len(keys), len(sets), dtype=numpy.int64)
    numpy.minimum.at(first, labels, numpy.arange(len(sets)))
    earliest = first.copy()
    for link in links(keys, bands, seed):
        left, right = link[0], link[1]
        earliest[left] = min(earliest[left], first[right])
        earliest[right] = min(earliest[right], first[left])
    filled = numpy.array([bool(k) for k in keys], dtype=bool)
    positions = numpy.arange(len(sets))
    return numpy.asarray(filled[labels] & (earliest[labels] < positions))
