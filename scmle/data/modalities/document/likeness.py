"""Near-duplicate documents: MinHash signatures, banded candidates, checks.

Two documents are near duplicates when the Jaccard resemblance of their
word-shingle sets reaches `THRESHOLD` (Broder, "On the resemblance and
containment of documents", 1997). Every pair of documents is never compared:
each document gets a MinHash signature of `BANDS` x `ROWS` permutations, the
signature is cut into bands, and only documents that agree on a whole band
become candidates; each candidate pair is then checked on its exact
resemblance. Near duplicates are joined transitively into clusters, so a
split that keeps a cluster on one side keeps every near-duplicate pair
together.
"""

import dataclasses
import re
import zlib

import numpy
import pandas

from scmle.contracts import errors

SHINGLE = 5
"""Words per shingle, the w of Broder's w-shingling (his choice for web
pages)."""

THRESHOLD = 0.8
"""Jaccard resemblance at or above which two documents are near duplicates.

Changing m scattered words of an n-word document alters at most m x SHINGLE
of its about n shingles, so its resemblance to the original is at least
(n - m x SHINGLE) / (n + m x SHINGLE). At 0.8 with SHINGLE = 5 that admits
edits to (1 - 0.8) / (5 x 1.8), about 2 % of the words, scattered (more when
the edits are contiguous): a re-saved, re-headed or lightly edited copy is
one document, while two texts on the same template with different content
are not.
"""

BANDS = 32
"""Bands of the MinHash signature."""

ROWS = 4
"""Signature rows per band.

A pair with resemblance s agrees on one band with probability s^ROWS, so it
becomes a candidate with probability 1 - (1 - s^ROWS)^BANDS. At the
threshold s = 0.8 that is 1 - (1 - 0.4096)^32 > 1 - 6e-8, so a near
duplicate is missed less than once in ten million pairs; the false
candidates the bands admit are removed by the exact check.
"""

PRIME = (1 << 31) - 1
"""The Mersenne prime of the universal hash family (a x + b) mod PRIME; with
a and x below 2^31 the product stays below 2^62, inside 64-bit integers."""

SEED = 20_260_101
"""Seed of the permutation coefficients, fixed so clusters are reproducible
across runs and machines."""


class Forest:
    """Disjoint sets of positions, joined pairwise (union-find).

    Attributes:
        parents: The parent of each position; a root is its own parent.
    """

    def __init__(self, count: int) -> None:
        """Starts every position in a set of its own.

        Args:
            count: The positions.
        """
        self.parents = list(range(count))

    def find(self, index: int) -> int:
        """The root of a position's set, halving the path to it.

        Args:
            index: The position.

        Returns:
            The root: the lowest position of the set.
        """
        while self.parents[index] != index:
            self.parents[index] = self.parents[self.parents[index]]
            index = self.parents[index]
        return index

    def join(self, first: int, second: int) -> None:
        """Merges the sets of two positions under the lower root.

        Args:
            first: One position.
            second: The other.
        """
        low, high = sorted((self.find(first), self.find(second)))
        self.parents[high] = low

    def roots(self) -> list[int]:
        """The root of every position.

        Returns:
            One root per position, in order.
        """
        return [self.find(index) for index in range(len(self.parents))]


def merge(*columns: pandas.Series) -> numpy.ndarray:
    """Joins rows that share a value in any of the columns.

    Args:
        columns: Aligned columns; a missing value joins nothing.

    Returns:
        For each row, the position of the first row of its component, as
        text, so rows tied through any chain of shared values share a key.
    """
    forest = Forest(len(columns[0]))
    for column in columns:
        first: dict[object, int] = {}
        for position, value in enumerate(column.tolist()):
            if pandas.isna(value):
                continue
            forest.join(first.setdefault(value, position), position)
    return numpy.asarray([str(root) for root in forest.roots()])


def shingle(text: str) -> frozenset[int]:
    """Hashes the word shingles of a text.

    Words are the lower-cased runs of word characters. A text with fewer
    than `SHINGLE` words is one shingle of all its words.

    Args:
        text: The document text.

    Returns:
        The CRC-32 of each distinct shingle, reduced modulo `PRIME`; empty
        when the text has no word.
    """
    words = re.findall(r"\w+", text.lower())
    if not words:
        return frozenset()
    width = min(SHINGLE, len(words))
    return frozenset(
        zlib.crc32(" ".join(words[start : start + width]).encode()) % PRIME
        for start in range(len(words) - width + 1)
    )


def resemblance(first: frozenset[int], second: frozenset[int]) -> float:
    """The Jaccard resemblance of two shingle sets.

    Args:
        first: One shingle set.
        second: The other.

    Returns:
        the shared shingles over the shingles of either; 1.0 for two empty sets.
    """
    union = len(first | second)
    return len(first & second) / union if union else 1.0


@dataclasses.dataclass(frozen=True, slots=True)
class Resemblance:
    """Clusters documents whose shingle sets resemble each other.

    Attributes:
        threshold: Least Jaccard resemblance of a near-duplicate pair, in
            (0, 1].
        bands: Bands of the MinHash signature.
        rows: Signature rows per band.
    """

    threshold: float = THRESHOLD
    bands: int = BANDS
    rows: int = ROWS

    def __post_init__(self) -> None:
        """Checks the bounds.

        Raises:
            InputError: If the threshold is outside (0, 1] or a band count
                or row count is below one.
        """
        if not 0 < self.threshold <= 1:
            raise errors.InputError("threshold must be in (0, 1]")
        if self.bands < 1 or self.rows < 1:
            raise errors.InputError("bands and rows must be at least one")

    def permutations(self) -> tuple[numpy.ndarray, numpy.ndarray]:
        """The coefficients of the signature's hash permutations.

        Returns:
            The slopes (in [1, PRIME)) and shifts (in [0, PRIME)) of the
            `bands` x `rows` permutations, drawn from `SEED`.
        """
        generator = numpy.random.default_rng(SEED)
        count = self.bands * self.rows
        return (
            generator.integers(1, PRIME, count, dtype=numpy.uint64),
            generator.integers(0, PRIME, count, dtype=numpy.uint64),
        )

    def sign(
        self,
        shingles: frozenset[int],
        permutations: tuple[numpy.ndarray, numpy.ndarray],
    ) -> numpy.ndarray:
        """The MinHash signature of one shingle set.

        Args:
            shingles: The shingle hashes.
            permutations: The slopes and shifts (`permutations`).

        Returns:
            One minimum per permutation; every entry is `PRIME` for an empty
            set.
        """
        slopes, shifts = permutations
        if not shingles:
            return numpy.full(len(slopes), PRIME, dtype=numpy.uint64)
        values = numpy.fromiter(shingles, dtype=numpy.uint64)
        hashed = (slopes[:, None] * values[None, :] + shifts[:, None]) % PRIME
        return numpy.asarray(hashed.min(axis=1))

    def cluster(self, texts: list[str]) -> list[int]:
        """Assigns every text to its near-duplicate cluster.

        Exact copies share a shingle set and are joined first; the distinct
        sets are then banded, and a candidate pair joins its clusters when
        its exact resemblance reaches the threshold.

        Args:
            texts: The documents, in order.

        Returns:
            For each text, the position of the first text of its cluster, so
            the labels are deterministic and a text alone in its cluster is
            labelled with its own position.
        """
        forest = Forest(len(texts))
        sets: dict[frozenset[int], int] = {}
        for index, text in enumerate(texts):
            forest.join(sets.setdefault(shingle(text), index), index)
        distinct = list(sets.items())
        permutations = self.permutations()
        signatures = [self.sign(entry[0], permutations) for entry in distinct]
        for first, second in self.candidates(signatures):
            score = resemblance(distinct[first][0], distinct[second][0])
            if score >= self.threshold:
                forest.join(distinct[first][1], distinct[second][1])
        return forest.roots()

    def candidates(
        self, signatures: list[numpy.ndarray]
    ) -> set[tuple[int, int]]:
        """The pairs of signatures that agree on at least one whole band.

        Args:
            signatures: The MinHash signatures.

        Returns:
            Each candidate pair once, as positions, lower first.
        """
        pairs: set[tuple[int, int]] = set()
        for band in range(self.bands):
            buckets: dict[bytes, list[int]] = {}
            window = slice(band * self.rows, (band + 1) * self.rows)
            for slot, signature in enumerate(signatures):
                buckets.setdefault(signature[window].tobytes(), []).append(slot)
            pairs.update(
                (first, second)
                for members in buckets.values()
                for position, first in enumerate(members)
                for second in members[position + 1 :]
            )
        return pairs
