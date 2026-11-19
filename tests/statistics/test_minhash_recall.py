"""MinHash finds planted near-copies at the rate its banding promises.

Documents are random word sequences; near copies change exactly the share
of characters per shingle that `minhash.similarity` allows (the worst case
the threshold admits) or half of it. Recall at the threshold must not be
significantly below ``1 - MISS`` (exact binomial test at 0.001), recall at
half the edits must be complete, precision is exact (every reported pair is
verified), and no pair of independent documents is reported. Rows planted
with a changed share of cells are found the same way.
"""

import string

import numpy
import pandas
from scipy import stats

from scmle.assurance.detectors import duplicates
from scmle.assurance.detectors import minhash

BASES = 600
COPIES = 300
LEVEL = 0.001


def corpus(rng):
    letters = list(string.ascii_lowercase)
    words = [
        "".join(rng.choice(letters, size=rng.integers(3, 9)))
        for count in range(4000)
    ]
    return [" ".join(rng.choice(words, size=40)) for row in range(BASES)]


def edit(rng, text, share):
    """Substitutes letters for a share of the text's distinct shingles."""
    count = int(share * len(minhash.shingles(text)))
    chars = list(text)
    letters = [p for p, c in enumerate(chars) if c != " "]
    for position in rng.choice(letters, size=count, replace=False):
        chars[position] = "z" if chars[position] != "z" else "q"
    return "".join(chars)


def planted(seed, share):
    rng = numpy.random.default_rng(seed)
    bases = corpus(rng)
    copies = [edit(rng, bases[i], share) for i in range(COPIES)]
    return [minhash.shingles(t) for t in bases + copies]


def test_recall_at_the_threshold_meets_the_banding():
    sets = planted(0, minhash.EDITS)
    threshold = minhash.similarity()
    truth = [
        i for i in range(COPIES)
        if minhash.jaccard(sets[i], sets[BASES + i]) >= threshold
    ]  # fmt: skip
    found = {(i, j) for i, j, value in minhash.pairs(sets)}
    hits = sum((i, BASES + i) in found for i in truth)
    assert (len(truth), hits) == (COPIES, 295)
    assert stats.binomtest(len(truth) - hits, len(truth), minhash.MISS,
                           alternative="greater").pvalue > LEVEL  # fmt: skip
    assert found <= {(i, BASES + i) for i in range(COPIES)}


def test_recall_below_the_threshold_is_complete_and_bases_are_clean():
    sets = planted(1, minhash.EDITS / 2)
    marked = minhash.flags(sets)
    assert not marked[:BASES].any()
    assert marked[BASES:].all()


def test_planted_rows_are_found_by_their_cells():
    rng = numpy.random.default_rng(2)
    columns = 20
    frame = pandas.DataFrame(
        rng.integers(0, 10**6, size=(BASES, columns)).astype(str),
        columns=[f"c{i}" for i in range(columns)],
    )
    copies = frame.iloc[:COPIES].copy()
    changed = int(0.05 * columns)
    for row in range(COPIES):
        cells = rng.choice(columns, size=changed, replace=False)
        copies.iloc[row, cells] = "edited"
    rows = minhash.tokens(pandas.concat([frame, copies], ignore_index=True))
    threshold = minhash.similarity(0.05, 1)
    marked = minhash.flags(rows, threshold)
    assert not marked[:BASES].any()
    misses = int((~marked[BASES:]).sum())
    assert stats.binomtest(misses, COPIES, minhash.MISS,
                           alternative="greater").pvalue > LEVEL  # fmt: skip
    table = pandas.concat([frame, copies], ignore_index=True)
    joined = table.agg(" ".join, axis=1).to_frame("text")
    assert duplicates.resemble(joined, ["text"])[:BASES].sum() == 0
