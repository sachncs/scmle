"""MinHash near-duplicates: the derived threshold, bands, pairs and rows."""

import math

import numpy
import pandas
import pytest

from scmle.assurance.detectors import duplicates
from scmle.assurance.detectors import minhash
from scmle.contracts import errors


def test_the_threshold_follows_the_edit_bound():
    assert minhash.similarity() == pytest.approx(0.9 / 1.1, rel=1e-15)
    assert minhash.similarity(0.1, 1) == pytest.approx(0.9 / 1.1, rel=1e-15)
    assert minhash.similarity(0.0) == 1.0
    for edits, width in ((0.2, 5), (-0.1, 5), (0.01, 0), (0.01, 1.5)):
        with pytest.raises(errors.InputError):
            minhash.similarity(edits, width)


def test_the_bands_are_the_tallest_that_reach_the_miss_rate():
    bands = minhash.Bands(minhash.similarity())
    assert (bands.rows, bands.count) == (8, 14)
    assert bands.rows * bands.count <= minhash.PERMUTATIONS
    assert bands.chance(bands.threshold) >= 1 - minhash.MISS
    taller = 1 - (1 - bands.threshold**9) ** (minhash.PERMUTATIONS // 9)
    assert taller < 1 - minhash.MISS
    assert minhash.Bands(1.0).layout() == (minhash.PERMUTATIONS, 1)
    for changes in ({"threshold": 0.0}, {"miss": 1.0}, {"cap": 1.5}):
        with pytest.raises(errors.InputError):
            minhash.Bands(**({"threshold": 0.8} | changes))
    with pytest.raises(errors.InputError, match="no banding"):
        minhash.Bands(0.05, 0.001, cap=4)


def test_shingles_fold_case_and_space():
    assert minhash.shingles("Ab  CdE", 3) == {"ab ", "b c", " cd", "cde"}
    assert minhash.shingles("ab", 3) == {"ab"}
    assert minhash.shingles("   ", 3) == frozenset()


def test_tokens_round_floats_and_mark_missing():
    frame = pandas.DataFrame({"a": [1.0000001, numpy.nan], "b": ["x", "y"]})
    assert minhash.tokens(frame) == [
        {"a\x1f1.0", "b\x1fx"},
        {"a\x1f<missing>", "b\x1fy"},
    ]
    assert minhash.tokens(frame[[]]) == [frozenset(), frozenset()]


def test_jaccard_is_exact():
    assert minhash.jaccard(frozenset("abc"), frozenset("bcd")) == 0.5
    assert minhash.jaccard(frozenset(), frozenset()) == 0.0


def test_signatures_are_seeded_and_ignore_empty_sets():
    sets = [frozenset({"a", "b"}), frozenset()]
    first = minhash.signatures(sets, 16)
    assert (first == minhash.signatures(sets, 16)).all()
    assert (first[1] == minhash.PRIME).all()
    assert (first[0] < minhash.PRIME).all()
    assert not (first == minhash.signatures(sets, 16, seed=1)).all()


def test_pairs_are_verified_and_identical_sets_pair_at_one():
    text = "the quick brown fox jumps over the lazy dog " * 4
    sets = [
        minhash.shingles(text),
        minhash.shingles("an unrelated sentence about something else"),
        minhash.shingles(text.replace("lazy", "hazy", 1)),
        minhash.shingles(text),
        frozenset(),
        frozenset(),
    ]
    found = minhash.pairs(sets)
    assert [(i, j) for i, j, value in found] == [(0, 2), (0, 3), (2, 3)]
    assert found[1][2] == 1.0
    assert all(value >= minhash.similarity() for i, j, value in found)
    assert minhash.flags(sets).tolist() == [
        False, False, True, True, False, False,
    ]  # fmt: skip


def test_rows_with_near_copied_text_are_near_duplicates():
    review = "this blender is loud but it crushes ice in seconds, recommended"
    frame = pandas.DataFrame(
        {
            "text": [review, "a different review entirely, nothing alike",
                     review.replace("loud", "lowd"), review + "!"],
            "stars": [4.0, 4.0, 4.0, 1.0],
        }
    )  # fmt: skip
    flags = duplicates.resemble(frame, ["text", "stars"])
    assert flags.tolist() == [False, False, True, False]
    report = duplicates.assess(frame)
    assert (report.exact_extra, report.near, report.similar) == (0, 0, 1)
    assert report.similarity == duplicates.DEFAULT_SIMILARITY
    assert duplicates.flag(frame).tolist() == [
        False, False, True, False,
    ]  # fmt: skip
    text = frame[["text"]]
    assert duplicates.resemble(text, ["text"]).tolist() == [
        False, False, True, True,
    ]  # fmt: skip
    with pytest.raises(errors.InputError):
        duplicates.resemble(frame, ["text"], similarity=0.0)
    with pytest.raises(errors.InputError):
        duplicates.resemble(frame, ["text"], cut=math.inf)


def test_missing_text_reads_as_nan_and_still_pairs():
    review = "this blender is loud but it crushes ice in seconds, recommended"
    frame = pandas.DataFrame(
        {
            "text": [review, review.replace("loud", "lowd"), None],
            "note": [numpy.nan, numpy.nan, "only one note"],
        },
        dtype=object,
    )
    flags = duplicates.resemble(frame, ["text", "note"])
    assert flags.tolist() == [False, True, False]


def test_numeric_only_tables_have_no_text_pass():
    frame = pandas.DataFrame({"a": [1.0, 1.0, 2.0]})
    assert not duplicates.resemble(frame, ["a"]).any()


def test_the_report_writes_the_text_group_and_moves_back():
    found = duplicates.Duplicates(10, 1, 0.1, (0.0, 0.4), 2, 0.01, 3, 1, 4, 0.8)
    written = found.to_dict()
    assert written["text"] == {"near": 4, "threshold": 0.8}
    assert written["near"] == 2 and written["exact"]["extra"] == 1
    older = duplicates.records.downgrade(found.migrations, dict(written), 2)
    assert "text" not in older
    assert older["similar"] == 4 and older["similarity"] == 0.8
