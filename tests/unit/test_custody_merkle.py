"""Merkle roots over row content: sensitivity, order and platform text."""

import hashlib

import numpy
import pandas

from scmle.data.custody import merkle


def table():
    return pandas.DataFrame(
        {
            "a": [1.5, numpy.nan, 3.0, 4.25, 0.1],
            "b": [1, 2, 3, 4, 5],
            "c": ["x", "y", None, "z", "w"],
            "d": [True, False, True, False, True],
        }
    )


def test_canonical_text_is_typed_and_platform_independent():
    assert merkle.canonical(1.0) == "f:1.0"
    assert merkle.canonical(numpy.float64(0.1)) == "f:0.1"
    assert merkle.canonical(numpy.float32(0.5)) == "f:0.5"
    assert merkle.canonical(numpy.int64(7)) == "i:7"
    assert merkle.canonical(True) == "b:1"
    assert merkle.canonical(numpy.bool_(False)) == "b:0"
    assert merkle.canonical(float("nan")) == "null"
    assert merkle.canonical(None) == "null"
    assert merkle.canonical(pandas.NaT) == "null"
    assert merkle.canonical("1") == "s:1"
    assert merkle.canonical(pandas.Timestamp("2024-01-02 03:04:05")) == (
        "t:2024-01-02T03:04:05"
    )


def test_an_integer_and_the_same_float_and_text_are_different_cells():
    texts = {merkle.canonical(v) for v in (1, 1.0, "1", True)}
    assert len(texts) == 4


def test_the_root_is_deterministic_and_a_sha256():
    root = merkle.frame_root(table(), [0, 1, 2, 3, 4])
    assert root == merkle.frame_root(table(), [0, 1, 2, 3, 4])
    assert len(root) == 64


def test_a_changed_value_in_one_row_changes_the_root():
    base = merkle.frame_root(table(), [0, 1, 2, 3, 4])
    for column, row, value in (
        ("a", 3, 4.2500001),
        ("b", 0, 99),
        ("c", 4, "v"),
        ("d", 2, False),
    ):
        changed = table()
        changed.loc[row, column] = value
        assert merkle.frame_root(changed, [0, 1, 2, 3, 4]) != base, column


def test_a_missing_value_differs_from_a_present_one():
    changed = table()
    changed.loc[1, "a"] = 0.0
    assert merkle.frame_root(changed, [1]) != merkle.frame_root(table(), [1])


def test_the_root_follows_ids_not_the_order_they_are_given_in():
    assert merkle.frame_root(table(), [3, 0, 2]) == merkle.frame_root(
        table(), [0, 2, 3]
    )


def test_a_different_set_of_ids_gives_a_different_root():
    assert merkle.frame_root(table(), [0, 1]) != merkle.frame_root(
        table(), [0, 2]
    )
    assert merkle.frame_root(table(), [0, 1]) != merkle.frame_root(
        table(), [0, 1, 2]
    )


def test_the_row_id_is_part_of_the_row():
    same = pandas.DataFrame({"a": [1.0, 1.0]})
    assert merkle.frame_root(same, [0]) != merkle.frame_root(same, [1])


def test_the_empty_set_has_a_fixed_root():
    assert merkle.frame_root(table(), []) == merkle.EMPTY_ROOT
    assert merkle.merkle_root([]) == merkle.EMPTY_ROOT


def test_leaves_and_nodes_are_domain_separated():
    leaf = merkle.row_hash(0, [("a", 1.0)])
    other = merkle.row_hash(1, [("a", 1.0)])
    assert merkle.merkle_root([leaf]) == leaf
    pair = merkle.merkle_root([leaf, other])
    assert pair == merkle.combine(leaf, other)
    # An inner node cannot be passed off as a leaf of a shorter tree.
    assert merkle.merkle_root([pair]) == pair
    assert merkle.merkle_root([leaf, other, leaf]) != pair


def test_an_odd_leaf_is_not_duplicated():
    a, b, c = (merkle.row_hash(i, [("v", i)]) for i in range(3))
    assert merkle.merkle_root([a, b, c]) != merkle.merkle_root([a, b, c, c])
    assert merkle.merkle_root([a, b, c]) == merkle.combine(
        merkle.combine(a, b), c
    )


def test_the_row_hash_matches_its_stated_encoding():
    text = "id=3\x1fa=f:1.5\x1fb=i:2"
    expected = hashlib.sha256(b"\x00" + text.encode()).hexdigest()
    assert merkle.row_hash(3, [("a", 1.5), ("b", 2)]) == expected


def test_a_fixed_table_has_a_pinned_root():
    # The same bytes on every platform: this constant is the cross-platform
    # check, because nothing in the encoding depends on the machine.
    root = merkle.frame_root(table(), [0, 1, 2, 3, 4])
    assert root == PINNED_ROOT


PINNED_ROOT = "a497fd948a5b789eb42c59d9db81b4067bab285c0dfa91ccd17f319ff84b3fba"


def test_the_pinned_root_equals_an_independent_construction():
    lines = [
        "id=0\x1fa=f:1.5\x1fb=i:1\x1fc=s:x\x1fd=b:1",
        "id=1\x1fa=null\x1fb=i:2\x1fc=s:y\x1fd=b:0",
        "id=2\x1fa=f:3.0\x1fb=i:3\x1fc=null\x1fd=b:1",
        "id=3\x1fa=f:4.25\x1fb=i:4\x1fc=s:z\x1fd=b:0",
        "id=4\x1fa=f:0.1\x1fb=i:5\x1fc=s:w\x1fd=b:1",
    ]

    def sha(data):
        return hashlib.sha256(data).digest()

    level = [sha(b"\x00" + text.encode()) for text in lines]
    while len(level) > 1:
        nxt = [
            sha(b"\x01" + level[i] + level[i + 1])
            for i in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            nxt.append(level[-1])
        level = nxt
    assert level[0].hex() == PINNED_ROOT
