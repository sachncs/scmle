"""Plan section 6: custody of the evidence sets."""

import json

import numpy
import pandas
import pytest

from benchmarks.evidence import custody


def item(name, stratum="a", source=None, value=None):
    """An item whose contents are `value` (its name by default)."""
    content = value if value is not None else name
    return custody.Item(name, stratum, source or f"src:{name}", lambda: content)


def bundle(name="toy", count=12, needs=None, decides=custody.ACCEPTANCE):
    """A set of `count` items in one stratum."""
    return custody.Set(
        name,
        "C0",
        decides,
        needs if needs is not None else {"a": 3},
        "test",
        tuple(item(f"{name}{i}") for i in range(count)),
    )


def splits(document, name):
    return next(s for s in document["sets"] if s["name"] == name)["items"]


def test_the_assignment_is_deterministic_and_depends_on_the_seed():
    sets = [bundle(count=40)]
    first = custody.assign(sets, 1)
    assert first == custody.assign(sets, 1)
    assert first != custody.assign(sets, 2)


def test_each_deciding_split_gets_its_need_and_development_the_rest():
    placed = custody.assign([bundle(count=12)])["toy"]
    counts = {s: list(placed.values()).count(s) for s in custody.SPLITS}
    assert counts == {
        custody.DEVELOPMENT: 6,
        custody.VALIDATION: 3,
        custody.ACCEPTANCE: 3,
    }


def test_a_short_stratum_fills_the_deciding_split_first():
    for decides, other in (
        (custody.ACCEPTANCE, custody.VALIDATION),
        (custody.VALIDATION, custody.ACCEPTANCE),
    ):
        each = bundle(count=4, decides=decides)
        placed = custody.assign([each])["toy"]
        values = list(placed.values())
        assert values.count(decides) == 3
        assert values.count(other) == 1
        strata = custody.shares(each, placed)
        assert strata["a"]["short"]


def test_the_stated_share_is_the_need_over_the_size():
    each = bundle(count=12)
    strata = custody.shares(each, custody.assign([each])["toy"])
    assert strata["a"]["share"] == {
        custody.DEVELOPMENT: 0.5,
        custody.VALIDATION: 0.25,
        custody.ACCEPTANCE: 0.25,
    }
    assert not strata["a"]["short"]


def test_a_source_is_never_split_and_keeps_its_split_across_sets():
    shared = [item(f"x{i}", source=f"table{i % 4}") for i in range(12)]
    first = custody.Set(
        "first", "C0", custody.ACCEPTANCE, {"a": 3}, "t", tuple(shared)
    )
    second = custody.Set(
        "second",
        "C0",
        custody.VALIDATION,
        {"a": 1},
        "t",
        tuple(item(f"y{i}", source=f"table{i}") for i in range(4)),
    )
    placed = custody.assign([first, second])
    by_source = {}
    for thing in shared:
        by_source.setdefault(thing.source, set()).add(
            placed["first"][thing.name]
        )
    assert all(len(found) == 1 for found in by_source.values())
    for i in range(4):
        assert placed["second"][f"y{i}"] == placed["first"][f"x{i}"]


def test_a_fresh_record_has_no_problems_and_states_the_seal():
    sets = [bundle(), custody.Set("empty", "C0", "acceptance", {}, "t", ())]
    document = custody.measure(sets)
    assert custody.problems(document, sets) == []
    record = document["sets"][0]
    assert record["acceptance"] == 3
    assert len(record["seal"]) == 64
    json.dumps(document)


def test_changed_acceptance_contents_are_reported():
    sets = [bundle()]
    document = custody.measure(sets)
    sealed = [
        n for n, s in splits(document, "toy").items() if s == custody.ACCEPTANCE
    ]
    changed = custody.Set(
        "toy",
        "C0",
        custody.ACCEPTANCE,
        {"a": 3},
        "test",
        tuple(
            item(t.name, value="edited" if t.name == sealed[0] else None)
            for t in sets[0].items
        ),
    )
    issues = custody.problems(document, [changed])
    assert any("acceptance contents changed" in i for i in issues)


def test_development_edits_do_not_break_the_seal():
    sets = [bundle()]
    document = custody.measure(sets)
    developed = [
        n
        for n, s in splits(document, "toy").items()
        if s == custody.DEVELOPMENT
    ]
    changed = custody.Set(
        "toy",
        "C0",
        custody.ACCEPTANCE,
        {"a": 3},
        "test",
        tuple(
            item(t.name, value="edited" if t.name == developed[0] else None)
            for t in sets[0].items
        ),
    )
    assert custody.problems(document, [changed]) == []


def test_added_removed_and_missing_sets_are_reported():
    document = custody.measure([bundle()])
    grown = bundle(count=13)
    issues = custody.problems(document, [grown])
    assert any("1 items added" in i for i in issues)
    shrunk = custody.Set(
        "toy", "C0", custody.ACCEPTANCE, {"a": 3}, "t",
        bundle().items[1:],
    )  # fmt: skip
    issues = custody.problems(document, [shrunk])
    assert any("removed" in i for i in issues)
    issues = custody.problems(document, [bundle(), bundle("other")])
    assert any("set other is not in custody" in i for i in issues)
    issues = custody.problems(document, [])
    assert any("set toy no longer exists" in i for i in issues)


def test_unreadable_acceptance_contents_are_reported():
    sets = [bundle()]
    document = custody.measure(sets)

    def fail():
        raise OSError("not cached")

    broken = custody.Set(
        "toy",
        "C0",
        custody.ACCEPTANCE,
        {"a": 3},
        "test",
        tuple(custody.Item(t.name, "a", t.source, fail) for t in sets[0].items),
    )
    issues = custody.problems(document, [broken])
    assert any("cannot be read" in i for i in issues)


def test_the_fingerprint_sees_cell_changes_in_frames_and_arrays():
    def digest(value):
        return custody.contents(item("x", value=value))

    frame = pandas.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    edited = frame.copy()
    edited.loc[1, "b"] = "z"
    assert digest(frame) == digest(frame.copy())
    assert digest(frame) != digest(edited)
    assert digest(numpy.zeros(3)) != digest(numpy.zeros(4))
    assert digest((b"ab", "c")) != digest((b"a", "bc"))


def test_the_summary_names_short_strata():
    document = custody.measure([bundle(count=4)])
    text, verdict = custody.summary(document)
    assert verdict == "fail"
    assert "toy/a" in text
    assert custody.summary(custody.measure([bundle()]))[1] == "pass"


def test_the_needs_come_from_the_acceptance_rules():
    assert custody.minimum("false_block_max") == 59
    assert custody.minimum("leak_recall_min") == 29
    assert custody.minimum("interval_coverage_min") == 42
    with pytest.raises(ValueError):
        custody.minimum("alpha")


def test_the_catalog_lists_every_set_in_order():
    sets = custody.catalog()
    assert [s.name for s in sets] == [
        "leakage",
        "refusal",
        "quality",
        "reading",
        "coverage",
        "practitioner",
    ]
    for each in sets:
        names = [i.name for i in each.items]
        assert len(names) == len(set(names))
    assert {i.stratum for i in sets[1].items} == {"real", "simulated"}
    assert sets[-1].items == ()


def test_the_stored_record_matches_the_current_items():
    document = custody.read(None)
    assert document is not None
    sets = custody.catalog()
    placed = custody.assign(sets, document["seed"])
    for record in document["sets"]:
        assert record["items"] == placed[record["name"]]


def test_no_stratum_of_the_catalog_is_short():
    sets = custody.catalog()
    placed = custody.assign(sets)
    for each in sets:
        for name, stratum in custody.shares(each, placed[each.name]).items():
            assert not stratum["short"], f"{each.name}/{name}"


@pytest.mark.parametrize(
    ("name", "strata"),
    [
        ("refusal", ("simulated",)),
        ("quality", ("binary", "multiclass", "regression")),
        ("reading", ("baseline", "empty columns")),
    ],
)
def test_the_grown_strata_fill_both_deciding_splits(name, strata):
    sets = custody.catalog()
    each = next(s for s in sets if s.name == name)
    found = custody.shares(each, custody.assign(sets)[name])
    for stratum in strata:
        counts = found[stratum]["counts"]
        need = found[stratum]["need"]
        assert counts[custody.VALIDATION] >= need
        assert counts[custody.ACCEPTANCE] >= need
