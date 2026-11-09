"""Hand-computed checks of the naive point-in-time reference."""

from __future__ import annotations

import math

import pandas
import pytest

from benchmarks.qualification.ingest_naive import aggregate_feature
from benchmarks.qualification.ingest_naive import known_time
from benchmarks.qualification.ingest_naive import lookup_feature
from benchmarks.qualification.ingest_naive import nested_feature
from benchmarks.qualification.ingest_naive import PRIMITIVES

NAN = float("nan")


def targets(moment: float = 10.0) -> pandas.DataFrame:
    return pandas.DataFrame({"k": ["a"], "t": [moment]})


def child(rows: list[tuple[str, float, float]]) -> pandas.DataFrame:
    return pandas.DataFrame(rows, columns=["k", "ev", "v"])


BASE = child(
    [
        ("a", 2.0, 1.0),
        ("a", 4.0, 3.0),
        ("a", 6.0, NAN),
        ("a", 8.0, 5.0),
        ("b", 9.0, 100.0),
    ]
)


def run(
    name: str,
    frame: pandas.DataFrame = BASE,
    known: str | None = None,
    window: float | None = None,
    lag: float = 0.0,
    moment: float = 10.0,
    delta: float = 0.5,
) -> float:
    return aggregate_feature(
        targets(moment),
        "k",
        "t",
        frame,
        "k",
        "ev",
        known,
        "v",
        name,
        window,
        lag,
        delta,
    )[0]


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("count", 4.0),
        ("sum", 9.0),
        ("mean", 3.0),
        ("min", 1.0),
        ("max", 5.0),
        ("std", 2.0),
        ("nunique", 3.0),
        ("last", 5.0),
        ("time_since_last", 2.0),
        ("slope", 9.0 / 14.0),
    ],
)
def test_primitives(name: str, expected: float) -> None:
    assert run(name) == pytest.approx(expected)


def test_all_primitives_covered() -> None:
    assert len(PRIMITIVES) == 10


def test_empty_scope() -> None:
    empty = child([("b", 1.0, 1.0)])
    expected = {"count": 0.0, "sum": 0.0, "nunique": 0.0}
    for name in PRIMITIVES:
        got = run(name, empty)
        if name in expected:
            assert got == expected[name]
        else:
            assert math.isnan(got)


def test_small_scopes() -> None:
    one = child([("a", 1.0, 7.0)])
    assert math.isnan(run("std", one))
    assert math.isnan(run("slope", one))
    assert run("mean", one) == 7.0
    same = child([("a", 1.0, 7.0), ("a", 1.0, 9.0)])
    assert math.isnan(run("slope", same))


def test_known_time_fallback() -> None:
    row = {"ev": 3.0, "kn": NAN}
    assert known_time(row, "ev", None, 0.5) == 3.5
    assert known_time(row, "ev", "kn", 0.5) == 3.5
    assert known_time({"ev": 3.0, "kn": 4.0}, "ev", "kn", 0.5) == 4.0
    assert known_time({"ev": 3.0, "kn": None}, "ev", "kn", 2.0) == 5.0


def test_fallback_excludes_row() -> None:
    frame = child([("a", 9.8, 1.0), ("a", 5.0, 2.0)])
    assert run("count", frame, delta=0.5) == 1.0
    assert run("count", frame, delta=0.1) == 2.0


def test_late_arrival_excluded() -> None:
    frame = BASE.copy()
    frame["kn"] = [2.0, 4.0, 6.0, 11.0, 9.0]
    assert run("count", frame, known="kn") == 3.0
    assert run("last", frame, known="kn") == 3.0
    frame["kn"] = [2.0, 4.0, 6.0, None, 9.0]
    assert run("count", frame, known="kn") == 4.0


def test_event_after_cutoff_excluded() -> None:
    frame = child([("a", 10.5, 1.0), ("a", 2.0, 2.0)])
    frame["kn"] = [0.0, 0.0]
    assert run("count", frame, known="kn") == 1.0


def test_window_boundary() -> None:
    frame = child([("a", 8.0, 1.0), ("a", 9.0, 2.0), ("a", 10.0, 4.0)])
    assert run("count", frame, window=2.0, delta=0.0) == 2.0
    assert run("sum", frame, window=2.0, delta=0.0) == 6.0
    assert run("count", frame, window=None, delta=0.0) == 3.0


def test_lag() -> None:
    assert run("count", moment=12.0, lag=2.0) == 4.0
    assert run("count", moment=12.0, lag=4.0) == 3.0
    assert run("time_since_last", moment=12.0, lag=4.0) == 2.0


def test_last_ties() -> None:
    frame = child([("a", 5.0, 1.0), ("a", 5.0, 2.0)])
    frame["kn"] = [6.0, 7.0]
    assert run("last", frame, known="kn") == 2.0
    frame["kn"] = [7.0, 6.0]
    assert run("last", frame, known="kn") == 1.0
    frame["kn"] = [6.0, 6.0]
    assert run("last", frame, known="kn") == 2.0
    frame = child([("a", 5.0, 1.0), ("a", 5.0, NAN)])
    assert run("last", frame) == 1.0


def test_count_includes_missing_values() -> None:
    assert run("count") == 4.0
    assert run("sum") == 9.0


def test_lookup() -> None:
    got = lookup_feature(
        targets(), "k", "t", BASE, "k", "ev", None, "v", 0, 0.5
    )
    assert got == [5.0]


def test_multiple_targets_in_order() -> None:
    tg = pandas.DataFrame({"k": ["b", "a"], "t": [10.0, 5.0]})
    got = aggregate_feature(
        tg, "k", "t", BASE, "k", "ev", None, "v", "sum", None, 0.0, 0.0
    )
    assert got == [100.0, 4.0]


MID = pandas.DataFrame(
    {
        "k": ["a", "a", "a", "b"],
        "id": [1, 2, 3, 4],
        "ev": [2.0, 5.0, 9.0, 3.0],
    }
)
LEAF = pandas.DataFrame(
    {
        "link": [1, 1, 2, 4, 2],
        "ev": [1.0, 2.0, 5.0, 3.0, 9.5],
        "v": [2.0, 4.0, 10.0, 99.0, 50.0],
        "kn": [1.0, 2.0, 5.0, 3.0, 12.0],
    }
)


def nested(inner: str, outer: str, **kw: float | None) -> float:
    return nested_feature(
        targets(),
        "k",
        "t",
        MID,
        "k",
        "id",
        "ev",
        None,
        LEAF,
        "link",
        "ev",
        "kn",
        "v",
        inner,
        outer,
        kw.get("inner_w"),
        kw.get("outer_w"),
        0.0,
        0.5,
    )[0]


def test_nested() -> None:
    # inner sums: m1=6, m2=10 (late leaf 9.5 excluded), m3=0.
    assert nested("sum", "mean") == pytest.approx(16.0 / 3.0)
    assert nested("sum", "count") == 3.0
    assert nested("sum", "last") == 0.0
    assert nested("sum", "time_since_last") == 1.0
    # inner means: 3, 10, NaN
    assert nested("mean", "sum") == 13.0
    assert nested("mean", "last") == 10.0
    assert nested("mean", "count") == 3.0
    assert nested("count", "max") == 2.0
    assert nested("sum", "slope") == pytest.approx(-35.0 / 37.0)


def test_nested_windows() -> None:
    # outer window 6 keeps mids with event > 4: m2, m3.
    assert nested("sum", "sum", outer_w=6.0) == 10.0
    # inner window 6 keeps leaf events > 4: only m2's ev 5 -> 10.
    assert nested("sum", "sum", inner_w=6.0) == 10.0
