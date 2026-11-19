"""Per-period drift: each time period of `X` against the first one."""

import numpy
import pandas
import pytest
from scipy import stats

from scmle.assurance.robustness import periods
from scmle.assurance.robustness import report
from scmle.assurance.robustness import shift


def frame(rows=400, jump=0.0, seed=3):
    rng = numpy.random.default_rng(seed)
    late = numpy.arange(rows) >= rows // 2
    return pandas.DataFrame(
        {
            "when": pandas.date_range("2024-01-01", periods=rows, freq="D"),
            "x": rng.normal(size=rows) + jump * late,
            "kind": rng.choice(["a", "b"], size=rows),
        }
    ).sample(frac=1.0, random_state=0)


def test_the_count_is_derived_from_the_rows_per_period():
    assert shift.MIN_ROWS == 50
    assert periods.MAX_PERIODS == 10
    assert periods.count(99) == 1
    assert periods.count(100) == 2
    assert periods.count(400) == 8
    assert periods.count(5000) == 10


def test_a_stable_series_passes_with_ordered_contiguous_periods():
    found = periods.assess(frame(), "when", ["x", "kind"])

    record = found.to_dict()
    assert found.status is shift.Status.PASS
    assert record["column"] == "when"
    assert [p["index"] for p in record["periods"]] == list(range(8))
    assert [p["rows"] for p in record["periods"]] == [50] * 8
    assert record["periods"][0]["start"] == "2024-01-01 00:00:00"
    assert record["periods"][0]["end"] == "2024-02-19 00:00:00"
    assert record["periods"][1]["start"] == "2024-02-20 00:00:00"
    assert record["periods"][0]["features"] == []
    assert [d["column"] for d in record["periods"][1]["features"]] == [
        "x",
        "kind",
    ]
    assert all(p["rejected"] == [] for p in record["periods"])


def test_a_level_jump_fails_in_the_later_periods_only():
    found = periods.assess(frame(jump=3.0), "when", ["x", "kind"])

    record = found.to_dict()
    assert found.status is shift.Status.FAIL
    assert record["reason"] == (
        "4 of 7 later periods differ from the first (Benjamini-Hochberg "
        "over 14 tests at 0.05)"
    )
    rejected = [p["rejected"] for p in record["periods"]]
    assert rejected == [[], [], [], [], ["x"], ["x"], ["x"], ["x"]]


def test_the_adjustment_spans_every_period_and_column():
    found = periods.assess(frame(jump=3.0), "when", ["x", "kind"])

    raw = [d.p_value for p in found.periods for d in p.features]
    adjusted = [d.adjusted_p_value for p in found.periods for d in p.features]
    expected = stats.false_discovery_control(raw, method="bh")
    assert adjusted == pytest.approx(list(expected))


@pytest.mark.parametrize(
    ("time", "columns", "rows", "reason"),
    [
        (None, ["x"], 400, "the data have no time column"),
        (
            "missing",
            ["x"],
            400,
            "the time column 'missing' is not among the exploration columns",
        ),
        ("when", [], 400, "no feature column to compare across periods"),
        (
            "when",
            ["x"],
            99,
            "99 timed rows make fewer than 2 periods of 50 rows",
        ),
    ],
)
def test_unverifiable_cases_say_why(time, columns, rows, reason):
    found = periods.assess(frame(rows=rows), time, columns)

    assert found.status is shift.Status.UNVERIFIABLE
    assert found.reason == reason
    assert found.periods == ()


def test_numeric_times_and_missing_times_are_handled():
    data = frame().reset_index(drop=True)
    data["when"] = numpy.arange(len(data), dtype=float)
    data.loc[:9, "when"] = numpy.nan

    found = periods.assess(data, "when", ["x"])

    assert sum(p.rows for p in found.periods) == 390
    assert found.periods[0].start == "10.0"


def test_unparseable_times_are_unverifiable():
    data = frame().assign(when="soon")

    found = periods.assess(data, "when", ["x"])

    assert found.status is shift.Status.UNVERIFIABLE
    assert found.reason == "the time column 'when' holds no orderable values"


def test_the_period_record_validates_old_and_new_spellings():
    found = periods.assess(frame(jump=3.0), "when", ["x"])

    report.periodic(found.to_dict())
    report.periodic({"status": "not_run", "reason": "older report"})
    with pytest.raises(
        ValueError, match=r"^shift report has an unknown per-period status$"
    ):
        report.periodic({**found.to_dict(), "status": "bogus"})
    with pytest.raises(
        ValueError,
        match="^shift report periods are missing "
        r"\['reason'\]$",
    ):
        report.periodic({"status": "pass"})
