"""Sentinel detector: spike test, dictionary, recode."""

import numpy
import pandas
import pytest

from scmle.data.detectors import sentinels


def column(n=400, seed=0):
    return numpy.random.default_rng(seed).normal(50.0, 10.0, n)


def test_grid_spacing_is_the_median_positive_gap():
    assert sentinels.grid_spacing(numpy.array([0.0, 1.0, 3.0, 6.0])) == 2.0
    assert sentinels.grid_spacing(numpy.array([5.0])) == 0.0


def test_expected_share_outside_the_range_by_hand():
    others = numpy.arange(100, dtype=float)
    # value 1000: distances 901..1000 ascending; k = max(30, 5) = 30 -> h = 930
    # all 30 inside; width = h (outside the support) -> 30/100 * 1 / 930
    share = sentinels.expected_share(others, 1000.0, 1.0)
    assert share == pytest.approx(30 / 100 / 930.0)


def test_expected_share_inside_the_range_by_hand():
    others = numpy.array([float(i) for i in range(101) if i != 50])
    share = sentinels.expected_share(others, 50.0, 1.0)
    # k=30 nearest: distances 1,1,2,2,... -> h = 15; inside = 30
    assert share == pytest.approx(30 / 100 * 1.0 / 30.0)
    assert sentinels.expected_share(others, 50.0, 1e9) == 1.0


def test_planted_code_is_flagged_and_clean_column_is_not():
    clean = pandas.DataFrame({"x": column()})
    assert sentinels.find_sentinels(clean).findings == ()
    planted = clean.copy()
    planted.loc[:39, "x"] = -999.0
    report = sentinels.find_sentinels(planted)
    assert [(f.column, f.value, f.count) for f in report.findings] == [
        ("x", -999.0, 40)
    ]
    assert report.findings[0].kind == "spike"
    assert report.findings[0].p_value < 1e-10
    assert sentinels.sentinel_rows(planted).sum() == 40
    assert report.tier == "unverified" and report.rows == 400


@pytest.mark.parametrize(
    ("count", "flagged"),
    [
        (sentinels.MIN_COUNT - 1, False),
        (sentinels.MIN_COUNT, True),
    ],
)
def test_minimum_count_boundary(count, flagged):
    frame = pandas.DataFrame({"x": column(300)})
    frame.loc[: count - 1, "x"] = -999.0
    assert bool(sentinels.find_sentinels(frame).findings) is flagged


def test_effect_floor_blocks_mild_heaping():
    values = numpy.round(column(2000, 1) / 5.0) * 5.0
    frame = pandas.DataFrame({"x": values})
    assert sentinels.find_sentinels(frame).findings == ()


def test_short_columns_and_booleans_are_skipped():
    short = pandas.DataFrame(
        {"x": [-999.0] * 10 + [1.0] * (sentinels.MIN_VALUES - 11)}
    )
    assert sentinels.find_sentinels(short).findings == ()
    flags = pandas.DataFrame({"b": [True] * 40 + [False] * 40})
    assert sentinels.find_sentinels(flags).findings == ()


def test_declared_numeric_codes_are_reported_when_present():
    frame = pandas.DataFrame({"x": [1.0, 2.0, 999.0, 3.0]})
    report = sentinels.find_sentinels(frame, numeric_codes=[999, -1])
    assert [(f.value, f.kind, f.count) for f in report.findings] == [
        (999.0, "declared_numeric", 1)
    ]


def test_text_dictionary_ignores_case_and_spaces():
    frame = pandas.DataFrame({"t": ["a", " N/A ", "n/a", "ok", None, "NULL"]})
    report = sentinels.find_sentinels(frame)
    found = {(f.value.strip().lower(), f.count) for f in report.findings}
    assert found == {("n/a", 2), ("null", 1)}
    custom = sentinels.find_sentinels(frame, text_codes=["ok"])
    assert [f.count for f in custom.findings] == [1]
    assert sentinels.sentinel_mask(frame, report)["t"].sum() == 3


def test_recode_sets_missing_logs_and_keeps_the_original():
    frame = pandas.DataFrame({"x": column(200), "i": numpy.arange(200)})
    frame.loc[:19, "x"] = -999.0
    frame["t"] = "ok"
    frame.loc[:3, "t"] = "N/A"
    report = sentinels.find_sentinels(frame)
    out, log = sentinels.recode_missing(frame, report)
    assert out["x"].isna().sum() == 20 and out["t"].isna().sum() == 4
    assert (frame["x"] == -999.0).sum() == 20
    assert {(r["column"], r["recoded"]) for r in log} == {("x", 20), ("t", 4)}
    assert out["i"].equals(frame["i"].astype("int64")) or out["i"].notna().all()


def test_integer_column_becomes_float_on_recode():
    frame = pandas.DataFrame({"n": [999] * 10 + list(range(60))})
    report = sentinels.find_sentinels(frame, numeric_codes=[999])
    out = sentinels.recode_missing(frame, report)[0]
    assert out["n"].isna().sum() == 10 and out["n"].dtype == "float64"


def test_alpha_is_validated():
    frame = pandas.DataFrame({"x": column()})
    for bad in (0.0, 1.0):
        with pytest.raises(ValueError, match="alpha"):
            sentinels.find_sentinels(frame, alpha=bad)


def test_report_serialises():
    frame = pandas.DataFrame({"t": ["N/A"] * 3 + ["a"] * 10})
    record = sentinels.find_sentinels(frame).to_dict()
    assert record["findings"][0]["kind"] == "dictionary"
