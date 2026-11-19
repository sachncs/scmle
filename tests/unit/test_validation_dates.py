"""ISO date recognition: one layout, one zone, and no exception as a verdict."""

import itertools

import pandas
import pytest

from scmle.data.validation import columns
from scmle.data.validation import dates


@pytest.mark.parametrize(
    "values",
    [
        ["2012-01-01T10:00+01:00", "2012-01-01T10:00+02:00"],
        ["2012-01-01T10:00:00-0500", "2012-01-01T10:00:00+0100"],
    ],
)
def test_mixed_zones_are_not_one_clock(values):
    # The parser cannot place two offsets on one column of moments, so the
    # column stays categorical instead of being read in some zone.
    series = pandas.Series(values, dtype=object)
    assert not dates.iso(series)
    assert columns.categorize(series).name == "categorical"


def test_one_zone_is_a_date():
    series = pandas.Series(
        ["2012-01-01T10:00+01:00", "2013-05-06T11:30+01:00"], dtype=object
    )
    assert dates.iso(series)
    assert columns.categorize(series).name == "datetime"


def test_temporal_reads_only_what_the_datetime_kind_takes():
    frame = pandas.DataFrame(
        {
            "when": ["2012-09-04", "2013-03-08"],
            "zones": ["2012-01-01T10:00+01:00", "2012-01-01T10:00+02:00"],
            "stamp": pandas.to_datetime(["2012-09-04", "2013-03-08"]),
        }
    )
    parsed = columns.temporal(frame)
    assert pandas.api.types.is_datetime64_any_dtype(parsed["when"])
    assert parsed["zones"].tolist() == frame["zones"].tolist()
    assert parsed["stamp"].tolist() == frame["stamp"].tolist()


def test_no_layout_of_the_pattern_raises():
    # Every combination of the optional parts, with valid and invalid
    # fields, gets a verdict rather than an exception.
    stems = ["2012-09-04", "2012-13-45", "0000-01-01", "9999-12-31"]
    times = ["", "T10:15", " 23:59:59", "T00:00:00.123456789123"]
    zones = ["", "Z", "+01:00", "-0530", "+99"]
    values = [s + t + z for s, t, z in itertools.product(stems, times, zones)]
    for left, right in itertools.combinations(values, 2):
        verdict = dates.iso(pandas.Series([left, right], dtype=object))
        assert isinstance(verdict, bool)
