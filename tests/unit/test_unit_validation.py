"""Inference validation compares incoming data with the training schema."""

import numpy
import pandas
import pytest

from scmle.contracts import artifacts
from scmle.contracts import errors
from scmle.inference import validation

SCHEMA = artifacts.Interface(
    target="y",
    task="binary_classification",
    numeric=("age", "income"),
    categorical=("city",),
    dtypes={"age": "float64", "income": "float64", "city": "object"},
    categories={"city": ["east", "north"]},
    classes=(0, 1),
)


def test_at_most_twenty_unseen_values_are_reported():
    assert validation.MAX_REPORTED_UNSEEN == 20


def test_a_conforming_frame_reports_nothing():
    frame = pandas.DataFrame(
        {"age": [1.0], "income": [2.0], "city": ["east"], "y": [1]}
    )
    report = validation.Validator(SCHEMA).report(frame)
    assert report.missing == ()
    assert report.extra == ()
    assert report.missing_value_counts == {}
    assert report.unseen_categories == {}
    assert report.valid


def test_a_report_lists_missing_extra_nulls_and_unseen_values():
    frame = pandas.DataFrame(
        {
            "age": [1.0, numpy.nan, numpy.nan],
            "city": ["west", None, "south"],
            "other": [0, 0, 0],
        }
    )
    report = validation.Validator(SCHEMA).report(frame)
    assert report.missing == ("income",)
    assert report.extra == ("other",)
    assert report.missing_value_counts == {"age": 2, "city": 1}
    assert report.unseen_categories == {"city": ["south", "west"]}
    assert not report.valid


def test_a_report_caps_unseen_values_at_twenty_sorted():
    values = [f"v{i:02d}" for i in range(25)]
    frame = pandas.DataFrame({"city": values})
    report = validation.Validator(SCHEMA).report(frame)
    assert report.unseen_categories["city"] == values[:20]


def test_a_report_compares_categories_as_strings():
    schema = artifacts.Interface(
        "y", "t", (), ("code",), {}, {"code": ["1", "2"]}, None
    )
    frame = pandas.DataFrame({"code": [1, 2, 3]})
    report = validation.Validator(schema).report(frame)
    assert report.unseen_categories == {"code": ["3"]}


def test_a_description_flags_each_rows_nulls_and_unseen_values():
    frame = pandas.DataFrame(
        {
            "age": [1.0, numpy.nan],
            "income": [numpy.nan, 2.0],
            "city": ["north", "west"],
        }
    )
    rows = validation.Validator(SCHEMA).describe(frame)
    assert rows == [
        {"missing": ["income"], "unseen": {}},
        {"missing": ["age"], "unseen": {"city": "west"}},
    ]


def test_a_description_does_not_call_a_null_category_unseen():
    frame = pandas.DataFrame({"city": [None, "east"]})
    rows = validation.Validator(SCHEMA).describe(frame)
    assert rows == [
        {"missing": ["city"], "unseen": {}},
        {"missing": [], "unseen": {}},
    ]


def test_a_description_of_a_frame_without_features_is_empty_per_row():
    frame = pandas.DataFrame({"other": [1, 2]})
    rows = validation.Validator(SCHEMA).describe(frame)
    assert rows == [{"missing": [], "unseen": {}}] * 2


def test_a_check_raises_on_missing_columns_with_details():
    frame = pandas.DataFrame({"age": [1.0]})
    with pytest.raises(errors.SchemaError) as caught:
        validation.Validator(SCHEMA).check(frame)
    assert str(caught.value) == (
        "input is missing required columns: ['income', 'city']"
    )
    assert caught.value.details == {"missing": ["income", "city"]}


def test_a_check_raises_on_non_numeric_values():
    frame = pandas.DataFrame(
        {"age": ["x", "2"], "income": [1.0, None], "city": ["a", "b"]}
    )
    with pytest.raises(errors.SchemaError) as caught:
        validation.Validator(SCHEMA).check(frame)
    assert str(caught.value) == "non-numeric values in numeric columns: ['age']"
    assert caught.value.details == {"columns": ["age"]}


def test_a_check_accepts_nulls_numeric_strings_and_unseen_categories():
    frame = pandas.DataFrame(
        {"age": ["1.5", None], "income": [1.0, None], "city": ["zzz", "q"]}
    )
    assert validation.Validator(SCHEMA).check(frame) is None
