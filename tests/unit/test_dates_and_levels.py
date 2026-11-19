"""ISO date strings are dates, and wide categoricals keep their tail levels."""

import numpy
import pandas
import pytest

from scmle.contracts import data
from scmle.contracts import tasks
from scmle.data.readers import tabular
from scmle.data.validation import columns
from scmle.models.sklearn import classical
from scmle.pipeline import profiler
from scmle.preparation import preprocessing


def days(rows, seed=0):
    rng = numpy.random.default_rng(seed)
    start = pandas.Timestamp("2012-07-30")
    offsets = rng.integers(0, 700, size=rows)
    return [
        (start + pandas.Timedelta(days=int(o))).strftime("%Y-%m-%d")
        for o in offsets
    ]


@pytest.mark.parametrize(
    "values",
    [
        ["2012-09-04", "2013-03-08", None, "2014-08-21"],
        ["2012-09-04T10:15:00", "2013-03-08T23:59:59", "2014-08-21T00:00:00"],
        ["2012-09-04 10:15", "2013-03-08 23:59", "2014-08-21 00:00"],
    ],
)
def test_iso_date_strings_are_dates(values):
    series = pandas.Series(values, dtype=object)
    assert columns.categorize(series).name == "datetime"
    assert columns.categorize(series.astype("string")).name == "datetime"
    assert columns.categorize(series.astype("category")).name == "datetime"


@pytest.mark.parametrize(
    "values",
    [
        # Day first: 04/09/2012 is 4 September or 9 April.
        ["04/09/2012", "13/03/2013", "21/08/2014"],
        ["09/04/2012", "03/13/2013", "08/21/2014"],
        ["04.09.2012", "13.03.2013"],
        # Not a calendar date, though it looks like one.
        ["2012-13-45", "2013-03-08"],
        # One value that is not a date keeps the column categorical.
        ["2012-09-04", "2013-03-08", "unknown"],
        # Years alone and codes with dashes are not dates.
        ["2012", "2013", "2014"],
        ["12-34-5678", "12-34-5679"],
        # Mixed layouts are read differently by different parsers.
        ["2012-09-04", "2013-03-08T10:00:00"],
        ["a", "b"],
    ],
)
def test_other_text_is_not_read_as_dates(values):
    series = pandas.Series(values, dtype=object)
    assert columns.categorize(series).name != "datetime"


def test_numbers_and_empty_columns_are_not_dates():
    assert columns.categorize(pandas.Series([20120904, 20130308])).name == (
        "numeric"
    )
    empty = pandas.Series([None, None], dtype=object)
    assert columns.categorize(empty).name != "datetime"


def test_temporal_parses_only_the_date_text_columns():
    frame = pandas.DataFrame(
        {
            "when": ["2012-09-04", "2013-03-08"],
            "code": ["04/09/2012", "13/03/2013"],
            "x": [1.0, 2.0],
        }
    )
    parsed = columns.temporal(frame)
    assert pandas.api.types.is_datetime64_any_dtype(parsed["when"])
    assert parsed["code"].tolist() == ["04/09/2012", "13/03/2013"]
    # The input is not changed.
    assert not pandas.api.types.is_datetime64_any_dtype(frame["when"])


def campaign(rows=400, seed=0):
    rng = numpy.random.default_rng(seed)
    stamps = days(rows, seed)
    order = pandas.to_datetime(pandas.Series(stamps))
    late = (order > pandas.Timestamp("2013-09-01")).to_numpy()
    flip = rng.random(rows) < 0.1
    return pandas.DataFrame(
        {
            "Dt_Customer": stamps,
            "Income": rng.normal(50, 10, size=rows),
            "y": (late ^ flip).astype(int),
        }
    )


def portray(frame, cutoff=None, budget=600):
    spec = tasks.Config(
        data=tasks.Dataset(uri="m", target="y", cutoff=cutoff),
        resources=tasks.Resources(time_budget_seconds=budget),
    )
    manifest = tabular.manifested(frame, "y")
    return profiler.Profiler(spec).profile(
        frame, manifest, tasks.Task.BINARY, 0.0
    )


def test_a_date_text_column_becomes_calendar_features():
    plan = portray(campaign()).feature_plan
    assert "Dt_Customer" not in plan.categorical
    names = {node.name for node in plan.derived}
    # The ordinal day number carries the order the categorical codes lost.
    assert "Dt_Customer__timestamp" in names
    assert {"Dt_Customer__year", "Dt_Customer__month"} <= names
    assert "Dt_Customer__dayofweek" in names
    assert "Dt_Customer" in plan.extra_inputs


def test_the_calendar_features_fit_from_the_raw_text():
    frame = campaign()
    plan = portray(frame).feature_plan
    model = preprocessing.instantiate(
        plan, classical.HIST_GRADIENT_BOOSTING, tasks.Task.BINARY, 0
    )
    features = frame.drop(columns="y")
    model.fit(features, frame.y)
    graph = model.named_steps["graph"].transform(features)
    expected = (
        pandas.to_datetime(frame.Dt_Customer) - pandas.Timestamp("1970-01-01")
    ) / pandas.Timedelta(days=1)
    assert graph["Dt_Customer__timestamp"].tolist() == expected.tolist()


def dated(profile):
    return [
        f
        for f in profile.leakage["findings"]
        if f["feature"] == "Dt_Customer" and f["mechanism"] == "dated"
    ]


def test_the_cutoff_screen_sees_date_text_as_dates():
    # Every value after the declared prediction time is not yet known
    # when the prediction is made, so the column is late.
    profile = portray(campaign(), cutoff="2013-01-01")
    found = dated(profile)
    assert found and found[0]["significant"]
    assert "Dt_Customer" in profile.leakage["blocking"]


def test_a_cutoff_after_every_date_flags_nothing():
    assert not dated(portray(campaign(), cutoff="2020-01-01"))


def wide(levels=400, rows=4000, seed=0):
    effect = numpy.random.default_rng(0).normal(size=levels)
    rng = numpy.random.default_rng(seed + 1)
    codes = rng.integers(0, levels, size=rows)
    frame = pandas.DataFrame(
        {
            "code": [f"c{c}" for c in codes],
            "x": rng.normal(size=rows),
        }
    )
    signal = effect[codes] + 0.1 * frame.x.to_numpy()
    return frame, pandas.Series((signal > 0).astype(int))


PLAN = data.Blueprint(
    numeric=("x",), categorical=("code",), high_cardinality=("code",)
)


def test_levels_past_the_native_limit_are_target_encoded_after_the_rest():
    frame, target = wide()
    model = preprocessing.instantiate(
        PLAN, classical.HIST_GRADIENT_BOOSTING, tasks.Task.BINARY, 0
    )
    model.fit(frame, target)
    prep = model.named_steps["prep"]
    names = list(prep.get_feature_names_out())
    # Native codes first (their positions are declared categorical), the
    # numeric columns next, the overflow encoding last.
    assert names[0] == "code"
    assert names[1] == "x"
    assert names[-1] == "code__target"
    assert model.named_steps["model"].get_params()["categorical_features"] == [
        0
    ]
    matrix = prep.transform(frame)
    assert matrix.shape[1] == 3


def test_the_overflow_encoding_is_cross_fitted():
    # Every level is seen once, so a plain target mean would be the row's
    # own label; the cross-fitted encoding of a training row never sees it.
    rows = 2000
    codes = pandas.DataFrame({"code": [f"c{i}" for i in range(rows)]})
    target = pandas.Series(numpy.random.default_rng(0).integers(0, 2, rows))
    name, encoder, chooser = preprocessing.overflow(("code",))
    assert (name, chooser(codes)) == ("overflow", ["code"])
    encoded = encoder.fit_transform(codes, target)
    assert abs(numpy.corrcoef(encoded[:, 0], target)[0, 1]) < 0.1


def test_columns_within_the_native_limit_add_no_overflow():
    frame, target = wide(levels=40)
    model = preprocessing.instantiate(
        PLAN, classical.HIST_GRADIENT_BOOSTING, tasks.Task.BINARY, 0
    )
    model.fit(frame, target)
    names = list(model.named_steps["prep"].get_feature_names_out())
    assert names == ["code", "x"]


def test_the_overflow_encoding_recovers_the_tail_signal():
    # The label is the sign of a per-level effect: the pooled tail levels
    # are predictable only through their overflow encoding.
    frame, target = wide(levels=1000, rows=20000)
    test, labels = wide(levels=1000, rows=4000, seed=1)
    model = preprocessing.instantiate(
        PLAN, classical.HIST_GRADIENT_BOOSTING, tasks.Task.BINARY, 0
    )
    model.fit(frame, target)
    assert model.score(test, labels) > 0.9


def test_multiclass_overflow_keeps_the_layout():
    frame, target = wide()
    target = pandas.Series(
        numpy.random.default_rng(0).integers(0, 3, len(frame))
    )
    model = preprocessing.instantiate(
        PLAN, classical.HIST_GRADIENT_BOOSTING, tasks.Task.MULTICLASS, 0
    )
    model.fit(frame, target)
    names = list(model.named_steps["prep"].get_feature_names_out())
    assert names[:2] == ["code", "x"]
    assert len(names) == 2 + 3
