"""Panel data and the forecast families: seasonal naive, smoothing, boosting."""

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.models import forecast
from scmle.models.forecast import families
from scmle.models.forecast import panel


def seasonal(entities=6, length=140, period=7, noise=0.0, seed=0, slope=0.0):
    """Series of one weekly shape at different levels, rising by `slope`."""
    rng = numpy.random.default_rng(seed)
    rows = []
    for entity in range(entities):
        steps = numpy.arange(length)
        values = (
            50.0 * entity
            + slope * steps
            + 10.0 * numpy.sin(2 * numpy.pi * steps / period)
        )
        values = values + rng.normal(scale=noise, size=length)
        rows.append(
            pandas.DataFrame(
                {
                    "store": f"s{entity}",
                    "day": pandas.date_range("2024-01-01", periods=length),
                    "sales": values,
                }
            )
        )
    return pandas.concat(rows, ignore_index=True)


def build(frame):
    return panel.Panel.build(frame, "day", "sales", "store")


def test_a_panel_orders_rows_by_entity_then_time():
    frame = seasonal(entities=2, length=3).iloc[::-1]
    built = build(frame)
    assert built.frame["store"].tolist() == ["s0"] * 3 + ["s1"] * 3
    assert list(built.series()) == ["s0", "s1"]


def test_a_single_series_panel_is_keyed_by_its_value_column():
    frame = pandas.DataFrame({"t": [3, 1, 2], "v": [30.0, 10.0, 20.0]})
    built = panel.Panel.build(frame, "t", "v")
    assert {k: v.tolist() for k, v in built.series().items()} == {
        "v": [10.0, 20.0, 30.0]
    }


def test_a_panel_cuts_history_and_reads_the_future():
    frame = pandas.DataFrame(
        {"g": ["a"] * 4 + ["b"] * 2, "t": [1, 2, 3, 4, 1, 2], "v": range(6)}
    )
    built = panel.Panel.build(frame, "t", "v", "g")
    history = built.history(2)
    assert {k: v.tolist() for k, v in history.series().items()} == {
        "a": [0.0, 1.0],
        "b": [4.0, 5.0],
    }
    assert {k: v.tolist() for k, v in built.ahead(2, 3).items()} == {
        "a": [2.0, 3.0],
        "b": [],
    }
    assert built.moments().tolist() == [1, 2, 3, 4]


@pytest.mark.parametrize(
    ("frame", "kind", "message"),
    [
        (pandas.DataFrame({"t": [1], "v": [1.0]}), errors.InputError, "'g'"),
        (
            pandas.DataFrame({"g": [1, 1], "t": [1, 1], "v": [1.0, 2.0]}),
            errors.DataError,
            "1 repeated",
        ),
        (
            pandas.DataFrame({"g": [1, 1], "t": [1, 2], "v": [1.0, None]}),
            errors.DataError,
            "1 missing or non-numeric",
        ),
        (
            pandas.DataFrame({"g": [1, 1], "t": ["x", "2024"], "v": [1, 2]}),
            errors.DataError,
            "unparseable",
        ),
    ],
)
def test_a_panel_refuses_bad_input(frame, kind, message):
    with pytest.raises(kind, match=message):
        panel.Panel.build(frame, "t", "v", "g")


def test_seasonal_naive_repeats_the_last_season():
    frame = pandas.DataFrame({"t": range(10), "v": numpy.arange(1.0, 11.0)})
    built = panel.Panel.build(frame, "t", "v")
    model = families.Seasonal(4).fit(built, 6)
    assert model.predict()["v"].tolist() == [7.0, 8.0, 9.0, 10.0, 7.0, 8.0]
    naive = families.Seasonal().fit(built, 2)
    assert naive.predict()["v"].tolist() == [10.0, 10.0]


def test_seasonal_naive_falls_back_to_the_last_value_on_short_history():
    frame = pandas.DataFrame({"t": range(3), "v": [1.0, 2.0, 3.0]})
    built = panel.Panel.build(frame, "t", "v")
    assert families.Seasonal(7).fit(built, 2).predict()["v"].tolist() == [
        3.0,
        3.0,
    ]


def test_smoothing_tracks_a_random_walk_and_holds_a_constant():
    walk = pandas.DataFrame({"t": range(5), "v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    model = families.Smoothing().fit(panel.Panel.build(walk, "t", "v"), 3)
    assert model.predict()["v"] == pytest.approx([5.0] * 3, abs=1e-3)
    flat = pandas.DataFrame({"t": range(4), "v": [2.0] * 4})
    model = families.Smoothing().fit(panel.Panel.build(flat, "t", "v"), 2)
    assert model.predict()["v"].tolist() == [2.0, 2.0]


def test_boosting_learns_a_shared_season_and_trend_across_levels():
    built = build(seasonal(noise=0.5, slope=0.5))
    history = built.history(built.moments()[-8])
    truth = built.ahead(built.moments()[-8], 7)
    model = families.Boosted(period=7, seed=0).fit(history, 7)
    guess = model.predict()
    naive = families.Seasonal(7).fit(history, 7).predict()
    boosted = numpy.mean([numpy.abs(guess[k] - truth[k]) for k in truth])
    last = numpy.mean([numpy.abs(naive[k] - truth[k]) for k in truth])
    # Seasonal naive trails the trend by one week's rise, 7 * 0.5.
    assert last == pytest.approx(3.498, abs=1e-3)
    assert boosted == pytest.approx(0.605, abs=1e-3)


def test_boosting_is_deterministic_for_a_seed():
    built = build(seasonal(noise=1.0))
    first = families.Boosted(seed=3).fit(built, 2).predict()
    second = families.Boosted(seed=3).fit(built, 2).predict()
    assert {k: v.tolist() for k, v in first.items()} == {
        k: v.tolist() for k, v in second.items()
    }


def test_boosting_refuses_predicting_before_fitting():
    with pytest.raises(errors.OrderError, match="fit"):
        families.Boosted().predict()


def test_boosting_refuses_lags_below_one():
    with pytest.raises(errors.InputError, match="lags"):
        families.Boosted(lags=(0, 2))


def test_boosting_refuses_a_panel_too_short_for_the_horizon():
    frame = pandas.DataFrame({"t": range(3), "v": [1.0, 2.0, 3.0]})
    built = panel.Panel.build(frame, "t", "v")
    with pytest.raises(errors.DataError, match="step 1 has 2 row"):
        families.Boosted().fit(built, 3)


def test_the_package_names_every_family():
    assert {
        "seasonal_naive": families.Seasonal,
        "exponential_smoothing": families.Smoothing,
        "lag_boosting": families.Boosted,
    } == forecast.FAMILIES
