"""Grouped-temporal and spatial blocked splits: groups and blocks never cross."""

import numpy
import pandas
import pytest

from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.data.modalities.geospatial import blocks
from scmle.data.splitting.splitters import spatial
from scmle.data.splitting.splitters import splitter
from scmle.data.splitting.splitters import temporal


def spec(folds=3, embargo=0, **data):
    return tasks.Config(
        data=tasks.Dataset(uri="m", target="y", **data),
        validation=tasks.Validation(folds=folds, seed=7, embargo_rows=embargo),
    )


def weeks(groups=12, rows=10):
    """Back-to-back groups of `rows` hourly rows each, in time order."""
    total = groups * rows
    return pandas.DataFrame(
        {
            "x": numpy.arange(total, dtype=float),
            "g": numpy.repeat(numpy.arange(groups), rows),
            "t": pandas.date_range("2024-01-01", periods=total, freq="h"),
            "y": numpy.arange(total, dtype=float) % 3,
        }
    )


def ahead(embargo=0):
    """A three-step forecast over the groups of `weeks`."""
    settings = tasks.Config(
        task="forecast",
        data=tasks.Dataset(
            uri="m", target="y", group_column="g", time_column="t", horizon=3
        ),
        validation=tasks.Validation(folds=3, seed=7, embargo_rows=embargo),
    )
    return temporal.Succession(settings, tasks.Task.FORECAST)


def grouped(frame, **options):
    settings = spec(group_column="g", time_column="t", **options)
    return temporal.Succession(settings, tasks.Task.REGRESSION).partition(frame)


def test_the_registry_offers_both_new_strategies():
    assert splitter.STRATEGIES["grouped_temporal"] is temporal.Succession
    assert splitter.STRATEGIES["spatial_block"] is spatial.Tiling


def test_grouped_temporal_holds_out_the_latest_groups_whole():
    part = grouped(weeks())
    assert part.test_ids.tolist() == list(range(100, 120))
    assert part.dev_ids.tolist() == list(range(100))


def test_grouped_temporal_folds_validate_later_runs_of_groups():
    part = grouped(weeks())
    shapes = [(fit.tolist(), held.tolist()) for fit, held in part.folds]
    assert shapes == [
        (list(range(80)), list(range(80, 100))),
        (list(range(60)), list(range(60, 80))),
        (list(range(30)), list(range(30, 60))),
    ]


def test_grouped_temporal_never_puts_a_group_on_both_sides():
    frame = weeks().sample(frac=1.0, random_state=1).reset_index(drop=True)
    part = grouped(frame)
    labels = frame["g"].to_numpy()
    times = frame["t"].to_numpy()
    assert not set(labels[part.test_ids]) & set(labels[part.dev_ids])
    assert times[part.dev_ids].max() < times[part.test_ids].min()
    local = labels[part.dev_ids]
    clock = times[part.dev_ids]
    for fit, held in part.folds:
        assert not set(local[fit]) & set(local[held])
        assert clock[fit].max() < clock[held].min()


def test_grouped_temporal_leaves_out_rows_that_overlap_the_test_window():
    frame = weeks()
    # Group 3 also has a row stamped inside the test window.
    frame.loc[35, "t"] = frame.loc[110, "t"] + pandas.Timedelta(minutes=30)
    part = grouped(frame)
    assert 35 not in part.dev_ids.tolist()
    assert 35 not in part.test_ids.tolist()
    assert len(part.dev_ids) == 99
    assert any("1 row(s) of development groups" in n for n in part.notes)


def test_grouped_temporal_embargo_trims_the_latest_training_rows():
    part = grouped(weeks(), embargo=4)
    assert part.dev_ids.tolist() == list(range(96))
    assert part.folds[-1][0].tolist() == list(range(26))


def test_grouped_temporal_embargoes_a_horizon_of_one_series_for_back_to_back_groups():
    strategy = ahead()
    assert strategy.concurrency(weeks()) == 1
    assert strategy.embargo(weeks()) == 3
    part = strategy.partition(weeks())
    assert part.dev_ids.tolist() == list(range(97))
    shapes = [(fit.tolist(), held.tolist()) for fit, held in part.folds]
    assert shapes == [
        (list(range(77)), list(range(80, 97))),
        (list(range(57)), list(range(60, 80))),
        (list(range(27)), list(range(30, 60))),
    ]


def test_grouped_temporal_embargo_counts_groups_running_at_once():
    lines = weeks()
    frame = pandas.concat(
        [lines, lines.assign(g=lines["g"] + 100)], ignore_index=True
    )
    strategy = ahead(embargo=4)
    assert strategy.concurrency(frame) == 2
    assert strategy.embargo(frame) == 6


def test_grouped_temporal_keeps_the_configured_embargo_when_it_is_larger():
    assert ahead(embargo=4).embargo(weeks()) == 4


def test_the_timed_strategies_include_grouped_temporal():
    assert {"temporal", "rolling", "grouped_temporal"} == splitter.TIMED


def test_grouped_temporal_refuses_too_few_groups():
    with pytest.raises(errors.BlockedError, match="needs at least 5 groups"):
        grouped(weeks(groups=4))


def test_grouped_temporal_needs_its_columns():
    settings = spec(time_column="t")
    strategy = temporal.Succession(settings, tasks.Task.REGRESSION)
    with pytest.raises(errors.BlockedError, match=r"data\.group_column"):
        strategy.partition(weeks())


def test_grouped_temporal_refuses_missing_groups():
    frame = weeks().astype({"g": "float"})
    frame.loc[3, "g"] = numpy.nan
    with pytest.raises(errors.BlockedError, match="missing values"):
        grouped(frame)


def test_the_splitter_applies_grouped_temporal_end_to_end():
    settings = spec(group_column="g", time_column="t")
    settings = settings.model_copy(
        update={
            "validation": settings.validation.model_copy(
                update={"strategy": "grouped_temporal"}
            )
        }
    )
    chosen = splitter.Splitter(settings, tasks.Task.REGRESSION)
    split = chosen.split(weeks())
    assert split.manifest.strategy == "grouped_temporal"
    assert split.manifest.entities is None
    assert len(split.manifest.test_ids) == 20
    assert len(split.manifest.folds) == 3


def field(names, rows=600, seed=0):
    """Points on a 1000 x 1000 square with a smooth spatial label."""
    rng = numpy.random.default_rng(seed)
    east = rng.uniform(0, 1000, rows)
    north = rng.uniform(0, 1000, rows)
    label = numpy.sin(east / 150) + numpy.cos(north / 150)
    return pandas.DataFrame(
        {
            names[0]: east,
            names[1]: north,
            "f": rng.normal(size=rows),
            "y": label + rng.normal(scale=0.1, size=rows),
        }
    )


def tiled(folds=4):
    return spatial.Tiling(spec(folds=folds), tasks.Task.REGRESSION)


def test_spatial_blocks_keep_every_block_on_one_side():
    frame = field(names=("easting", "northing"))
    strategy = tiled()
    part = strategy.partition(frame)
    points = frame[["easting", "northing"]].to_numpy()
    side = strategy.side(frame)
    places = blocks.label(points, side)
    assert not set(places[part.test_ids]) & set(places[part.dev_ids])
    local = places[part.dev_ids]
    validated = numpy.concatenate([fold[1] for fold in part.folds])
    assert sorted(validated.tolist()) == list(range(len(part.dev_ids)))
    assert len(part.folds) == 4
    for fit, held in part.folds:
        assert not set(local[fit]) & set(local[held])
        assert len(fit) + len(held) == len(part.dev_ids)


def test_spatial_block_side_is_derived_from_the_label():
    frame = field(names=("easting", "northing"))
    points = frame[["easting", "northing"]].to_numpy()
    gram = blocks.variogram(points, blocks.encode(frame["y"], False), 7)
    expected = blocks.design(points, gram, 5)[0]
    assert tiled().side(frame) == expected
    assert expected >= 100.0


def test_spatial_blocks_hold_at_least_the_test_share():
    frame = field(names=("longitude", "latitude"))
    part = tiled().partition(frame)
    assert len(part.test_ids) >= round(0.2 * len(frame))
    assert len(part.test_ids) + len(part.dev_ids) == len(frame)
    assert part.notes[0].startswith("spatial blocks of side ")


def test_spatial_blocks_find_coordinates_by_name_ignoring_case():
    frame = field(names=("Lon", "Lat"))
    assert tiled().columns(frame) == ("Lon", "Lat")


def test_spatial_blocks_never_take_the_target_as_a_coordinate():
    frame = field(names=("x", "northing")).rename(columns={"northing": "z"})
    with pytest.raises(errors.BlockedError, match="coordinate columns"):
        tiled().partition(frame)


def test_spatial_blocks_refuse_a_frame_without_coordinates():
    frame = field(names=("a", "b"))
    with pytest.raises(errors.BlockedError, match="coordinate columns"):
        tiled().partition(frame)


def test_spatial_blocks_refuse_missing_coordinates():
    frame = field(names=("lng", "lat"))
    frame.loc[4, "lat"] = numpy.nan
    with pytest.raises(errors.BlockedError, match="missing or non-numeric"):
        tiled().partition(frame)


def test_spatial_blocks_refuse_too_few_places():
    frame = pandas.DataFrame(
        {"lon": [0.0, 0.0, 1.0] * 4, "lat": [0.0, 1.0, 1.0] * 4, "y": 1.0}
    )
    with pytest.raises(errors.BlockedError, match="distinct places"):
        tiled().partition(frame)


def test_the_splitter_applies_spatial_block_end_to_end():
    frame = field(names=("lon", "lat"))
    frame["y"] = numpy.where(frame["y"] > 0, "high", "low")
    settings = spec(folds=4)
    settings = settings.model_copy(
        update={
            "validation": settings.validation.model_copy(
                update={"strategy": "spatial_block"}
            )
        }
    )
    split = splitter.Splitter(settings, tasks.Task.BINARY).split(frame)
    manifest = split.manifest
    assert manifest.strategy == "spatial_block"
    assert manifest.entities is None
    assert len(manifest.folds) == 4
    points = frame[["lon", "lat"]].to_numpy()
    strategy = spatial.Tiling(settings, tasks.Task.BINARY)
    places = blocks.label(points, strategy.side(frame))
    tested = set(places[list(manifest.test_ids)])
    assert not tested & set(places[list(manifest.train_ids)])
    assert len(manifest.test_ids) + len(manifest.train_ids) == len(frame)
