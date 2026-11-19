"""The geospatial vector kind on the shared stages (4B, 1.I; MM-2, MM-9, MM-11).

Each register row closed here has a test written to fail on the code before
the change: MM-2 (3-D coordinates enter bounds and measures), MM-9 (an
uncapped quadratic neighbour loop; equal labels crash curation) and MM-11
(blocks of 1 CRS unit whose origin moves with the data). The A9 property
tests check that no spatial block, duplicate geometry or declared group sits
on both sides of the split or of a fold.
"""

import json
import time

from hypothesis import given
from hypothesis import settings
from hypothesis import strategies
import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.contracts.data
from scmle.data.modalities.geospatial import blocks
from scmle.data.modalities.geospatial import dataset
from scmle.data.modalities.geospatial import geometry
from scmle.data.modalities.geospatial import tables
from scmle.data.readers import tabular
from scmle.data.splitting import splitters
from scmle.pipeline import kinds
from scmle.pipeline import modality_lifecycle
from scmle.pipeline import profiler
from scmle.pipeline.kinds import geospatial
from scmle.pipeline.kinds import rows


def feature(geometry, identifier=None, **properties):
    found = {"type": "Feature", "geometry": geometry, "properties": properties}
    if identifier is not None:
        found["id"] = identifier
    return found


def collection(features, crs=None):
    found = {"type": "FeatureCollection", "features": list(features)}
    if crs is not None:
        found["crs"] = {"type": "name", "properties": {"name": crs}}
    return found


def field(count=240, seed=0, scale=1000.0):
    """Points on a metre grid whose label follows a smooth spatial trend."""
    generator = numpy.random.default_rng(seed)
    points = generator.uniform(0, scale, size=(count, 2))
    trend = numpy.sin(points[:, 0] / scale * 3) + points[:, 1] / scale
    noise = generator.normal(scale=0.1, size=count)
    return points, trend + noise


def written(tmp_path, features, crs="EPSG:3857", name="data.geojson"):
    path = tmp_path / name
    path.write_text(json.dumps(collection(features, crs)), encoding="utf-8")
    return path


def labelled(tmp_path, count=240, regression=False, crs="EPSG:3857"):
    points, values = field(count)
    features = [
        feature(
            {"type": "Point", "coordinates": [float(x), float(y)]},
            f"f{index}",
            soil=float(values[index] * 2 + 1),
            zone="north" if y > 500 else "south",
            y=float(value)
            if regression
            else ("high" if value > 0.9 else "low"),
        )
        for index, ((x, y), value) in enumerate(
            zip(points, values, strict=True)
        )
    ]
    return written(tmp_path, features, crs)


def config(path, **data):
    return tasks.Config(
        data=tasks.Dataset(
            uri=str(path), target="y", modality="geospatial", **data
        )
    )


# The kind and its reader.


def test_the_geospatial_kind_is_a_kind_named_after_its_modality():
    assert isinstance(geospatial.GEOSPATIAL, kinds.Kind)
    assert geospatial.GEOSPATIAL.name == "geospatial"


def test_the_reader_turns_features_into_property_and_geometry_columns(
    tmp_path,
):
    square = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]]],
    }
    hole = {
        "type": "Polygon",
        "coordinates": [
            [[0, 0], [4, 0], [4, 4], [0, 4], [0, 0]],
            [[1, 1], [2, 1], [2, 2], [1, 2], [1, 1]],
        ],
    }
    line = {"type": "LineString", "coordinates": [[0, 0], [3, 4], [3, 8]]}
    path = written(
        tmp_path,
        [
            feature(square, "a", y="x", tag=[1, 2]),
            feature(hole, "b", y="z"),
            feature(line, "c", y="x"),
        ],
    )
    frame, stream = geospatial.GEOSPATIAL.read(config(path))
    assert stream is None
    assert list(frame["y"]) == ["x", "z", "x"]
    assert list(frame["geo_area"]) == [16.0, 15.0, 0.0]
    assert list(frame["geo_length"]) == [16.0, 20.0, 9.0]
    assert list(frame["geo_vertices"]) == [5, 10, 3]
    assert list(frame["geo_type"]) == ["Polygon", "Polygon", "LineString"]
    assert list(frame["geo_x"]) == [2.0, 2.0, 1.5]
    assert frame.loc[0, "tag"] == "[1,2]"
    assert pandas.isna(frame.loc[1, "tag"])


def test_three_dimensional_positions_measure_only_x_and_y():
    """MM-2: an elevation never enters a bound, an area or a length."""
    flat = {"type": "LineString", "coordinates": [[0, 0], [3, 4]]}
    raised = {"type": "LineString", "coordinates": [[0, 0, 100], [3, 4, -50]]}
    frame = tables.tabulate(
        collection([feature(flat, y=1), feature(raised, y=1)])
    )
    columns = ["geo_x", "geo_y", "geo_width", "geo_height", "geo_length"]
    assert frame.loc[0, columns].tolist() == frame.loc[1, columns].tolist()
    assert frame.loc[1, "geo_length"] == 5.0


def test_a_property_named_like_a_geometry_column_is_refused():
    point = {"type": "Point", "coordinates": [0, 0]}
    with pytest.raises(geometry.GeometryError, match="geo_area"):
        tables.tabulate(collection([feature(point, geo_area=3, y=1)]))


def test_the_declared_crs_is_read_and_rfc7946_defaults_to_crs84(tmp_path):
    point = {"type": "Point", "coordinates": [1, 2]}
    declared = written(tmp_path, [feature(point, y=1)], "EPSG:27700")
    assert tables.load(declared)[1] == "EPSG:27700"
    plain = tmp_path / "plain.geojson"
    plain.write_text(json.dumps(collection([feature(point, y=1)])), "utf-8")
    assert tables.load(plain)[1] == tables.DEFAULT


def test_a_feature_sequence_file_is_read_line_by_line(tmp_path):
    path = tmp_path / "data.geojsonl"
    lines = [
        json.dumps(feature({"type": "Point", "coordinates": [i, i]}, y=i))
        for i in range(3)
    ]
    path.write_text("\x1e" + "\n\x1e".join(lines) + "\n\n", "utf-8")
    document, crs = tables.load(path)
    assert len(document["features"]) == 3
    assert crs == tables.DEFAULT


def test_a_format_needing_a_missing_library_names_it(tmp_path, monkeypatch):
    path = tmp_path / "data.gpkg"
    path.write_bytes(b"")
    monkeypatch.setattr(tables, "LIBRARY", "scmle_missing_geo_library")
    with pytest.raises(capabilities.UnsupportedError, match="geopandas") as hit:
        tables.load(path)
    assert hit.value.details["library"] == "scmle_missing_geo_library"


def test_malformed_files_raise_typed_errors(tmp_path):
    broken = tmp_path / "broken.geojson"
    broken.write_text("{", "utf-8")
    with pytest.raises(errors.DataError):
        tables.load(broken)
    with pytest.raises(errors.DataError):
        tables.load(tmp_path / "absent.geojson")
    wrong = tmp_path / "wrong.geojson"
    wrong.write_text(json.dumps({"type": "Point", "coordinates": [0, 0]}))
    with pytest.raises(geometry.GeometryError):
        tables.load(wrong)


# MM-11: a block size derived from the data, on a fixed origin.


def test_the_block_side_follows_the_autocorrelation_range_not_one_unit():
    """MM-11: a metre CRS gets blocks of the label's range, not of 1 m."""
    points, values = field(600)
    gram = blocks.variogram(points, values[:, None], 0)
    side = blocks.snap(gram.reach)
    assert 50 <= gram.reach <= 1000
    assert side >= gram.reach
    assert side in {m * 10.0**e for m in (1, 2, 5) for e in range(-3, 6)}


def test_a_label_without_spatial_structure_gets_the_smallest_lag():
    generator = numpy.random.default_rng(1)
    points = generator.uniform(0, 1000, size=(600, 2))
    values = generator.normal(size=(600, 1))
    gram = blocks.variogram(points, values, 0)
    assert gram.reach == gram.lags[0]


def test_the_block_origin_is_fixed_so_a_new_feature_moves_no_block():
    """MM-11: adding one feature far away changes no existing block."""
    points = field(100)[0]
    before = blocks.label(points, 200.0)
    after = blocks.label(numpy.vstack([points, [[-5000.0, -7000.0]]]), 200.0)
    assert list(after[:-1]) == list(before)


def test_the_variogram_sample_is_capped_and_order_free():
    points, values = field(blocks.SAMPLE + 500)
    started = time.perf_counter()
    gram = blocks.variogram(points, values[:, None], 3)
    assert time.perf_counter() - started < blocks.SECONDS
    assert gram.pairs <= blocks.SAMPLE * (blocks.SAMPLE - 1) // 2
    order = numpy.random.default_rng(5).permutation(len(points))
    again = blocks.variogram(points[order], values[order, None], 3)
    assert again.reach == gram.reach


def test_the_splitter_records_the_block_design_in_the_manifest(tmp_path):
    path = labelled(tmp_path)
    spec = config(path)
    frame = geospatial.GEOSPATIAL.read(spec)[0]
    splitter = geospatial.GEOSPATIAL.constrain(spec, tasks.Task.BINARY)
    assert isinstance(splitter, splitters.Splitter)
    split = splitter.split(frame)
    assert split.manifest.strategy == "group"
    assert any("spatial blocks of" in note for note in split.manifest.notes)
    assert rows.KEY not in split.dev_features.columns
    assert rows.KEY not in split.locked_test.features.columns
    assert len(split.dev_features) + len(split.locked_test.features) == 240


def test_too_few_distinct_places_refuse_the_spatial_split(tmp_path):
    point = {"type": "Point", "coordinates": [7.0, 7.0]}
    path = written(
        tmp_path, [feature(point, y=i % 2, soil=i) for i in range(40)]
    )
    spec = config(path)
    frame = geospatial.GEOSPATIAL.read(spec)[0]
    with pytest.raises(errors.BlockedError, match="spatial blocks"):
        geospatial.GEOSPATIAL.constrain(spec, tasks.Task.BINARY).split(frame)


def test_a_declared_time_column_keeps_the_temporal_split_and_says_so(
    tmp_path,
):
    points, values = field(120)
    features = [
        feature(
            {"type": "Point", "coordinates": [float(x), float(y)]},
            when=f"2020-01-{1 + index % 28:02d}",
            y=float(values[index]),
        )
        for index, (x, y) in enumerate(points)
    ]
    spec = config(written(tmp_path, features), time_column="when")
    frame = geospatial.GEOSPATIAL.read(spec)[0]
    split = geospatial.GEOSPATIAL.constrain(spec, tasks.Task.REGRESSION).split(
        frame
    )
    assert split.manifest.strategy == "temporal"
    assert any("not spatial blocks" in note for note in split.manifest.notes)


# A9: no block, duplicate geometry or group on both sides.


def sides(split, count):
    side = numpy.full(count, "", dtype=object)
    side[list(split.manifest.train_ids)] = "dev"
    side[list(split.manifest.test_ids)] = "test"
    return side


@settings(max_examples=25, deadline=None)
@given(
    seed=strategies.integers(0, 10_000),
    count=strategies.integers(60, 200),
    duplicates=strategies.integers(0, 20),
)
def test_no_block_duplicate_or_group_straddles_a_split(seed, count, duplicates):
    generator = numpy.random.default_rng(seed)
    points = generator.uniform(0, 5000, size=(count, 2))
    if duplicates:
        points[:duplicates] = points[-duplicates:]
    trend = points[:, 0] / 5000 + generator.normal(scale=0.2, size=count)
    frame = pandas.DataFrame(
        {
            "geo_x": points[:, 0],
            "geo_y": points[:, 1],
            "team": generator.integers(0, 12, size=count),
            "y": trend,
        }
    )
    spec = tasks.Config(
        data=tasks.Dataset(
            uri="x.geojson",
            target="y",
            modality="geospatial",
            group_column="team",
        )
    )
    splitter = geospatial.GEOSPATIAL.constrain(spec, tasks.Task.REGRESSION)
    try:
        split = splitter.split(frame)
    except errors.BlockedError:
        return
    side = sides(split, count)
    side_of = {}
    places = blocks.label(points, splitter.side)
    for key in (places, frame["team"].to_numpy(), [tuple(p) for p in points]):
        side_of.clear()
        for row, value in enumerate(key):
            assert side_of.setdefault(value, side[row]) == side[row]
    development = numpy.asarray(split.manifest.train_ids)
    for train, valid in split.manifest.folds:
        seen = set(places[development[list(train)]])
        assert seen.isdisjoint(places[development[list(valid)]])


# The rest of the protocol.


def test_the_survey_adds_spatial_findings_to_the_profilers_portrait(tmp_path):
    point = {"type": "Point", "coordinates": [0, 0]}
    line = {"type": "LineString", "coordinates": [[0, 0], [1, 1]]}
    features = [
        feature(point if i % 2 else line, y=i % 2, soil=float(i))
        for i in range(40)
    ]
    spec = config(written(tmp_path, features))
    frame = geospatial.GEOSPATIAL.read(spec)[0]
    manifest = tabular.manifested(frame, "y", spec.data.uri, "geospatial")
    found = geospatial.GEOSPATIAL.survey(
        spec, frame, manifest, tasks.Task.BINARY
    )
    expected = profiler.Profiler(spec).profile(
        frame, manifest, tasks.Task.BINARY
    )
    assert isinstance(found, scmle.contracts.data.Portrait)
    assert found.feature_plan == expected.feature_plan
    assert any("geometry types" in item for item in found.quality_findings)
    assert any("EPSG:3857" in item for item in found.quality_findings)


def test_the_featuriser_decoder_and_cost_follow_the_shared_contract():
    plan = scmle.contracts.data.Blueprint(
        numeric=("geo_x", "geo_y"), categorical=("geo_type",)
    )
    table = tables.tabulate(
        collection(
            feature({"type": "Point", "coordinates": [i, i * 2]}, y=i % 2)
            for i in range(10)
        )
    )
    values = geospatial.GEOSPATIAL.featurise(
        plan, scmle.contracts.data.Preprocessing()
    ).fit_transform(table)
    assert values.shape[0] == 10
    assert geospatial.GEOSPATIAL.decode(("a", "b")) == ("a", "b")
    assert geospatial.GEOSPATIAL.cost(1000, 10, 2, 600.0) > profiler.estimate(
        1000, 10, 2, 600.0
    )


def test_the_declared_capabilities_cover_real_families_and_regression():
    declared = geospatial.GEOSPATIAL.declare()
    keys = {(item.key.task, item.key.model_family) for item in declared}
    assert (tasks.Task.REGRESSION, "boosting") in keys
    assert (tasks.Task.BINARY, "tree") in keys
    assert (tasks.Task.MULTICLASS, "neighbors") in keys
    assert {item.key.modality for item in declared} == {
        capabilities.Modality.GEOSPATIAL
    }
    assert {item.status for item in declared} == {geospatial.GEOSPATIAL.status}
    assert all(item.limitations for item in declared)


# MM-9: a capped neighbour search; equal labels do not crash curation.


def test_the_neighbour_search_is_not_a_quadratic_python_loop():
    """MM-9: 20,000 features find their neighbours in well under a second."""
    generator = numpy.random.default_rng(0)
    points = generator.uniform(0, 1000, size=(20_000, 2))
    features = tuple(
        geometry.Placemark(
            str(index),
            {"type": "Point", "coordinates": [float(x), float(y)]},
        )
        for index, (x, y) in enumerate(points)
    )
    collected = dataset.Dataset(features, crs="EPSG:3857")
    started = time.perf_counter()
    edges = collected.connect(4)
    assert time.perf_counter() - started < 1.0
    assert len(edges) >= 20_000 * 4 // 2


def test_neighbours_break_distance_ties_by_feature_index():
    marks = tuple(
        geometry.Placemark(
            str(index), {"type": "Point", "coordinates": [float(x), 0.0]}
        )
        for index, x in enumerate([0.0, 1.0, -1.0, 2.0, -2.0])
    )
    edges = dataset.Dataset(marks, crs="EPSG:3857").connect(1)
    assert edges == ((0, 1), (0, 2), (1, 3), (2, 4))


def test_equal_numeric_labels_report_an_unestimable_autocorrelation():
    """MM-9: a constant label is reported, not raised from curation."""
    marks = tuple(
        geometry.Placemark(
            str(index),
            {"type": "Point", "coordinates": [float(index), 0.0]},
            label=1.0,
        )
        for index in range(6)
    )
    report = dataset.Dataset(marks, crs="EPSG:3857").curate()
    autocorrelation = report.profile["spatial_autocorrelation"]
    assert autocorrelation["status"] == "not_estimable"


def test_the_lifecycle_preparer_derives_its_block_side_by_default():
    """MM-11: the old lifecycle no longer defaults to blocks of 1 unit."""
    points, values = field(200)
    marks = tuple(
        geometry.Placemark(
            str(index),
            {"type": "Point", "coordinates": [float(x), float(y)]},
            label="high" if value > 0.9 else "low",
        )
        for index, ((x, y), value) in enumerate(
            zip(points, values, strict=True)
        )
    )
    preparer = modality_lifecycle.GeospatialPreparer(
        dataset.Dataset(marks, crs="EPSG:3857")
    )
    assert preparer.side > 1.0
    assert len(preparer.split(0).groups["test"]) >= 1
