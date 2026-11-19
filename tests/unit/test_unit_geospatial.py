"""Behavioural tests for the geospatial modality."""

import dataclasses
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from datetime import UTC
import hashlib
import json
import pathlib

import numpy
import pytest

from scmle.contracts import errors
from scmle.data.modalities.geospatial import dataset
from scmle.data.modalities.geospatial import geometry
from scmle.data.modalities.geospatial import linkage
from scmle.data.modalities.geospatial import raster
from scmle.data.modalities.geospatial import splits
from scmle.data.modalities.geospatial import survey
from scmle.data.modalities.geospatial import topology


def point(x, y):
    return {"type": "Point", "coordinates": [x, y]}


def mark(identifier, x, y, **kwargs):
    return geometry.Placemark(identifier, point(x, y), **kwargs)


def grid(crs="EPSG:3857", labels=None):
    features = tuple(
        mark(
            str(i),
            float(i),
            float(i % 2),
            label=None if labels is None else labels[i],
        )
        for i in range(4)
    )
    return dataset.Dataset(features, crs=crs)


def test_verify_accepts_only_lowercase_digests():
    geometry.verify("f", "a" * 64)
    with pytest.raises(errors.InputError, match="f must be a SHA-256"):
        geometry.verify("f", "A" * 64)
    with pytest.raises(errors.InputError, match="SHA-256"):
        geometry.verify("f", "a" * 63)


def test_canonical_source_normalises_paths():
    assert geometry.canonical_source(" ") == "memory://"
    assert geometry.canonical_source("s3://x") == "s3://x"
    assert geometry.canonical_source("f.json") == (
        pathlib.Path("f.json").resolve().as_uri()
    )


def test_observation_times_are_normalised_to_utc():
    parsed = geometry.canonical_observation_time("2020-01-01T05:00:00+05:00")
    assert parsed == datetime(2020, 1, 1, tzinfo=UTC)
    assert parsed.tzinfo is not None
    assert geometry.canonical_observation_time("2020-01-01T00:00:00z") == parsed
    aware = datetime(2020, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    assert geometry.canonical_observation_time(aware) == parsed


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("not a time", "invalid_observation_time"),
        (12, "invalid_observation_time"),
        ("2020-01-01T00:00:00", "timezone_required"),
        (datetime.fromisoformat("2020-01-01"), "timezone_required"),
    ],
)
def test_bad_observation_times_are_rejected(value, code):
    with pytest.raises(geometry.TemporalError) as raised:
        geometry.canonical_observation_time(value)
    assert raised.value.details["code"] == code


def test_flatten_reads_nested_positions():
    assert geometry.flatten([[1, 2], [3, 4.5]]) == (1.0, 2.0, 3.0, 4.5)
    assert geometry.flatten((1, 2, 3)) == (1.0, 2.0, 3.0)


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ([], "empty_coordinates"),
        ([1], "invalid_coordinates"),
        ([1, float("inf")], "invalid_coordinates"),
        ([True, 1], "invalid_coordinates"),
        ("1,2", "invalid_coordinates"),
        ([[1, 2], []], "empty_coordinates"),
    ],
)
def test_flatten_rejects_bad_coordinates(value, code):
    with pytest.raises(geometry.GeometryError) as raised:
        geometry.flatten(value)
    assert raised.value.details["code"] == code


def test_unpack_handles_collections_and_rejects_bad_types():
    collection = {
        "type": "GeometryCollection",
        "geometries": [point(1, 2), point(3, 4)],
    }
    assert geometry.unpack(collection) == (1.0, 2.0, 3.0, 4.0)
    for bad, code in (
        ({"type": "Circle"}, "unsupported_geometry"),
        (
            {"type": "GeometryCollection", "geometries": []},
            "empty_geometry_collection",
        ),
        ({"type": "GeometryCollection"}, "empty_geometry_collection"),
        (
            {"type": "GeometryCollection", "geometries": [1]},
            "invalid_geometry_collection",
        ),
    ):
        with pytest.raises(geometry.GeometryError) as raised:
            geometry.unpack(bad)
        assert raised.value.details["code"] == code
    assert len(geometry.GEOMETRY_TYPES) == 7


def test_autocorrelate_computes_morans_i():
    # Exact in real arithmetic; the platform's dot product may round the
    # last bit (0.5000000000000001 on Windows).
    clustered = geometry.autocorrelate(numpy.array([1.0, 1.0, 3.0]), [(0, 1)])
    assert clustered == pytest.approx(0.5)
    line = geometry.autocorrelate(
        numpy.array([1.0, 2.0, 3.0]), [(0, 1), (1, 2)]
    )
    assert line == pytest.approx(0.0, abs=1e-12)
    negative = geometry.autocorrelate(numpy.array([1.0, 3.0, 1.0]), [(0, 1)])
    assert negative == pytest.approx(-1.0)


def test_autocorrelate_rejects_unusable_inputs():
    with pytest.raises(errors.InputError, match="at least three"):
        geometry.autocorrelate(numpy.array([1.0, 2.0]), [(0, 1)])
    with pytest.raises(errors.InputError, match="at least three"):
        geometry.autocorrelate(numpy.zeros((3, 1)), [(0, 1)])
    with pytest.raises(errors.InputError, match="finite"):
        geometry.autocorrelate(numpy.array([1.0, 2.0, numpy.nan]), [(0, 1)])
    with pytest.raises(errors.InputError, match="more than one unique"):
        geometry.autocorrelate(numpy.ones(3), [(0, 1)])
    with pytest.raises(errors.InputError, match="neighbor pair"):
        geometry.autocorrelate(numpy.array([1.0, 2.0, 3.0]), [])


def test_substitute_replaces_one_axis_and_keeps_structure():
    line = {"type": "LineString", "coordinates": [[1, 2, 9], [3, 4, 9]]}
    assert geometry.substitute(line, 1, 0) == {
        "type": "LineString",
        "coordinates": [[1.0, 0.0, 9.0], [3.0, 0.0, 9.0]],
    }
    collection = {"type": "GeometryCollection", "geometries": [point(1, 2)]}
    masked = geometry.substitute(collection, 0, 5)
    assert masked["geometries"][0]["coordinates"] == [5.0, 2.0]
    with pytest.raises(errors.InputError, match="axis"):
        geometry.substitute(line, 2, 0)
    with pytest.raises(errors.InputError, match="baseline"):
        geometry.substitute(line, 0, float("nan"))
    with pytest.raises(geometry.GeometryError, match="non-empty"):
        geometry.substitute({"type": "Point", "coordinates": []}, 0, 0)
    with pytest.raises(geometry.GeometryError, match="must contain"):
        geometry.substitute({"type": "GeometryCollection"}, 0, 0)


def test_a_placemark_hashes_its_canonical_geometry():
    feature = mark("a", 1, 2, properties={"k": 1}, label=3)
    payload = json.dumps(
        point(1, 2), sort_keys=True, separators=(",", ":")
    ).encode()
    assert feature.geometry_hash == hashlib.sha256(payload).hexdigest()
    assert feature.geometry_type == "Point"
    assert feature.bounds == (1.0, 2.0, 1.0, 2.0)
    assert feature.centroid == (1.0, 2.0)
    assert dict(feature.properties) == {"k": 1}
    with pytest.raises(TypeError):
        feature.properties["k"] = 2
    assert mark("a", 1, 2, label=4).content_hash != feature.content_hash
    timed = mark("a", 1, 2, observation_time="2020-01-01T00:00:00Z")
    assert timed.observation_time == datetime(2020, 1, 1, tzinfo=UTC)


def test_a_placemark_bounds_a_polygon():
    square = {
        "type": "Polygon",
        "coordinates": [[[0, 0], [4, 0], [4, 2], [0, 2], [0, 0]]],
    }
    feature = geometry.Placemark("p", square)
    assert feature.bounds == (0.0, 0.0, 4.0, 2.0)
    assert feature.centroid == (2.0, 1.0)
    moved = feature.substituted(0, 7)
    assert moved.bounds == (7.0, 0.0, 7.0, 2.0)
    assert moved.feature_id == "p"


def test_a_placemark_with_elevation_bounds_only_x_and_y():
    feature = geometry.Placemark(
        "p", {"type": "LineString", "coordinates": [[0, 0, 100], [1, 1, 100]]}
    )
    assert feature.bounds == (0.0, 0.0, 1.0, 1.0)
    point = geometry.Placemark(
        "q", {"type": "Point", "coordinates": [3, 4, 9, 7]}
    )
    assert point.bounds == (3.0, 4.0, 3.0, 4.0)


def test_a_collection_of_mixed_dimensions_bounds_only_x_and_y():
    collection = {
        "type": "GeometryCollection",
        "geometries": [
            {"type": "Point", "coordinates": [5, -2]},
            {"type": "LineString", "coordinates": [[0, 1, 50], [2, 3, -50]]},
        ],
    }
    assert geometry.Placemark("c", collection).bounds == (0.0, -2.0, 5.0, 3.0)


def test_a_placemark_rejects_bad_metadata():
    with pytest.raises(errors.InputError, match="feature_id"):
        mark(" ", 0, 0)
    with pytest.raises(geometry.GeometryError, match="must be an object"):
        geometry.Placemark("a", [0, 0])
    with pytest.raises(geometry.GeometryError, match="property names"):
        mark("a", 0, 0, properties={1: 2})
    with pytest.raises(geometry.GeometryError, match="JSON-native"):
        mark("a", 0, 0, properties={"k": object()})
    with pytest.raises(geometry.GeometryError, match="JSON-native"):
        mark("a", 0, 0, label=float("nan"))


def test_placemarks_read_a_feature_collection():
    document = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": 7, "geometry": point(0, 0)},
            {
                "type": "Feature",
                "geometry": point(1, 1),
                "properties": {"a": 1},
            },
        ],
    }
    features = geometry.placemarks(document)
    assert [f.feature_id for f in features] == ["7", "1"]
    assert dict(features[1].properties) == {"a": 1}


@pytest.mark.parametrize(
    ("document", "code"),
    [
        ({"type": "Feature"}, "unsupported_geojson_root"),
        ({"type": "FeatureCollection", "features": {}}, "invalid_features"),
        (
            {"type": "FeatureCollection", "features": [{"type": "X"}]},
            "invalid_feature",
        ),
        (
            {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "geometry": None}],
            },
            "invalid_feature",
        ),
    ],
)
def test_placemarks_reject_malformed_documents(document, code):
    with pytest.raises(geometry.GeometryError) as raised:
        geometry.placemarks(document)
    assert raised.value.details["code"] == code


def test_probe_checks_the_predicate_result():
    first, second = mark("a", 0, 0), mark("b", 1, 1)
    assert linkage.probe(lambda a, b: True, first, second, "p") is True
    falsy = linkage.probe(lambda a, b: numpy.bool_(False), first, second, "p")
    assert isinstance(falsy, numpy.bool_)
    assert not falsy
    with pytest.raises(geometry.GeometryError) as raised:
        linkage.probe(lambda a, b: 1, first, second, "p")
    assert raised.value.details["result_type"] == "int"
    with pytest.raises(geometry.GeometryError) as raised:
        linkage.probe(lambda a, b: 1 / 0, first, second, "p")
    assert raised.value.details["exception"] == "ZeroDivisionError"


def test_pairs_enforces_the_match_limit():
    left = [mark("a", 0, 0), mark("b", 1, 1)]
    right = [mark("c", 0, 0)]
    matches = linkage.pairs(left, right, lambda a, b: True, "p", 2)
    assert [(m.left_feature_id, m.right_feature_id) for m in matches] == [
        ("a", "c"),
        ("b", "c"),
    ]
    with pytest.raises(geometry.GeometryError, match="match limit"):
        linkage.pairs(left, right, lambda a, b: True, "p", 1)
    assert linkage.pairs(left, right, lambda a, b: False, "p", 1) == []


def test_transform_guards_feature_metadata():
    # Enforces website/src/pages/docs/api.astro:484.
    features = [mark("a", 1, 2, properties={"k": 1})]

    def shift(feature, source, target):
        return feature.substituted(0, 9)

    moved = linkage.transform(features, shift, "A", "B", "t")
    assert moved[0].bounds[0] == 9.0

    def rename(feature, source, target):
        return dataclasses.replace(feature, feature_id="z")

    def relabel(feature, source, target):
        return dataclasses.replace(feature, label=1)

    def failing(feature, source, target):
        raise KeyError

    for bad, code in (
        (rename, "transformer_metadata_changed"),
        (relabel, "transformer_metadata_changed"),
        (failing, "transformer_failure"),
        (lambda f, s, t: "x", "invalid_transformer_result"),
    ):
        with pytest.raises(geometry.GeometryError) as raised:
            linkage.transform(features, bad, "A", "B", "t")
        assert raised.value.details["code"] == code


def test_a_dataset_hash_binds_crs_and_ignores_order():
    # Enforces website/src/pages/docs/api.astro:485.
    first, second = mark("a", 0, 0), mark("b", 1, 1)
    forward = dataset.Dataset((first, second), crs=" EPSG:4326 ")
    assert forward.crs == "EPSG:4326"
    assert dataset.Dataset((second, first), crs="EPSG:4326").content_hash == (
        forward.content_hash
    )
    assert dataset.Dataset((first, second)).content_hash != forward.content_hash
    with pytest.raises(geometry.GeometryError, match="at least one"):
        dataset.Dataset(())
    with pytest.raises(geometry.GeometryError, match="duplicate"):
        dataset.Dataset((first, first))
    with pytest.raises(errors.KindError, match="crs"):
        dataset.Dataset((first,), crs=None)


def test_from_geojson_builds_a_dataset():
    built = dataset.Dataset.from_geojson(
        {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "geometry": point(0, 0)}],
        },
        source="s3://x",
        crs="EPSG:4326",
    )
    assert built.source == "s3://x"
    assert built.features[0].feature_id == "0"


def test_spatial_join_reports_matches_and_unmatched_counts():
    left = grid()
    right = dataset.Dataset((mark("r", 0, 0),), crs="EPSG:3857")
    report = left.spatial_join(
        right, lambda a, b: a.bounds == b.bounds, " near/v1 "
    )
    assert report.predicate_name == "near/v1"
    assert report.comparison_count == 4
    assert report.match_count == 1
    assert report.unmatched_left_count == 3
    assert report.unmatched_right_count == 0
    assert report.matches[0].left_feature_id == "0"
    left.spatial_join(right, lambda a, b: False, "p", comparison_limit=4)
    with pytest.raises(geometry.GeometryError, match="comparison limit"):
        left.spatial_join(right, lambda a, b: False, "p", comparison_limit=3)


def test_vet_rejects_bad_join_arguments():
    left = grid()
    with pytest.raises(errors.KindError, match="other"):
        left.vet("x", lambda a, b: True, "p", 1, 1)
    with pytest.raises(errors.KindError, match="callable"):
        left.vet(left, None, "p", 1, 1)
    with pytest.raises(errors.InputError, match="predicate_name"):
        left.vet(left, lambda a, b: True, " ", 1, 1)
    with pytest.raises(errors.InputError, match="join limits"):
        left.vet(left, lambda a, b: True, "p", 0, 1)
    with pytest.raises(errors.InputError, match="join limits"):
        left.vet(left, lambda a, b: True, "p", 1, 0)
    with pytest.raises(geometry.GeometryError, match="declared CRS"):
        left.vet(grid(crs=""), lambda a, b: True, "p", 1, 1)
    with pytest.raises(geometry.GeometryError, match="matching CRS"):
        left.vet(grid(crs="EPSG:4326"), lambda a, b: True, "p", 1, 1)


def test_reproject_binds_source_and_target():
    source = grid()
    result = source.reproject(
        "EPSG:4326", lambda f, s, t: f.substituted(1, 0), "flat/v1"
    )
    assert result.dataset.crs == "EPSG:4326"
    assert result.report.source_dataset_hash == source.content_hash
    assert result.report.feature_count == 4
    assert result.to_dict()["dataset_hash"] == result.dataset.content_hash
    assert result.to_dict()["crs"] == "EPSG:4326"
    with pytest.raises(errors.InputError, match="differ"):
        source.reproject("EPSG:3857", lambda f, s, t: f, "t")
    with pytest.raises(errors.InputError, match="target_crs"):
        source.reproject(" ", lambda f, s, t: f, "t")
    with pytest.raises(errors.KindError, match="callable"):
        source.reproject("X", None, "t")
    with pytest.raises(errors.InputError, match="transformer_name"):
        source.reproject("X", lambda f, s, t: f, " ")
    with pytest.raises(geometry.GeometryError, match="source CRS"):
        grid(crs="").reproject("X", lambda f, s, t: f, "t")


def test_a_reprojection_must_match_its_report():
    result = grid().reproject("B", lambda f, s, t: f, "t")
    other = grid(crs="C")
    with pytest.raises(errors.InputError, match="must match the transformed"):
        dataset.Reprojection(other, result.report)
    renamed = dataclasses.replace(
        result.report, target_dataset_hash=other.content_hash
    )
    with pytest.raises(errors.InputError, match="CRS must match"):
        dataset.Reprojection(other, renamed)


def test_profile_spans_every_feature():
    profile = grid().profile()
    assert profile.feature_count == 4
    assert profile.bounds == (0.0, 0.0, 3.0, 1.0)
    assert profile.geometry_types == ("Point",)


def test_timeline_reports_temporal_completeness():
    first = mark("a", 0, 0, observation_time="2020-01-02T00:00:00Z")
    second = mark("b", 0, 0, observation_time="2020-01-01T00:00:00Z")
    assert survey.timeline([first, second]) == {
        "status": "complete",
        "observation_time_min": "2020-01-01T00:00:00+00:00",
        "observation_time_max": "2020-01-02T00:00:00+00:00",
    }
    assert survey.timeline([first, mark("c", 0, 0)]) == {
        "status": "mixed",
        "timed_feature_count": 1,
        "untimed_feature_count": 1,
    }
    assert survey.timeline([mark("c", 0, 0)]) == {"status": "absent"}


def test_curate_without_crs_blocks_and_skips_autocorrelation():
    report = grid(crs="", labels=[1, None, 2, 3]).curate()
    assert report.profile["spatial_autocorrelation"] == {
        "status": "blocked_missing_crs"
    }
    assert [f.identifier for f in report.quality_findings] == [
        "quality.geospatial-missing-crs"
    ]
    assert report.quality_findings[0].blocking
    assert [f.identifier for f in report.risk_findings] == [
        "risk.geospatial-mixed-label-state"
    ]


def test_curate_measures_autocorrelation_for_numeric_labels():
    report = grid(labels=[1.0, 2.0, 3.0, 5.0]).curate()
    measured = report.profile["spatial_autocorrelation"]
    assert measured["features"] == 4
    assert measured["neighbors"] == 3
    assert measured["permutations"] == 99
    assert report.quality_findings == ()
    text = grid(labels=["a", "b", "c", "d"]).curate()
    assert text.profile["spatial_autocorrelation"]["status"] == "not_applicable"
    few = dataset.Dataset((mark("a", 0, 0), mark("b", 1, 1)), crs="X").curate()
    assert few.profile["spatial_autocorrelation"]["status"] == "not_estimable"


def test_findings_flag_mixed_geometry_and_temporal_state():
    line = geometry.Placemark(
        "l",
        {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
        observation_time="2020-01-01T00:00:00Z",
    )
    data = dataset.Dataset((mark("a", 0, 0), line), crs="X")
    names = [f.identifier for f in data.curate().quality_findings]
    assert names == [
        "quality.geospatial-mixed-temporal-state",
        "quality.geospatial-mixed-geometry",
    ]


def test_connect_builds_symmetric_nearest_neighbour_edges():
    data = dataset.Dataset(
        (mark("a", 0, 0), mark("b", 1, 0), mark("c", 5, 0)), crs="X"
    )
    assert data.connect(1) == ((0, 1), (1, 2))
    assert data.connect(2) == ((0, 1), (0, 2), (1, 2))
    with pytest.raises(errors.InputError, match="neighbor_count"):
        data.connect(3)
    with pytest.raises(errors.InputError, match="neighbor_count"):
        data.connect(0)
    with pytest.raises(geometry.GeometryError, match="CRS"):
        dataset.Dataset(data.features).connect(1)


def test_collect_checks_value_count_and_type():
    data = grid(labels=[1, 2, 3, 4])
    assert data.collect(None).tolist() == [1.0, 2.0, 3.0, 4.0]
    assert data.collect([4, 3, 2, 1]).tolist() == [4.0, 3.0, 2.0, 1.0]
    with pytest.raises(geometry.GeometryError) as raised:
        data.collect([1])
    assert raised.value.details["value_count"] == 1
    with pytest.raises(geometry.GeometryError, match="numeric"):
        data.collect([1, 2, 3, True])


def test_spatial_autocorrelation_validates_and_is_seeded():
    data = grid()
    values = [1.0, 2.0, 3.0, 5.0]
    first = data.spatial_autocorrelation(values, neighbor_count=9, seed=3)
    assert first.neighbor_count == 3
    assert first == data.spatial_autocorrelation(values, 9, 99, 3)
    assert 0.0 < first.p_value <= 1.0
    skipped = data.spatial_autocorrelation(values, permutations=0)
    assert skipped.p_value is None
    data.spatial_autocorrelation(values, permutations=10_000 // 1000)
    with pytest.raises(errors.InputError, match="permutations"):
        data.spatial_autocorrelation(values, permutations=-1)
    with pytest.raises(errors.InputError, match="permutations"):
        data.spatial_autocorrelation(values, permutations=10_001)
    with pytest.raises(errors.InputError, match="neighbor_count"):
        data.spatial_autocorrelation(values, neighbor_count=0)
    with pytest.raises(geometry.GeometryError) as raised:
        data.spatial_autocorrelation([1, 1, 1, 1])
    assert raised.value.details["code"] == "statistic_not_estimable"
    two = dataset.Dataset((mark("a", 0, 0), mark("b", 1, 1)), crs="X")
    with pytest.raises(geometry.GeometryError, match="three features"):
        two.spatial_autocorrelation([1, 2])
    assert topology.MAX_PERMUTATIONS == 10_000


def test_permutation_p_value_counts_exceedances():
    values = numpy.array([1.0, 2.0, 3.0, 4.0])
    edges = [(0, 1), (2, 3)]
    assert topology.permutation_p_value(values, edges, 0.0, 0, 0) is None
    assert topology.permutation_p_value(values, edges, 0.0, 9, 0) == 1.0
    assert topology.permutation_p_value(values, edges, 99.0, 9, 0) == 0.1


def test_prepare_features_lists_bounds_and_centroids():
    line = geometry.Placemark(
        "l", {"type": "LineString", "coordinates": [[0, 0], [2, 4]]}
    )
    prepared = dataset.Dataset((line,), crs="X").prepare_features()
    assert prepared.values == ((0.0, 0.0, 2.0, 4.0, 2.0, 4.0, 1.0, 2.0, 4.0),)
    assert prepared.feature_names[-1] == "coordinate_value_count"
    assert prepared.crs == "X"
    with pytest.raises(geometry.GeometryError, match="CRS"):
        dataset.Dataset((line,)).prepare_features()


def test_envelop_and_straddle_count_blocks():
    # Enforces website/src/pages/docs/api.astro:408.
    line = geometry.Placemark(
        "l", {"type": "LineString", "coordinates": [[0, 0], [2.5, 0.5]]}
    )
    data = dataset.Dataset((line, mark("p", 1, 0)), crs="X")
    assert data.envelop(1.0) == {"l": (0, 0, 2, 0), "p": (1, 0, 1, 0)}
    leakage = data.straddle(1.0, max_blocks_per_feature=2)
    assert leakage.feature_block_counts == (("l", 3), ("p", 1))
    assert leakage.crossing_feature_ids == ("l",)
    assert leakage.truncated_feature_ids == ("l",)
    assert (
        data.straddle(1.0, max_blocks_per_feature=3).truncated_feature_ids == ()
    )
    with pytest.raises(errors.InputError, match="max_blocks_per_feature"):
        data.straddle(1.0, max_blocks_per_feature=0)
    with pytest.raises(errors.InputError, match="block_size"):
        data.envelop(0.0)
    with pytest.raises(errors.InputError, match="block_size"):
        data.envelop(float("inf"))
    with pytest.raises(geometry.GeometryError, match="CRS"):
        dataset.Dataset((line,)).envelop(1.0)


def test_tile_keeps_each_block_in_one_partition():
    # Enforces website/src/pages/docs/api.astro:408.
    # Enforces website/src/pages/docs/data.astro:349.
    features = tuple(mark(str(i), float(i), 0.0) for i in range(10))
    data = dataset.Dataset(features, crs="X")
    partition = data.tile(2.0, 0.2, 0.2, seed=0)
    assert len(partition.blocks["validation"]) == 1
    assert len(partition.blocks["test"]) == 1
    assert len(partition.blocks["train"]) == 3
    assert len(partition.train) == 6
    assert partition.block_size == 2.0
    assert data.tile(2.0, 0.2, 0.2, seed=0) == partition
    assert set(data.tile(100.0).blocks["train"]) == {"0:0"}
    with pytest.raises(errors.InputError, match="block_size"):
        data.tile(-1.0)
    with pytest.raises(errors.InputError, match="validation_fraction"):
        data.tile(1.0, validation_fraction=1.0)


def test_cutover_puts_the_cutoff_in_training():
    early = mark("a", 0, 0, observation_time="2020-01-01T00:00:00Z")
    late = mark("b", 0, 0, observation_time="2020-01-02T00:00:00Z")
    data = dataset.Dataset((early, late))
    chronology = data.cutover("2020-01-01T00:00:00+00:00")
    assert [f.feature_id for f in chronology.train] == ["a"]
    assert [f.feature_id for f in chronology.future] == ["b"]
    assert chronology.cutoff == datetime(2020, 1, 1, tzinfo=UTC)
    with pytest.raises(geometry.TemporalError, match="every feature"):
        dataset.Dataset((early, mark("c", 0, 0))).cutover("2020-01-01T00:00Z")
    assert isinstance(data, splits.Splits)


def payload_tile(**kwargs):
    raw = b"raster"
    arguments = {
        "source": "s",
        "encoded_bytes": raw,
        "file_hash": hashlib.sha256(raw).hexdigest(),
        "width": 2,
        "height": 3,
        "bands": 1,
        "dtype": "uint8",
        "crs": "X",
        "transform": (1.0, 0.0, 10.0, 0.0, -1.0, 20.0),
    } | kwargs
    return raster.Tile(**arguments)


def test_a_tile_validates_its_metadata():
    # Enforces website/src/pages/docs/data.astro:349.
    tile = payload_tile(nodata_value=0)
    assert tile.byte_size == 6
    assert tile.nodata_value == 0.0
    assert isinstance(tile.nodata_value, float)
    assert tile.nodata_status == "declared"
    assert payload_tile().nodata_status == "unknown"
    assert payload_tile(nodata_fraction=0.0).nodata_status == "measured"
    assert payload_tile(nodata_fraction=1.0).nodata_fraction == 1.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"source": " "}, "must not be empty"),
        ({"crs": ""}, "must not be empty"),
        ({"dtype": ""}, "must not be empty"),
        ({"width": 0}, "must be positive"),
        ({"bands": 0}, "must be positive"),
        ({"transform": (1.0,)}, "transform must be empty"),
        ({"nodata_value": True}, "must be numeric"),
        ({"nodata_value": "0"}, "must be numeric"),
        ({"nodata_value": float("inf")}, "must be finite"),
        ({"nodata_fraction": 1.5}, r"in \[0, 1\]"),
        ({"nodata_fraction": -0.1}, r"in \[0, 1\]"),
        ({"file_hash": "0" * 64}, "does not match"),
    ],
)
def test_a_tile_rejects_invalid_metadata(kwargs, message):
    with pytest.raises(errors.InputError, match=message):
        payload_tile(**kwargs)


def test_an_empty_tile_payload_is_rejected():
    with pytest.raises(geometry.GeometryError, match="empty"):
        payload_tile(encoded_bytes=b"", file_hash="0" * 64)


def test_extent_projects_the_window_boundaries():
    tile = payload_tile()
    assert tile.extent(0, 3, 0, 2) == (10.0, 17.0, 12.0, 20.0)
    assert tile.extent(1, 2, 1, 2) == (11.0, 18.0, 12.0, 19.0)
    assert payload_tile(transform=()).extent(0, 1, 0, 1) is None
    for window in ((0, 0, 0, 1), (0, 4, 0, 1), (-1, 1, 0, 1), (0, 1, 0, 3)):
        with pytest.raises(errors.InputError, match="pixel window"):
            tile.extent(*window)


def test_a_homogeneous_projector_divides_and_detects_singularity():
    project = raster.projector((2, 0, 0, 0, 2, 0, 0, 0, 2))
    assert project(3, 4) == (3.0, 4.0)
    singular = raster.projector((1, 0, 0, 0, 1, 0, 1, 0, 0))
    assert singular(1, 0) == (1.0, 0.0)
    with pytest.raises(geometry.GeometryError, match="singular"):
        singular(0, 5)
    tile = payload_tile(transform=(1, 0, 0, 0, 1, 0, 0, 0, 1))
    assert tile.extent(0, 1, 0, 1) == (0.0, 0.0, 1.0, 1.0)


def test_decode_returns_read_only_float_pixels():
    tile = payload_tile()
    pixels = tile.decode(lambda raw: numpy.ones((3, 2), dtype=numpy.uint8))
    assert pixels.dtype == numpy.float64
    assert not pixels.flags.writeable
    banded = payload_tile(bands=2).decode(lambda raw: numpy.zeros((3, 2, 2)))
    assert banded.shape == (3, 2, 2)
    assert tile.decode(lambda raw: numpy.zeros((3, 2)), max_pixels=6).size == 6


@pytest.mark.parametrize(
    ("pixels", "code"),
    [
        (numpy.zeros(6), "invalid_pixel_shape"),
        (numpy.zeros((3, 2), dtype=complex), "invalid_pixel_type"),
        (numpy.array([["a", "b"]] * 3), "invalid_pixel_type"),
        (numpy.full((3, 2), numpy.nan), "non_finite_pixels"),
        (numpy.zeros((2, 3)), "pixel_dimension_mismatch"),
        (numpy.zeros((3, 2, 2)), "pixel_dimension_mismatch"),
        (numpy.zeros((0, 2)), "pixel_limit"),
    ],
)
def test_decode_rejects_bad_pixels(pixels, code):
    with pytest.raises(geometry.GeometryError) as raised:
        payload_tile().decode(lambda raw: pixels)
    assert raised.value.details["code"] == code


def test_decode_enforces_its_arguments():
    tile = payload_tile()
    with pytest.raises(geometry.GeometryError) as raised:
        tile.decode(lambda raw: numpy.zeros((3, 2)), max_pixels=5)
    assert raised.value.details["code"] == "pixel_limit"
    with pytest.raises(errors.KindError, match="callable"):
        tile.decode(None)
    with pytest.raises(errors.InputError, match="max_pixels"):
        tile.decode(lambda raw: numpy.zeros((3, 2)), max_pixels=0)
    with pytest.raises(geometry.GeometryError, match="decoding failed"):
        tile.decode(lambda raw: 1 / 0)


def test_measured_nodata_fraction_counts_the_sentinel():
    pixels = numpy.array([[0, 1], [0, 0], [2, 3]])
    assert (
        payload_tile(nodata_value=0).measured_nodata_fraction(
            lambda raw: pixels
        )
        == 0.5
    )
    assert payload_tile().measured_nodata_fraction(lambda raw: pixels) is None
