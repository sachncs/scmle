"""The raster kind on the shared stages (plan commit 44, register rows MM-4,
MM-6, MM-10, MM-11 and MM-16, and the A9 split-leakage property).

Tiles are listed in a manifest, described in bounded strips, tied into
cohorts (copies, near-copies, declared groups and spatial blocks anchored
at the CRS origin) and run through the same engine and gates as a table.
"""

import hashlib
import importlib.util
import json
import os
import pathlib
import tracemalloc

import numpy
import pandas
import pytest

import scmle.api.project
from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.data.modalities import geospatial
from scmle.pipeline import kinds
from scmle.pipeline import modality_lifecycle
from scmle.pipeline.kinds import raster
from scmle.pipeline.kinds import rows
from tests import support
from tests.unit import test_kind_audio
from tests.unit import test_modality_lifecycle

GOLDENS = pathlib.Path(__file__).parents[1] / "fixtures" / "modality" / "raster"
WRITE = os.environ.get("SCMLE_WRITE_GOLDENS") == "1"
SIDE = 28
"""Tiles per side of the test grid: four blocks of seven a side, 16 in all,
enough for five group folds and a locked share."""


def tile(rng, label, pixels=16, bands=3):
    """A smooth random field per band; label 1 lifts the second band."""
    coarse = rng.normal(size=(pixels // 4 + 2, pixels // 4 + 2))
    field = numpy.kron(coarse, numpy.ones((4, 4)))[:pixels, :pixels]
    field += 0.1 * rng.normal(size=(pixels, pixels))
    return numpy.stack(
        [
            field * (1 + 0.5 * band) + (2.0 if label and band == 1 else 0.0)
            for band in range(bands)
        ],
        axis=-1,
    )


def grid(root, side=SIDE, seed=0, value=None):
    """A labelled manifest of `side` x `side` georeferenced tiles.

    The label follows a checkerboard of 7 x 7 tile patches with a fifth of
    the labels drawn at random, so neighbouring tiles share labels as
    spatially autocorrelated data do. `value` gives a numeric target
    column instead of the label.
    """
    rng = numpy.random.default_rng(seed)
    (root / "tiles").mkdir(parents=True, exist_ok=True)
    entries = []
    for row in range(side):
        for column in range(side):
            label = (
                int((row // 7 + column // 7) % 2 == 0)
                if rng.uniform() < 0.8
                else int(rng.uniform() < 0.5)
            )
            values = tile(rng, label)
            name = f"tiles/{row:02d}_{column:02d}.npy"
            numpy.save(root / name, values)
            entries.append(
                {
                    "path": name,
                    "label": label,
                    "west": column * 100.0,
                    "east": column * 100.0 + 100.0,
                    "south": row * 100.0,
                    "north": row * 100.0 + 100.0,
                    "crs": "EPSG:32633",
                }
            )
    listed = pandas.DataFrame(entries)
    if value is not None:
        listed = listed.drop(columns=["label"]).assign(level=value(listed))
    listed.to_csv(root / "manifest.csv", index=False)
    return root / "manifest.csv"


def specification(uri, target="label"):
    """A run specification for a raster manifest."""
    return tasks.Config(
        data=tasks.Dataset(uri=str(uri), target=target, modality="raster")
    )


@pytest.fixture
def registered():
    """The raster kind, as `kinds.KINDS` registers it."""
    assert kinds.KINDS[raster.RASTER.name] is raster.RASTER
    return raster.RASTER


def test_raster_curation_names_the_raster_modality():
    """MM-4: curation reported `geospatial-raster` for a `raster` adapter."""
    report = test_modality_lifecycle.raster_adapter().curate()
    assert report.modality == "raster"


def test_raster_curation_decodes_each_tile_once():
    """MM-6: curation decoded a tile with nodata twice and kept every array."""
    calls = []

    def decoder(payload):
        calls.append(payload)
        return numpy.ones((4, 4, 3))

    tiles = []
    for index in range(5):
        payload = bytes([index + 1]) * 8
        tiles.append(
            geospatial.Tile(
                source=f"memory://tile-{index}",
                encoded=payload,
                digest=hashlib.sha256(payload).hexdigest(),
                width=4,
                height=4,
                bands=3,
                dtype="float64",
                crs="EPSG:3857",
                transform=(),
                sentinel=0,
            )
        )
    preparer = modality_lifecycle.Imagery(
        tiles,
        {item.source: "a" for item in tiles},
        decoder,
        "ones 1.0",
    )

    profile = preparer.curate().profile

    assert len(calls) == len(tiles)
    assert profile["pixels"]["measured_nodata_fraction"] == [0.0] * 5


def test_describing_a_large_tile_holds_a_bounded_working_set(tmp_path):
    """MM-6: memory follows the strip, not the tile.

    Decoding the 1,500 x 1,500 x 2 tile at once holds 36 MB of float64; the
    description stays within the dozen or so strip-sized temporaries its
    statistics make (16 strips).
    """
    rng = numpy.random.default_rng(1)
    path = tmp_path / "large.npy"
    numpy.save(path, rng.standard_normal((1_500, 1_500, 2)).astype("float32"))
    bound = 16 * geospatial.raster.STRIP

    tracemalloc.start()
    try:
        picture = geospatial.raster.describe(path)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()

    assert picture.features["raster_bands"] == 2.0
    assert peak < bound, f"peak {peak} bytes over the bound {bound}"


def test_nan_and_declared_nodata_are_left_out(tmp_path):
    """MM-16: NaN-nodata rasters are described, not refused."""
    values = numpy.arange(64.0).reshape(8, 8)
    values[:2] = numpy.nan
    values[7, 7] = -9999.0
    numpy.save(tmp_path / "tile.npy", values)

    picture = geospatial.raster.describe(tmp_path / "tile.npy", nodata=-9999.0)

    features = picture.features
    assert features["raster_band0_valid"] == pytest.approx(47 / 64)
    assert features["raster_band0_max"] == 62.0
    assert features["raster_band0_min"] == 16.0
    assert all(numpy.isfinite(list(features.values())))


def test_an_all_nodata_tile_is_a_geometry_error(tmp_path):
    numpy.save(tmp_path / "void.npy", numpy.full((4, 4), numpy.nan))
    with pytest.raises(geospatial.GeometryError) as caught:
        geospatial.raster.describe(tmp_path / "void.npy")
    assert caught.value.details["code"] == "all_nodata"


def test_a_geotiff_without_rasterio_names_the_library(tmp_path):
    if importlib.util.find_spec("rasterio") is not None:
        pytest.skip("rasterio is installed, so a GeoTIFF is decoded")
    path = tmp_path / "tile.tif"
    path.write_bytes(b"II*\0" + bytes(64))
    with pytest.raises(geospatial.GeometryError, match="rasterio"):
        geospatial.raster.describe(path)


def test_the_block_side_follows_the_locked_share():
    """CROSSING: 2p(1 - p) / k <= 5% of adjacent pairs may straddle."""
    assert raster.span(0.2) == 7
    assert raster.span(0.5) == 10
    for fraction in (0.1, 0.2, 0.3, 0.5):
        straddle = 2 * fraction * (1 - fraction) / raster.span(fraction)
        assert straddle <= raster.CROSSING


def test_blocks_are_anchored_at_the_origin(tmp_path):
    """MM-11: adding a far tile moves no other tile's block."""
    manifest = grid(tmp_path, side=14)
    before = raster.RASTER.read(specification(manifest))[0]
    listed = pandas.read_csv(manifest)
    far = tile(numpy.random.default_rng(9), 1)
    numpy.save(tmp_path / "tiles" / "far.npy", far)
    extra = listed.iloc[[0]].assign(
        path="tiles/far.npy", west=-50_000.0, east=-49_900.0
    )
    pandas.concat([listed, extra]).to_csv(tmp_path / "more.csv", index=False)

    after = raster.RASTER.read(specification(tmp_path / "more.csv"))[0]

    assert before[rows.KEY].nunique() == 4  # 14 x 14 tiles, 7 x 7 blocks
    pandas.testing.assert_series_equal(  # the far tile sorts last
        after[rows.KEY].iloc[: len(before)], before[rows.KEY]
    )
    assert after[rows.KEY].iloc[-1] not in set(before[rows.KEY])


def test_a_block_never_straddles_the_split(tmp_path):
    """A9 and MM-11: a block of neighbours lies wholly on one side."""
    manifest = grid(tmp_path)
    spec = specification(manifest)
    frame = raster.RASTER.read(spec)[0]
    listed = (
        pandas.read_csv(manifest).sort_values("path").reset_index(drop=True)
    )
    block = (
        (listed["west"] // 700).astype(int).astype(str)
        + ":"
        + (listed["south"] // 700).astype(int).astype(str)
    )

    split = raster.RASTER.constrain(spec, tasks.Task.BINARY).split(frame)

    train = numpy.asarray(split.manifest.training)
    test = numpy.asarray(split.manifest.test)
    assert not set(block[train]) & set(block[test])
    assert len(set(block[test])) >= 1
    for fit, held in split.manifest.folds:
        assert not set(block[train[list(fit)]]) & set(block[train[list(held)]])


def test_copies_and_near_copies_share_a_cohort(tmp_path):
    """MM-10: exact, rescaled and noisy copies fall in one cohort."""
    rng = numpy.random.default_rng(4)
    source = tile(rng, 1, pixels=32)
    spread = float(source.std())
    copies = {
        "orig.npy": source,
        "copy.npy": source.copy(),
        "scaled.npy": 3.0 * source + 10.0,
        "noisy.npy": source
        + rng.normal(0.0, spread * geospatial.raster.RADIUS, source.shape),
    }
    entries = []
    for name, values in copies.items():
        numpy.save(tmp_path / name, values)
        entries.append({"path": name, "label": 1})
    for index in range(10):
        numpy.save(tmp_path / f"s{index}.npy", tile(rng, index % 2, pixels=32))
        entries.append({"path": f"s{index}.npy", "label": index % 2})
    pandas.DataFrame(entries).to_csv(tmp_path / "m.csv", index=False)

    frame = raster.RASTER.read(specification(tmp_path / "m.csv"))[0]
    names = sorted(row["path"] for row in entries)
    cohort = dict(zip(names, frame[rows.KEY], strict=True))

    assert len({cohort[name] for name in copies}) == 1
    others = {cohort[f"s{index}.npy"] for index in range(10)}
    assert cohort["orig.npy"] not in others
    assert len(others) >= 8


def test_declared_columns_are_no_features(tmp_path):
    frame = raster.RASTER.read(specification(grid(tmp_path, side=7)))[0]
    assert not {*raster.SIDES, raster.NODATA, raster.CRS} & set(frame.columns)
    assert "raster_band1_mean" in frame.columns


def test_the_kind_declares_the_tabular_tasks():
    declared = raster.RASTER.declare()
    assert {item.key.task for item in declared} == set(rows.TASKS)
    assert {item.key.modality for item in declared} == {
        capabilities.Modality.RASTER
    }


def test_a_classification_run_takes_the_shared_stages(registered, tmp_path):
    manifest = grid(tmp_path)
    run = scmle.api.project.Project(
        data=str(manifest),
        target="label",
        modality="raster",
        models=support.optin(tasks.Task.BINARY),
        budget=60,
        directory=str(tmp_path / "out"),
    ).run()

    found = test_kind_audio.outcome(run)
    golden = GOLDENS / "engine.json"
    if WRITE:
        golden.write_text(json.dumps(found, indent=1) + "\n", encoding="utf-8")
    assert found == json.loads(golden.read_text(encoding="utf-8"))
    assert run.evaluation["opened"]

    fresh = tmp_path / "fresh.npy"
    numpy.save(fresh, tile(numpy.random.default_rng(11), 1))
    predicted = run.predict(registered.describe([fresh]))
    assert len(predicted) == 1


def test_a_regression_run_takes_the_shared_stages(registered, tmp_path):
    manifest = grid(
        tmp_path,
        value=lambda listed: numpy.random.default_rng(3).normal(
            size=len(listed)
        ),
    )
    frame = raster.RASTER.read(specification(manifest, "level"))[0]
    listed = pandas.read_csv(manifest).sort_values("path")
    listed["level"] = (
        frame["raster_band1_mean"].to_numpy() + 0.1 * listed["level"].to_numpy()
    )
    listed.to_csv(manifest, index=False)

    run = scmle.api.project.Project(
        data=str(manifest),
        target="level",
        modality="raster",
        models=support.optin(tasks.Task.REGRESSION),
        budget=60,
        directory=str(tmp_path / "out"),
    ).run()

    assert run.evaluation["task"] == tasks.Task.REGRESSION.value
    assert run.evaluation["metrics"]


def test_a_geotiff_brings_its_own_bounds_crs_and_nodata(tmp_path):
    rasterio = pytest.importorskip(
        "rasterio", reason="the optional rasterio library is not installed"
    )
    values = tile(numpy.random.default_rng(13), 1).astype("float32")
    values[0, 0, :] = -1.0
    with rasterio.open(
        tmp_path / "tile.tif",
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=values.shape[2],
        dtype="float32",
        crs="EPSG:32633",
        transform=rasterio.transform.from_origin(500.0, 900.0, 10.0, 10.0),
        nodata=-1.0,
    ) as opened:
        opened.write(numpy.moveaxis(values, -1, 0))

    picture = geospatial.raster.describe(tmp_path / "tile.tif")

    assert picture.bounds == (500.0, 740.0, 660.0, 900.0)
    assert "32633" in picture.crs
    assert picture.features["raster_band0_valid"] == pytest.approx(255 / 256)
