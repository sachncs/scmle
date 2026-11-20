"""A geospatial vector file through `Project.run` on the shared engine (4B).

The kind is registered in `kinds.KINDS` and its capability records in
`roster.enrol`; the run reads the GeoJSON, splits it by derived spatial
blocks, searches the shared model families, tests once, explains and
packages, for classification and for regression.
"""

import json

import numpy
import pytest

from scmle.api import project
from scmle.contracts import capabilities
from scmle.contracts import tasks
from scmle.data.modalities.geospatial import tables
import scmle.models.registry
from scmle.pipeline import kinds
from scmle.pipeline.kinds import geospatial

ROWS = 240


def document(seed, regression):
    generator = numpy.random.default_rng(seed)
    points = generator.uniform(0, 1000, size=(ROWS, 2))
    trend = numpy.sin(points[:, 0] / 300) + points[:, 1] / 1000
    values = trend + generator.normal(scale=0.1, size=ROWS)
    return {
        "type": "FeatureCollection",
        "crs": {"type": "name", "properties": {"name": "EPSG:3857"}},
        "features": [
            {
                "type": "Feature",
                "id": f"f{index}",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(x), float(y)],
                },
                "properties": {
                    "soil": float(value * 2 + generator.normal()),
                    "y": float(value)
                    if regression
                    else ("high" if value > 0.9 else "low"),
                },
            }
            for index, ((x, y), value) in enumerate(
                zip(points, values, strict=True)
            )
        ],
    }


@pytest.fixture
def registered():
    """The geospatial kind, as `kinds.KINDS` registers it."""
    assert kinds.KINDS["geospatial"] is geospatial.GEOSPATIAL
    return geospatial.GEOSPATIAL


def accepted(task):
    """The automatic candidates of a task, opted in as experimental."""
    adapters = scmle.models.registry.Catalog.stock().adapters
    return sorted(
        name
        for name, adapter in adapters.items()
        if adapter.candidate
        and adapter.capabilities.standalone
        and adapter.objection(task, ROWS) is None
    )


@pytest.mark.parametrize("regression", [False, True])
def test_a_geojson_file_runs_through_the_shared_engine(
    registered, tmp_path, regression
):
    del registered
    path = tmp_path / "fields.geojson"
    path.write_text(json.dumps(document(0, regression)), "utf-8")
    task = tasks.Task.REGRESSION if regression else tasks.Task.BINARY
    result = project.Project(
        str(path),
        "y",
        budget=60,
        modality="geospatial",
        enforcement="cooperative",
        models=tasks.Candidates(experimental_models=accepted(task)),
        directory=str(tmp_path / "runs"),
    ).run()
    folder = result.directory
    split = json.loads((folder / "validation_manifest.json").read_text())
    assert split["strategy"] == "group"
    assert any("spatial blocks of" in note for note in split["notes"])
    evaluation = json.loads((folder / "evaluation.json").read_text())
    assert evaluation["task"] == task.value
    assert evaluation["confidence"]["available"]
    assert len(evaluation["confidence"]["interval"]) == 2
    assert evaluation["confidence"]["candidates"] > 1
    manifest = json.loads((folder / "dataset_manifest.json").read_text())
    assert manifest["modality"] == "geospatial"
    assert (folder / "model" / "pipeline.joblib").is_file()
    assert (folder / "explanations" / "permutation_importance.json").is_file()
    fresh = tables.tabulate(document(1, regression)).drop(columns=["y"])
    predicted = result.predict(fresh)
    assert len(predicted) == ROWS


def test_without_the_opt_in_the_experimental_kind_is_refused(
    registered, tmp_path
):
    del registered
    path = tmp_path / "fields.geojson"
    path.write_text(json.dumps(document(0, False)), "utf-8")
    declared = project.Project(
        str(path),
        "y",
        budget=60,
        modality="geospatial",
        enforcement="cooperative",
        directory=str(tmp_path / "runs"),
    )
    with pytest.raises(capabilities.UnsupportedError, match="experimental"):
        declared.run()
