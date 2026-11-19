"""`scmle run --modality <kind>` on a tiny real example of every kind.

Each kind reads its source through `kinds.KINDS`, and every kind but
tabular is experimental, so the run's `--config` file opts the task's
automatic candidates in (`models.experimental_models`). A kind whose
optional library or program is missing skips, naming it.
"""

import json
import pathlib
import shutil

import numpy
import pandas
import pytest

from scmle import cli
from scmle.contracts import tasks
from scmle.pipeline import kinds
from tests import support
from tests.integration import test_geospatial_run
from tests.integration import test_image_kind
from tests.integration import test_video_kind
from tests.unit import test_kind_audio
from tests.unit import test_kind_document
from tests.unit import test_kind_graph
from tests.unit import test_kind_raster


def tabular(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A table whose label follows its first column."""
    generator = numpy.random.default_rng(0)
    signal = generator.normal(size=120)
    path = root / "rows.csv"
    pandas.DataFrame(
        {
            "a": signal + generator.normal(0, 0.3, 120),
            "b": generator.normal(size=120),
            "label": (signal > 0).astype(int),
        }
    ).to_csv(path, index=False)
    return path, "label"


def document(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Class folders of documents."""
    return test_kind_document.folders(root / "corpus"), "label"


def graph(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """An untimed graph whose node label follows its feature."""
    folder = root / "graph"
    test_kind_graph.network(folder, timed=False)
    return folder, "label"


def image(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """Class folders of PNG textures; needs Pillow."""
    pytest.importorskip(
        "PIL.Image", reason="the optional Pillow extra (multimodal) is missing"
    )
    test_image_kind.collection(root / "pictures")
    return root / "pictures", "animal"


def video(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A listing of moving and still clips; needs ffmpeg and ffprobe."""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("the optional ffmpeg and ffprobe programs are missing")
    table = test_video_kind.listing(root)[0]
    table.drop(columns=["when"]).to_csv(root / "clips.csv", index=False)
    return root / "clips.csv", "label"


def audio(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A manifest of WAV clips (WAV needs no optional library)."""
    return test_kind_audio.collection(root), "label"


def raster(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A manifest of `.npy` tiles with bounds (no optional library)."""
    return test_kind_raster.grid(root), "label"


def geospatial(root: pathlib.Path) -> tuple[pathlib.Path, str]:
    """A GeoJSON file of labelled points."""
    path = root / "fields.geojson"
    path.write_text(
        json.dumps(test_geospatial_run.document(0, regression=False)), "utf-8"
    )
    return path, "y"


SOURCES = {
    "tabular": tabular,
    "document": document,
    "graph": graph,
    "image": image,
    "video": video,
    "audio": audio,
    "raster": raster,
    "geospatial": geospatial,
}
"""A tiny binary or multiclass source of each kind."""

TASKS = {"audio": tasks.Task.MULTICLASS}
"""Kinds whose example is not binary."""


def test_every_registered_kind_has_an_example():
    assert set(SOURCES) == set(kinds.KINDS)


@pytest.mark.parametrize("modality", sorted(SOURCES))
def test_the_command_line_runs_every_kind(modality, tmp_path):
    source, target = SOURCES[modality](tmp_path)
    task = TASKS.get(modality, tasks.Task.BINARY)
    models = support.optin(
        task, exclude_families=["bagging", "deep_learning", "foundation"]
    )
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "models": models.model_dump(mode="json"),
                "search": {"top_models": 0},
                "strategies": {"top_models": 0},
                "ensemble": {"size": 0},
            }
        ),
        encoding="utf-8",
    )
    code = cli.main(
        [
            "run",
            str(source),
            "--target",
            target,
            "--modality",
            modality,
            "--budget",
            "120",
            "--folds",
            "3",
            "--enforcement",
            "cooperative",
            "--config",
            str(config),
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code == cli.EXIT_OK
    [folder] = [
        path.parent for path in (tmp_path / "runs").rglob("evaluation.json")
    ]
    manifest = json.loads((folder / "dataset_manifest.json").read_text("utf-8"))
    assert manifest["modality"] == modality
    evaluation = json.loads((folder / "evaluation.json").read_text("utf-8"))
    assert evaluation["task"] == task.value
    assert evaluation["opened"]
    assert (folder / "model").is_dir()


def test_an_experimental_kind_without_the_opt_in_is_refused(tmp_path, caplog):
    source, target = audio(tmp_path)
    code = cli.main(
        [
            "run",
            str(source),
            "--target",
            target,
            "--modality",
            "audio",
            "--budget",
            "120",
            "--output-dir",
            str(tmp_path / "runs"),
        ]
    )
    assert code != cli.EXIT_OK
    assert "experimental" in caplog.text
