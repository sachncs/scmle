"""Images through the shared stages: `Project.run` with `modality="image"`.

The run plans, trains several model families, cross-validates, tests once
on the locked rows, brackets the score and packages a bundle, as a table
run does. Copies and edited copies of a picture never straddle the split
(MM-10). The golden `tests/fixtures/modality/image/kind.json` records the
stage list, the held-out interval and the custody events of such a run:

    SCMLE_WRITE_GOLDENS=1 uv run pytest tests/integration/test_image_kind.py
"""

import io
import json
import os
import pathlib

import numpy
import pandas
import pytest

import scmle.api.project
from scmle.artifacts.bundle import archive
from scmle.contracts import errors
from scmle.contracts import tasks
from scmle.pipeline import kinds
from scmle.pipeline import roster
from scmle.pipeline.kinds import image
from scmle.pipeline.kinds import video
from tests import support

GOLDENS = pathlib.Path(__file__).parents[1] / "fixtures" / "modality"
WRITE = os.environ.get("SCMLE_WRITE_GOLDENS") == "1"


@pytest.fixture
def registered():
    """The media kinds, as `kinds.KINDS` registers them."""
    for kind in (image.IMAGE, video.VIDEO):
        assert kinds.KINDS[kind.name] is kind
        assert {item.key for item in kind.declare()} <= set(
            roster.enrol().records
        )


def texture(seed: int, tint: int) -> numpy.ndarray:
    """A blocky random texture whose green channel carries the class."""
    rng = numpy.random.default_rng(seed)
    field = numpy.kron(rng.uniform(30, 200, (4, 4)), numpy.ones((12, 12)))
    noise = rng.normal(0, 20, field.shape)
    green = numpy.clip(field / 2 + tint + noise, 0, 255)
    return numpy.stack([field, green, 255 - field], axis=-1).astype(numpy.uint8)


def write(path: pathlib.Path, pixels: numpy.ndarray) -> None:
    """Writes RGB pixels as a PNG file."""
    pillow = pytest.importorskip(
        "PIL.Image", reason="the optional Pillow extra (multimodal) is missing"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    output = io.BytesIO()
    pillow.fromarray(pixels).save(output, format="PNG")
    path.write_bytes(output.getvalue())


def collection(root: pathlib.Path, count: int = 30) -> dict[str, str]:
    """Writes two classes of textures plus exact and edited copies.

    Returns:
        Each copy's file name to the file name of its original.
    """
    copies = {}
    for label, tint in (("cat", 0), ("dog", 60)):
        for index in range(count):
            pixels = texture(tint * 1000 + index, tint)
            write(root / label / f"{index:02d}.png", pixels)
            if index < 4:
                copy = f"{label}/{index:02d}copy.png"
                write(root / copy, pixels)
                edit = f"{label}/{index:02d}edit.png"
                brighter = numpy.clip(pixels * 0.9 + 12, 0, 255)
                write(root / edit, brighter.astype(numpy.uint8))
                copies[copy] = copies[edit] = f"{label}/{index:02d}.png"
    return copies


def project(
    data: pathlib.Path,
    target: str,
    output: pathlib.Path,
    task: tasks.Task,
    **extra,
):
    """A small, quick project over a media collection, opted in."""
    return scmle.api.project.Project(
        data=str(data),
        target=target,
        modality=extra.pop("modality", "image"),
        models=support.optin(task),
        budget=40,
        folds=3,
        directory=str(output),
        search=tasks.Search(top_models=0),
        strategies=tasks.Strategies(top_models=0),
        ensemble=tasks.Stacking(size=0),
        **extra,
    )


def events(run_dir: pathlib.Path) -> list[str]:
    """The kinds of the run's events, in order."""
    lines = (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line)["kind"] for line in lines if line]


def shape(value):
    """The keys and value types of JSON data, without its values."""
    if isinstance(value, dict):
        return {str(key): shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def capture(run) -> dict[str, object]:
    """What the golden records of a media run."""
    run_dir = pathlib.Path(run.directory)
    kinds_seen = events(run_dir)
    return {
        "stages": sorted(
            path.stem for path in (run_dir / "stages").glob("*.json")
        ),
        "interval": shape(run.evaluation["confidence"]),
        "custody": [
            kind
            for kind in kinds_seen
            if kind
            in {
                "split.created",
                "selection.frozen",
                "test.opened",
                "model.shipped",
            }
        ],
        "model": sorted(path.name for path in (run_dir / "model").iterdir()),
    }


def check(name: str, found: dict[str, object]) -> None:
    """Compares a run with its golden, or writes the golden."""
    golden = GOLDENS / name / "kind.json"
    if WRITE:
        golden.write_text(
            json.dumps(found, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
    assert found == json.loads(golden.read_text(encoding="utf-8"))


def manifest(run_dir: pathlib.Path) -> dict[str, object]:
    """The run's split manifest."""
    path = run_dir / "validation_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def together(
    run_dir: pathlib.Path, names: list[str], copies: dict[str, str]
) -> None:
    """Asserts each copy fell on its original's side of the split."""
    split = manifest(run_dir)
    side = {names[row]: "development" for row in split["training"]}
    side.update({names[row]: "test" for row in split["test"]})
    for copy, original in copies.items():
        assert side[copy] == side[original], (copy, original)


def test_an_image_directory_runs_the_shared_stages_end_to_end(
    registered, tmp_path
):
    """ML-1, MM-10, MM-17: train, test once, bracket and package images.

    The package is a standard bundle: its model is the bundle's own
    `pipeline.joblib`, and a manifest naming a file outside it is refused.
    """
    copies = collection(tmp_path / "pictures")
    run = project(
        tmp_path / "pictures", "animal", tmp_path / "runs", tasks.Task.BINARY
    ).run()
    run_dir = pathlib.Path(run.directory)
    assert run.evaluation["task"] == "binary_classification"
    assert run.evaluation["opened"] is True
    assert run.evaluation["confidence"]["available"] is True
    assert events(run_dir).count("test.opened") == 1
    models = set(run.leaderboard()["model"])
    assert len(models - {"dummy", "stacking", "blending"}) >= 3, models
    files = sorted(
        str(path.relative_to(tmp_path / "pictures"))
        for path in (tmp_path / "pictures").rglob("*.png")
    )
    assert manifest(run_dir)["strategy"] == "group"
    together(run_dir, files, copies)
    spec = tasks.Config(
        data=tasks.Dataset(
            uri=str(tmp_path / "pictures"), target="animal", modality="image"
        )
    )
    frame = image.IMAGE.read(spec)[0]
    predicted = run.predict(frame.drop(columns=["animal"]).head(6))
    assert set(numpy.asarray(predicted).tolist()) <= {"cat", "dog"}
    check("image", capture(run))
    loaded = archive.Archive.load(run_dir / "model", trusted=True)
    assert len(loaded.predict(frame.drop(columns=["animal"]).head(2))) == 2
    record = json.loads(
        (run_dir / "model" / "model_manifest.json").read_text(encoding="utf-8")
    )
    (tmp_path / "outside.joblib").write_bytes(b"not the packaged model")
    record["files"]["../../outside.joblib"] = record["files"].pop(
        "pipeline.joblib"
    )
    (run_dir / "model" / "model_manifest.json").write_text(
        json.dumps(record), encoding="utf-8"
    )
    with pytest.raises(errors.ArtifactError):
        archive.Archive.load(run_dir / "model", trusted=True)


def test_a_listed_table_with_a_numeric_target_trains_a_regression(
    registered, tmp_path
):
    root = tmp_path / "pictures"
    rows = []
    for index in range(50):
        pixels = texture(index, 0)
        write(root / f"{index:02d}.png", pixels)
        rows.append(
            {
                "path": f"{index:02d}.png",
                "brightness": float(pixels.mean()) + index % 3,
            }
        )
    pandas.DataFrame(rows).sample(frac=1.0, random_state=0).to_csv(
        root / "listed.csv", index=False
    )
    run = project(
        root / "listed.csv",
        "brightness",
        tmp_path / "runs",
        tasks.Task.REGRESSION,
    ).run()
    assert run.evaluation["task"] == "regression"
    assert run.evaluation["opened"] is True
    assert run.evaluation["confidence"]["available"] is True
