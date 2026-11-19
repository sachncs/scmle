"""Video through the shared stages: `Project.run` with `modality="video"`.

Clips are encoded with FFmpeg from random textures that either move or
stand still. A listed table with an observation time gives a time-ordered
split that finishes with validation folds (MM-3); copied clips stay on one
side (MM-10); the rows do not depend on the listing order (MM-12). The
golden `tests/fixtures/modality/video/kind.json` records the stages, the
held-out interval and the custody events:

    SCMLE_WRITE_GOLDENS=1 uv run pytest tests/integration/test_video_kind.py
"""

import pathlib
import shutil
import subprocess

import numpy
import pandas
import pytest

from scmle.contracts import tasks
from scmle.pipeline import kinds
from scmle.pipeline.kinds import video
from tests.integration import test_image_kind


@pytest.fixture
def registered():
    """The media kinds, as `kinds.KINDS` registers them."""
    assert kinds.KINDS[video.VIDEO.name] is video.VIDEO


pytestmark = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="the optional ffmpeg and ffprobe programs are missing",
)


def clip(path: pathlib.Path, seed: int, moving: bool) -> None:
    """Encodes a one-second 48 x 48 clip of a random texture."""
    rng = numpy.random.default_rng(seed)
    texture = numpy.kron(rng.uniform(20, 230, (6, 6, 3)), numpy.ones((8, 8, 1)))
    frames = numpy.stack(
        [
            numpy.roll(texture, 4 * step if moving else 0, axis=1)
            for step in range(10)
        ]
    ).astype(numpy.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "48x48",
            "-r",
            "10",
            "-i",
            "-",
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            str(path),
        ],
        input=frames.tobytes(),
        check=True,
    )


def listing(
    root: pathlib.Path, count: int = 60
) -> tuple[pandas.DataFrame, dict[str, str]]:
    """Writes moving and still clips and three copies; lists them in time.

    Returns:
        The listing (path, label, time) and each copy's original.
    """
    rows = []
    copies = {}
    for index in range(count):
        moving = index % 2 == 0
        name = f"clips/{index:02d}.mp4"
        clip(root / name, index, moving)
        label = "moving" if moving else "still"
        rows.append({"path": name, "label": label, "when": float(index)})
        if index < 3:
            copy = f"clips/{index:02d}copy.mp4"
            shutil.copyfile(root / name, root / copy)
            rows.append(
                {"path": copy, "label": label, "when": float(index) + 0.5}
            )
            copies[copy] = name
    return pandas.DataFrame(rows), copies


def test_a_timed_video_listing_runs_the_shared_stages_end_to_end(
    registered, tmp_path
):
    """ML-1, MM-3, MM-10: a time-ordered video split finishes with folds."""
    table, copies = listing(tmp_path)
    table.to_csv(tmp_path / "listed.csv", index=False)
    run = test_image_kind.project(
        tmp_path / "listed.csv",
        "label",
        tmp_path / "runs",
        tasks.Task.BINARY,
        modality="video",
        time_column="when",
    ).run()
    run_dir = pathlib.Path(run.run_dir)
    split = test_image_kind.manifest(run_dir)
    assert split["strategy"] == "temporal"
    assert split["folds"] and all(fold["val"] for fold in split["folds"])
    assert run.evaluation["opened"] is True
    assert run.evaluation["confidence"]["available"] is True
    names = sorted(table["path"])
    test_image_kind.together(run_dir, names, copies)
    when = table.set_index("path")["when"]
    tested = [when[names[row]] for row in split["test"]]
    trained = [when[names[row]] for row in split["training"]]
    assert min(tested) > min(trained)
    test_image_kind.check("video", test_image_kind.capture(run))


def test_video_rows_do_not_depend_on_the_listing_order(tmp_path):
    """MM-12: a shuffled listing reads to the same rows."""
    table = listing(tmp_path, 6)[0]
    table.to_csv(tmp_path / "listed.csv", index=False)
    table.iloc[::-1].to_csv(tmp_path / "reversed.csv", index=False)
    read = [
        video.VIDEO.read(
            tasks.Config(
                data=tasks.Dataset(
                    uri=str(tmp_path / name), target="label", modality="video"
                )
            )
        )[0]
        for name in ("listed.csv", "reversed.csv")
    ]
    pandas.testing.assert_frame_equal(read[0], read[1])
