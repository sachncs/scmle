"""The image and video kinds: descriptors, near-duplicate links, reading.

Pillow (the `multimodal` extra) and the FFmpeg programs are optional; a
test that needs one skips, naming it, only when it is missing.
"""

import io
import math
import pathlib
import shutil
import subprocess
import tracemalloc

import numpy
import pandas
import pytest

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import tasks
import scmle.data.modalities.image
from scmle.data.modalities.image import descriptor
from scmle.data.modalities.image import linkage
from scmle.data.modalities.video import stream
from scmle.pipeline import kinds
from scmle.pipeline import profiler
from scmle.pipeline import roster
from scmle.pipeline.kinds import image
from scmle.pipeline.kinds import rows
from scmle.pipeline.kinds import video

FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe")
NEEDS_FFMPEG = pytest.mark.skipif(
    not FFMPEG, reason="the optional ffmpeg and ffprobe programs are missing"
)


def pillow():
    """The Pillow image module, or a skip naming the missing extra."""
    return pytest.importorskip(
        "PIL.Image", reason="the optional Pillow extra (multimodal) is missing"
    )


def texture(seed: int, side: int = 24) -> numpy.ndarray:
    """A blocky random RGB texture, distinct for every seed."""
    rng = numpy.random.default_rng(seed)
    field = numpy.kron(
        rng.uniform(30, 220, (4, 4, 3)), numpy.ones((side // 4, side // 4, 1))
    )
    return field.astype(numpy.uint8)


def png(pixels: numpy.ndarray) -> bytes:
    """Encodes RGB pixels as PNG."""
    output = io.BytesIO()
    pillow().fromarray(pixels).save(output, format="PNG")
    return output.getvalue()


def thumbs(rows: numpy.ndarray) -> pandas.DataFrame:
    """A frame holding only the thumbnail columns of rows x 64 x 3 values."""
    names = [name for name in descriptor.names() if name.startswith("thumb_")]
    return pandas.DataFrame(
        rows.transpose(0, 2, 1).reshape(len(rows), -1), columns=names
    )


def link(frame, *declared):
    """Each row's cohort: linked pictures and declared groups joined."""
    left, right = linkage.edges(frame)
    return rows.unite(
        len(frame), zip(left.tolist(), right.tolist(), strict=True), *declared
    )


def test_the_link_distance_is_the_largest_with_a_null_tail_below_one_false_link():
    assert linkage.tail(linkage.DISTANCE) <= linkage.FALSE
    assert linkage.tail(linkage.DISTANCE + 1) > linkage.FALSE
    assert math.isclose(linkage.FALSE * math.comb(linkage.REFERENCE, 2), 1.0)


def test_a_descriptor_row_matches_its_names_and_reads_back_its_thumbnail():
    pixels = texture(3, descriptor.WORK)
    row = descriptor.describe(pixels, 640, 480)
    assert row.shape == (len(descriptor.names()),)
    assert numpy.isfinite(row).all()
    frame = pandas.DataFrame([row], columns=descriptor.names())
    back = linkage.thumbnails(frame)[0]
    step = descriptor.WORK // descriptor.SIDE
    expected = (pixels / 255.0).reshape(8, step, 8, step, 3).mean(axis=(1, 3))
    numpy.testing.assert_allclose(back, expected.reshape(64, 3))
    assert row[-1] == pytest.approx(640 / 480)


def test_descriptors_refuse_pixels_of_the_wrong_shape():
    with pytest.raises(errors.InputError, match="RGB pixels"):
        descriptor.describe(numpy.zeros((8, 8, 3)), 8, 8)


def test_exact_and_brightness_edited_copies_link_and_unrelated_pictures_do_not():
    """MM-10: duplicates and near-duplicates are found, not just warned of."""
    rng = numpy.random.default_rng(0)
    base = rng.uniform(0.1, 0.8, (5, 64, 3))
    edited = base[0] * 0.9 + 0.05
    pictures = numpy.concatenate([base, base[:1], edited[None]])
    labels = link(thumbs(pictures))
    assert labels[0] == labels[5] == labels[6]
    assert len(set(labels[:5])) == 5


def test_flat_pictures_link_only_when_their_colours_agree():
    red = numpy.tile([0.8, 0.1, 0.1], (64, 1))
    green = numpy.tile([0.1, 0.8, 0.1], (64, 1))
    pictures = numpy.stack([red, green, red + 0.01])
    labels = link(thumbs(pictures))
    assert labels[0] == labels[2] != labels[1]


def test_a_declared_group_joins_its_rows_into_one_cohort():
    pictures = numpy.random.default_rng(1).uniform(0.1, 0.8, (4, 64, 3))
    labels = link(thumbs(pictures), ["a", "b", "a", "c"])
    assert labels[0] == labels[2]
    assert len(set(labels)) == 3


def test_a_row_without_a_declared_group_is_tied_by_nothing_else():
    frame = pandas.DataFrame(
        {
            "x": numpy.arange(40.0),
            "y": numpy.arange(40.0),
            "g": [None, *(f"g{index % 10}" for index in range(39))],
        }
    )
    spec = tasks.Config(
        data=tasks.Dataset(
            uri="x.csv", target="y", modality="image", group_column="g"
        )
    )
    split = image.IMAGE.constrain(spec, tasks.Task.REGRESSION).split(frame)
    groups = frame["g"].to_numpy()
    train = list(split.manifest.train_ids)
    test = list(split.manifest.test_ids)
    assert not ({*groups[train]} & {*groups[test]}) - {None}


def test_linking_without_thumbnail_columns_is_a_data_error():
    with pytest.raises(errors.DataError, match="thumbnail"):
        linkage.edges(pandas.DataFrame({"x": [1.0, 2.0]}))


def test_media_kinds_declare_experimental_capabilities_beside_the_tabular_ones():
    registry = roster.enrol()
    for kind in (image.IMAGE, video.VIDEO):
        declared = kind.declare()
        assert {item.key.modality.value for item in declared} == {kind.name}
        assert {item.key.task for item in declared} == set(rows.TASKS)
        assert all(
            item.status is capabilities.Status.EXPERIMENTAL for item in declared
        )
        assert {item.key for item in declared} <= set(registry.records)
    assert image.IMAGE.declare()[0].required_extras == ("multimodal",)


def test_media_cost_adds_the_per_file_bound_to_the_profiling_bound():
    base = profiler.estimate(100, 250, 2, 600)
    assert image.IMAGE.cost(100, 250, 2, 600) == pytest.approx(
        base + 100 * image.IMAGE.seconds + 100 * 100 * image.PAIR
    )
    assert video.VIDEO.cost(100, 250, 2, 600) == pytest.approx(
        base + 100 * video.VIDEO.seconds + 100 * 100 * image.PAIR
    )


def test_media_kinds_satisfy_the_kind_protocol():
    assert isinstance(image.IMAGE, kinds.Kind)
    assert isinstance(video.VIDEO, kinds.Kind)
    assert image.IMAGE.decode(("low", "high")) == ("low", "high")


def test_reading_images_without_pillow_names_the_missing_extra(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(scmle.data.modalities.image, "PILLOW_AVAILABLE", False)
    spec = tasks.Config(
        data=tasks.Dataset(uri=str(tmp_path), target="y", modality="image")
    )
    with pytest.raises(
        scmle.data.modalities.image.DependencyError, match="Pillow"
    ):
        image.IMAGE.read(spec)


def test_reading_video_without_ffmpeg_names_the_missing_program(
    monkeypatch, tmp_path
):
    monkeypatch.setattr(stream.shutil, "which", lambda name: None)
    spec = tasks.Config(
        data=tasks.Dataset(uri=str(tmp_path), target="y", modality="video")
    )
    with pytest.raises(errors.DataError, match="ffprobe") as caught:
        video.VIDEO.read(spec)
    assert caught.value.details["code"] == "missing_dependency"


def folder(root: pathlib.Path, count: int = 4) -> pathlib.Path:
    """Writes `count` textured PNGs per class into class subdirectories."""
    for label, offset in (("cat", 0), ("dog", 100)):
        (root / label).mkdir(parents=True)
        for index in range(count):
            (root / label / f"{index}.png").write_bytes(
                png(texture(offset + index))
            )
    return root


def test_a_class_directory_reads_one_descriptor_row_per_image(tmp_path):
    root = folder(tmp_path / "pictures")
    spec = tasks.Config(
        data=tasks.Dataset(uri=str(root), target="animal", modality="image")
    )
    frame, streamed = image.IMAGE.read(spec)
    assert streamed is None
    assert len(frame) == 8
    assert list(frame.columns) == ["animal", *descriptor.names(), rows.KEY]
    assert sorted(frame["animal"].unique()) == ["cat", "dog"]


def test_the_rows_do_not_depend_on_the_order_files_are_listed(tmp_path):
    """MM-12: a shuffled listing reads to the same rows."""
    root = folder(tmp_path / "pictures")
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*.png"))
    table = pandas.DataFrame(
        {"path": files, "y": [float(len(name)) for name in files]}
    )
    table.to_csv(root / "listed.csv", index=False)
    table.iloc[::-1].to_csv(root / "reversed.csv", index=False)
    read = [
        image.IMAGE.read(
            tasks.Config(
                data=tasks.Dataset(
                    uri=str(root / name), target="y", modality="image"
                )
            )
        )[0]
        for name in ("listed.csv", "reversed.csv")
    ]
    pandas.testing.assert_frame_equal(read[0], read[1])


def test_a_table_without_a_path_column_must_name_the_files_in_one_column(
    tmp_path,
):
    root = folder(tmp_path / "pictures", 2)
    files = sorted(str(path.relative_to(root)) for path in root.rglob("*.png"))
    pandas.DataFrame({"picture": files, "y": range(len(files))}).to_csv(
        root / "one.csv", index=False
    )
    frame = image.IMAGE.read(
        tasks.Config(
            data=tasks.Dataset(
                uri=str(root / "one.csv"), target="y", modality="image"
            )
        )
    )[0]
    assert "picture" not in frame.columns and len(frame) == len(files)
    pandas.DataFrame({"a": files, "b": files, "y": range(len(files))}).to_csv(
        root / "two.csv", index=False
    )
    with pytest.raises(errors.DataError, match="'path' column") as caught:
        image.IMAGE.read(
            tasks.Config(
                data=tasks.Dataset(
                    uri=str(root / "two.csv"), target="y", modality="image"
                )
            )
        )
    assert caught.value.details["candidates"] == ["a", "b"]


def test_every_unreadable_image_is_named_in_one_error(tmp_path):
    root = folder(tmp_path / "pictures", 2)
    (root / "cat" / "broken.png").write_bytes(b"not an image")
    (root / "dog" / "broken.png").write_bytes(b"")
    with pytest.raises(errors.DataError) as caught:
        image.IMAGE.read(
            tasks.Config(
                data=tasks.Dataset(uri=str(root), target="y", modality="image")
            )
        )
    assert caught.value.details["count"] == 2
    assert {item["source"] for item in caught.value.details["failures"]} == {
        str((root / folder / "broken.png").resolve())
        for folder in ("cat", "dog")
    }


def test_a_missing_or_empty_source_is_a_data_error(tmp_path):
    pillow()
    with pytest.raises(errors.DataError, match="does not exist"):
        image.IMAGE.read(
            tasks.Config(
                data=tasks.Dataset(
                    uri=str(tmp_path / "nope"), target="y", modality="image"
                )
            )
        )
    with pytest.raises(errors.DataError, match="names no file"):
        image.IMAGE.read(
            tasks.Config(
                data=tasks.Dataset(
                    uri=str(tmp_path), target="y", modality="image"
                )
            )
        )


def clip(
    path: pathlib.Path, source: str, seconds: float = 1.0, size: str = "64x48"
) -> pathlib.Path:
    """Encodes a synthetic MPEG-4 clip with FFmpeg."""
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"{source}=size={size}:rate=10:duration={seconds}",
            "-c:v",
            "mpeg4",
            str(path),
        ],
        check=True,
    )
    return path


@NEEDS_FFMPEG
def test_a_video_is_probed_and_described_from_sampled_frames(tmp_path):
    path = clip(tmp_path / "a.mp4", "testsrc", 2.0)
    metadata = stream.probe(path)
    assert (metadata.width, metadata.height, metadata.frame_count) == (
        64,
        48,
        20,
    )
    named = video.VIDEO.trace(path, pandas.Series(dtype=object)).features
    assert list(named) == list(stream.names())
    assert numpy.isfinite(list(named.values())).all()
    assert named["timing_seconds"] == pytest.approx(2.0)
    assert named["motion_mean"] > 0


def peak(path: pathlib.Path) -> int:
    """Bytes this process held at most while describing a video."""
    tracemalloc.start()
    try:
        video.VIDEO.trace(path, pandas.Series(dtype=object))
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@NEEDS_FFMPEG
def test_describing_a_video_never_holds_its_encoded_bytes(tmp_path):
    """MM-6: the memory a video takes is its sampled frames, not its file."""
    small = clip(tmp_path / "small.mp4", "testsrc2", 1.0, "64x48")
    large = clip(tmp_path / "large.mp4", "testsrc2", 20.0, "640x480")
    assert large.stat().st_size > 50 * small.stat().st_size
    slack = 64 * 1024
    assert peak(large) < peak(small) + slack
    assert peak(large) < large.stat().st_size


@NEEDS_FFMPEG
def test_an_unreadable_video_is_a_typed_error(tmp_path):
    bad = tmp_path / "bad.mp4"
    bad.write_bytes(b"not a video")
    with pytest.raises(errors.DataError) as caught:
        video.VIDEO.trace(bad, pandas.Series(dtype=object))
    assert caught.value.details["code"] in {"decode_failure", "no_video_stream"}


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("30000/1001", 30000 / 1001),
        ("2.5", 2.5),
        ("N/A", 0.0),
        ("0/0", 0.0),
        ("1/2/3", 0.0),
    ],
)
def test_ffprobe_numbers_read_as_plain_or_ratio(text, value):
    assert stream.fraction(text) == pytest.approx(value)


def test_occlusion_reads_every_mask_on_the_class_the_picture_was_given(
    tmp_path,
):
    """MM-13: a mask that flips the predicted class keeps its sign.

    The model calls a picture bright when its mean thumbnail exceeds 0.5.
    The picture is just bright (p = 0.77), so masking any region makes it
    confidently dark (p = 0.99): the bright score falls for every region,
    while a re-predicted class would read the dark score and report a
    negative change.
    """
    path = tmp_path / "grey.png"
    path.write_bytes(png(numpy.full((16, 16, 3), 129, dtype=numpy.uint8)))
    names = [name for name in descriptor.names() if name.startswith("thumb_")]

    def score(frame):
        bright = 1 / (1 + numpy.exp(-200 * (frame[names].mean(axis=1) - 0.5)))
        return numpy.column_stack([1 - bright, bright])

    terms = image.IMAGE.occlude(score, path)
    assert len(terms) == image.REGIONS**2
    assert all(change > 0.5 for name, change in terms)
    assert terms[0][0].startswith("region[")
    single = image.IMAGE.occlude(lambda frame: frame[names].mean(axis=1), path)
    assert all(change > 0 for name, change in single)
