"""Behavioural tests for the image modality, run on a stand-in for Pillow.

The stand-in decodes a tiny self-describing format, so every decoding path
runs, and gives the same answers, whether or not Pillow is installed.
"""

import hashlib
import io
import json
import pathlib
import types

import numpy
import pytest

from scmle.contracts import data
from scmle.contracts import errors
from scmle.data.modalities import image
from scmle.data.modalities.image import dataset
from scmle.data.modalities.image import duplicates
from scmle.data.modalities.image import evidence
from scmle.data.modalities.image import failures
from scmle.data.modalities.image import pixels
from scmle.data.modalities.image import sample
from scmle.data.modalities.image import survey

MAGIC = b"FAKE"


class Picture:
    """A decoded stand-in image holding a (height, width, channels) array."""

    def __init__(self, values, mode, exif=None, kind="FAKE"):
        self.values = values
        self.mode = mode
        self.exif = exif or {}
        self.format = kind

    def __enter__(self):
        return self

    def __exit__(self, *arguments):
        return False

    @property
    def size(self):
        return (self.values.shape[1], self.values.shape[0])

    def verify(self):
        return None

    def load(self):
        return None

    def getbands(self):
        return ("L",) if self.mode == "L" else tuple(self.mode)

    def getexif(self):
        return dict(self.exif)

    def tobytes(self):
        return self.values.tobytes()

    def convert(self, mode):
        rgb = self.values[..., :3]
        if self.values.shape[2] == 1:
            rgb = numpy.repeat(self.values, 3, axis=2)
        if mode == "L":
            gray = numpy.rint(rgb.astype(float).mean(axis=2, keepdims=True))
            return Picture(gray.astype(numpy.uint8), "L")
        if mode == "RGB":
            return Picture(numpy.ascontiguousarray(rgb), "RGB")
        alpha = numpy.full((*rgb.shape[:2], 1), 255, dtype=numpy.uint8)
        return Picture(numpy.concatenate([rgb, alpha], axis=2), "RGBA")

    def resize(self, size, resampling):
        width, height = size
        rows = numpy.arange(height) * self.values.shape[0] // height
        columns = numpy.arange(width) * self.values.shape[1] // width
        return Picture(self.values[rows][:, columns], self.mode)

    def __array__(self, dtype=None, copy=None):
        values = self.values[..., 0] if self.mode == "L" else self.values
        return values if dtype is None else values.astype(dtype)


class Library:
    """The parts of `PIL.Image` the image modality calls."""

    Resampling = types.SimpleNamespace(BILINEAR="bilinear")

    @staticmethod
    def open(stream):
        raw = stream.read()
        if not raw.startswith(MAGIC):
            raise OSError("cannot identify image file")
        header, body = raw[len(MAGIC) :].split(b"\n", 1)
        meta = json.loads(header)
        values = numpy.frombuffer(body, dtype=numpy.uint8).reshape(
            meta["shape"]
        )
        exif = {int(key): value for key, value in meta["exif"].items()}
        return Picture(values, meta["mode"], exif, meta["format"])


def encode(values, mode="RGB", exif=None, kind="FAKE"):
    array = numpy.asarray(values, dtype=numpy.uint8)
    if array.ndim == 2:
        array = array[..., numpy.newaxis]
    header = {
        "shape": list(array.shape),
        "mode": mode,
        "exif": {str(k): v for k, v in (exif or {}).items()},
        "format": kind,
    }
    return MAGIC + json.dumps(header).encode() + b"\n" + array.tobytes()


def flat(value, width=2, height=2, mode="RGB"):
    channels = 1 if mode == "L" else len(mode)
    return encode(numpy.full((height, width, channels), value), mode)


@pytest.fixture(autouse=True)
def pillow(monkeypatch):
    monkeypatch.setattr(image, "PillowModule", Library)
    monkeypatch.setattr(image, "PILLOW_AVAILABLE", True)


def make(value, source, **kwargs):
    return sample.Sample.parse(flat(value), source=source, **kwargs)


def test_image_errors_carry_their_context():
    error = failures.ImageError("m", code="c", source="s", details={"k": 1})
    assert error.details == {"code": "c", "source": "s", "k": 1}
    assert (error.code, error.source) == ("c", "s")
    assert failures.ImageError("m", code="c").details == {"code": "c"}
    missing = failures.DependencyError("op")
    assert missing.details["operation"] == "op"
    assert missing.code == "missing_dependency"
    assert missing.category == "dependency"


def test_the_dependency_error_is_raised_without_pillow(monkeypatch):
    monkeypatch.setattr(image, "PILLOW_AVAILABLE", False)
    with pytest.raises(failures.DependencyError):
        sample.Sample.parse(flat(1))


@pytest.mark.parametrize(
    "size", [(0, 1), (1,), (1, 2, 3), (1.0, 2), (True, 2), (2, -1)]
)
def test_configure_rejects_bad_sizes(size):
    with pytest.raises(errors.InputError, match="two positive integers"):
        pixels.configure(size, "RGB")


def test_configure_accepts_the_supported_modes():
    for mode in ("L", "RGB", "RGBA"):
        assert pixels.configure((1, 1), mode) == (1, 1)
    with pytest.raises(errors.InputError, match="mode must be one of"):
        pixels.configure((1, 1), "CMYK")


def test_bounds_checks_dimensions_and_pixel_limit():
    sample.bounds(2, 3, 6, "s")
    sample.bounds(2, 3, None, "s")
    with pytest.raises(failures.ViolationError) as raised:
        sample.bounds(2, 3, 5, "s")
    assert raised.value.details == {
        "code": "pixel_count_limit",
        "source": "s",
        "pixel_count": 6,
        "max_pixels": 5,
    }
    with pytest.raises(failures.ViolationError, match="positive"):
        sample.bounds(0, 3, None, "s")
    with pytest.raises(failures.ViolationError, match="positive"):
        sample.bounds(3, 0, None, "s")


def test_canonical_source_normalises_paths():
    assert sample.identify("") == "memory://"
    assert sample.identify(" mem://a ") == "mem://a"
    assert sample.identify("x.png") == (
        pathlib.Path("x.png").resolve().as_uri()
    )


def test_imprint_hashes_dimensions_and_rgba_pixels():
    picture = Library.open(io.BytesIO(flat(7, 1, 1)))
    payload = b"scmle-image-content-v1\x001x1\x00" + bytes([7, 7, 7, 255])
    assert sample.imprint(picture, 1, 1) == hashlib.sha256(payload).hexdigest()


def test_from_bytes_reads_metadata_and_identity():
    raw = encode(
        numpy.zeros((3, 2, 3)), exif={271: "cam", 9: "x", 34853: "gps"}
    )
    built = sample.Sample.parse(raw, source="mem://a", label=1)
    assert (built.width, built.height, built.channels) == (2, 3, 3)
    assert built.mode == "RGB"
    assert built.format == "FAKE"
    assert built.size == len(raw)
    assert built.file_hash == hashlib.sha256(raw).hexdigest()
    assert built.tags == ("271", "34853", "9")
    assert built.metadata == ("camera_make", "gps_info")
    unknown = sample.Sample.parse(encode(numpy.zeros((1, 1)), "L", kind=None))
    assert unknown.format == "unknown"
    assert unknown.channels == 1


def test_the_content_hash_is_independent_of_the_encoding():
    rgb = sample.Sample.parse(encode(numpy.full((1, 1, 3), 9)))
    rgba = sample.Sample.parse(encode(numpy.array([[[9, 9, 9, 255]]]), "RGBA"))
    assert rgb.content_hash == rgba.content_hash
    assert rgb.file_hash != rgba.file_hash


def test_from_bytes_enforces_limits():
    raw = flat(1)
    with pytest.raises(failures.ViolationError) as raised:
        sample.Sample.parse(b"")
    assert raised.value.code == "empty_payload"
    with pytest.raises(failures.ViolationError) as raised:
        sample.Sample.parse(raw, max_bytes=len(raw) - 1)
    assert raised.value.code == "encoded_size_limit"
    assert sample.Sample.parse(raw, max_bytes=len(raw)).size == len(raw)
    with pytest.raises(failures.ViolationError) as raised:
        sample.Sample.parse(raw, max_pixels=3)
    assert raised.value.code == "pixel_count_limit"
    assert sample.Sample.parse(raw, max_pixels=4).width == 2
    with pytest.raises(failures.ViolationError) as raised:
        sample.Sample.parse(b"junk")
    assert raised.value.code == "corrupt_payload"
    assert raised.value.details["exception"] == "OSError"


@pytest.mark.parametrize("limit", [0, -1, 1.5, True, "1"])
def test_validate_rejects_bad_limits(limit):
    with pytest.raises(errors.InputError, match="max_bytes must be"):
        sample.Sample.validate(limit, None)
    with pytest.raises(errors.InputError, match="max_pixels must be"):
        sample.Sample.validate(None, limit)


def test_from_path_reads_files_and_checks_them(tmp_path):
    path = tmp_path / "a.fake"
    path.write_bytes(flat(5))
    built = sample.Sample.load(path, label="x", group_id="g")
    assert built.source == path.resolve().as_uri()
    assert (built.label, built.group) == ("x", "g")
    with pytest.raises(failures.ViolationError) as raised:
        sample.Sample.load(tmp_path / "missing")
    assert raised.value.code == "missing_file"
    with pytest.raises(failures.ViolationError) as raised:
        sample.Sample.load(tmp_path)
    assert raised.value.code == "not_a_file"
    with pytest.raises(failures.ViolationError) as raised:
        sample.Sample.load(path, max_bytes=3)
    assert raised.value.details["byte_size"] == len(flat(5))


def test_a_sample_validates_its_fields():
    built = make(1, "mem://a")
    fields = {
        "source": built.source,
        "encoded": built.encoded,
        "file_hash": built.file_hash,
        "content_hash": built.content_hash,
        "width": 2,
        "height": 2,
        "channels": 3,
        "mode": "RGB",
        "format": "FAKE",
        "size": built.size,
    }
    for change, kind, message in (
        ({"source": " "}, errors.InputError, "source"),
        ({"group": ""}, errors.InputError, "group_id"),
        ({"augmentation": "x"}, errors.KindError, "augmentation"),
        ({"size": 1}, errors.InputError, "byte_size"),
        ({"width": 0}, errors.InputError, "dimensions"),
        ({"height": 0}, errors.InputError, "dimensions"),
        ({"channels": 0}, errors.InputError, "channels"),
        ({"file_hash": "a"}, errors.InputError, "SHA-256"),
        ({"content_hash": "a" * 65}, errors.InputError, "SHA-256"),
    ):
        with pytest.raises(kind, match=message):
            sample.Sample(**(fields | change))
    assert sample.Sample(**fields).tags == ()


def test_to_dict_omits_bytes_and_nests_digests():
    built = make(1, "mem://a", group_id="g")
    record = built.to_dict()
    assert "encoded_bytes" not in record
    assert record["digests"] == {
        "file": built.file_hash,
        "content": built.content_hash,
    }
    assert record["group"] == "g"
    assert record["size"] == built.size
    assert record["exif"] == []
    assert record["augmentation"] is None


def test_augmentation_normalises_its_provenance():
    record = sample.Augmentation(
        " mem://p ", " flip ", {"b": 1, "a": [1, 2]}, tool=" t "
    )
    assert record.source == "mem://p"
    assert record.operation == "flip"
    assert record.tool == "t"
    assert dict(record.parameters) == {"a": [1, 2], "b": 1}
    assert record.to_dict() == {
        "parent": "mem://p",
        "operation": "flip",
        "parameters": {"a": [1, 2], "b": 1},
        "tool": "t",
    }
    assert record.identity == (
        '{"operation":"flip","parameters":{"a":[1,2],"b":1},'
        '"parent_source":"mem://p","tool":"t"}'
    )
    with pytest.raises(TypeError):
        record.parameters["c"] = 1


@pytest.mark.parametrize(
    ("operation", "parameters", "message"),
    [
        (" ", {}, "operation"),
        ("x", {"": 1}, "keys"),
        ("x", {1: 1}, "keys"),
        ("x", {"a": float("nan")}, "JSON-native"),
        ("x", {"a": object()}, "JSON-native"),
    ],
)
def test_augmentation_rejects_bad_provenance(operation, parameters, message):
    with pytest.raises(errors.InputError, match=message):
        sample.Augmentation("p", operation, parameters)


def test_decode_converts_and_resizes():
    built = sample.Sample.parse(encode(numpy.arange(12).reshape(2, 2, 3)))
    rgb = built.decode()
    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == numpy.uint8
    assert rgb[0, 1].tolist() == [3, 4, 5]
    gray = built.decode("L", (1, 1))
    assert gray.shape == (1, 1, 1)
    assert gray[0, 0, 0] == 1
    assert built.decode("RGBA").shape == (2, 2, 4)
    with pytest.raises(errors.InputError, match="mode must be one of"):
        built.decode("P")
    with pytest.raises(errors.InputError, match="two positive integers"):
        built.decode("RGB", (0, 1))


def test_decode_wraps_failures():
    built = make(1, "mem://a")
    broken = sample.Sample(
        **(
            {
                field: getattr(built, field)
                for field in (
                    "source",
                    "file_hash",
                    "content_hash",
                    "width",
                    "height",
                    "channels",
                    "mode",
                    "format",
                )
            }
            | {"encoded": b"junk", "size": 4}
        )
    )
    with pytest.raises(failures.ViolationError) as raised:
        broken.decode()
    assert raised.value.code == "decode_failure"
    assert raised.value.source == "mem://a"


def test_fingerprint_quantises_a_luminance_thumbnail():
    built = make(100, "mem://a")
    assert built.fingerprint(2) == "6666"
    assert len(built.fingerprint()) == 64
    assert len(make(255, "mem://b").fingerprint(32)) == 1024
    for bad in (1, 33, 2.0, True):
        with pytest.raises(errors.InputError, match="between 2 and 32"):
            built.fingerprint(bad)
    assert (pixels.MIN_HASH_SIZE, pixels.MAX_HASH_SIZE) == (2, 32)


def test_islands_returns_components_of_two_or_more():
    neighbors = {0: {2}, 1: set(), 2: {0, 3}, 3: {2}, 4: {5}, 5: {4}}
    assert duplicates.islands(neighbors) == [{0, 2, 3}, {4, 5}]
    assert duplicates.islands({0: set()}) == []


def test_near_duplicate_groups_need_distinct_content():
    data = dataset.Dataset(
        (
            make(100, "mem://a"),
            make(101, "mem://b"),
            make(100, "mem://c"),
            make(250, "mem://d"),
        )
    )
    assert data.group(hash_size=2, max_hamming_distance=0) == (
        ("mem://a", "mem://b", "mem://c"),
    )
    assert data.group(max_samples=1) == ()
    identical = dataset.Dataset((make(1, "mem://a"), make(1, "mem://b")))
    assert identical.group() == ()
    far = data.group(hash_size=2, max_hamming_distance=64)
    assert far == (("mem://a", "mem://b", "mem://c", "mem://d"),)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_samples": 0}, "max_samples"),
        ({"max_samples": True}, "max_samples"),
        ({"max_samples": 1.0}, "max_samples"),
        ({"max_hamming_distance": -1}, "max_hamming_distance"),
        ({"max_hamming_distance": False}, "max_hamming_distance"),
    ],
)
def test_near_duplicate_groups_validate_limits(kwargs, message):
    data = dataset.Dataset((make(1, "mem://a"),))
    with pytest.raises(errors.InputError, match=message):
        data.group(**kwargs)


def test_a_dataset_hash_ignores_order_and_binds_augmentation():
    first, second = make(1, "mem://a"), make(2, "mem://b")
    forward = dataset.Dataset((first, second))
    assert dataset.Dataset((second, first)).digest == forward.digest
    augmented = make(
        2, "mem://b", augmentation=sample.Augmentation("mem://a", "flip")
    )
    assert dataset.Dataset((first, augmented)).digest != forward.digest
    with pytest.raises(errors.KindError, match="only Sample"):
        dataset.Dataset(("x",))
    assert forward.size == 2


def test_deduplicate_keeps_the_smallest_source_per_content():
    data = dataset.Dataset(
        (make(1, "mem://c"), make(1, "mem://a"), make(2, "mem://b"))
    )
    kept = data.deduplicate()
    assert sorted(s.source for s in kept.samples) == ["mem://a", "mem://b"]
    hashes = [s.content_hash for s in kept.samples]
    assert hashes == sorted(hashes)
    assert isinstance(kept, dataset.Dataset)


def test_flatten_builds_one_row_per_sample():
    data = dataset.Dataset((make(51, "mem://a"), make(102, "mem://b")))
    features = data.flatten((1, 1), "RGB")
    assert features.shape == (2, 3)
    assert features.dtype == numpy.float32
    assert features[0].tolist() == pytest.approx([0.2, 0.2, 0.2])
    raw = data.flatten((2, 1), "L", normalize=False)
    assert raw.dtype == numpy.uint8
    assert raw.tolist() == [[51, 51], [102, 102]]
    empty = dataset.Dataset().flatten((2, 3), "RGBA")
    assert empty.shape == (0, 24)
    assert empty.dtype == numpy.float32
    assert dataset.Dataset().flatten((1, 1), "L", False).dtype == numpy.uint8


def test_split_keeps_groups_whole():
    samples = tuple(
        make(i, f"mem://{i}", group_id=f"g{i % 5}") for i in range(10)
    )
    split = dataset.Dataset(samples).split(0.2, 0.2, seed=1)
    cohorts = split.cohorts()
    assert [len(cohorts[k]) for k in ("train", "validation", "test")] == [
        3,
        1,
        1,
    ]
    assert len(split.train) == 6
    with pytest.raises(errors.InputError, match="empty image dataset"):
        dataset.Dataset().split()
    with pytest.raises(errors.InputError, match="must be < 1"):
        dataset.Dataset(samples).split(0.5, 0.5)


def test_a_split_lists_its_cohorts():
    split = evidence.Split(
        (make(1, "mem://b"), make(2, "mem://a", group_id="g")), (), ()
    )
    assert split.cohorts() == {
        "train": ("g", "mem://b"),
        "validation": (),
        "test": (),
    }


def test_profile_measures_pixels_and_shapes():
    data = dataset.Dataset(
        (
            sample.Sample.parse(flat(100, 2, 2), "mem://a"),
            sample.Sample.parse(flat(200, 4, 2), "mem://b"),
            sample.Sample.parse(flat(100, 2, 2, "L"), "mem://c"),
        )
    )
    profile = data.profile(max_samples=2, profile_size=(2, 2))
    assert profile.samples == 3
    assert profile.profiled == 2
    assert profile.truncated
    assert profile.unique == 2
    assert profile.duplicates == 0
    assert profile.modes == (("RGB", 2),)
    assert profile.channels == ((3, 2),)
    assert (profile.width_min, profile.width_max, profile.width_mean) == (
        2,
        4,
        3.0,
    )
    assert profile.height_mean == 2.0
    assert profile.mean_pixel == (150.0, 150.0, 150.0)
    assert profile.pixel_std == (0.0, 0.0, 0.0)
    full = data.profile(profile_size=(2, 2))
    assert not full.truncated
    assert full.modes == (("L", 1), ("RGB", 2))
    assert full.channels == ((1, 1), (3, 2))
    assert full.duplicates == 1
    record = full.to_dict()
    assert record["width"] == {"low": 2, "high": 4, "mean": 8 / 3}
    assert record["samples"] == 3


def test_profile_of_an_empty_dataset_is_all_zero():
    assert dataset.Dataset().profile() == survey.EMPTY_PROFILE
    with pytest.raises(errors.InputError, match="max_samples"):
        dataset.Dataset().profile(max_samples=0)
    with pytest.raises(errors.InputError, match="max_samples"):
        dataset.Dataset().profile(max_samples=False)
    with pytest.raises(errors.InputError, match="two positive"):
        dataset.Dataset().profile(profile_size=(0, 0))


def test_palette_averages_channel_statistics():
    split = sample.Sample.parse(
        encode(numpy.array([[[0, 0, 0], [100, 100, 100]]]))
    )
    mean, variance = survey.palette([split], (2, 1))
    assert mean.tolist() == [50.0, 50.0, 50.0]
    assert variance.tolist() == [2500.0, 2500.0, 2500.0]


def test_augmentation_summary_lists_external_parents():
    inside = make(1, "mem://a")
    child = make(
        2, "mem://b", augmentation=sample.Augmentation("mem://a", "flip")
    )
    orphan = make(
        3, "mem://c", augmentation=sample.Augmentation("mem://z", "crop")
    )
    assert survey.augmentation([inside]) == {
        "status": "absent",
        "augmented_sample_count": 0,
        "operations": [],
        "external_parent_sources": [],
    }
    assert survey.augmentation([inside, child, orphan]) == {
        "status": "declared",
        "augmented_sample_count": 2,
        "operations": ["crop", "flip"],
        "external_parent_sources": ["mem://z"],
    }


def test_defects_lists_quality_findings_in_order():
    profile = dataset.Dataset(
        (make(1, "mem://a"), make(1, "mem://b"), make(2, "mem://c"))
    ).profile(max_samples=2)
    findings = survey.defects(3, profile, [("mem://a", "mem://b")], 2)
    assert [f.identifier for f in findings] == [
        "quality.image-duplicate-content",
        "quality.image-near-duplicate-content",
        "quality.image-profile-truncated",
    ]
    assert findings[0].evidence == {"duplicate_count": 1}
    assert findings[2].evidence == {"max_samples": 2, "profiled_count": 2}
    empty = survey.defects(0, survey.EMPTY_PROFILE, [], 1)
    assert [f.identifier for f in empty] == ["quality.image-empty-dataset"]
    assert empty[0].severity == data.FindingSeverity.CRITICAL
    assert empty[0].blocking


def test_hazards_lists_risk_findings_in_order():
    findings = survey.hazards([{"identity": "x"}], ["gps_info"], {"a": 1})
    assert [f.identifier for f in findings] == [
        "risk.image-conflicting-duplicate-labels",
        "risk.image-sensitive-metadata",
    ]
    assert findings[1].evidence == {"tags": ["gps_info"]}
    mixed = survey.hazards([], [], {"str:a": 1, "unlabeled": 1})
    assert [f.identifier for f in mixed] == ["risk.image-mixed-label-state"]
    assert survey.hazards([], [], {"unlabeled": 2}) == []


def test_curate_assembles_profile_and_findings():
    sensitive = sample.Sample.parse(
        encode(numpy.full((2, 2, 3), 1), exif={315: "me"}),
        source="mem://s",
        label="x",
    )
    data = dataset.Dataset((sensitive, make(1, "mem://t", label="y")))
    report = data.curate()
    assert report.modality == "image"
    assert report.samples == 2
    assert report.dataset_hash == data.digest
    assert report.profile["sensitive_metadata"] == ["artist"]
    assert report.profile["label_counts"] == {"str:x": 1, "str:y": 1}
    assert report.profile["near_duplicate_groups"] == []
    assert report.profile["augmentation"]["status"] == "absent"
    assert [f.identifier for f in report.risk_findings] == [
        "risk.image-conflicting-duplicate-labels",
        "risk.image-sensitive-metadata",
    ]
    assert [f.identifier for f in report.findings] == [
        "quality.image-duplicate-content"
    ]
