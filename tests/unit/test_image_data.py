"""Tests for the optional-Pillow image data foundation."""

import io

import numpy
import pytest

from scmle.data.modalities import image

try:
    from PIL import Image
except ImportError:
    Image = None


PILLOW_TESTS_AVAILABLE = Image is not None
PILLOW_SKIP = pytest.mark.skipif(
    not PILLOW_TESTS_AVAILABLE,
    reason="Pillow-backed image test requires the optional dependency",
)


def png_bytes(color, size=(4, 3), mode="RGB"):
    """Creates a deterministic PNG fixture when Pillow is available."""
    canvas = Image.new(mode, size, color)
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def patterned_png(delta=0):
    """Creates two distinct but visually near-identical fixtures."""
    canvas = Image.new("RGB", (8, 8), (20, 40, 60))
    for index in range(8):
        value = 100 + delta
        canvas.putpixel((index, index), (value, value + 20, value + 40))
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def test_dependency_error_is_structured_when_dependency_is_disabled(
    monkeypatch,
):
    monkeypatch.setattr(image, "PILLOW_AVAILABLE", False)

    with pytest.raises(image.ImageDependencyError) as raised:
        image.require_pillow("test operation")

    assert raised.value.category == "dependency"
    assert raised.value.details == {
        "code": "missing_dependency",
        "dependency": "Pillow",
        "operation": "test operation",
        "install": "pip install Pillow",
    }


@PILLOW_SKIP
def test_sample_has_file_and_content_identity():
    payload = png_bytes((10, 20, 30))
    first = image.Sample.from_bytes(payload, "first.png")
    second = image.Sample.from_bytes(payload, "second.png")

    assert first.file_hash == second.file_hash
    assert first.content_hash == second.content_hash
    assert first.source.endswith("first.png")
    assert (first.width, first.height, first.channels, first.mode) == (
        4,
        3,
        3,
        "RGB",
    )


@PILLOW_SKIP
def test_empty_and_corrupt_payloads_are_structured():
    for payload, code in [
        (b"", "empty_payload"),
        (b"not an image", "corrupt_payload"),
    ]:
        with pytest.raises(image.ImageValidationError) as raised:
            image.Sample.from_bytes(payload)
        assert raised.value.code == code
        assert raised.value.details["source"] == "memory://"


@PILLOW_SKIP
def test_oversized_payload_and_pixel_limits_are_rejected():
    payload = png_bytes((1, 2, 3), size=(10, 10))
    with pytest.raises(image.ImageValidationError, match="bytes") as by_bytes:
        image.Sample.from_bytes(payload, max_bytes=len(payload) - 1)
    assert by_bytes.value.code == "encoded_size_limit"

    with pytest.raises(image.ImageValidationError, match="pixels") as by_pixels:
        image.Sample.from_bytes(payload, max_pixels=99)
    assert by_pixels.value.code == "pixel_count_limit"


@PILLOW_SKIP
def test_dataset_deduplication_is_content_based_and_order_independent(tmp_path):
    duplicate = png_bytes((4, 5, 6))
    first_path = tmp_path / "z.png"
    second_path = tmp_path / "a.png"
    first_path.write_bytes(duplicate)
    second_path.write_bytes(duplicate)
    unique_path = tmp_path / "unique.png"
    unique_path.write_bytes(png_bytes((200, 5, 6)))

    left = image.Dataset(
        tuple(
            image.Sample.from_path(path)
            for path in [first_path, second_path, unique_path]
        )
    ).deduplicate()
    right = image.Dataset(
        tuple(
            image.Sample.from_path(path)
            for path in [unique_path, second_path, first_path]
        )
    ).deduplicate()

    assert left == right
    assert left.sample_count == 2
    assert any(sample.source.endswith("a.png") for sample in left.samples)


@PILLOW_SKIP
def test_profile_is_bounded_and_detects_shift():
    dataset = image.Dataset(
        (
            image.Sample.from_bytes(png_bytes((0, 0, 0)), "dark.png"),
            image.Sample.from_bytes(png_bytes((255, 255, 255)), "light.png"),
        )
    )
    profile = dataset.profile(max_samples=1, profile_size=(2, 2))

    assert profile.dataset_count == 2
    assert profile.profiled_count == 1
    assert profile.truncated is True
    assert profile.mean_pixel == (0.0, 0.0, 0.0)
    shifted = image.Dataset(
        (image.Sample.from_bytes(png_bytes((255, 255, 255)), "shifted.png"),)
    ).profile(profile_size=(2, 2))
    assert all(
        before < after
        for before, after in zip(
            profile.mean_pixel, shifted.mean_pixel, strict=True
        )
    )


@PILLOW_SKIP
def test_curation_reports_duplicate_and_sampling_findings():
    """Image curation makes duplicate and bounded-profile risk visible."""
    payload = png_bytes((0, 0, 0))
    dataset = image.Dataset(
        (
            image.Sample.from_bytes(payload, "first.png"),
            image.Sample.from_bytes(payload, "second.png"),
        )
    )

    report = dataset.curate(max_samples=1, profile_size=(2, 2))

    assert report.modality == "image"
    assert report.profile["duplicate_count"] == 0
    assert report.profile["truncated"] is True
    assert [finding.finding_id for finding in report.quality_findings] == [
        "quality.image-profile-truncated",
    ]
    assert report.profile["label_counts"] == {"unlabeled": 2}


@PILLOW_SKIP
def test_near_duplicate_and_augmentation_provenance_are_explicit():
    """Curation records visual similarity and declared transformation lineage."""
    original = image.Sample.from_bytes(patterned_png(), "original.png")
    augmented = image.Sample.from_bytes(
        patterned_png(1),
        "augmented.png",
        augmentation=image.Augmentation(
            parent_source="original.png",
            operation="color_jitter",
            parameters={"brightness": 0.1},
            tool="fixture-transform/1",
        ),
    )
    dataset = image.Dataset((original, augmented))

    report = dataset.curate()

    assert report.dataset_hash == dataset.content_hash
    assert report.profile["near_duplicate_groups"] == [
        sorted([original.source, augmented.source])
    ]
    assert report.profile["augmentation"] == {
        "status": "declared",
        "augmented_sample_count": 1,
        "operations": ["color_jitter"],
        "external_parent_sources": [],
    }
    assert "quality.image-near-duplicate-content" in {
        finding.finding_id for finding in report.quality_findings
    }


@PILLOW_SKIP
def test_augmentation_provenance_changes_dataset_identity():
    """Transformation lineage is part of the image dataset identity."""
    payload = patterned_png()
    plain = image.Dataset((image.Sample.from_bytes(payload, "source.png"),))
    transformed = image.Dataset(
        (
            image.Sample.from_bytes(
                payload,
                "source.png",
                augmentation=image.Augmentation("raw.png", "resize"),
            ),
        )
    )

    assert plain.content_hash != transformed.content_hash


@PILLOW_SKIP
def test_curation_reports_mixed_label_state():
    """Image curation makes an unlabeled training boundary visible."""
    report = image.Dataset(
        (
            image.Sample.from_bytes(
                png_bytes((0, 0, 0)), "labelled.png", label="dark"
            ),
            image.Sample.from_bytes(png_bytes((255, 255, 255)), "raw.png"),
        )
    ).curate()

    assert report.profile["label_counts"] == {
        "str:dark": 1,
        "unlabeled": 1,
    }
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.image-mixed-label-state",
    ]


@PILLOW_SKIP
def test_curation_blocks_conflicting_duplicate_labels():
    """Duplicate pixels cannot silently carry incompatible labels."""
    payload = png_bytes((10, 20, 30))
    report = image.Dataset(
        (
            image.Sample.from_bytes(payload, "first.png", label="object"),
            image.Sample.from_bytes(payload, "second.png", label="background"),
        )
    ).curate()

    assert report.profile["conflicting_label_groups"]
    assert report.release_ready is False
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.image-conflicting-duplicate-labels",
    ]


@PILLOW_SKIP
def test_curation_blocks_sensitive_exif_metadata():
    """GPS and author metadata require an explicit privacy disposition."""
    canvas = Image.new("RGB", (4, 3), (10, 20, 30))
    exif = Image.Exif()
    exif[34853] = {}
    exif[315] = "annotator"
    output = io.BytesIO()
    canvas.save(output, format="JPEG", exif=exif)

    sample = image.Sample.from_bytes(output.getvalue(), "private.jpg")
    report = image.Dataset((sample,)).curate()

    assert report.profile["sensitive_metadata"] == ["artist", "gps_info"]
    assert report.release_ready is False
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.image-sensitive-metadata",
    ]


@PILLOW_SKIP
def test_split_keeps_explicit_capture_groups_together():
    """Grouped splitting prevents capture-session leakage."""
    samples = tuple(
        image.Sample.from_bytes(
            png_bytes((index * 20, 0, 0)),
            source=f"image-{index}.png",
            label="red",
            group_id=f"session-{index // 2}",
        )
        for index in range(6)
    )
    dataset = image.Dataset(samples)

    first = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)
    second = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)

    assignments = first.group_ids()
    groups = [set(values) for values in assignments.values()]
    assert sum("session-0" in values for values in groups) == 1
    assert set().union(*groups) == {
        "session-0",
        "session-1",
        "session-2",
    }
    assert first == second


@PILLOW_SKIP
def test_features_are_flattened_with_stable_shape_and_normalization():
    dataset = image.Dataset(
        (image.Sample.from_bytes(png_bytes((255, 0, 128)), "sample.png"),)
    )

    features = dataset.prepare_tabular_features((2, 2), "RGB")
    raw_features = dataset.prepare_tabular_features(
        (2, 2), "RGB", normalize=False
    )

    assert features.shape == (1, 12)
    assert features.dtype == numpy.float32
    assert raw_features.dtype == numpy.uint8
    numpy.testing.assert_allclose(features[0, :3], [1.0, 0.0, 128 / 255])


@PILLOW_SKIP
def test_empty_dataset_has_typed_feature_matrix():
    features = image.Dataset().prepare_tabular_features((3, 2), "L")

    assert features.shape == (0, 6)
    assert features.dtype == numpy.float32
