"""Tests for the optional-Pillow image data foundation."""

import io

import numpy as np
import pytest

from scml.data.modalities import image

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
    first = image.ImageSample.from_bytes(payload, "first.png")
    second = image.ImageSample.from_bytes(payload, "second.png")

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
            image.ImageSample.from_bytes(payload)
        assert raised.value.code == code
        assert raised.value.details["source"] == "memory://"


@PILLOW_SKIP
def test_oversized_payload_and_pixel_limits_are_rejected():
    payload = png_bytes((1, 2, 3), size=(10, 10))
    with pytest.raises(image.ImageValidationError, match="bytes") as by_bytes:
        image.ImageSample.from_bytes(payload, max_bytes=len(payload) - 1)
    assert by_bytes.value.code == "encoded_size_limit"

    with pytest.raises(image.ImageValidationError, match="pixels") as by_pixels:
        image.ImageSample.from_bytes(payload, max_pixels=99)
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

    left = image.ImageDataset.from_paths(
        [first_path, second_path, unique_path]
    ).deduplicate()
    right = image.ImageDataset.from_paths(
        [unique_path, second_path, first_path]
    ).deduplicate()

    assert left == right
    assert left.sample_count == 2
    assert any(sample.source.endswith("a.png") for sample in left.samples)


@PILLOW_SKIP
def test_profile_is_bounded_and_detects_shift():
    dataset = image.ImageDataset(
        (
            image.ImageSample.from_bytes(png_bytes((0, 0, 0)), "dark.png"),
            image.ImageSample.from_bytes(
                png_bytes((255, 255, 255)), "light.png"
            ),
        )
    )
    profile = dataset.profile(max_samples=1, profile_size=(2, 2))

    assert profile.dataset_count == 2
    assert profile.profiled_count == 1
    assert profile.truncated is True
    assert profile.mean_pixel == (0.0, 0.0, 0.0)
    shifted = image.ImageDataset(
        (
            image.ImageSample.from_bytes(
                png_bytes((255, 255, 255)), "shifted.png"
            ),
        )
    ).profile(profile_size=(2, 2))
    assert all(
        before < after
        for before, after in zip(
            profile.mean_pixel, shifted.mean_pixel, strict=True
        )
    )


@PILLOW_SKIP
def test_features_are_flattened_with_stable_shape_and_normalization():
    dataset = image.ImageDataset(
        (image.ImageSample.from_bytes(png_bytes((255, 0, 128)), "sample.png"),)
    )

    features = dataset.prepare_tabular_features((2, 2), "RGB")
    raw_features = dataset.prepare_tabular_features(
        (2, 2), "RGB", normalize=False
    )

    assert features.shape == (1, 12)
    assert features.dtype == np.float32
    assert raw_features.dtype == np.uint8
    np.testing.assert_allclose(features[0, :3], [1.0, 0.0, 128 / 255])


@PILLOW_SKIP
def test_empty_dataset_has_typed_feature_matrix():
    features = image.ImageDataset().prepare_tabular_features((3, 2), "L")

    assert features.shape == (0, 6)
    assert features.dtype == np.float32
