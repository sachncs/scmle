"""Tests for the bounded image modelling lifecycle."""

import io
import json
import math

import pytest

from scmle.data.modalities import image
from scmle.models.images import ImageClassifier
from scmle.models.images import ImageModelConfig
from scmle.models.images import ImageMonitor

try:
    from PIL import Image
except ImportError:
    Image = None


PILLOW_SKIP = pytest.mark.skipif(
    Image is None,
    reason="Pillow-backed image model test requires the optional dependency",
)


def png_bytes(color, size=(4, 4)):
    """Create a deterministic RGB PNG fixture."""
    canvas = Image.new("RGB", size, color)
    output = io.BytesIO()
    canvas.save(output, format="PNG")
    return output.getvalue()


def sample(source: str, color, label: str) -> image.ImageSample:
    """Create a labelled image fixture."""
    return image.ImageSample.from_bytes(png_bytes(color), source, label=label)


@pytest.fixture
def training_data() -> image.ImageDataset:
    """Return a deterministic binary image-training fixture."""
    return image.ImageDataset(
        (
            sample("red-1.png", (255, 0, 0), "red"),
            sample("red-2.png", (220, 10, 10), "red"),
            sample("blue-1.png", (0, 0, 255), "blue"),
            sample("blue-2.png", (10, 10, 220), "blue"),
        )
    )


@pytest.fixture
def evaluation_data() -> image.ImageDataset:
    """Return held-out images with both labels represented."""
    return image.ImageDataset(
        (
            sample("red-test.png", (250, 20, 20), "red"),
            sample("blue-test.png", (20, 20, 250), "blue"),
        )
    )


def make_classifier() -> ImageClassifier:
    """Return a small deterministic image baseline."""
    return ImageClassifier(ImageModelConfig(target_size=(2, 2), max_iter=200))


@PILLOW_SKIP
def test_image_classifier_fits_evaluates_and_explains(
    training_data,
    evaluation_data,
):
    """The baseline exposes image modelling and pixel evidence."""
    classifier = make_classifier().fit(training_data)

    evaluation = classifier.evaluate(evaluation_data)
    global_explanation = classifier.global_explanation(top_k=5)
    local_explanation = classifier.local_explanation(evaluation_data.samples[0])

    assert classifier.fitted
    assert set(classifier.predict(evaluation_data)) == {"red", "blue"}
    assert len(classifier.predict_proba(evaluation_data)) == 2
    assert evaluation.accuracy == 1.0
    assert evaluation.macro_f1 == 1.0
    assert math.isfinite(evaluation.log_loss)
    assert global_explanation.input_hash is None
    assert global_explanation.terms
    assert (
        local_explanation.input_hash == evaluation_data.samples[0].content_hash
    )
    assert local_explanation.terms

    occlusion = classifier.occlusion_explanation(
        evaluation_data.samples[0], top_k=3, patch_size=(1, 1)
    )
    assert occlusion.method == "occlusion_pixel_region"
    assert occlusion.input_hash == evaluation_data.samples[0].content_hash
    assert len(occlusion.terms) == 3
    assert all("region[" in term[0] for term in occlusion.terms)
    assert all(math.isfinite(value) and term for term, value in occlusion.terms)


@PILLOW_SKIP
def test_image_occlusion_rejects_invalid_configuration(training_data):
    """Occlusion validates its output and transformed patch boundaries."""
    classifier = make_classifier().fit(training_data)
    sample_value = training_data.samples[0]

    with pytest.raises(ValueError, match="top_k"):
        classifier.occlusion_explanation(sample_value, top_k=0)
    with pytest.raises(ValueError, match="patch_size"):
        classifier.occlusion_explanation(sample_value, patch_size=(0, 1))


@PILLOW_SKIP
def test_image_classifier_rejects_unsafe_training_data(training_data):
    """Training refuses missing labels, duplicates, and one-class data."""
    with pytest.raises(ValueError, match="every training image"):
        make_classifier().fit(
            image.ImageDataset(
                (image.ImageSample.from_bytes(png_bytes((1, 2, 3)), "raw"),)
            )
        )

    with pytest.raises(ValueError, match="duplicate content"):
        make_classifier().fit(
            image.ImageDataset(
                (
                    sample("one.png", (100, 0, 0), "red"),
                    sample("two.png", (100, 0, 0), "blue"),
                )
            )
        )

    with pytest.raises(ValueError, match="two classes"):
        make_classifier().fit(
            image.ImageDataset(
                (
                    sample("one.png", (100, 0, 0), "red"),
                    sample("two.png", (120, 0, 0), "red"),
                )
            )
        )


@PILLOW_SKIP
def test_image_classifier_package_round_trip_and_tamper_rejection(
    tmp_path,
    training_data,
    evaluation_data,
):
    """Image packages require trust and verify their manifest identities."""
    classifier = make_classifier().fit(training_data)
    path = tmp_path / "image-model"
    artifact = classifier.save(path)

    with pytest.raises(PermissionError, match="trusted=True"):
        ImageClassifier.load(path)

    loaded = ImageClassifier.load(path, trusted=True)
    assert loaded.model_digest == classifier.model_digest
    assert loaded.predict(evaluation_data) == classifier.predict(
        evaluation_data
    )

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="model digest"):
        ImageClassifier.load(artifact.path, trusted=True)


@PILLOW_SKIP
def test_image_monitor_detects_mean_pixel_shift(training_data):
    """Monitoring compares image profile evidence against a fixed snapshot."""
    reference = training_data.curate(profile_size=(2, 2))
    observed = image.ImageDataset(
        (sample("observed.png", (0, 255, 0), "green"),)
    )

    report = ImageMonitor(reference, threshold=0.1).observe(observed)

    assert report.reference_hash == reference.dataset_hash
    assert report.observed_hash == observed.curate().dataset_hash
    assert report.mean_pixel_distance > 0.1
    assert report.shifted
