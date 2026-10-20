"""Tests for the bounded document modelling lifecycle."""

import math

import pytest

from scml.data.modalities.document import DocumentDataset
from scml.data.modalities.document import DocumentSample
from scml.models.documents import DocumentClassifier
from scml.models.documents import DocumentModelConfig
from scml.models.documents import DocumentMonitor


def sample(document_id: str, content: str, label: str) -> DocumentSample:
    """Build a labelled document fixture."""
    return DocumentSample(document_id, content, label=label)


@pytest.fixture
def training_samples() -> tuple[DocumentSample, ...]:
    """Return a deterministic binary text-training fixture."""
    return (
        sample("train-1", "red apple orchard fruit", "fruit"),
        sample("train-2", "red orange orchard harvest", "fruit"),
        sample("train-3", "blue ocean water sailing", "water"),
        sample("train-4", "blue lake water boat", "water"),
    )


@pytest.fixture
def evaluation_samples() -> tuple[DocumentSample, ...]:
    """Return held-out documents with both labels represented."""
    return (
        sample("test-1", "red fruit harvest", "fruit"),
        sample("test-2", "orange orchard", "fruit"),
        sample("test-3", "blue water sailing", "water"),
        sample("test-4", "lake boat", "water"),
    )


def make_classifier() -> DocumentClassifier:
    """Return a bounded configuration suitable for a unit fixture."""
    return DocumentClassifier(
        DocumentModelConfig(max_features=32, ngram_range=(1, 1))
    )


def test_document_classifier_fits_evaluates_and_explains(
    training_samples,
    evaluation_samples,
):
    """The baseline exposes modelling, evaluation, and explanation evidence."""
    classifier = make_classifier().fit(training_samples)

    evaluation = classifier.evaluate(evaluation_samples)
    global_explanation = classifier.global_explanation(top_k=5)
    local_explanation = classifier.local_explanation(evaluation_samples[0])

    assert classifier.fitted
    assert set(classifier.predict(evaluation_samples)) == {"fruit", "water"}
    assert len(classifier.predict_proba(evaluation_samples)) == 4
    assert evaluation.sample_count == 4
    assert evaluation.accuracy >= 0.75
    assert evaluation.macro_f1 >= 0.75
    assert math.isfinite(evaluation.log_loss)
    assert global_explanation.input_hash is None
    assert global_explanation.terms
    assert local_explanation.input_hash == evaluation_samples[0].content_hash
    assert local_explanation.terms
    assert classifier.model_digest == classifier.model_digest


def test_document_classifier_rejects_unlabelled_or_single_class_training():
    """Training fails before feature preparation when labels are unsafe."""
    with pytest.raises(ValueError, match="every training document"):
        make_classifier().fit((DocumentSample("missing", "some text"),))

    with pytest.raises(ValueError, match="two classes"):
        make_classifier().fit(
            (
                sample("one", "same class one", "only"),
                sample("two", "same class two", "only"),
            )
        )


def test_document_classifier_atomic_trusted_package_round_trip(
    tmp_path,
    training_samples,
    evaluation_samples,
):
    """Packages round-trip only after an explicit trust decision."""
    classifier = make_classifier().fit(training_samples)
    path = tmp_path / "document-model"
    artifact = classifier.save(path)

    assert artifact.path == str(path)
    assert len(artifact.digest) == 64
    with pytest.raises(PermissionError, match="trusted=True"):
        DocumentClassifier.load(path)

    loaded = DocumentClassifier.load(path, trusted=True)
    assert loaded.model_digest == classifier.model_digest
    assert loaded.predict(evaluation_samples) == classifier.predict(
        evaluation_samples
    )
    with pytest.raises(FileExistsError):
        classifier.save(path)

    (path / "model.joblib").write_bytes(b"corrupted package")
    with pytest.raises(
        (EOFError, KeyError, ModuleNotFoundError, TypeError, ValueError)
    ):
        DocumentClassifier.load(path, trusted=True)


def test_document_monitor_detects_profile_shift(training_samples):
    """Monitoring compares observed corpus evidence with a fixed snapshot."""
    reference = DocumentDataset(training_samples).curate()
    observed = DocumentDataset(
        (
            DocumentSample("observed-1", "नमस्ते दुनिया", label="fruit"),
            DocumentSample("observed-2", "नमस्ते पानी", label="water"),
        )
    )

    report = DocumentMonitor(reference, threshold=0.1).observe(observed)

    assert report.reference_hash == reference.dataset_hash
    assert report.observed_hash == observed.content_hash
    assert report.language_distance > 0.1
    assert report.shifted
