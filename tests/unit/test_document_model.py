"""Tests for the bounded document modelling lifecycle."""

import json
import math

import pytest

from scmle.data.modalities.document import Dataset
from scmle.data.modalities.document import Reader
from scmle.data.modalities.document import Sample
from scmle.models import documents
from scmle.models.documents import Classifier
from scmle.models.documents import Config
from scmle.models.documents import Monitor


def sample(document_id: str, content: str, label: str) -> Sample:
    """Build a labelled document fixture."""
    return Sample(document_id, content, label=label)


@pytest.fixture
def training_samples() -> tuple[Sample, ...]:
    """Return a deterministic binary text-training fixture."""
    return (
        sample("train-1", "red apple orchard fruit", "fruit"),
        sample("train-2", "red orange orchard harvest", "fruit"),
        sample("train-3", "blue ocean water sailing", "water"),
        sample("train-4", "blue lake water boat", "water"),
    )


@pytest.fixture
def evaluation_samples() -> tuple[Sample, ...]:
    """Return held-out documents with both labels represented."""
    return (
        sample("test-1", "red fruit harvest", "fruit"),
        sample("test-2", "orange orchard", "fruit"),
        sample("test-3", "blue water sailing", "water"),
        sample("test-4", "lake boat", "water"),
    )


def make_classifier() -> Classifier:
    """Return a bounded configuration suitable for a unit fixture."""
    return Classifier(Config(max_features=32, ngram_range=(1, 1)))


def test_document_classifier_fits_evaluates_and_explains(
    training_samples,
    evaluation_samples,
):
    """The baseline exposes modelling, evaluation, and explanation evidence."""
    classifier = make_classifier().fit(training_samples)

    evaluation = classifier.evaluate(evaluation_samples)
    global_explanation = classifier.global_explanation(top_k=5)
    local_explanation = classifier.local_explanation(evaluation_samples[0])
    ablation = classifier.term_ablation_explanation(
        evaluation_samples[0], top_k=3, max_terms=5
    )

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
    assert ablation.method == "term_ablation"
    assert ablation.input_hash == evaluation_samples[0].content_hash
    assert len(ablation.terms) == 3
    assert all(math.isfinite(value) and term for term, value in ablation.terms)
    assert classifier.model_digest == classifier.model_digest


def test_document_explanations_preserve_source_citations():
    """Local explanations carry source identity supplied by the reader."""
    reader = Reader()
    training = (
        reader.load(b"alpha document", document_id="alpha", label="yes"),
        reader.load(b"beta document", document_id="beta", label="no"),
    )
    classifier = make_classifier().fit(training)

    explanation = classifier.local_explanation(training[0])
    payload = explanation.to_dict()

    assert explanation.citations
    assert payload["citations"][0]["source_digest"] == (
        training[0].source_digest
    )


def test_document_term_ablation_rejects_invalid_configuration(
    training_samples,
):
    """Term ablation validates both result and work bounds."""
    classifier = make_classifier().fit(training_samples)
    sample_value = training_samples[0]

    with pytest.raises(ValueError, match="top_k"):
        classifier.term_ablation_explanation(sample_value, top_k=0)
    with pytest.raises(ValueError, match="max_terms"):
        classifier.term_ablation_explanation(sample_value, max_terms=0)


def test_document_classifier_rejects_unlabelled_or_single_class_training():
    """Training fails before feature preparation when labels are unsafe."""
    with pytest.raises(ValueError, match="every training document"):
        make_classifier().fit((Sample("missing", "some text"),))

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
        Classifier.load(path)

    loaded = Classifier.load(path, trusted=True)
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
        Classifier.load(path, trusted=True)


def test_document_classifier_rejects_tampered_manifest(
    tmp_path,
    training_samples,
):
    """Manifest changes cannot silently alter the trusted package identity."""
    classifier = make_classifier().fit(training_samples)
    path = tmp_path / "document-model"
    classifier.save(path)

    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model_digest"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="model digest"):
        Classifier.load(path, trusted=True)


def test_document_classifier_cleans_failed_atomic_replacement(
    monkeypatch,
    tmp_path,
    training_samples,
):
    """A failed package replacement leaves no misleading partial artifact."""
    classifier = make_classifier().fit(training_samples)
    target = tmp_path / "document-model"
    real_replace = documents.os.replace

    def fail_replace(source, destination):
        """Inject a failure at the atomic directory replacement boundary."""
        if destination == target:
            raise OSError("injected replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(documents.os, "replace", fail_replace)

    with pytest.raises(OSError, match="injected replacement failure"):
        classifier.save(target)

    assert not target.exists()
    assert tuple(tmp_path.glob(".document-model.*")) == ()


def test_document_monitor_detects_profile_shift(training_samples):
    """Monitoring compares observed corpus evidence with a fixed snapshot."""
    reference = Dataset(training_samples).curate()
    observed = Dataset(
        (
            Sample("observed-1", "नमस्ते दुनिया", label="fruit"),
            Sample("observed-2", "नमस्ते पानी", label="water"),
        )
    )

    report = Monitor(reference, threshold=0.1).observe(observed)

    assert report.reference_hash == reference.dataset_hash
    assert report.observed_hash == observed.content_hash
    assert report.language_distance > 0.1
    assert report.shifted
