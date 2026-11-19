"""Exact findings and profile keys of an audio survey."""

import io
import wave

import numpy

from scmle.data.modalities.audio import dataset
from scmle.data.modalities.audio import sample
from scmle.data.modalities.audio import survey
from scmle.data.modalities.audio.metadata import Transcript


def clip(source, rate=8, label=None, transcript=None):
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(rate)
        writer.writeframes(numpy.zeros(4, dtype="<i2").tobytes())
    return sample.Sample.from_bytes(
        payload.getvalue(), source=source, label=label, transcript=transcript
    )


def text():
    return Transcript(
        text="hi",
        source="s",
        provider="p",
        provider_version="1",
        language="en",
        segments=(),
        confidence=None,
    )


def test_confidence_remediation_is_exact():
    first = clip("mem://a", transcript=text())

    (finding,) = survey.confidence([first], [text()])

    assert finding.remediation == (
        "Obtain confidence from the transcript provider or record why "
        "confidence is unavailable before using transcript-dependent claims."
    )


def test_drift_remediations_are_exact():
    profiles = [clip("mem://a").profile()]

    formats, shapes = survey.drift(profiles, ["a", "b"], [8, 16], [1])

    assert formats.remediation == (
        "Select an explicit codec normalization policy before training."
    )
    assert shapes.remediation == (
        "Resample or channel-normalize with the transformation recorded."
    )


def test_conflicting_labels_carry_their_groups_as_evidence():
    groups = [{"hash": "h", "labels": ["x", "y"]}]

    (finding,) = survey.risks("h", {"str:x": 1}, groups)

    assert finding.evidence == {"groups": groups}


def test_curate_reports_confidence_and_shape_drift_with_sources():
    samples = (
        clip("mem://a", rate=8, label="x", transcript=text()),
        clip("mem://b", rate=16, label="x"),
    )
    data = dataset.Dataset(samples)

    report = data.curate()

    confidence, shape = report.quality_findings
    assert (
        confidence.identifier == "quality.audio-transcript-confidence-missing"
    )
    assert confidence.scope == {"sources": ["mem://a"]}
    assert shape.identifier == "quality.audio-shape-drift"
    assert shape.scope == {"sources": ["mem://a", "mem://b"]}
    assert report.profile["total_bytes"] == sum(
        item.profile().byte_size for item in samples
    )
    assert report.profile["label_counts"] == {"str:x": 2}
    assert report.profile["conflicting_label_groups"] == []
