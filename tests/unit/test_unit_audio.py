"""Behavioural tests for the audio modality: WAV, samples, dataset, survey."""

import hashlib
import io
import pathlib
import wave

import numpy
import pytest

from scmle.contracts import errors
from scmle.data.modalities.audio import dataset
from scmle.data.modalities.audio import evidence
from scmle.data.modalities.audio import sample
from scmle.data.modalities.audio import survey
from scmle.data.modalities.audio import wav
from scmle.data.modalities.audio.metadata import DecoderError
from scmle.data.modalities.audio.metadata import Metadata
from scmle.data.modalities.audio.metadata import Segment
from scmle.data.modalities.audio.metadata import Transcript
from scmle.data.modalities.audio.metadata import ValidationError


def encode(frames, width=2, channels=1, rate=8):
    """A WAV payload holding the given integer frames."""
    payload = io.BytesIO()
    with wave.open(payload, "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(width)
        writer.setframerate(rate)
        writer.writeframes(frames)
    return payload.getvalue()


def tone(values, rate=8):
    raw = numpy.asarray(values, dtype="<i2").tobytes()
    return encode(raw, rate=rate)


def clip(values, source, label=None, group=None, transcript=None):
    return sample.Sample.decode(
        tone(values),
        source=source,
        label=label,
        group_id=group,
        transcript=transcript,
    )


def text(confidence=None, segments=()):
    return Transcript(
        text="hi",
        source="s",
        provider="p",
        version="1",
        language="en",
        segments=segments,
        confidence=confidence,
    )


def test_parse_reads_wav_metadata():
    metadata = wav.parse(tone([0, 1, 2, 3], rate=4))
    assert metadata == Metadata("wav", 4, 1, 4, 2)
    with pytest.raises(ValidationError, match="not a readable WAV"):
        wav.parse(b"nonsense")


def test_scaled_decodes_every_supported_width():
    samples, scale = wav.scaled(bytes([0, 128, 255]), 1)
    assert samples.tolist() == [-1.0, 0.0, 127 / 128]
    assert scale == 1.0
    samples, scale = wav.scaled(
        numpy.array([16384, -32768], "<i2").tobytes(), 2
    )
    assert samples.tolist() == [0.5, -1.0]
    assert scale == 32768.0
    samples, scale = wav.scaled(numpy.array([2**30], "<i4").tobytes(), 4)
    assert samples.tolist() == [0.5]
    assert scale == 2.0**31
    samples, scale = wav.scaled(bytes([0, 0, 0x40, 0, 0, 0x80]), 3)
    assert samples.tolist() == [0.5, -1.0]
    assert scale == 2.0**23
    with pytest.raises(ValidationError, match="sample width is unsupported"):
        wav.scaled(b"\x00" * 5, 5)


def test_statistics_measure_a_window_exactly():
    samples = numpy.array([1.0, -1.0, 0.0, 0.5])
    values = wav.statistics(samples, 2, 4, 2, 32768.0)
    assert values == (
        0.125,
        0.625,
        pytest.approx(numpy.sqrt(2.25 / 4)),
        1.0,
        0.5,
        0.25,
        0.5,
        2.0,
        32768.0,
    )
    with pytest.raises(ValidationError, match="must be finite"):
        wav.statistics(numpy.array([numpy.nan]), 1, 1, 1, 1.0)


def test_statistics_thresholds_are_inclusive():
    edge = numpy.array([wav.CLIPPING_LEVEL, wav.SILENCE_LEVEL])
    values = wav.statistics(edge, 1, 1, 2, 1.0)
    assert values[4] == 0.5
    assert values[5] == 0.5


def test_extract_reads_a_bounded_window():
    payload = tone([0, 16384, -16384, 32767], rate=4)
    values, observed = wav.extract(payload, 2, start_frame=1)
    assert observed == 2
    assert values[0] == 0.0
    assert values[3] == 0.5
    assert values[6] == 0.5
    assert values[7] == 2.0
    observed = wav.extract(payload, 100)[1]
    assert observed == 4
    observed = wav.extract(payload, 1, start_frame=3)[1]
    assert observed == 1


def test_extract_rejects_bad_windows():
    payload = tone([0, 1])
    with pytest.raises(errors.InputError, match="max_frames"):
        wav.extract(payload, 0)
    with pytest.raises(errors.InputError, match="start_frame"):
        wav.extract(payload, 1, start_frame=-1)
    with pytest.raises(ValidationError, match="beyond the available"):
        wav.extract(payload, 1, start_frame=2)
    with pytest.raises(ValidationError, match="not a readable"):
        wav.extract(b"junk", 1)


def test_pcm_round_trips_through_the_encoder():
    for width in (1, 2, 3, 4):
        parameters = (1, width, 8, 3, "NONE", "not compressed")
        values = numpy.array([[0.5], [-0.5], [0.0]])
        payload = wav.encode(parameters, values)
        decoded_parameters, decoded = wav.pcm(payload)
        assert decoded_parameters[:3] == (1, width, 8)
        assert decoded_parameters[3] == 3
        assert decoded == pytest.approx(values, abs=1 / 127)


def test_the_encoder_clips_and_rounds_to_the_width():
    parameters = (2, 2, 8, 1, "NONE", "not compressed")
    payload = wav.encode(parameters, numpy.array([[2.0, -2.0]]))
    decoded = wav.pcm(payload)[1]
    assert decoded.tolist() == [[32767 / 32768, -32767 / 32768]]
    one = wav.encode(
        (1, 1, 8, 1, "NONE", "not compressed"), numpy.array([[1.0]])
    )
    assert wav.pcm(one)[1].tolist() == [[127 / 128]]


def test_the_encoder_rejects_bad_matrices_and_widths():
    parameters = (1, 2, 8, 1, "NONE", "not compressed")
    with pytest.raises(errors.InputError, match="frame-by-channel"):
        wav.encode(parameters, numpy.array([0.0]))
    with pytest.raises(errors.InputError, match="frame-by-channel"):
        wav.encode(parameters, numpy.array([[numpy.inf]]))
    with pytest.raises(ValidationError, match="unsupported"):
        wav.encode((1, 5, 8, 1, "NONE", "x"), numpy.array([[0.0]]))
    with pytest.raises(ValidationError, match="not a readable"):
        wav.pcm(b"junk")


def test_canonical_source_normalises_paths_and_uris():
    assert sample.identify("  ") == "memory://"
    assert sample.identify(" s3://b/k ") == "s3://b/k"
    assert sample.identify("a.wav") == (
        pathlib.Path("a.wav").resolve().as_uri()
    )


def test_from_bytes_builds_a_hashed_sample():
    payload = tone([0, 1, 2, 3])
    built = sample.Sample.decode(payload, source="mem://a", label=1)
    assert built.file_hash == hashlib.sha256(payload).hexdigest()
    assert built.size == len(payload)
    assert built.metadata.frame_count == 4
    assert built.label == 1
    profile = built.profile()
    assert profile.duration_seconds == 0.5
    assert profile.byte_size == len(payload)
    assert not profile.transcript_present
    assert profile.transcript_segment_count == 0


def test_from_bytes_enforces_its_limits():
    payload = tone([0, 1])
    with pytest.raises(ValidationError, match="empty"):
        sample.Sample.decode(b"")
    with pytest.raises(ValidationError, match="byte limit"):
        sample.Sample.decode(payload, max_bytes=len(payload) - 1)
    with pytest.raises(ValidationError, match="byte limit"):
        sample.Sample.decode(payload, max_bytes=0)
    assert sample.Sample.decode(payload, max_bytes=len(payload))
    assert sample.Sample.decode(payload, max_bytes=None)


def test_from_bytes_needs_a_probe_for_other_containers():
    with pytest.raises(DecoderError, match="explicit codec probe"):
        sample.Sample.decode(b"OggS....")

    def broken(raw):
        raise RuntimeError(raw)

    with pytest.raises(DecoderError, match="could not decode"):
        sample.Sample.decode(b"OggS", probe=broken)

    def refusing(raw):
        raise ValidationError("bad", {"code": "x"})

    with pytest.raises(ValidationError, match="bad"):
        sample.Sample.decode(b"OggS", probe=refusing)
    built = sample.Sample.decode(
        b"OggS", probe=lambda raw: Metadata("ogg", 10, 1, 5)
    )
    assert built.metadata.format == "ogg"


def test_a_sample_validates_its_identity():
    payload = tone([0])
    digest = hashlib.sha256(payload).hexdigest()
    metadata = wav.parse(payload)
    with pytest.raises(errors.InputError, match="source"):
        sample.Sample(" ", payload, digest, metadata)
    with pytest.raises(errors.InputError, match="group_id"):
        sample.Sample("s", payload, digest, metadata, group=" ")
    with pytest.raises(errors.InputError, match="SHA-256 hex digest"):
        sample.Sample("s", payload, digest.upper(), metadata)
    with pytest.raises(errors.InputError, match="SHA-256 hex digest"):
        sample.Sample("s", payload, digest[:-1], metadata)
    with pytest.raises(errors.InputError, match="does not match"):
        sample.Sample("s", payload, "0" * 64, metadata)
    with pytest.raises(ValidationError, match="empty"):
        sample.Sample("s", b"", digest, metadata)


def test_a_transcript_may_end_exactly_at_the_audio_end():
    within = text(segments=(Segment(0.0, 0.5, "a"),))
    built = clip([0, 1, 2, 3], "mem://a", transcript=within)
    assert built.profile().transcript_segment_count == 1
    assert built.profile().transcript_present
    beyond = text(segments=(Segment(0.0, 0.6, "a"),))
    with pytest.raises(ValidationError, match="exceeds audio duration"):
        clip([0, 1, 2, 3], "mem://a", transcript=beyond)


def test_features_report_their_shape_and_a_writable_copy():
    features = evidence.Features(
        ("a",), ("x", "y"), ((1.0, 2.0),), (1,), (False,)
    )
    assert features.shape == (1, 2)
    matrix = features.materialize()
    matrix[0, 0] = 9.0
    assert features.values == ((1.0, 2.0),)


def test_a_split_lists_its_cohorts():
    first = clip([0], "mem://a", group="g")
    second = clip([1], "mem://b")
    split = evidence.Split(train=(first, second), validation=(), test=())
    assert split.cohorts() == {
        "train": ("g", "mem://b"),
        "validation": (),
        "test": (),
    }


def test_a_dataset_rejects_empty_and_duplicate_sources():
    with pytest.raises(ValidationError, match="at least one sample"):
        dataset.Dataset(())
    one = clip([0], "mem://a")
    with pytest.raises(ValidationError, match="duplicate sources"):
        dataset.Dataset((one, one))


def test_a_dataset_hash_depends_on_order_and_content():
    first = clip([0], "mem://a")
    second = clip([1], "mem://b")
    forward = dataset.Dataset((first, second))
    expected = hashlib.sha256(
        f"mem://a\0{first.file_hash}\nmem://b\0{second.file_hash}".encode()
    ).hexdigest()
    assert forward.digest == expected
    assert dataset.Dataset((second, first)).digest != expected


def test_prepare_features_appends_metadata_and_truncation():
    data = dataset.Dataset((clip([0, 16384, 0, 0], "mem://a"),))
    features = data.prepare(max_frames=2)
    assert features.shape == (1, 13)
    assert features.names[0] == "mean_amplitude"
    assert features.names[-1] == "sample_width"
    row = features.values[0]
    assert row[7] == 2.0
    assert row[9:] == (0.5, 8.0, 1.0, 2.0)
    assert features.observed == (2,)
    assert features.truncated == (True,)
    full = data.prepare()
    assert full.truncated == (False,)
    with pytest.raises(errors.InputError, match="max_frames"):
        data.prepare(max_frames=0)
    with pytest.raises(errors.InputError, match="start_frame"):
        data.prepare(start_frame=-1)


def test_prepare_features_needs_wav():
    other = sample.Sample.decode(
        b"OggS", source="mem://o", probe=lambda raw: Metadata("ogg", 1, 1, 1)
    )
    with pytest.raises(DecoderError, match="mem://o"):
        dataset.Dataset((other,)).prepare()


def test_split_keeps_groups_whole_and_is_seeded():
    clips = tuple(clip([i], f"mem://{i}", group=f"g{i % 5}") for i in range(10))
    data = dataset.Dataset(clips)
    split = data.split(0.2, 0.2, seed=3)
    cohorts = split.cohorts()
    assert len(cohorts["validation"]) == 1
    assert len(cohorts["test"]) == 1
    assert len(cohorts["train"]) == 3
    assert not set(cohorts["train"]) & set(cohorts["test"])
    assert len(split.train) + len(split.validation) + len(split.test) == 10
    assert data.split(0.2, 0.2, seed=3) == split
    with pytest.raises(errors.InputError, match="validation_fraction"):
        data.split(1.0, 0.0)


def test_curate_profiles_a_clean_dataset():
    data = dataset.Dataset(
        (clip([0, 0], "mem://a", label="x"), clip([0, 0, 0, 0], "mem://b"))
    )
    report = data.curate()
    assert report.modality == "audio"
    assert report.samples == 2
    assert report.dataset_hash == data.digest
    assert report.profile["formats"] == ["wav"]
    assert report.profile["duration_min_seconds"] == 0.25
    assert report.profile["duration_max_seconds"] == 0.5
    assert report.profile["duration_mean_seconds"] == 0.375
    assert report.profile["transcript_coverage"] == 0.0
    assert report.profile["transcript_confidence_min"] is None
    assert report.findings == ()
    assert [f.identifier for f in report.risk_findings] == [
        "risk.audio-mixed-label-state"
    ]
    assert not report.risk_findings[0].blocking


def test_curate_blocks_conflicting_labels_on_identical_audio():
    data = dataset.Dataset(
        (clip([0], "mem://a", label="x"), clip([0], "mem://b", label="y"))
    )
    findings = data.curate().risk_findings
    assert [f.identifier for f in findings] == [
        "risk.audio-conflicting-duplicate-labels"
    ]
    assert findings[0].blocking


def test_confidence_warns_only_without_any_confidence():
    first = clip([0], "mem://a", transcript=text())
    assert survey.confidence([first], [text(confidence=0.5)]) == []
    scored = text(segments=(Segment(0.0, 0.1, "a", 0.4),))
    assert survey.confidence([first], [scored]) == []
    findings = survey.confidence([first], [text(), text()])
    assert (
        findings[0].identifier == "quality.audio-transcript-confidence-missing"
    )
    assert findings[0].evidence == {"sample_count": 2}
    assert findings[0].scope == {"sources": ["mem://a"]}


def test_drift_flags_formats_rates_and_channels():
    profiles = [clip([0], "mem://a").profile()]
    assert survey.drift(profiles, ["wav"], [8], [1]) == []
    names = [
        f.identifier for f in survey.drift(profiles, ["a", "b"], [8], [1, 2])
    ]
    assert names == ["quality.audio-format-drift", "quality.audio-shape-drift"]
    rate = survey.drift(profiles, ["wav"], [8, 16], [1])
    assert rate[0].evidence == {"sample_rates": [8, 16], "channels": [1]}


def test_risks_need_both_labelled_and_unlabelled_clips():
    assert survey.risks("h", {"unlabeled": 2}, ()) == []
    assert survey.risks("h", {"str:a": 2}, ()) == []
    mixed = survey.risks("h", {"str:a": 1, "unlabeled": 1}, ())
    assert mixed[0].scope == {"dataset": "h"}


def test_transcription_summarises_coverage_and_confidence():
    first = text(confidence=0.9, segments=(Segment(0.0, 0.1, "a", 0.2),))
    summary = survey.transcription([first, text()], 4)
    assert summary == {
        "transcript_coverage": 0.5,
        "transcript_sources": ["s"],
        "transcript_providers": ["p"],
        "transcript_languages": ["en"],
        "transcript_confidence_min": 0.2,
        "transcript_confidence_max": 0.9,
        "transcript_segment_count": 1,
    }
