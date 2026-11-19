"""Tests for dependency-light audio, video, and geospatial foundations."""

from datetime import datetime
from datetime import UTC
import hashlib
import io
import wave

import numpy
import pytest

from scmle.data.modalities import audio
from scmle.data.modalities import geospatial
from scmle.data.modalities import video


def wav_bytes(frame_count: int = 8) -> bytes:
    """Build a minimal deterministic mono WAV fixture."""
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8)
        writer.writeframes(b"\x00\x00" * frame_count)
    return output.getvalue()


def failing_probe(payload: bytes) -> audio.Metadata:
    """Represent a caller-owned codec probe that fails deterministically."""
    raise RuntimeError(payload)


def flac_probe(payload: bytes) -> audio.Metadata:
    """Represent metadata returned by an explicit non-WAV codec adapter."""
    return audio.Metadata("flac", 16_000, 1, len(payload))


def test_audio_wav_ingestion_preserves_identity_and_profile():
    """WAV metadata and bytes are bound to the same immutable sample."""
    sample = audio.Sample.decode(wav_bytes(), source="clip.wav")

    assert sample.source.endswith("/clip.wav")
    assert sample.metadata.sample_rate == 8
    assert sample.metadata.channels == 1
    assert sample.profile().duration_seconds == 1.0
    assert sample.file_hash == hashlib.sha256(sample.encoded).hexdigest()


def test_audio_non_wav_requires_explicit_probe():
    """Unknown containers do not silently acquire a codec dependency."""
    with pytest.raises(audio.DecoderError):
        audio.Sample.decode(b"opaque-audio", source="clip.flac")

    with pytest.raises(audio.DecoderError):
        audio.Sample.decode(
            b"opaque-audio",
            source="clip.flac",
            probe=failing_probe,
        )


def test_audio_dataset_rejects_duplicate_sources():
    """Duplicate source identities are a curation error, not two samples."""
    first = audio.Sample.decode(wav_bytes(), source="one.wav")
    duplicate = audio.Sample.decode(wav_bytes(), source="one.wav")

    with pytest.raises(audio.ValidationError, match="duplicate sources"):
        audio.Dataset((first, duplicate))


def test_audio_curation_reports_metadata_and_label_composition():
    """Audio data understanding is explicit before a model is selected."""
    sample = audio.Sample.decode(wav_bytes(), source="one.wav")

    report = audio.Dataset((sample,)).curate()

    assert report.modality == "audio"
    assert report.samples == 1
    assert report.profile["sample_rates"] == [8]
    assert report.profile["label_counts"] == {"unlabeled": 1}
    assert report.release


def test_audio_curation_hash_matches_source_order_identity():
    """Audio curation remains attached when source order is non-sorted."""
    samples = (
        audio.Sample.decode(wav_bytes(), source="z-last.wav"),
        audio.Sample.decode(wav_bytes(), source="a-first.wav"),
    )
    dataset = audio.Dataset(samples)

    assert dataset.curate().dataset_hash == dataset.digest


def test_audio_transcript_provenance_is_bounded_and_curated():
    """Transcript metadata remains tied to the source audio and provider."""
    transcript = audio.Transcript(
        text="hello world",
        source="transcriber://run-7",
        provider="acme-asr",
        version="2026.10",
        language="en",
        segments=(
            audio.Segment(0.0, 0.5, "hello", 0.9),
            audio.Segment(0.5, 1.0, "world", 0.8),
        ),
        confidence=0.85,
    )
    sample = audio.Sample.decode(
        wav_bytes(), source="spoken.wav", transcript=transcript
    )

    report = audio.Dataset((sample,)).curate()

    assert sample.profile().transcript_present
    assert sample.profile().transcript_segment_count == 2
    assert transcript.to_dict()["version"] == "2026.10"
    assert report.profile["transcript_coverage"] == 1.0
    assert report.profile["transcript_sources"] == ["transcriber://run-7"]
    assert report.profile["transcript_segment_count"] == 2
    assert report.profile["transcript_confidence_min"] == 0.8
    assert report.findings == ()


def test_audio_transcript_confidence_and_timing_fail_closed():
    """Missing confidence is visible and segments cannot exceed duration."""
    uncertain = audio.Transcript(
        text="hello",
        source="transcriber://run-8",
        provider="acme-asr",
        version="2026.10",
        language="en",
        segments=(audio.Segment(0.0, 0.5, "hello"),),
    )
    report = audio.Dataset(
        (
            audio.Sample.decode(
                wav_bytes(), source="uncertain.wav", transcript=uncertain
            ),
        )
    ).curate()

    assert [finding.identifier for finding in report.findings] == [
        "quality.audio-transcript-confidence-missing",
    ]
    out_of_range = audio.Transcript(
        text="too long",
        source="transcriber://run-9",
        provider="acme-asr",
        version="2026.10",
        language="en",
        segments=(audio.Segment(0.0, 2.0, "too long"),),
    )
    with pytest.raises(audio.ValidationError, match="exceeds audio"):
        audio.Sample.decode(
            wav_bytes(), source="bad.wav", transcript=out_of_range
        )


def test_audio_curation_blocks_conflicting_duplicate_labels():
    """Duplicate audio bytes cannot silently carry incompatible labels.

    Also enforces website/src/pages/docs/api.astro:157.
    """
    dataset = audio.Dataset(
        (
            audio.Sample.decode(
                wav_bytes(), source="first.wav", label="speech"
            ),
            audio.Sample.decode(
                wav_bytes(), source="second.wav", label="music"
            ),
        )
    )

    report = dataset.curate()

    assert report.profile["conflicting_label_groups"]
    assert report.release is False
    assert [finding.identifier for finding in report.risk_findings] == [
        "risk.audio-conflicting-duplicate-labels",
    ]


def test_audio_feature_preparation_is_bounded_and_explainable():
    """Signal features expose the observed prefix and deterministic names."""
    sample = audio.Sample.decode(wav_bytes(frame_count=8), source="one.wav")

    prepared = audio.Dataset((sample,)).prepare(max_frames=3)

    assert prepared.shape == (1, 13)
    assert prepared.observed == (3,)
    assert prepared.truncated == (True,)
    assert prepared.names[:4] == (
        "mean_amplitude",
        "mean_absolute_amplitude",
        "rms_amplitude",
        "peak_amplitude",
    )
    assert prepared.materialize()[0, 2] == 0.0


def test_audio_feature_preparation_supports_a_bounded_window():
    """A decoded WAV window remains tied to its source frame boundary."""
    sample = audio.Sample.decode(wav_bytes(frame_count=8), source="one.wav")

    prepared = audio.Dataset((sample,)).prepare(max_frames=3, start_frame=4)

    assert prepared.observed == (3,)
    assert prepared.materialize()[0, 7] == 3.0


def test_audio_feature_preparation_requires_a_wav_decoder():
    """Explicit probe metadata does not silently imply signal decoding."""
    sample = audio.Sample.decode(
        b"opaque-audio", source="clip.flac", probe=flac_probe
    )

    with pytest.raises(audio.DecoderError, match="WAV decoder"):
        audio.Dataset((sample,)).prepare()


def test_audio_split_keeps_explicit_speaker_groups_together():
    """Grouped splitting prevents speaker leakage across partitions.

    Also enforces website/src/pages/docs/modalities.astro:42.
    """
    samples = tuple(
        audio.Sample.decode(
            wav_bytes(),
            source=f"clip-{index}.wav",
            label="speech",
            group_id=f"speaker-{index // 2}",
        )
        for index in range(6)
    )
    dataset = audio.Dataset(samples)

    first = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)
    second = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)

    assignments = first.cohorts()
    groups = [set(values) for values in assignments.values()]
    assert sum("speaker-0" in values for values in groups) == 1
    assert set().union(*groups) == {"speaker-0", "speaker-1", "speaker-2"}
    assert first == second


def test_video_metadata_is_explicit_and_profiled():
    """Video data records probe output instead of guessing from a suffix."""
    metadata = video.Metadata(
        format="mp4",
        width=1920,
        height=1080,
        frames=60,
        rate=30.0,
        codec="h264",
        tracks=1,
    )
    sample = video.Sample.decode(b"video-bytes", metadata, source="sample.mp4")

    assert sample.profile().duration == 2.0
    assert sample.profile().tracks == 1
    assert sample.source.endswith("/sample.mp4")


def test_video_curation_records_explicit_audio_video_synchronization():
    """Synchronization claims require measured timing and tolerance."""
    metadata = video.Metadata(
        "mp4",
        2,
        2,
        60,
        30.0,
        "h264",
        1,
        audio_duration_seconds=2.02,
        offset=0.05,
        tolerance=0.1,
    )
    sample = video.Sample.decode(
        b"synchronized-video", metadata, source="synced.mp4"
    )

    report = video.Dataset((sample,)).curate()

    assert metadata.synchrony == "within_tolerance"
    assert sample.profile().synchrony == "within_tolerance"
    assert report.profile["synchronization_statuses"] == [
        "within_tolerance",
    ]
    assert report.findings == ()


def test_video_curation_hash_matches_source_order_identity():
    """Video curation remains attached when source order is non-sorted."""
    metadata = video.Metadata("mp4", 2, 2, 1, 1.0, "h264")
    dataset = video.Dataset(
        (
            video.Sample.decode(b"z-video", metadata, "z-last.mp4"),
            video.Sample.decode(b"a-video", metadata, "a-first.mp4"),
        )
    )

    assert dataset.curate().dataset_hash == dataset.digest


def test_video_temporal_split_normalizes_time_and_records_coverage():
    """Observation times support UTC-safe future evaluation evidence.

    Also enforces website/src/pages/docs/api.astro:356.
    Also enforces website/src/pages/docs/data.astro:277.
    Also enforces website/src/pages/docs/modalities.astro:42.
    """
    metadata = video.Metadata("mp4", 2, 2, 1, 1.0, "h264")
    samples = tuple(
        video.Sample.decode(
            f"video-{index}".encode(),
            metadata,
            source=f"video-{index}.mp4",
            observation_time=timestamp,
        )
        for index, timestamp in enumerate(
            (
                "2026-01-01T01:00:00+01:00",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00+00:00",
            )
        )
    )
    dataset = video.Dataset(samples)

    split = dataset.cutover("2026-01-02T00:00:00+00:00")
    report = dataset.curate()

    assert dataset.samples[0].time == datetime(2026, 1, 1, tzinfo=UTC)
    assert len(split.train) == 2
    assert len(split.future) == 1
    assert not split.lookahead
    assert split.cutoff.tzinfo == UTC
    assert report.profile["temporal"]["status"] == "complete"


def test_video_temporal_state_fails_closed_when_incomplete():
    """Missing observation times remain explicit before temporal splitting.

    Also enforces website/src/pages/docs/api.astro:356.
    Also enforces website/src/pages/docs/data.astro:277.
    Also enforces website/src/pages/docs/data.astro:356.
    """
    metadata = video.Metadata("mp4", 2, 2, 1, 1.0, "h264")
    dataset = video.Dataset(
        (
            video.Sample.decode(
                b"timed-video",
                metadata,
                source="timed.mp4",
                observation_time="2026-01-01T00:00:00Z",
            ),
            video.Sample.decode(
                b"untimed-video", metadata, source="untimed.mp4"
            ),
        )
    )

    report = dataset.curate()

    assert report.profile["temporal"] == {
        "status": "mixed",
        "timed_video_count": 1,
        "untimed_video_count": 1,
    }
    assert "quality.video-mixed-temporal-state" in {
        finding.identifier for finding in report.findings
    }
    with pytest.raises(video.TemporalError, match="every video"):
        dataset.cutover("2026-01-02T00:00:00Z")
    with pytest.raises(video.TemporalError, match="timezone"):
        video.timestamp("2026-01-01T00:00:00")


def test_video_curation_blocks_measured_sync_drift_and_surfaces_unknown_sync():
    """Sync drift blocks timing-dependent data while missing evidence is visible."""
    drifted = video.Metadata(
        "mp4",
        2,
        2,
        60,
        30.0,
        "h264",
        1,
        audio_duration_seconds=2.4,
        offset=0.4,
        tolerance=0.1,
    )
    unknown = video.Metadata("mp4", 2, 2, 60, 30.0, "h264", 1)

    drift_report = video.Dataset(
        (video.Sample.decode(b"drift", drifted, "drift.mp4"),)
    ).curate()
    unknown_report = video.Dataset(
        (video.Sample.decode(b"unknown", unknown, "unknown.mp4"),)
    ).curate()

    assert drift_report.release is False
    assert drift_report.findings[0].identifier == (
        "quality.video-audio-sync-out-of-tolerance"
    )
    assert unknown_report.release
    assert unknown_report.findings[0].identifier == (
        "quality.video-audio-sync-unknown"
    )


def test_video_rejects_hash_mismatch():
    """A persisted video cannot claim an identity different from its bytes."""
    metadata = video.Metadata("mp4", 2, 2, 1, 1.0, "raw")

    with pytest.raises(ValueError, match="file_hash"):
        video.Sample(
            source="memory://video",
            encoded=b"video",
            file_hash="0" * 64,
            metadata=metadata,
        )


def test_video_curation_surfaces_encoding_and_dimension_drift():
    """Video curation records normalization risks instead of guessing."""
    first_metadata = video.Metadata("mp4", 2, 2, 1, 1.0, "h264")
    second_metadata = video.Metadata("webm", 4, 4, 2, 1.0, "vp9")
    dataset = video.Dataset(
        (
            video.Sample.decode(b"one", first_metadata, "one.mp4"),
            video.Sample.decode(b"two", second_metadata, "two.webm"),
        )
    )

    report = dataset.curate()

    assert report.profile["formats"] == ["mp4", "webm"]
    assert report.profile["dimensions"] == [[2, 2], [4, 4]]
    assert [finding.identifier for finding in report.findings] == [
        "quality.video-encoding-drift",
        "quality.video-dimension-drift",
    ]


def test_video_curation_blocks_conflicting_duplicate_labels():
    """Duplicate video bytes cannot silently carry incompatible labels.

    Also enforces website/src/pages/docs/api.astro:157.
    """
    metadata = video.Metadata("mp4", 2, 2, 1, 1.0, "h264")
    dataset = video.Dataset(
        (
            video.Sample.decode(
                b"same-video", metadata, "first.mp4", label="event"
            ),
            video.Sample.decode(
                b"same-video", metadata, "second.mp4", label="background"
            ),
        )
    )

    report = dataset.curate()

    assert report.profile["conflicting_label_groups"]
    assert report.release is False
    assert [finding.identifier for finding in report.risk_findings] == [
        "risk.video-conflicting-duplicate-labels",
    ]


def test_video_metadata_features_are_numeric_and_decoder_free():
    """Video preparation is explicit metadata evidence, not frame sampling.

    Also enforces website/src/pages/docs/data.astro:277.
    Also enforces website/src/pages/docs/data.astro:355.
    """
    metadata = video.Metadata("mp4", 1920, 1080, 60, 30.0, "h264", 1)
    sample = video.Sample.decode(b"video-bytes", metadata, source="sample.mp4")

    prepared = video.Dataset((sample,)).numerize()

    assert prepared.shape == (1, 10)
    assert prepared.names[:5] == (
        "duration_seconds",
        "width",
        "height",
        "pixel_count",
        "aspect_ratio",
    )
    assert prepared.materialize()[0, 0] == 2.0
    assert prepared.materialize()[0, 4] == 1920 / 1080


def test_video_metadata_features_preserve_sample_order():
    """Feature rows remain bound to canonical source order."""
    first = video.Sample.decode(
        b"one", video.Metadata("mp4", 2, 2, 2, 2.0, "h264"), "one.mp4"
    )
    second = video.Sample.decode(
        b"two", video.Metadata("mp4", 4, 2, 4, 2.0, "h264"), "two.mp4"
    )

    prepared = video.Dataset((first, second)).numerize()

    assert prepared.sources[0].endswith("/one.mp4")
    assert prepared.sources[1].endswith("/two.mp4")
    assert prepared.materialize()[:, 1].tolist() == [2.0, 4.0]


def test_video_frame_features_use_a_bounded_explicit_decoder():
    """Temporal features record evenly sampled frame identities."""
    metadata = video.Metadata("mp4", 4, 4, 5, 2.0, "h264")
    sample = video.Sample.decode(b"video-bytes", metadata, source="sample.mp4")
    calls: list[int] = []

    def decoder(payload: bytes, frame_index: int) -> numpy.ndarray:
        assert payload == sample.encoded
        calls.append(frame_index)
        return numpy.full((2, 2, 3), frame_index, dtype=numpy.uint8)

    prepared = video.Dataset((sample,)).summarize(
        decoder, max_frames=3, max_frame_pixels=32
    )

    assert calls == [0, 2, 4]
    assert prepared.frames == ((0, 2, 4),)
    assert prepared.shape == (1, 15)
    assert prepared.names[0] == "frame[0].mean"
    assert prepared.materialize()[0, 0] == 0.0
    assert prepared.materialize()[0, 5] == 2.0
    assert prepared.materialize()[0, 14] == 1.0


def test_video_frame_features_reject_oversized_decoded_frames():
    """A decoder cannot bypass the configured per-frame resource budget.

    Also enforces website/src/pages/docs/api.astro:354.
    """
    metadata = video.Metadata("mp4", 4, 4, 1, 1.0, "h264")
    sample = video.Sample.decode(b"video-bytes", metadata, source="sample.mp4")

    with pytest.raises(video.VideoError, match="pixel limit"):
        video.Dataset((sample,)).summarize(
            lambda payload, frame_index: numpy.zeros((8, 8), dtype=numpy.uint8),
            max_frame_pixels=16,
        )


def test_video_split_keeps_explicit_subject_groups_together():
    """Grouped splitting prevents subject leakage across partitions."""
    samples = tuple(
        video.Sample.decode(
            f"video-{index}".encode(),
            video.Metadata("mp4", 2, 2, 2, 2.0, "h264"),
            source=f"clip-{index}.mp4",
            label="action",
            group_id=f"subject-{index // 2}",
        )
        for index in range(6)
    )
    dataset = video.Dataset(samples)

    first = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)
    second = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)

    assignments = first.cohorts()
    groups = [set(values) for values in assignments.values()]
    assert sum("subject-0" in values for values in groups) == 1
    assert set().union(*groups) == {
        "subject-0",
        "subject-1",
        "subject-2",
    }
    assert first == second


def test_geospatial_feature_collection_preserves_bounds_and_crs():
    """GeoJSON validation produces deterministic geometry understanding."""
    dataset = geospatial.Dataset.parse(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "a",
                    "geometry": {"type": "Point", "coordinates": [2, 1]},
                    "properties": {"class": "park"},
                },
                {
                    "type": "Feature",
                    "id": "b",
                    "geometry": {
                        "type": "LineString",
                        "coordinates": [[-3, 4], [5, -2]],
                    },
                    "properties": {},
                },
            ],
        },
        source="features.geojson",
        crs="EPSG:4326",
    )

    profile = dataset.profile()
    assert profile.features == 2
    assert profile.geometries == ("LineString", "Point")
    assert profile.crs == "EPSG:4326"
    assert profile.bounds == (-3.0, -2.0, 5.0, 4.0)
    assert dataset.source.endswith("/features.geojson")


def test_geospatial_identity_binds_crs_and_ignores_feature_order():
    """Equivalent features share identity only when their CRS also matches.

    Also enforces website/src/pages/docs/api.astro:485.
    Also enforces website/src/pages/docs/data.astro:274.
    """
    first_feature = geospatial.Placemark(
        "first", {"type": "Point", "coordinates": [0, 0]}
    )
    second_feature = geospatial.Placemark(
        "second", {"type": "Point", "coordinates": [1, 1]}
    )
    first = geospatial.Dataset(
        (first_feature, second_feature), crs=" EPSG:4326 "
    )
    reordered = geospatial.Dataset(
        (second_feature, first_feature), crs="EPSG:4326"
    )
    projected = geospatial.Dataset(
        (first_feature, second_feature), crs="EPSG:3857"
    )

    assert first.crs == "EPSG:4326"
    assert first.digest == reordered.digest
    assert first.digest != projected.digest


def test_geospatial_identity_binds_labels_and_properties():
    """Feature metadata changes produce a new immutable dataset identity.

    Also enforces website/src/pages/docs/data.astro:274.
    """
    geometry = {"type": "Point", "coordinates": [0, 0]}
    baseline = geospatial.Dataset(
        (geospatial.Placemark("feature", geometry, {"kind": "park"}, "open"),),
        crs="EPSG:4326",
    )
    changed_label = geospatial.Dataset(
        (
            geospatial.Placemark(
                "feature", geometry, {"kind": "park"}, "closed"
            ),
        ),
        crs="EPSG:4326",
    )
    changed_property = geospatial.Dataset(
        (
            geospatial.Placemark(
                "feature", geometry, {"kind": "school"}, "open"
            ),
        ),
        crs="EPSG:4326",
    )

    assert baseline.digest != changed_label.digest
    assert baseline.digest != changed_property.digest
    with pytest.raises(TypeError):
        baseline.features[0].properties["kind"] = "mutated"


def test_geospatial_spatial_join_records_bounded_same_crs_evidence():
    """A caller predicate produces deterministic, hash-bound join evidence.

    Also enforces website/src/pages/docs/capabilities.astro:22.
    Also enforces website/src/pages/docs/data.astro:350.
    """
    left = geospatial.Dataset(
        (
            geospatial.Placemark(
                "left-a", {"type": "Point", "coordinates": [0, 0]}
            ),
            geospatial.Placemark(
                "left-b", {"type": "Point", "coordinates": [10, 0]}
            ),
        ),
        crs="EPSG:3857",
    )
    right = geospatial.Dataset(
        (
            geospatial.Placemark(
                "right-a", {"type": "Point", "coordinates": [1, 0]}
            ),
            geospatial.Placemark(
                "right-b", {"type": "Point", "coordinates": [20, 0]}
            ),
        ),
        crs="EPSG:3857",
    )

    report = left.join(
        right,
        lambda first, second: abs(first.centroid[0] - second.centroid[0]) < 5,
        "centroid_distance_lt_5/v1",
    )

    assert report.crs == "EPSG:3857"
    assert report.comparisons == 4
    assert report.matched == 1
    assert report.unmatched_left_count == 1
    assert report.unmatched_right_count == 1
    assert report.matches == (geospatial.Match("left-a", "right-a"),)
    assert report.left_dataset_hash == left.digest
    assert report.right_dataset_hash == right.digest
    assert report.to_dict()["predicate"] == "centroid_distance_lt_5/v1"


def test_geospatial_spatial_join_rejects_crs_mismatch_and_unbounded_work():
    """Joins fail closed instead of comparing incompatible or huge inputs.

    Also enforces website/src/pages/docs/data.astro:345.
    Also enforces website/src/pages/docs/data.astro:350.
    """
    feature = geospatial.Placemark(
        "feature", {"type": "Point", "coordinates": [0, 0]}
    )
    left = geospatial.Dataset((feature,), crs="EPSG:4326")
    right = geospatial.Dataset((feature,), crs="EPSG:3857")

    with pytest.raises(geospatial.GeometryError, match="matching CRS"):
        left.join(right, lambda first, second: True, "always/v1")

    with pytest.raises(
        geospatial.GeometryError,
        match="comparison limit",
    ):
        left.join(
            geospatial.Dataset(
                tuple(
                    geospatial.Placemark(
                        f"right-{index}",
                        {"type": "Point", "coordinates": [index, 0]},
                    )
                    for index in range(3)
                ),
                crs="EPSG:4326",
            ),
            lambda first, second: True,
            "always/v1",
            comparison_limit=2,
        )


def test_geospatial_spatial_join_rejects_invalid_predicate_output():
    """Join evidence cannot encode an ambiguous predicate result."""
    feature = geospatial.Placemark(
        "feature", {"type": "Point", "coordinates": [0, 0]}
    )
    dataset = geospatial.Dataset((feature,), crs="EPSG:4326")

    with pytest.raises(
        geospatial.GeometryError,
        match="must return a boolean",
    ):
        dataset.join(dataset, lambda first, second: "yes", "bad/v1")


def test_geospatial_reprojection_records_source_and_target_identity():
    """An explicit transformer preserves metadata and binds both snapshots.

    Also enforces website/src/pages/docs/api.astro:484.
    Also enforces website/src/pages/docs/capabilities.astro:22.
    Also enforces website/src/pages/docs/data.astro:350.
    """
    observation_time = datetime(2026, 1, 1, tzinfo=UTC)
    dataset = geospatial.Dataset(
        (
            geospatial.Placemark(
                "feature",
                {"type": "Point", "coordinates": [1, 2]},
                {"kind": "park"},
                label="open",
                time=observation_time,
            ),
        ),
        crs="EPSG:4326",
    )

    def transform(
        feature: geospatial.Placemark, source_crs: str, target_crs: str
    ) -> geospatial.Placemark:
        assert source_crs == "EPSG:4326"
        assert target_crs == "EPSG:3857"
        return geospatial.Placemark(
            feature.identifier,
            {
                "type": "Point",
                "coordinates": [
                    feature.centroid[0] * 2,
                    feature.centroid[1] * 2,
                ],
            },
            dict(feature.properties),
            feature.label,
            feature.time,
        )

    result = dataset.reproject("EPSG:3857", transform, "scale/v1")

    assert result.dataset.crs == "EPSG:3857"
    assert result.dataset.features[0].geometry["coordinates"] == [2.0, 4.0]
    assert result.report.source_dataset_hash == dataset.digest
    assert result.report.target_dataset_hash == result.dataset.digest
    assert result.report.transformer == "scale/v1"
    assert result.to_dict()["report"]["features"] == 1


def test_geospatial_reprojection_rejects_metadata_changes():
    """A transformer cannot silently alter labels or feature ownership.

    Also enforces website/src/pages/docs/api.astro:484.
    Also enforces website/src/pages/docs/data.astro:350.
    """
    feature = geospatial.Placemark(
        "feature",
        {"type": "Point", "coordinates": [0, 0]},
        label="open",
    )
    dataset = geospatial.Dataset((feature,), crs="EPSG:4326")

    def changes_label(
        value: geospatial.Placemark, source_crs: str, target_crs: str
    ) -> geospatial.Placemark:
        return geospatial.Placemark(
            value.identifier,
            value.geometry,
            dict(value.properties),
            label="closed",
        )

    with pytest.raises(
        geospatial.GeometryError,
        match="changed feature metadata",
    ):
        dataset.reproject("EPSG:3857", changes_label, "bad/v1")


def test_geospatial_rejects_invalid_geometry():
    """Malformed geometry is rejected before it reaches feature preparation.

    Also enforces website/src/pages/docs/data.astro:345.
    Also enforces website/src/pages/docs/extensions.astro:199.
    """
    with pytest.raises(geospatial.GeometryError):
        geospatial.Placemark(
            "broken", {"type": "Point", "coordinates": [float("nan"), 1]}
        )


def test_geospatial_curation_blocks_missing_crs():
    """Spatial coordinates cannot enter modelling without a reference CRS."""
    dataset = geospatial.Dataset.parse(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "a",
                    "geometry": {"type": "Point", "coordinates": [0, 0]},
                }
            ],
        }
    )

    report = dataset.curate()

    assert report.release is False
    assert report.blockers[0].identifier == ("quality.geospatial-missing-crs")


def test_geospatial_curation_reports_mixed_label_state():
    """Spatial curation exposes mixed labels before spatial modelling."""
    dataset = geospatial.Dataset(
        (
            geospatial.Placemark(
                "a",
                {"type": "Point", "coordinates": [0, 0]},
                label="park",
            ),
            geospatial.Placemark("b", {"type": "Point", "coordinates": [1, 1]}),
        ),
        crs="EPSG:4326",
    )

    report = dataset.curate()

    assert report.profile["label_counts"] == {
        "str:park": 1,
        "unlabeled": 1,
    }
    assert [finding.identifier for finding in report.risk_findings] == [
        "risk.geospatial-mixed-label-state",
    ]


def test_geospatial_curation_records_deterministic_spatial_autocorrelation():
    """Numeric labels expose bounded spatial-dependence evidence."""
    dataset = geospatial.Dataset(
        tuple(
            geospatial.Placemark(
                f"feature-{index}",
                {"type": "Point", "coordinates": [float(index), 0]},
                label=float(index // 2),
            )
            for index in range(5)
        ),
        crs="EPSG:3857",
    )

    first = dataset.autocorrelate(neighbor_count=1, permutations=19, seed=11)
    second = dataset.autocorrelate(neighbor_count=1, permutations=19, seed=11)
    report = dataset.curate()

    assert first == second
    assert first.edges == 4
    assert first.statistic > 0
    assert 0.0 <= first.p_value <= 1.0
    evidence = report.profile["spatial_autocorrelation"]
    assert evidence["features"] == 5
    assert evidence["neighbors"] == 4
    assert evidence["permutations"] == 99
    assert evidence["p_value"] is not None


def test_spatial_autocorrelation_rejects_non_numeric_and_constant_values():
    """Spatial dependence requires an estimable numeric variable."""
    features = tuple(
        geospatial.Placemark(
            f"feature-{index}",
            {"type": "Point", "coordinates": [float(index), 0]},
            label="same",
        )
        for index in range(3)
    )
    dataset = geospatial.Dataset(features, crs="EPSG:3857")

    with pytest.raises(geospatial.GeometryError, match="numeric"):
        dataset.autocorrelate()

    with pytest.raises(geospatial.GeometryError, match="more than one unique"):
        dataset.autocorrelate(values=[1.0, 1.0, 1.0])


def test_geospatial_spatial_split_keeps_blocks_together():
    """Spatial blocks remain isolated across train and evaluation data.

    Also enforces website/src/pages/docs/api.astro:372.
    Also enforces website/src/pages/docs/capabilities.astro:22.
    Also enforces website/src/pages/docs/data.astro:278.
    Also enforces website/src/pages/docs/data.astro:349.
    Also enforces website/src/pages/docs/modalities.astro:42.
    """
    features = tuple(
        geospatial.Placemark(
            f"feature-{index}",
            {"type": "Point", "coordinates": [float(index * 10), 0]},
            label="class",
        )
        for index in range(6)
    )
    dataset = geospatial.Dataset(features, crs="EPSG:3857")

    first = dataset.tile(
        block_size=5, validation_fraction=0.25, test_fraction=0.25, seed=3
    )
    second = dataset.tile(
        block_size=5, validation_fraction=0.25, test_fraction=0.25, seed=3
    )

    block_assignments = first.tiles()
    blocks = [set(values) for values in block_assignments.values()]
    assert len(set().union(*blocks)) == 6
    assert first == second
    assert first.block_size == 5


def test_geospatial_boundary_leakage_reports_conservative_envelope_crossings():
    """Bounds crossing blocks are surfaced before centroid-based splitting.

    Also enforces website/src/pages/docs/api.astro:408.
    Also enforces website/src/pages/docs/capabilities.astro:22.
    Also enforces website/src/pages/docs/data.astro:278.
    Also enforces website/src/pages/docs/modalities.astro:63.
    """
    dataset = geospatial.Dataset(
        (
            geospatial.Placemark(
                "point", {"type": "Point", "coordinates": [0, 0]}
            ),
            geospatial.Placemark(
                "line",
                {"type": "LineString", "coordinates": [[0, 0], [2.2, 0]]},
            ),
        ),
        crs="EPSG:3857",
    )

    evidence = dataset.straddle(block_size=1.0, max_blocks_per_feature=2)

    assert evidence.status == "potential_leakage"
    assert evidence.crossing == ("line",)
    assert evidence.truncated == ("line",)
    assert evidence.blocks == (("point", 1), ("line", 3))
    assert evidence.to_dict()["spread"] == 2


def test_geospatial_temporal_split_normalizes_time_and_curates_coverage():
    """Observation times support UTC-safe future evaluation evidence."""
    features = tuple(
        geospatial.Placemark(
            f"feature-{index}",
            {"type": "Point", "coordinates": [float(index), 0]},
            time=timestamp,
        )
        for index, timestamp in enumerate(
            (
                "2026-01-01T01:00:00+01:00",
                "2026-01-02T00:00:00Z",
                "2026-01-03T00:00:00+00:00",
            )
        )
    )
    dataset = geospatial.Dataset(features, crs="EPSG:3857")

    split = dataset.cutover("2026-01-02T00:00:00+00:00")
    report = dataset.curate()

    assert dataset.features[0].time == datetime(2026, 1, 1, tzinfo=UTC)
    assert len(split.train) == 2
    assert len(split.future) == 1
    assert not split.lookahead
    assert split.cutoff.tzinfo == UTC
    assert report.profile["temporal"]["status"] == "complete"


def test_geospatial_temporal_state_fails_closed_when_incomplete():
    """Missing or malformed observation times remain explicit before splitting."""
    mixed = geospatial.Dataset(
        (
            geospatial.Placemark(
                "timed",
                {"type": "Point", "coordinates": [0, 0]},
                time=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            geospatial.Placemark(
                "untimed", {"type": "Point", "coordinates": [1, 1]}
            ),
        ),
        crs="EPSG:3857",
    )

    report = mixed.curate()
    quality_ids = {finding.identifier for finding in report.findings}

    assert report.profile["temporal"] == {
        "status": "mixed",
        "timed_feature_count": 1,
        "untimed_feature_count": 1,
    }
    assert "quality.geospatial-mixed-temporal-state" in quality_ids
    with pytest.raises(geospatial.TemporalError, match="every feature"):
        mixed.cutover("2026-01-02T00:00:00Z")
    with pytest.raises(geospatial.TemporalError, match="timezone"):
        geospatial.timestamp("2026-01-01T00:00:00")


def test_geospatial_feature_preparation_preserves_crs_units():
    """Geometry envelopes are prepared only after CRS provenance exists.

    Also enforces website/src/pages/docs/api.astro:420.
    """
    dataset = geospatial.Dataset(
        (
            geospatial.Placemark(
                "point", {"type": "Point", "coordinates": [2, 1]}
            ),
            geospatial.Placemark(
                "line",
                {"type": "LineString", "coordinates": [[-3, 4], [5, -2]]},
            ),
        ),
        crs="EPSG:4326",
    )

    prepared = dataset.prepare()

    assert prepared.shape == (2, 9)
    assert prepared.crs == "EPSG:4326"
    assert prepared.feature_ids == ("point", "line")
    assert prepared.materialize()[1, 4:6].tolist() == [8.0, 6.0]


def test_geospatial_raw_axis_mask_preserves_nested_geometry_structure():
    """Raw-coordinate masking preserves geometry type and untouched axes."""
    feature = geospatial.Placemark(
        "collection",
        {
            "type": "GeometryCollection",
            "geometries": [
                {"type": "Point", "coordinates": [2, 1]},
                {
                    "type": "LineString",
                    "coordinates": [[-3, 4], [5, -2]],
                },
            ],
        },
    )

    masked = feature.substituted(0, 0.5)

    assert masked.geometry["type"] == "GeometryCollection"
    geometries = masked.geometry["geometries"]
    assert geometries[0]["coordinates"] == [0.5, 1.0]
    assert geometries[1]["coordinates"] == [[0.5, 4.0], [0.5, -2.0]]
    assert masked.geometry_hash != feature.geometry_hash


def test_geospatial_raw_axis_mask_rejects_invalid_axis():
    """Only x and y axes may be perturbed by the raw evidence policy."""
    feature = geospatial.Placemark(
        "point", {"type": "Point", "coordinates": [2, 1]}
    )

    with pytest.raises(ValueError, match="axis"):
        feature.substituted(2, 0.0)


def test_geospatial_feature_preparation_rejects_missing_crs():
    """Coordinates cannot become model features without a reference CRS.

    Also enforces website/src/pages/docs/api.astro:420.
    Also enforces website/src/pages/docs/data.astro:345.
    """
    dataset = geospatial.Dataset(
        (
            geospatial.Placemark(
                "point", {"type": "Point", "coordinates": [0, 0]}
            ),
        )
    )

    with pytest.raises(geospatial.GeometryError, match="CRS"):
        dataset.prepare()


def test_raster_identity_requires_spatial_metadata():
    """Raster bytes are not useful without dimensions and CRS provenance."""
    payload = b"raster"
    raster = geospatial.Tile(
        source="memory://raster",
        encoded=payload,
        digest=hashlib.sha256(payload).hexdigest(),
        width=10,
        height=20,
        bands=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=(1, 0, 0, 0, -1, 0),
    )

    assert raster.size == len(payload)
    assert raster.crs == "EPSG:3857"


def test_raster_decoder_is_bounded_and_shape_checked():
    """Raster decoding preserves metadata and enforces the pixel budget."""
    payload = b"raster"
    raster = geospatial.Tile(
        source="memory://raster",
        encoded=payload,
        digest=hashlib.sha256(payload).hexdigest(),
        width=2,
        height=3,
        bands=2,
        dtype="uint8",
        crs="EPSG:3857",
        transform=(1, 0, 0, 0, -1, 0),
        sentinel=0,
    )

    pixels = raster.decode(
        lambda value: numpy.full((3, 2, 2), value[0], dtype=numpy.uint8),
        max_pixels=12,
    )

    assert pixels.shape == (3, 2, 2)
    assert not pixels.flags.writeable
    assert (
        raster.measure(
            lambda value: numpy.full((3, 2, 2), value[0], dtype=numpy.uint8)
        )
        == 0.0
    )

    with pytest.raises(geospatial.GeometryError, match="limit"):
        raster.decode(
            lambda value: numpy.zeros((3, 2, 2), dtype=numpy.uint8),
            max_pixels=11,
        )


def test_raster_decoder_rejects_metadata_mismatch():
    """A caller-owned decoder cannot override declared raster dimensions."""
    payload = b"raster"
    raster = geospatial.Tile(
        source="memory://raster",
        encoded=payload,
        digest=hashlib.sha256(payload).hexdigest(),
        width=2,
        height=3,
        bands=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=(1, 0, 0, 0, -1, 0),
    )

    with pytest.raises(geospatial.GeometryError, match="dimensions"):
        raster.decode(lambda value: numpy.zeros((2, 2), dtype=numpy.uint8))
