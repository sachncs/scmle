"""Tests for dependency-light audio, video, and geospatial foundations."""

from datetime import datetime
from datetime import UTC
import hashlib
import io
import wave

import numpy as np
import pytest

from scml.data.modalities import audio
from scml.data.modalities import geospatial
from scml.data.modalities import video


def wav_bytes(frame_count: int = 8) -> bytes:
    """Build a minimal deterministic mono WAV fixture."""
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(8)
        writer.writeframes(b"\x00\x00" * frame_count)
    return output.getvalue()


def failing_probe(payload: bytes) -> audio.AudioMetadata:
    """Represent a caller-owned codec probe that fails deterministically."""
    raise RuntimeError(payload)


def flac_probe(payload: bytes) -> audio.AudioMetadata:
    """Represent metadata returned by an explicit non-WAV codec adapter."""
    return audio.AudioMetadata("flac", 16_000, 1, len(payload))


def test_audio_wav_ingestion_preserves_identity_and_profile():
    """WAV metadata and bytes are bound to the same immutable sample."""
    sample = audio.AudioSample.from_bytes(wav_bytes(), source="clip.wav")

    assert sample.source.endswith("/clip.wav")
    assert sample.metadata.sample_rate == 8
    assert sample.metadata.channels == 1
    assert sample.profile().duration_seconds == 1.0
    assert sample.file_hash == hashlib.sha256(sample.encoded_bytes).hexdigest()


def test_audio_non_wav_requires_explicit_probe():
    """Unknown containers do not silently acquire a codec dependency."""
    with pytest.raises(audio.AudioDependencyError):
        audio.AudioSample.from_bytes(b"opaque-audio", source="clip.flac")

    with pytest.raises(audio.AudioDependencyError):
        audio.AudioSample.from_bytes(
            b"opaque-audio",
            source="clip.flac",
            probe=failing_probe,
        )


def test_audio_dataset_rejects_duplicate_sources():
    """Duplicate source identities are a curation error, not two samples."""
    first = audio.AudioSample.from_bytes(wav_bytes(), source="one.wav")
    duplicate = audio.AudioSample.from_bytes(wav_bytes(), source="one.wav")

    with pytest.raises(audio.AudioValidationError, match="duplicate sources"):
        audio.AudioDataset((first, duplicate))


def test_audio_curation_reports_metadata_and_label_composition():
    """Audio data understanding is explicit before a model is selected."""
    sample = audio.AudioSample.from_bytes(wav_bytes(), source="one.wav")

    report = audio.AudioDataset((sample,)).curate()

    assert report.modality == "audio"
    assert report.sample_count == 1
    assert report.profile["sample_rates"] == [8]
    assert report.profile["label_counts"] == {"unlabeled": 1}
    assert report.release_ready


def test_audio_transcript_provenance_is_bounded_and_curated():
    """Transcript metadata remains tied to the source audio and provider."""
    transcript = audio.AudioTranscript(
        text="hello world",
        source="transcriber://run-7",
        provider="acme-asr",
        provider_version="2026.10",
        language="en",
        segments=(
            audio.AudioTranscriptSegment(0.0, 0.5, "hello", 0.9),
            audio.AudioTranscriptSegment(0.5, 1.0, "world", 0.8),
        ),
        confidence=0.85,
    )
    sample = audio.AudioSample.from_bytes(
        wav_bytes(), source="spoken.wav", transcript=transcript
    )

    report = audio.AudioDataset((sample,)).curate()

    assert sample.profile().transcript_present
    assert sample.profile().transcript_segment_count == 2
    assert transcript.to_dict()["provider_version"] == "2026.10"
    assert report.profile["transcript_coverage"] == 1.0
    assert report.profile["transcript_sources"] == ["transcriber://run-7"]
    assert report.profile["transcript_segment_count"] == 2
    assert report.profile["transcript_confidence_min"] == 0.8
    assert report.quality_findings == ()


def test_audio_transcript_confidence_and_timing_fail_closed():
    """Missing confidence is visible and segments cannot exceed duration."""
    uncertain = audio.AudioTranscript(
        text="hello",
        source="transcriber://run-8",
        provider="acme-asr",
        provider_version="2026.10",
        language="en",
        segments=(audio.AudioTranscriptSegment(0.0, 0.5, "hello"),),
    )
    report = audio.AudioDataset(
        (
            audio.AudioSample.from_bytes(
                wav_bytes(), source="uncertain.wav", transcript=uncertain
            ),
        )
    ).curate()

    assert [finding.finding_id for finding in report.quality_findings] == [
        "quality.audio-transcript-confidence-missing",
    ]
    out_of_range = audio.AudioTranscript(
        text="too long",
        source="transcriber://run-9",
        provider="acme-asr",
        provider_version="2026.10",
        language="en",
        segments=(audio.AudioTranscriptSegment(0.0, 2.0, "too long"),),
    )
    with pytest.raises(audio.AudioValidationError, match="exceeds audio"):
        audio.AudioSample.from_bytes(
            wav_bytes(), source="bad.wav", transcript=out_of_range
        )


def test_audio_curation_blocks_conflicting_duplicate_labels():
    """Duplicate audio bytes cannot silently carry incompatible labels."""
    dataset = audio.AudioDataset(
        (
            audio.AudioSample.from_bytes(
                wav_bytes(), source="first.wav", label="speech"
            ),
            audio.AudioSample.from_bytes(
                wav_bytes(), source="second.wav", label="music"
            ),
        )
    )

    report = dataset.curate()

    assert report.profile["conflicting_label_groups"]
    assert report.release_ready is False
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.audio-conflicting-duplicate-labels",
    ]


def test_audio_feature_preparation_is_bounded_and_explainable():
    """Signal features expose the observed prefix and deterministic names."""
    sample = audio.AudioSample.from_bytes(
        wav_bytes(frame_count=8), source="one.wav"
    )

    prepared = audio.AudioDataset((sample,)).prepare_features(max_frames=3)

    assert prepared.shape == (1, 13)
    assert prepared.frames_observed == (3,)
    assert prepared.frames_truncated == (True,)
    assert prepared.feature_names[:4] == (
        "mean_amplitude",
        "mean_absolute_amplitude",
        "rms_amplitude",
        "peak_amplitude",
    )
    assert prepared.to_numpy()[0, 2] == 0.0


def test_audio_feature_preparation_supports_a_bounded_window():
    """A decoded WAV window remains tied to its source frame boundary."""
    sample = audio.AudioSample.from_bytes(
        wav_bytes(frame_count=8), source="one.wav"
    )

    prepared = audio.AudioDataset((sample,)).prepare_features(
        max_frames=3, start_frame=4
    )

    assert prepared.frames_observed == (3,)
    assert prepared.to_numpy()[0, 7] == 3.0


def test_audio_feature_preparation_requires_a_wav_decoder():
    """Explicit probe metadata does not silently imply signal decoding."""
    sample = audio.AudioSample.from_bytes(
        b"opaque-audio", source="clip.flac", probe=flac_probe
    )

    with pytest.raises(audio.AudioDependencyError, match="WAV decoder"):
        audio.AudioDataset((sample,)).prepare_features()


def test_audio_split_keeps_explicit_speaker_groups_together():
    """Grouped splitting prevents speaker leakage across partitions."""
    samples = tuple(
        audio.AudioSample.from_bytes(
            wav_bytes(),
            source=f"clip-{index}.wav",
            label="speech",
            group_id=f"speaker-{index // 2}",
        )
        for index in range(6)
    )
    dataset = audio.AudioDataset(samples)

    first = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)
    second = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)

    assignments = first.group_ids()
    groups = [set(values) for values in assignments.values()]
    assert sum("speaker-0" in values for values in groups) == 1
    assert set().union(*groups) == {"speaker-0", "speaker-1", "speaker-2"}
    assert first == second


def test_video_metadata_is_explicit_and_profiled():
    """Video data records probe output instead of guessing from a suffix."""
    metadata = video.VideoMetadata(
        format="mp4",
        width=1920,
        height=1080,
        frame_count=60,
        frames_per_second=30.0,
        codec="h264",
        audio_tracks=1,
    )
    sample = video.VideoSample.from_bytes(
        b"video-bytes", metadata, source="sample.mp4"
    )

    assert sample.profile().duration_seconds == 2.0
    assert sample.profile().audio_tracks == 1
    assert sample.source.endswith("/sample.mp4")


def test_video_curation_records_explicit_audio_video_synchronization():
    """Synchronization claims require measured timing and tolerance."""
    metadata = video.VideoMetadata(
        "mp4",
        2,
        2,
        60,
        30.0,
        "h264",
        1,
        audio_duration_seconds=2.02,
        sync_offset_seconds=0.05,
        sync_tolerance_seconds=0.1,
    )
    sample = video.VideoSample.from_bytes(
        b"synchronized-video", metadata, source="synced.mp4"
    )

    report = video.VideoDataset((sample,)).curate()

    assert metadata.synchronization_status == "within_tolerance"
    assert sample.profile().synchronization_status == "within_tolerance"
    assert report.profile["synchronization_statuses"] == [
        "within_tolerance",
    ]
    assert report.quality_findings == ()


def test_video_curation_blocks_measured_sync_drift_and_surfaces_unknown_sync():
    """Sync drift blocks timing-dependent data while missing evidence is visible."""
    drifted = video.VideoMetadata(
        "mp4",
        2,
        2,
        60,
        30.0,
        "h264",
        1,
        audio_duration_seconds=2.4,
        sync_offset_seconds=0.4,
        sync_tolerance_seconds=0.1,
    )
    unknown = video.VideoMetadata("mp4", 2, 2, 60, 30.0, "h264", 1)

    drift_report = video.VideoDataset(
        (video.VideoSample.from_bytes(b"drift", drifted, "drift.mp4"),)
    ).curate()
    unknown_report = video.VideoDataset(
        (video.VideoSample.from_bytes(b"unknown", unknown, "unknown.mp4"),)
    ).curate()

    assert drift_report.release_ready is False
    assert drift_report.quality_findings[0].finding_id == (
        "quality.video-audio-sync-out-of-tolerance"
    )
    assert unknown_report.release_ready
    assert unknown_report.quality_findings[0].finding_id == (
        "quality.video-audio-sync-unknown"
    )


def test_video_rejects_hash_mismatch():
    """A persisted video cannot claim an identity different from its bytes."""
    metadata = video.VideoMetadata("mp4", 2, 2, 1, 1.0, "raw")

    with pytest.raises(ValueError, match="file_hash"):
        video.VideoSample(
            source="memory://video",
            encoded_bytes=b"video",
            file_hash="0" * 64,
            metadata=metadata,
        )


def test_video_curation_surfaces_encoding_and_dimension_drift():
    """Video curation records normalization risks instead of guessing."""
    first_metadata = video.VideoMetadata("mp4", 2, 2, 1, 1.0, "h264")
    second_metadata = video.VideoMetadata("webm", 4, 4, 2, 1.0, "vp9")
    dataset = video.VideoDataset(
        (
            video.VideoSample.from_bytes(b"one", first_metadata, "one.mp4"),
            video.VideoSample.from_bytes(b"two", second_metadata, "two.webm"),
        )
    )

    report = dataset.curate()

    assert report.profile["formats"] == ["mp4", "webm"]
    assert report.profile["dimensions"] == [[2, 2], [4, 4]]
    assert [finding.finding_id for finding in report.quality_findings] == [
        "quality.video-encoding-drift",
        "quality.video-dimension-drift",
    ]


def test_video_curation_blocks_conflicting_duplicate_labels():
    """Duplicate video bytes cannot silently carry incompatible labels."""
    metadata = video.VideoMetadata("mp4", 2, 2, 1, 1.0, "h264")
    dataset = video.VideoDataset(
        (
            video.VideoSample.from_bytes(
                b"same-video", metadata, "first.mp4", label="event"
            ),
            video.VideoSample.from_bytes(
                b"same-video", metadata, "second.mp4", label="background"
            ),
        )
    )

    report = dataset.curate()

    assert report.profile["conflicting_label_groups"]
    assert report.release_ready is False
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.video-conflicting-duplicate-labels",
    ]


def test_video_metadata_features_are_numeric_and_decoder_free():
    """Video preparation is explicit metadata evidence, not frame sampling."""
    metadata = video.VideoMetadata("mp4", 1920, 1080, 60, 30.0, "h264", 1)
    sample = video.VideoSample.from_bytes(
        b"video-bytes", metadata, source="sample.mp4"
    )

    prepared = video.VideoDataset((sample,)).prepare_metadata_features()

    assert prepared.shape == (1, 10)
    assert prepared.feature_names[:5] == (
        "duration_seconds",
        "width",
        "height",
        "pixel_count",
        "aspect_ratio",
    )
    assert prepared.to_numpy()[0, 0] == 2.0
    assert prepared.to_numpy()[0, 4] == 1920 / 1080


def test_video_metadata_features_preserve_sample_order():
    """Feature rows remain bound to canonical source order."""
    first = video.VideoSample.from_bytes(
        b"one", video.VideoMetadata("mp4", 2, 2, 2, 2.0, "h264"), "one.mp4"
    )
    second = video.VideoSample.from_bytes(
        b"two", video.VideoMetadata("mp4", 4, 2, 4, 2.0, "h264"), "two.mp4"
    )

    prepared = video.VideoDataset((first, second)).prepare_metadata_features()

    assert prepared.sources[0].endswith("/one.mp4")
    assert prepared.sources[1].endswith("/two.mp4")
    assert prepared.to_numpy()[:, 1].tolist() == [2.0, 4.0]


def test_video_frame_features_use_a_bounded_explicit_decoder():
    """Temporal features record evenly sampled frame identities."""
    metadata = video.VideoMetadata("mp4", 4, 4, 5, 2.0, "h264")
    sample = video.VideoSample.from_bytes(
        b"video-bytes", metadata, source="sample.mp4"
    )
    calls: list[int] = []

    def decoder(payload: bytes, frame_index: int) -> np.ndarray:
        assert payload == sample.encoded_bytes
        calls.append(frame_index)
        return np.full((2, 2, 3), frame_index, dtype=np.uint8)

    prepared = video.VideoDataset((sample,)).prepare_frame_features(
        decoder, max_frames=3, max_frame_pixels=32
    )

    assert calls == [0, 2, 4]
    assert prepared.sampled_frame_indices == ((0, 2, 4),)
    assert prepared.shape == (1, 15)
    assert prepared.feature_names[0] == "frame[0].mean"
    assert prepared.to_numpy()[0, 0] == 0.0
    assert prepared.to_numpy()[0, 5] == 2.0
    assert prepared.to_numpy()[0, 14] == 1.0


def test_video_frame_features_reject_oversized_decoded_frames():
    """A decoder cannot bypass the configured per-frame resource budget."""
    metadata = video.VideoMetadata("mp4", 4, 4, 1, 1.0, "h264")
    sample = video.VideoSample.from_bytes(
        b"video-bytes", metadata, source="sample.mp4"
    )

    with pytest.raises(video.VideoValidationError, match="pixel limit"):
        video.VideoDataset((sample,)).prepare_frame_features(
            lambda payload, frame_index: np.zeros((8, 8), dtype=np.uint8),
            max_frame_pixels=16,
        )


def test_video_split_keeps_explicit_subject_groups_together():
    """Grouped splitting prevents subject leakage across partitions."""
    samples = tuple(
        video.VideoSample.from_bytes(
            f"video-{index}".encode(),
            video.VideoMetadata("mp4", 2, 2, 2, 2.0, "h264"),
            source=f"clip-{index}.mp4",
            label="action",
            group_id=f"subject-{index // 2}",
        )
        for index in range(6)
    )
    dataset = video.VideoDataset(samples)

    first = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)
    second = dataset.split(validation_fraction=0.25, test_fraction=0.25, seed=3)

    assignments = first.group_ids()
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
    dataset = geospatial.GeospatialDataset.from_geojson(
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
    assert profile.feature_count == 2
    assert profile.geometry_types == ("LineString", "Point")
    assert profile.crs == "EPSG:4326"
    assert profile.bounds == (-3.0, -2.0, 5.0, 4.0)
    assert dataset.source.endswith("/features.geojson")


def test_geospatial_identity_binds_crs_and_ignores_feature_order():
    """Equivalent features share identity only when their CRS also matches."""
    first_feature = geospatial.GeoFeature(
        "first", {"type": "Point", "coordinates": [0, 0]}
    )
    second_feature = geospatial.GeoFeature(
        "second", {"type": "Point", "coordinates": [1, 1]}
    )
    first = geospatial.GeospatialDataset(
        (first_feature, second_feature), crs=" EPSG:4326 "
    )
    reordered = geospatial.GeospatialDataset(
        (second_feature, first_feature), crs="EPSG:4326"
    )
    projected = geospatial.GeospatialDataset(
        (first_feature, second_feature), crs="EPSG:3857"
    )

    assert first.crs == "EPSG:4326"
    assert first.content_hash == reordered.content_hash
    assert first.content_hash != projected.content_hash


def test_geospatial_identity_binds_labels_and_properties():
    """Feature metadata changes produce a new immutable dataset identity."""
    geometry = {"type": "Point", "coordinates": [0, 0]}
    baseline = geospatial.GeospatialDataset(
        (geospatial.GeoFeature("feature", geometry, {"kind": "park"}, "open"),),
        crs="EPSG:4326",
    )
    changed_label = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "feature", geometry, {"kind": "park"}, "closed"
            ),
        ),
        crs="EPSG:4326",
    )
    changed_property = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "feature", geometry, {"kind": "school"}, "open"
            ),
        ),
        crs="EPSG:4326",
    )

    assert baseline.content_hash != changed_label.content_hash
    assert baseline.content_hash != changed_property.content_hash
    with pytest.raises(TypeError):
        baseline.features[0].properties["kind"] = "mutated"


def test_geospatial_spatial_join_records_bounded_same_crs_evidence():
    """A caller predicate produces deterministic, hash-bound join evidence."""
    left = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "left-a", {"type": "Point", "coordinates": [0, 0]}
            ),
            geospatial.GeoFeature(
                "left-b", {"type": "Point", "coordinates": [10, 0]}
            ),
        ),
        crs="EPSG:3857",
    )
    right = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "right-a", {"type": "Point", "coordinates": [1, 0]}
            ),
            geospatial.GeoFeature(
                "right-b", {"type": "Point", "coordinates": [20, 0]}
            ),
        ),
        crs="EPSG:3857",
    )

    report = left.spatial_join(
        right,
        lambda first, second: abs(first.centroid[0] - second.centroid[0]) < 5,
        "centroid_distance_lt_5/v1",
    )

    assert report.crs == "EPSG:3857"
    assert report.comparison_count == 4
    assert report.match_count == 1
    assert report.unmatched_left_count == 1
    assert report.unmatched_right_count == 1
    assert report.matches == (geospatial.SpatialJoinMatch("left-a", "right-a"),)
    assert report.left_dataset_hash == left.content_hash
    assert report.right_dataset_hash == right.content_hash
    assert report.to_dict()["predicate_name"] == "centroid_distance_lt_5/v1"


def test_geospatial_spatial_join_rejects_crs_mismatch_and_unbounded_work():
    """Joins fail closed instead of comparing incompatible or huge inputs."""
    feature = geospatial.GeoFeature(
        "feature", {"type": "Point", "coordinates": [0, 0]}
    )
    left = geospatial.GeospatialDataset((feature,), crs="EPSG:4326")
    right = geospatial.GeospatialDataset((feature,), crs="EPSG:3857")

    with pytest.raises(
        geospatial.GeospatialValidationError, match="matching CRS"
    ):
        left.spatial_join(right, lambda first, second: True, "always/v1")

    with pytest.raises(
        geospatial.GeospatialValidationError,
        match="comparison limit",
    ):
        left.spatial_join(
            geospatial.GeospatialDataset(
                tuple(
                    geospatial.GeoFeature(
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
    feature = geospatial.GeoFeature(
        "feature", {"type": "Point", "coordinates": [0, 0]}
    )
    dataset = geospatial.GeospatialDataset((feature,), crs="EPSG:4326")

    with pytest.raises(
        geospatial.GeospatialValidationError,
        match="must return a boolean",
    ):
        dataset.spatial_join(dataset, lambda first, second: "yes", "bad/v1")


def test_geospatial_reprojection_records_source_and_target_identity():
    """An explicit transformer preserves metadata and binds both snapshots."""
    observation_time = datetime(2026, 1, 1, tzinfo=UTC)
    dataset = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "feature",
                {"type": "Point", "coordinates": [1, 2]},
                {"kind": "park"},
                label="open",
                observation_time=observation_time,
            ),
        ),
        crs="EPSG:4326",
    )

    def transform(
        feature: geospatial.GeoFeature, source_crs: str, target_crs: str
    ) -> geospatial.GeoFeature:
        assert source_crs == "EPSG:4326"
        assert target_crs == "EPSG:3857"
        return geospatial.GeoFeature(
            feature.feature_id,
            {
                "type": "Point",
                "coordinates": [
                    feature.centroid[0] * 2,
                    feature.centroid[1] * 2,
                ],
            },
            dict(feature.properties),
            feature.label,
            feature.observation_time,
        )

    result = dataset.reproject("EPSG:3857", transform, "scale/v1")

    assert result.dataset.crs == "EPSG:3857"
    assert result.dataset.features[0].geometry["coordinates"] == [2.0, 4.0]
    assert result.report.source_dataset_hash == dataset.content_hash
    assert result.report.target_dataset_hash == result.dataset.content_hash
    assert result.report.transformer_name == "scale/v1"
    assert result.to_dict()["report"]["feature_count"] == 1


def test_geospatial_reprojection_rejects_metadata_changes():
    """A transformer cannot silently alter labels or feature ownership."""
    feature = geospatial.GeoFeature(
        "feature",
        {"type": "Point", "coordinates": [0, 0]},
        label="open",
    )
    dataset = geospatial.GeospatialDataset((feature,), crs="EPSG:4326")

    def changes_label(
        value: geospatial.GeoFeature, source_crs: str, target_crs: str
    ) -> geospatial.GeoFeature:
        return geospatial.GeoFeature(
            value.feature_id,
            value.geometry,
            dict(value.properties),
            label="closed",
        )

    with pytest.raises(
        geospatial.GeospatialValidationError,
        match="changed feature metadata",
    ):
        dataset.reproject("EPSG:3857", changes_label, "bad/v1")


def test_geospatial_rejects_invalid_geometry():
    """Malformed geometry is rejected before it reaches feature preparation."""
    with pytest.raises(geospatial.GeospatialValidationError):
        geospatial.GeoFeature(
            "broken", {"type": "Point", "coordinates": [float("nan"), 1]}
        )


def test_geospatial_curation_blocks_missing_crs():
    """Spatial coordinates cannot enter modelling without a reference CRS."""
    dataset = geospatial.GeospatialDataset.from_geojson(
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

    assert report.release_ready is False
    assert report.unresolved_blocking_findings[0].finding_id == (
        "quality.geospatial-missing-crs"
    )


def test_geospatial_curation_reports_mixed_label_state():
    """Spatial curation exposes mixed labels before spatial modelling."""
    dataset = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "a",
                {"type": "Point", "coordinates": [0, 0]},
                label="park",
            ),
            geospatial.GeoFeature(
                "b", {"type": "Point", "coordinates": [1, 1]}
            ),
        ),
        crs="EPSG:4326",
    )

    report = dataset.curate()

    assert report.profile["label_counts"] == {
        "str:park": 1,
        "unlabeled": 1,
    }
    assert [finding.finding_id for finding in report.risk_findings] == [
        "risk.geospatial-mixed-label-state",
    ]


def test_geospatial_curation_records_deterministic_spatial_autocorrelation():
    """Numeric labels expose bounded spatial-dependence evidence."""
    dataset = geospatial.GeospatialDataset(
        tuple(
            geospatial.GeoFeature(
                f"feature-{index}",
                {"type": "Point", "coordinates": [float(index), 0]},
                label=float(index // 2),
            )
            for index in range(5)
        ),
        crs="EPSG:3857",
    )

    first = dataset.spatial_autocorrelation(
        neighbor_count=1, permutations=19, seed=11
    )
    second = dataset.spatial_autocorrelation(
        neighbor_count=1, permutations=19, seed=11
    )
    report = dataset.curate()

    assert first == second
    assert first.edge_count == 4
    assert first.statistic > 0
    assert 0.0 <= first.p_value <= 1.0
    evidence = report.profile["spatial_autocorrelation"]
    assert evidence["feature_count"] == 5
    assert evidence["neighbor_count"] == 4
    assert evidence["permutation_count"] == 99
    assert evidence["p_value"] is not None


def test_spatial_autocorrelation_rejects_non_numeric_and_constant_values():
    """Spatial dependence requires an estimable numeric variable."""
    features = tuple(
        geospatial.GeoFeature(
            f"feature-{index}",
            {"type": "Point", "coordinates": [float(index), 0]},
            label="same",
        )
        for index in range(3)
    )
    dataset = geospatial.GeospatialDataset(features, crs="EPSG:3857")

    with pytest.raises(geospatial.GeospatialValidationError, match="numeric"):
        dataset.spatial_autocorrelation()

    with pytest.raises(
        geospatial.GeospatialValidationError, match="more than one unique"
    ):
        dataset.spatial_autocorrelation(values=[1.0, 1.0, 1.0])


def test_geospatial_spatial_split_keeps_blocks_together():
    """Spatial blocks remain isolated across train and evaluation data."""
    features = tuple(
        geospatial.GeoFeature(
            f"feature-{index}",
            {"type": "Point", "coordinates": [float(index * 10), 0]},
            label="class",
        )
        for index in range(6)
    )
    dataset = geospatial.GeospatialDataset(features, crs="EPSG:3857")

    first = dataset.split_spatial(
        block_size=5, validation_fraction=0.25, test_fraction=0.25, seed=3
    )
    second = dataset.split_spatial(
        block_size=5, validation_fraction=0.25, test_fraction=0.25, seed=3
    )

    block_assignments = first.block_ids()
    blocks = [set(values) for values in block_assignments.values()]
    assert len(set().union(*blocks)) == 6
    assert first == second
    assert first.block_size == 5


def test_geospatial_temporal_split_normalizes_time_and_curates_coverage():
    """Observation times support UTC-safe future evaluation evidence."""
    features = tuple(
        geospatial.GeoFeature(
            f"feature-{index}",
            {"type": "Point", "coordinates": [float(index), 0]},
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
    dataset = geospatial.GeospatialDataset(features, crs="EPSG:3857")

    split = dataset.split_temporal("2026-01-02T00:00:00+00:00")
    report = dataset.curate()

    assert dataset.features[0].observation_time == datetime(
        2026, 1, 1, tzinfo=UTC
    )
    assert len(split.train) == 2
    assert len(split.future) == 1
    assert not split.has_future_leakage
    assert split.cutoff.tzinfo == UTC
    assert report.profile["temporal"]["status"] == "complete"


def test_geospatial_temporal_state_fails_closed_when_incomplete():
    """Missing or malformed observation times remain explicit before splitting."""
    mixed = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "timed",
                {"type": "Point", "coordinates": [0, 0]},
                observation_time=datetime(2026, 1, 1, tzinfo=UTC),
            ),
            geospatial.GeoFeature(
                "untimed", {"type": "Point", "coordinates": [1, 1]}
            ),
        ),
        crs="EPSG:3857",
    )

    report = mixed.curate()
    quality_ids = {finding.finding_id for finding in report.quality_findings}

    assert report.profile["temporal"] == {
        "status": "mixed",
        "timed_feature_count": 1,
        "untimed_feature_count": 1,
    }
    assert "quality.geospatial-mixed-temporal-state" in quality_ids
    with pytest.raises(
        geospatial.GeospatialTemporalError, match="every feature"
    ):
        mixed.split_temporal("2026-01-02T00:00:00Z")
    with pytest.raises(geospatial.GeospatialTemporalError, match="timezone"):
        geospatial.canonical_observation_time("2026-01-01T00:00:00")


def test_geospatial_feature_preparation_preserves_crs_units():
    """Geometry envelopes are prepared only after CRS provenance exists."""
    dataset = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "point", {"type": "Point", "coordinates": [2, 1]}
            ),
            geospatial.GeoFeature(
                "line",
                {"type": "LineString", "coordinates": [[-3, 4], [5, -2]]},
            ),
        ),
        crs="EPSG:4326",
    )

    prepared = dataset.prepare_features()

    assert prepared.shape == (2, 9)
    assert prepared.crs == "EPSG:4326"
    assert prepared.feature_ids == ("point", "line")
    assert prepared.to_numpy()[1, 4:6].tolist() == [8.0, 6.0]


def test_geospatial_raw_axis_mask_preserves_nested_geometry_structure():
    """Raw-coordinate masking preserves geometry type and untouched axes."""
    feature = geospatial.GeoFeature(
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

    masked = feature.masked_geometry_axis(0, 0.5)

    assert masked.geometry["type"] == "GeometryCollection"
    geometries = masked.geometry["geometries"]
    assert geometries[0]["coordinates"] == [0.5, 1.0]
    assert geometries[1]["coordinates"] == [[0.5, 4.0], [0.5, -2.0]]
    assert masked.geometry_hash != feature.geometry_hash


def test_geospatial_raw_axis_mask_rejects_invalid_axis():
    """Only x and y axes may be perturbed by the raw evidence policy."""
    feature = geospatial.GeoFeature(
        "point", {"type": "Point", "coordinates": [2, 1]}
    )

    with pytest.raises(ValueError, match="axis"):
        feature.masked_geometry_axis(2, 0.0)


def test_geospatial_feature_preparation_rejects_missing_crs():
    """Coordinates cannot become model features without a reference CRS."""
    dataset = geospatial.GeospatialDataset(
        (
            geospatial.GeoFeature(
                "point", {"type": "Point", "coordinates": [0, 0]}
            ),
        )
    )

    with pytest.raises(geospatial.GeospatialValidationError, match="CRS"):
        dataset.prepare_features()


def test_raster_identity_requires_spatial_metadata():
    """Raster bytes are not useful without dimensions and CRS provenance."""
    payload = b"raster"
    raster = geospatial.RasterSample(
        source="memory://raster",
        encoded_bytes=payload,
        file_hash=hashlib.sha256(payload).hexdigest(),
        width=10,
        height=20,
        bands=3,
        dtype="uint8",
        crs="EPSG:3857",
        transform=(1, 0, 0, 0, -1, 0),
    )

    assert raster.byte_size == len(payload)
    assert raster.crs == "EPSG:3857"


def test_raster_decoder_is_bounded_and_shape_checked():
    """Raster decoding preserves metadata and enforces the pixel budget."""
    payload = b"raster"
    raster = geospatial.RasterSample(
        source="memory://raster",
        encoded_bytes=payload,
        file_hash=hashlib.sha256(payload).hexdigest(),
        width=2,
        height=3,
        bands=2,
        dtype="uint8",
        crs="EPSG:3857",
        transform=(1, 0, 0, 0, -1, 0),
        nodata_value=0,
    )

    pixels = raster.decode(
        lambda value: np.full((3, 2, 2), value[0], dtype=np.uint8),
        max_pixels=12,
    )

    assert pixels.shape == (3, 2, 2)
    assert not pixels.flags.writeable
    assert (
        raster.measured_nodata_fraction(
            lambda value: np.full((3, 2, 2), value[0], dtype=np.uint8)
        )
        == 0.0
    )

    with pytest.raises(geospatial.GeospatialValidationError, match="limit"):
        raster.decode(
            lambda value: np.zeros((3, 2, 2), dtype=np.uint8),
            max_pixels=11,
        )


def test_raster_decoder_rejects_metadata_mismatch():
    """A caller-owned decoder cannot override declared raster dimensions."""
    payload = b"raster"
    raster = geospatial.RasterSample(
        source="memory://raster",
        encoded_bytes=payload,
        file_hash=hashlib.sha256(payload).hexdigest(),
        width=2,
        height=3,
        bands=1,
        dtype="uint8",
        crs="EPSG:3857",
        transform=(1, 0, 0, 0, -1, 0),
    )

    with pytest.raises(
        geospatial.GeospatialValidationError, match="dimensions"
    ):
        raster.decode(lambda value: np.zeros((2, 2), dtype=np.uint8))
