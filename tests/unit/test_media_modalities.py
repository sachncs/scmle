"""Tests for dependency-light audio, video, and geospatial foundations."""

import hashlib
import io
import wave

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


def test_geospatial_rejects_invalid_geometry():
    """Malformed geometry is rejected before it reaches feature preparation."""
    with pytest.raises(geospatial.GeospatialValidationError):
        geospatial.GeoFeature(
            "broken", {"type": "Point", "coordinates": [float("nan"), 1]}
        )


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
