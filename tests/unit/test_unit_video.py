"""Behavioural tests for the video modality."""

from datetime import datetime
from datetime import UTC
import hashlib
import pathlib

import numpy
import pytest

from scmle.contracts import errors
from scmle.data.modalities.video import dataset
from scmle.data.modalities.video import evidence
from scmle.data.modalities.video import frames
from scmle.data.modalities.video import sample
from scmle.data.modalities.video import splits
from scmle.data.modalities.video import survey


def meta(**kwargs):
    arguments = {
        "format": "mp4",
        "width": 4,
        "height": 2,
        "frames": 10,
        "rate": 5.0,
        "codec": "h264",
    } | kwargs
    return sample.Metadata(**arguments)


def clip(source, payload=None, metadata=None, **kwargs):
    return sample.Sample.decode(
        payload or source.encode(),
        metadata or meta(),
        source=source,
        **kwargs,
    )


def at(day):
    return datetime(2020, 1, day, tzinfo=UTC)


def test_metadata_reports_duration_and_synchrony():
    assert meta().seconds == 2.0
    assert meta().synchrony == "not_applicable"
    assert meta(tracks=1).synchrony == "unknown"
    synced = meta(
        tracks=1,
        audio_duration_seconds=2.0,
        offset=-0.1,
        tolerance=0.1,
    )
    assert synced.synchrony == "within_tolerance"
    drifted = meta(
        tracks=1,
        audio_duration_seconds=2.0,
        offset=0.2,
        tolerance=0.1,
    )
    assert drifted.synchrony == "out_of_tolerance"
    partial = meta(tracks=1, audio_duration_seconds=2.0, offset=0.0)
    assert partial.synchrony == "unknown"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"format": " "}, "format and codec"),
        ({"codec": ""}, "format and codec"),
        ({"width": 0}, "dimensions"),
        ({"height": 0}, "dimensions"),
        ({"frames": 0}, "frame timing"),
        ({"rate": 0.0}, "frame timing"),
        ({"tracks": -1}, "audio_tracks"),
        ({"offset": 0.0}, "requires at least one audio track"),
        (
            {"tracks": 1, "audio_duration_seconds": 0.0},
            "audio_duration_seconds",
        ),
        (
            {"tracks": 1, "audio_duration_seconds": float("inf")},
            "audio_duration_seconds",
        ),
        (
            {"tracks": 1, "offset": float("nan")},
            "sync_offset_seconds",
        ),
        (
            {"tracks": 1, "tolerance": -0.1},
            "sync_tolerance_seconds",
        ),
    ],
)
def test_metadata_rejects_invalid_values(kwargs, message):
    with pytest.raises(errors.InputError, match=message):
        meta(**kwargs)


def test_a_zero_tolerance_is_allowed():
    exact = meta(
        tracks=1,
        audio_duration_seconds=1.0,
        offset=0.0,
        tolerance=0.0,
    )
    assert exact.synchrony == "within_tolerance"


def test_canonical_source_and_time():
    assert sample.identify(" ") == "memory://"
    assert sample.identify("s3://v") == "s3://v"
    assert sample.identify("v.mp4") == (
        pathlib.Path("v.mp4").resolve().as_uri()
    )
    assert sample.timestamp("2020-01-01T00:00:00Z") == at(1)
    with pytest.raises(sample.TemporalError) as raised:
        sample.timestamp("2020-01-01T00:00:00")
    assert raised.value.details["code"] == "timezone_required"
    with pytest.raises(sample.TemporalError) as raised:
        sample.timestamp("x")
    assert raised.value.details["code"] == "invalid_observation_time"
    with pytest.raises(sample.TemporalError, match="datetime or text"):
        sample.timestamp(1)


def test_from_bytes_builds_a_sample_and_enforces_limits():
    built = clip("mem://a", b"abc", observation_time="2020-01-01T00:00:00Z")
    assert built.size == 3
    assert built.file_hash == hashlib.sha256(b"abc").hexdigest()
    assert built.time == at(1)
    assert clip("mem://a", b"abc", max_bytes=3).size == 3
    with pytest.raises(sample.VideoError, match="byte limit"):
        clip("mem://a", b"abc", max_bytes=2)
    with pytest.raises(sample.VideoError, match="byte limit"):
        clip("mem://a", b"abc", max_bytes=0)
    with pytest.raises(sample.VideoError, match="empty"):
        sample.Sample.decode(b"", meta())


def test_a_sample_validates_identity():
    digest = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(errors.InputError, match="source"):
        sample.Sample(" ", b"x", digest, meta())
    with pytest.raises(errors.InputError, match="group_id"):
        sample.Sample("s", b"x", digest, meta(), group=" ")
    with pytest.raises(errors.InputError, match="SHA-256 hex"):
        sample.Sample("s", b"x", digest.upper(), meta())
    with pytest.raises(errors.InputError, match="does not match"):
        sample.Sample("s", b"x", "0" * 64, meta())
    with pytest.raises(sample.VideoError, match="empty"):
        sample.Sample("s", b"", digest, meta())
    timed = sample.Sample("s", b"x", digest, meta(), time=at(1))
    assert timed.time == at(1)


def test_profile_copies_metadata():
    built = clip("mem://a", b"abcd", label=1)
    profile = built.profile()
    assert (profile.width, profile.height, profile.size) == (4, 2, 4)
    assert profile.duration == 2.0
    assert profile.synchrony == "not_applicable"
    assert profile.codec == "h264"


def test_a_dataset_rejects_empty_and_duplicate_sources():
    with pytest.raises(sample.VideoError, match="at least one"):
        dataset.Dataset(())
    one = clip("mem://a")
    with pytest.raises(sample.VideoError, match="duplicate"):
        dataset.Dataset((one, one))


def test_a_dataset_hash_binds_time_and_order():
    first = clip("mem://a", observation_time=at(1))
    second = clip("mem://b")
    data = dataset.Dataset((first, second))
    expected = hashlib.sha256(
        (
            f"mem://a\0{first.file_hash}\0{at(1).isoformat()}\n"
            f"mem://b\0{second.file_hash}"
        ).encode()
    ).hexdigest()
    assert data.digest == expected
    assert dataset.Dataset((second, first)).digest != expected


def test_numerize_lists_metadata_features():
    # Enforces website/src/pages/docs/data.astro:355.
    data = dataset.Dataset((clip("mem://a", b"x" * 20),))
    features = data.numerize()
    assert features.values == (
        (2.0, 4.0, 2.0, 8.0, 2.0, 10.0, 5.0, 0.0, 20.0, 2.0),
    )
    assert features.names[-1] == "bytes_per_frame"
    assert features.shape == (1, 10)
    infinite = dataset.Dataset(
        (clip("mem://b", metadata=meta(rate=float("inf"))),)
    )
    with pytest.raises(sample.VideoError, match="finite"):
        infinite.numerize()


@pytest.mark.parametrize(
    ("count", "limit", "expected"),
    [(10, 4, (0, 3, 6, 9)), (3, 8, (0, 1, 2)), (5, 1, (0,)), (1, 2, (0,))],
)
def test_stride_spaces_frame_indices_evenly(count, limit, expected):
    # Enforces website/src/pages/docs/api.astro:354.
    built = clip("mem://a", metadata=meta(frames=count))
    assert frames.Frames.stride(built, limit) == expected


def test_stride_needs_a_positive_limit():
    with pytest.raises(errors.InputError, match="max_frames"):
        frames.Frames.stride(clip("mem://a"), 0)


def test_summarize_records_frame_statistics_and_indices():
    # Enforces website/src/pages/docs/api.astro:354.
    data = dataset.Dataset((clip("mem://a", metadata=meta(frames=3)),))
    calls = []

    def decoder(raw, index):
        calls.append(index)
        return numpy.array([[index, index + 2.0]])

    features = data.summarize(decoder, max_frames=4)
    assert calls == [0, 1, 2]
    assert features.frames == ((0, 1, 2),)
    assert features.names[:5] == (
        "frame[0].mean",
        "frame[0].std",
        "frame[0].min",
        "frame[0].max",
        "frame[0].observed",
    )
    assert len(features.names) == 20
    row = features.values[0]
    assert row[:5] == (1.0, 1.0, 0.0, 2.0, 1.0)
    assert row[5:10] == (2.0, 1.0, 1.0, 3.0, 1.0)
    assert row[15:] == (0.0, 0.0, 0.0, 0.0, 0.0)


def test_summarize_validates_its_arguments():
    data = dataset.Dataset((clip("mem://a"),))
    with pytest.raises(errors.KindError, match="callable"):
        data.summarize(None)
    with pytest.raises(errors.InputError, match="frame bounds"):
        data.summarize(lambda raw, index: numpy.zeros((1, 1)), max_frames=0)
    with pytest.raises(errors.InputError, match="frame bounds"):
        data.summarize(
            lambda raw, index: numpy.zeros((1, 1)), max_frame_pixels=0
        )


@pytest.mark.parametrize(
    ("frame", "code"),
    [
        (numpy.zeros(3), "invalid_frame_shape"),
        (numpy.zeros((1, 1, 1, 1)), "invalid_frame_shape"),
        (numpy.zeros((0, 2)), "frame_pixel_limit"),
        (numpy.zeros((2, 4, 2)), "frame_pixel_limit"),
        (numpy.zeros((3, 1)), "frame_dimension_mismatch"),
        (numpy.zeros((1, 5)), "frame_dimension_mismatch"),
        (numpy.full((1, 1), numpy.nan), "non_finite_frame"),
    ],
)
def test_measure_rejects_bad_frames(frame, code):
    # Enforces website/src/pages/docs/api.astro:354.
    with pytest.raises(sample.VideoError) as raised:
        frames.measure(clip("mem://a"), lambda raw, index: frame, 7, 8)
    assert raised.value.details["code"] == code
    assert raised.value.details["frame_index"] == 7


def test_measure_accepts_frames_at_the_limits():
    full = frames.measure(
        clip("mem://a"), lambda raw, index: numpy.ones((2, 4)), 0, 8
    )
    assert full.dtype == numpy.float64
    assert full.shape == (2, 4)

    def failing(raw, index):
        raise RuntimeError

    with pytest.raises(sample.VideoError) as raised:
        frames.measure(clip("mem://a"), failing, 0, 8)
    assert raised.value.details["code"] == "frame_decode_failed"


def test_features_validate_their_shape():
    with pytest.raises(errors.InputError, match="rows must match"):
        evidence.Features(("a",), ("x",), ())
    with pytest.raises(errors.InputError, match="columns must match"):
        evidence.Features(("a",), ("x",), ((1.0, 2.0),))
    with pytest.raises(errors.InputError, match="finite"):
        evidence.Features(("a",), ("x",), ((numpy.nan,),))
    with pytest.raises(errors.InputError, match="sampled frame metadata"):
        evidence.Features(("a",), ("x",), ((1.0,),), ((0,), (1,)))
    with pytest.raises(errors.InputError, match="not be negative"):
        evidence.Features(("a",), ("x",), ((1.0,),), ((-1,),))
    built = evidence.Features(["a"], ["x"], [[1]], [[2.0]])
    assert built.values == ((1.0,),)
    assert built.frames == ((2,),)
    assert built.sources == ("a",)
    matrix = built.materialize()
    matrix[0, 0] = 5.0
    assert built.values == ((1.0,),)


def test_a_chronology_validates_and_detects_lookahead():
    early = clip("mem://a", observation_time=at(1))
    late = clip("mem://b", observation_time=at(3))
    assert not evidence.Chronology(at(2), (early,), (late,)).lookahead
    assert evidence.Chronology(at(2), (late,), ()).lookahead
    with pytest.raises(errors.InputError, match="cutoff"):
        evidence.Chronology(datetime.fromisoformat("2020-01-01"), (), ())
    with pytest.raises(errors.InputError, match="train videos"):
        evidence.Chronology(at(1), (clip("mem://c"),), ())
    with pytest.raises(errors.InputError, match="future videos"):
        evidence.Chronology(at(1), (), (clip("mem://c"),))


def test_split_keeps_groups_whole():
    clips = tuple(clip(f"mem://{i}", group_id=f"g{i % 5}") for i in range(10))
    data = dataset.Dataset(clips)
    split = data.split(0.2, 0.2, seed=2)
    cohorts = split.cohorts()
    assert [len(cohorts[k]) for k in ("train", "validation", "test")] == [
        3,
        1,
        1,
    ]
    assert len(split.train) == 6
    assert isinstance(data, splits.Splits)
    with pytest.raises(errors.InputError, match="test_fraction"):
        data.split(0.0, -0.1)


def test_cutover_puts_the_cutoff_in_training():
    data = dataset.Dataset(
        (
            clip("mem://a", observation_time=at(1)),
            clip("mem://b", observation_time=at(2)),
        )
    )
    chronology = data.cutover(at(1))
    assert [s.source for s in chronology.train] == ["mem://a"]
    assert [s.source for s in chronology.future] == ["mem://b"]
    mixed = dataset.Dataset(
        (clip("mem://a", observation_time=at(1)), clip("mem://c"))
    )
    with pytest.raises(sample.TemporalError, match="every video"):
        mixed.cutover(at(1))


def test_synchronization_findings_are_ordered():
    drifted = clip(
        "mem://a",
        metadata=meta(
            tracks=1,
            audio_duration_seconds=1.0,
            offset=1.0,
            tolerance=0.0,
        ),
    )
    unknown = clip("mem://b", metadata=meta(tracks=1))
    findings = survey.synchronization([drifted, unknown, clip("mem://c")])
    assert [f.identifier for f in findings] == [
        "quality.video-audio-sync-out-of-tolerance",
        "quality.video-audio-sync-unknown",
    ]
    assert findings[0].blocking
    assert findings[0].scope == {"sources": ["mem://a"]}
    assert findings[1].evidence == {"count": 1}
    assert survey.synchronization([clip("mem://c")]) == []


def test_drift_flags_encoding_and_dimensions():
    profiles = [clip("mem://a").profile()]
    assert survey.drift(profiles, ["mp4"], ["h264"], [(4, 2)]) == []
    codec = survey.drift(profiles, ["mp4"], ["a", "b"], [(4, 2)])
    assert [f.identifier for f in codec] == ["quality.video-encoding-drift"]
    sizes = survey.drift(profiles, ["mp4"], ["h264"], [(1, 1), (2, 2)])
    assert sizes[0].evidence == {"dimensions": [[1, 1], [2, 2]]}


def test_timeline_reports_temporal_completeness():
    # Enforces website/src/pages/docs/api.astro:356.
    # Enforces website/src/pages/docs/data.astro:356.
    first = clip("mem://a", observation_time=at(2))
    second = clip("mem://b", observation_time=at(1))
    assert survey.timeline([first, second]) == {
        "status": "complete",
        "observation_time_min": at(1).isoformat(),
        "observation_time_max": at(2).isoformat(),
    }
    assert survey.timeline([first, clip("mem://c")]) == {
        "status": "mixed",
        "timed_video_count": 1,
        "untimed_video_count": 1,
    }
    assert survey.timeline([clip("mem://c")]) == {"status": "absent"}


def test_risks_flag_mixed_and_conflicting_labels():
    assert survey.risks("h", {"str:a": 1}, ()) == []
    assert survey.risks("h", {"unlabeled": 1}, ()) == []
    names = [
        f.identifier
        for f in survey.risks("h", {"str:a": 1, "unlabeled": 1}, [{"x": 1}])
    ]
    assert names == [
        "risk.video-mixed-label-state",
        "risk.video-conflicting-duplicate-labels",
    ]


def test_curate_summarises_the_collection():
    data = dataset.Dataset(
        (
            clip("mem://a", b"same", label="x", observation_time=at(1)),
            clip(
                "mem://b",
                b"same",
                label="y",
                metadata=meta(frames=20, width=8),
            ),
        )
    )
    report = data.curate()
    assert report.modality == "video"
    assert report.samples == 2
    assert report.profile["dimensions"] == [[4, 2], [8, 2]]
    assert report.profile["duration_min_seconds"] == 2.0
    assert report.profile["duration_max_seconds"] == 4.0
    assert report.profile["duration_mean_seconds"] == 3.0
    assert report.profile["total_bytes"] == 8
    assert report.profile["synchronization_statuses"] == ["not_applicable"]
    assert [f.identifier for f in report.findings] == [
        "quality.video-dimension-drift",
        "quality.video-mixed-temporal-state",
    ]
    assert [f.identifier for f in report.risk_findings] == [
        "risk.video-conflicting-duplicate-labels"
    ]
    assert len(data.profiles()) == 2
