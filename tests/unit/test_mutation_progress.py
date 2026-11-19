"""Exact errors and timezone handling of streaming progress."""

import pandas
import pytest

from scmle.contracts import errors
from scmle.data.readers.streaming import failure
from scmle.data.readers.streaming import progress

DIGEST = "a" * 64


def checkpoint(offset=2, **changes):
    fields = {
        "source": "src",
        "digests": {"snapshot": DIGEST},
        "offset": offset,
    }
    fields.update(changes)
    return progress.Checkpoint(**fields)


def test_advance_compares_timestamps_with_different_offsets_in_utc():
    frame = pandas.DataFrame(
        {"at": ["2024-01-01T10:00:00+01:00", "2024-01-01T10:30:00+02:00"]}
    )

    latest = progress.advance(frame, "at", "", "src")

    assert latest == "2024-01-01T09:00:00+00:00"


def test_advance_reports_a_missing_column_with_its_source():
    with pytest.raises(failure.StreamingError) as caught:
        progress.advance(pandas.DataFrame({"x": [1]}), "at", "", "src")

    assert caught.value.source_identity == "src"
    assert caught.value.details == {
        "code": "watermark_column_missing",
        "source_identity": "src",
        "column": "at",
    }


def test_advance_reports_invalid_timestamps_exactly():
    frame = pandas.DataFrame({"at": ["not a time"]})

    with pytest.raises(failure.StreamingError) as caught:
        progress.advance(frame, "at", "", "src")

    assert caught.value.args[0] == (
        "watermark column 'at' contains invalid timestamps"
    )
    assert caught.value.source_identity == "src"
    assert caught.value.details == {
        "code": "invalid_watermark",
        "source_identity": "src",
        "column": "at",
    }


def test_advance_names_a_malformed_current_watermark():
    frame = pandas.DataFrame({"at": ["2024-01-01T00:00:00+00:00"]})

    with pytest.raises(errors.InputError) as caught:
        progress.advance(frame, "at", "yesterday", "src")

    assert str(caught.value) == "watermark must be ISO-8601"


def test_checkpoint_names_a_malformed_watermark():
    with pytest.raises(errors.InputError) as caught:
        checkpoint(watermark="yesterday")

    assert str(caught.value) == "watermark must be ISO-8601"


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"snapshot_sha256": "b" * 64},
            "batch and checkpoint snapshots differ",
        ),
        ({"start_offset": -1}, "start_offset must be non-negative"),
    ],
)
def test_batch_errors_are_exact(changes, message):
    fields = {
        "source_identity": "src",
        "snapshot_sha256": DIGEST,
        "start_offset": 0,
        "frame": pandas.DataFrame({"x": [1, 2]}),
        "next_checkpoint": checkpoint(),
    }
    fields.update(changes)

    with pytest.raises(errors.InputError) as caught:
        progress.Batch(**fields)

    assert str(caught.value) == message
