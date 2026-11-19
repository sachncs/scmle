"""Exact refusals of a video chronology."""

from datetime import datetime
from datetime import UTC

import pytest

from scmle.contracts import errors
from scmle.data.modalities.video import evidence
from scmle.data.modalities.video import sample


def clip(source):
    metadata = sample.Metadata(
        format="mp4",
        width=4,
        height=2,
        frames=10,
        rate=5.0,
        codec="h264",
    )
    return sample.Sample.decode(source.encode(), metadata, source=source)


def test_a_future_clip_without_a_time_is_refused_exactly():
    with pytest.raises(errors.InputError) as caught:
        evidence.Chronology(datetime(2020, 1, 1, tzinfo=UTC), (), (clip("a"),))

    assert str(caught.value) == "future videos must contain observation times"
