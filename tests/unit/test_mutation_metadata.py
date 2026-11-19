"""The audio transcript's refusals, word for word, and its boundaries."""

import math

import pytest

from scmle.contracts import errors
from scmle.data.modalities.audio import metadata


def transcript(**changes) -> metadata.Transcript:
    values = {
        "text": "hello there",
        "source": "clip.wav",
        "provider": "asr",
        "version": "1.0",
        "language": "en",
    }
    values.update(changes)
    return metadata.Transcript(**values)


def segment(start: float, end: float) -> metadata.Segment:
    return metadata.Segment(start, end, "word")


def test_a_blank_text_is_refused_by_its_lower_case_name() -> None:
    with pytest.raises(
        errors.InputError, match=r"^transcript text must not be empty$"
    ):
        transcript(text=" ")


def test_confidence_takes_both_ends_of_the_unit_interval() -> None:
    assert transcript(confidence=0.0).confidence == 0.0
    assert transcript(confidence=1.0).confidence == 1.0
    assert transcript(confidence=0.5).confidence == 0.5


def test_confidence_outside_the_unit_interval_or_not_finite_is_refused() -> (
    None
):
    for value in (-0.1, 1.1, math.nan, math.inf):
        with pytest.raises(
            errors.InputError,
            match=r"^transcript confidence must be in \[0, 1\]$",
        ):
            transcript(confidence=value)


def test_ordered_segments_are_kept_and_unordered_ones_refused() -> None:
    kept = transcript(segments=(segment(0.0, 1.0), segment(1.0, 2.0)))
    assert len(kept.segments) == 2
    assert transcript(segments=(segment(0.0, 1.0),)).segments
    with pytest.raises(
        errors.InputError,
        match=r"^transcript segments must be ordered by start time$",
    ):
        transcript(segments=(segment(1.0, 2.0), segment(0.0, 1.0)))


def test_segments_that_are_not_a_tuple_of_segments_are_refused() -> None:
    with pytest.raises(errors.KindError, match=r"^segments must be a tuple$"):
        transcript(segments=[segment(0.0, 1.0)])
    with pytest.raises(
        errors.KindError,
        match=r"^segments must contain transcript segments$",
    ):
        transcript(segments=("word",))
