"""The audio kind: labelled clips on the shared stages (`media.Media`).

A data block with `modality: audio` names a manifest table with a `path`
column next to the target and any other columns, or a directory whose
subdirectories name the labels of the clips inside them. Each clip is
described in bounded chunks (`spectrum.describe`) into numeric features.
Clips that are copies or near-copies of each other (their band envelopes
within `spectrum.RADIUS` over the bands above the floor) share a cohort,
which the shared splitter keeps on one side of every split (MM-10).
"""

import pathlib

import numpy
import pandas

from scmle.data.modalities.audio import spectrum
from scmle.pipeline.kinds import media

SECONDS = 0.05
"""Seconds to describe one clip, an upper bound set by observation: a
60-second 48 kHz stereo clip took 0.04 s here, and clips are mostly
shorter."""

PAIR = 2e-8
"""Seconds to compare two envelopes, set by observation (10,000 clips,
50 million pairs, took about 0.8 s here)."""


class Audio(media.Media):
    """Labelled audio clips, described by `spectrum.describe`."""

    name = "audio"
    suffixes = frozenset(
        {".wav", ".wave", ".flac", ".ogg", ".oga", ".aif", ".aiff"}
    )
    radius = spectrum.RADIUS
    seconds = SECONDS
    pair = PAIR
    evidence = "tests/unit/test_kind_audio.py"

    def limits(self) -> tuple[str, ...]:
        """The boundaries the audio capability states.

        Returns:
            Plain statements of what the kind does not cover.
        """
        return (
            (
                "clips are described into spectral features before any model "
                "sees them; no learned audio representation"
            ),
            "FLAC, Ogg and AIFF need the optional soundfile library",
            "a served model predicts from described rows (`describe`)",
        )

    def trace(
        self,
        path: pathlib.Path,
        row: pandas.Series,  # noqa: ARG002 -- media.Media.trace; a clip needs no declaration
    ) -> spectrum.Trace:
        """Describes one clip in bounded chunks.

        Args:
            path: The clip.
            row: Its manifest row; not consulted.

        Returns:
            Its features, digest and envelope (`spectrum.describe`).
        """
        return spectrum.describe(path)

    def mask(self, envelopes: numpy.ndarray) -> numpy.ndarray:
        """Marks the cells above the floor.

        Args:
            envelopes: Rows by cells, in dB below each clip's loudest.

        Returns:
            True where a cell is above `spectrum.FLOOR`.
        """
        return envelopes > -spectrum.FLOOR


AUDIO = Audio()
"""The audio kind."""
