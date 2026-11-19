"""The video kind: video files read as descriptor rows (`kinds.image.Image`).

A video is probed and `stream.COUNT` evenly spaced frames are decoded by
FFmpeg into a pipe (`video.stream`), so the encoded file never enters
memory (MM-6). Its row describes the mean sampled frame as an image is
described, so near-duplicate videos are linked and split together as
images are (MM-10), plus the brightness of each sampled frame in time
order, the motion between them and the timing. A declared time column
gives a time-ordered split with validation folds, each near-duplicate
island held out whole (MM-3); rows are in path order (MM-12).
"""

from collections.abc import Iterable
import pathlib
from typing import Any

import pandas

from scmle.data.modalities.video import stream
from scmle.pipeline.kinds import image
from scmle.pipeline.kinds import media


class Video(media.Media):
    """Video files decoded with the FFmpeg programs.

    The near-copy pairs are the image kind's, read from the mean frame's
    descriptor columns.
    """

    name = "video"
    suffixes = stream.SUFFIXES
    extras = ("core",)
    evidence = "tests/integration/test_video_kind.py"
    pair = image.PAIR
    seconds = 0.15
    """Observed: probing and decoding a ten-second 640 x 480 MPEG-4 video
    takes under 0.065 s on one core (seven runs); twice that, rounded up,
    covers a slower host. A longer video costs more, bounded by
    `stream.RATE`."""

    def limits(self) -> tuple[str, ...]:
        """The boundaries the video capability states.

        Returns:
            Plain statements of what the kind does not cover.
        """
        return (
            "needs the ffprobe and ffmpeg programs on the PATH",
            (
                "descriptor features of sampled frames (mean frame, brightness "
                "over time, motion); no learned temporal representation"
            ),
            (
                "a served model takes descriptor rows; raw-video requests are "
                "not decoded at serving time"
            ),
        )

    def ready(self) -> None:
        """Raises `VideoError` (`missing_dependency`) without FFmpeg."""
        stream.tool("ffprobe")
        stream.tool("ffmpeg")

    def trace(
        self,
        path: pathlib.Path,
        row: pandas.Series,  # noqa: ARG002 -- media.Media.trace; a video needs no declaration
    ) -> media.Description:
        """Probes, samples and describes one video file.

        Args:
            path: The video file.
            row: Its manifest row; not consulted.

        Returns:
            The `stream.describe` features and the file's digest, read in
            chunks (the file never enters memory whole, MM-6).
        """
        return media.Description(
            dict(
                zip(stream.names(), stream.describe(path).tolist(), strict=True)
            ),
            media.digest(path),
        )

    def pairs(
        self, traces: list[Any], features: pandas.DataFrame
    ) -> Iterable[tuple[int, int]]:
        """The near-copy pairs of the mean frames, as images pair.

        Args:
            traces: Each file's trace.
            features: The descriptor rows, with the mean frame's thumbnail.

        Returns:
            The linked pairs (`image.Image.pairs`).
        """
        return image.IMAGE.pairs(traces, features)


VIDEO = Video()
"""The video kind."""
