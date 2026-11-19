"""The image kind: picture files on the shared stages (`media.Media`).

Each picture is validated, decoded with Pillow and described
(`descriptor.describe`: thumbnail, colour, edges and shape) into one row,
without any fitted state; every fitted step runs inside the folds. Two
pictures are near-copies when their average hashes differ in at most
`linkage.DISTANCE` bits (`linkage.edges`), so copies, near-copies and
declared groups stay on one side of every split (MM-10).
"""

from collections.abc import Callable
from collections.abc import Iterable
import itertools
import pathlib
from typing import Any

import numpy
import pandas

from scmle.data.modalities import image
from scmle.data.modalities.image import descriptor
from scmle.data.modalities.image import linkage
from scmle.pipeline.kinds import media

REGIONS = 4
"""Regions per side an occlusion explanation masks: 16 regions, each a
sixteenth of the picture, as the 4 x 4 grid of the modality lifecycle."""

PAIR = 1e-8
"""Seconds per squared row to compare average hashes, an upper bound set
by observation: `linkage.edges` linked 10,000 random rows in 0.31 s here
(3.1e-9 s per squared row); 1e-8 allows three times that."""


class Image(media.Media):
    """Image files (PNG, JPEG, GIF, BMP, TIFF, WebP) decoded with Pillow."""

    name = "image"
    suffixes = frozenset(
        {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tif", ".tiff", ".webp"}
    )
    extras = ("multimodal",)
    seconds = 0.05
    """Observed: describing a 1-megapixel JPEG takes under 0.022 s on one
    core (seven runs: validation, the content hash and the resize); twice
    that, rounded up, covers a slower host. A file is capped at
    `image.DEFAULT_MAX_PIXELS`."""
    pair = PAIR
    evidence = "tests/integration/test_image_kind.py"

    def limits(self) -> tuple[str, ...]:
        """The boundaries the image capability states.

        Returns:
            Plain statements of what the kind does not cover.
        """
        return (
            (
                "descriptor features (thumbnail, colour, edges, shape); no "
                "learned visual representation"
            ),
            (
                "a served model takes descriptor rows; raw-image requests are "
                "not decoded at serving time"
            ),
            (
                "near-duplicates are found by average hash, not semantic "
                "similarity"
            ),
        )

    def ready(self) -> None:
        """Raises `DependencyError` when Pillow is missing."""
        image.pillow("image ingestion")

    def trace(
        self,
        path: pathlib.Path,
        row: pandas.Series,  # noqa: ARG002 -- media.Media.trace; a picture needs no declaration
    ) -> media.Description:
        """Validates, decodes and describes one image file.

        Args:
            path: The image file.
            row: Its manifest row; not consulted.

        Returns:
            The `descriptor.describe` features of its RGB pixels and the
            file's digest.
        """
        sample = image.Sample.load(path)
        pixels = sample.decode("RGB", (descriptor.WORK, descriptor.WORK))
        values = descriptor.describe(pixels, sample.width, sample.height)
        return media.Description(
            dict(zip(descriptor.names(), values.tolist(), strict=True)),
            media.digest(path),
        )

    def pairs(
        self,
        traces: list[Any],  # noqa: ARG002 -- media.Media.pairs; the hashes are in the features
        features: pandas.DataFrame,
    ) -> Iterable[tuple[int, int]]:
        """The near-copy pairs: average hashes within `linkage.DISTANCE`.

        Args:
            traces: Each file's trace.
            features: The descriptor rows.

        Returns:
            The linked pairs (`linkage.edges`).
        """
        left, right = linkage.edges(features)
        return zip(left.tolist(), right.tolist(), strict=True)

    def occlude(
        self,
        score: Callable[[pandas.DataFrame], Any],
        path: pathlib.Path,
    ) -> tuple[tuple[str, float], ...]:
        """Explains one image by masking regions of its source pixels.

        Each of `REGIONS` x `REGIONS` regions of the decoded source is set
        to zero, the masked picture is described again and scored. The
        score tracked is the one the unmasked picture's most probable class
        gets, for every mask, so a mask that flips the predicted class is
        still read on the original class (MM-13).

        Args:
            score: The fitted model's scores of descriptor rows: class
                probabilities (rows x classes) or one value per row.
            path: The image file.

        Returns:
            Each region (`region[top:bottom,left:right]` in source pixels)
            with the tracked score of the unmasked picture minus that of
            the masked one, largest change first and ties by region.
        """
        pillow = image.pillow("image explanation")
        sample = image.Sample.load(path)
        source = sample.decode("RGB")
        height, width = source.shape[:2]
        bounds = [
            (top, bottom, left, right)
            for top, bottom in steps(height)
            for left, right in steps(width)
        ]
        pictures = [source]
        for top, bottom, left, right in bounds:
            masked = source.copy()
            masked[top:bottom, left:right] = 0
            pictures.append(masked)
        side = (descriptor.WORK, descriptor.WORK)
        rows = [
            descriptor.describe(
                numpy.asarray(
                    pillow.fromarray(picture).resize(
                        side, pillow.Resampling.BILINEAR
                    )
                ),
                width,
                height,
            )
            for picture in pictures
        ]
        values = numpy.asarray(
            score(pandas.DataFrame(rows, columns=list(descriptor.names()))),
            dtype=float,
        )
        if values.ndim > 1:
            values = values[:, int(numpy.argmax(values[0]))]
        terms = [
            (f"region[{top}:{bottom},{left}:{right}]", float(values[0] - value))
            for (top, bottom, left, right), value in zip(
                bounds, values[1:], strict=True
            )
        ]
        return tuple(sorted(terms, key=lambda term: (-abs(term[1]), term[0])))


def steps(length: int) -> list[tuple[int, int]]:
    """Cuts a side into `REGIONS` spans.

    Args:
        length: Pixels along the side.

    Returns:
        The start and end of each non-empty span.
    """
    cuts = numpy.linspace(0, length, REGIONS + 1).round().astype(int)
    return [
        (int(start), int(end))
        for start, end in itertools.pairwise(cuts)
        if end > start
    ]


IMAGE = Image()
"""The image kind."""
