"""Numeric descriptors of decoded pixels: the rows the shared stages train on.

A descriptor is computed from one image alone, so it can be computed before
the split without leaking: every statistic a model fits (scaling, selection)
is fitted later, inside the folds, by the shared preprocessing. It holds

- `thumb_<channel>_<row>_<column>`: the block-mean RGB thumbnail of
  `SIDE` x `SIDE` cells, in [0, 1]; the near-duplicate signature
  (`linkage.signatures`) is read back from these columns;
- `hist_<channel>_<bin>`: the share of pixels in each of `BINS` intensity
  bins per channel;
- `edge_<row>_<column>_<angle>`: the gradient-magnitude-weighted share of
  each of `ANGLES` unsigned orientations in each of `GRID` x `GRID` cells,
  and `edge_density`, the mean gradient magnitude;
- `shape_width`, `shape_height`, `shape_aspect`: the source geometry.

Video reuses it on the mean of its sampled frames (`video.stream`).
"""

import numpy

from scmle.contracts import errors

SIDE = 8
"""Cells per side of the thumbnail: 64 cells, one bit each in the signature."""

GRID = 2
"""Cells per side of the orientation histograms."""

WORK = 32
"""Side every image is resized to before it is described; a multiple of
`SIDE` and `GRID`, so the thumbnail and the cells are exact block means."""

BINS = 8
"""Intensity bins per channel of the colour histogram."""

ANGLES = 8
"""Unsigned orientation bins over [0, pi) of the edge histograms."""

CHANNELS = ("r", "g", "b")
"""The channel names, in RGB order."""

LUMA = numpy.array([0.299, 0.587, 0.114])
"""ITU-R BT.601 luma weights of R, G and B."""


def names() -> tuple[str, ...]:
    """Names the descriptor columns, in the order `describe` fills them.

    Returns:
        The thumbnail, histogram, edge and shape column names.
    """
    thumb = tuple(
        f"thumb_{channel}_{row}_{column}"
        for channel in CHANNELS
        for row in range(SIDE)
        for column in range(SIDE)
    )
    hist = tuple(
        f"hist_{channel}_{index}"
        for channel in CHANNELS
        for index in range(BINS)
    )
    edge = tuple(
        f"edge_{row}_{column}_{angle}"
        for row in range(GRID)
        for column in range(GRID)
        for angle in range(ANGLES)
    )
    return (
        *thumb,
        *hist,
        *edge,
        "edge_density",
        "shape_width",
        "shape_height",
        "shape_aspect",
    )


def describe(pixels: numpy.ndarray, width: int, height: int) -> numpy.ndarray:
    """Describes resized RGB pixels.

    Args:
        pixels: `WORK` x `WORK` x 3 RGB values, `uint8` or in [0, 255].
        width: Width of the source in pixels.
        height: Height of the source in pixels.

    Returns:
        One float row, named by `names`.

    Raises:
        InputError: If the pixels are not `WORK` x `WORK` x 3, or a
            dimension is not positive.
    """
    if pixels.shape != (WORK, WORK, len(CHANNELS)):
        raise errors.InputError(
            f"descriptors need {WORK} x {WORK} RGB pixels, got {pixels.shape}"
        )
    if width <= 0 or height <= 0:
        raise errors.InputError("image dimensions must be positive")
    values = numpy.asarray(pixels, dtype=numpy.float64) / 255.0
    step = WORK // SIDE
    thumb = values.reshape(SIDE, step, SIDE, step, 3).mean(axis=(1, 3))
    hist = numpy.stack(
        [
            numpy.histogram(values[..., index], bins=BINS, range=(0.0, 1.0))[0]
            for index in range(len(CHANNELS))
        ]
    ) / float(WORK * WORK)
    gray = values @ LUMA
    rise, run = numpy.gradient(gray)
    magnitude = numpy.hypot(rise, run)
    angle = numpy.mod(numpy.arctan2(rise, run), numpy.pi)
    slot = numpy.minimum((angle / numpy.pi * ANGLES).astype(int), ANGLES - 1)
    cell = WORK // GRID
    edge = []
    for row in range(GRID):
        for column in range(GRID):
            part = numpy.s_[
                row * cell : (row + 1) * cell,
                column * cell : (column + 1) * cell,
            ]
            weights = numpy.bincount(
                slot[part].ravel(),
                weights=magnitude[part].ravel(),
                minlength=ANGLES,
            )
            total = weights.sum()
            edge.append(weights / total if total > 0 else weights)
    return numpy.concatenate(
        [
            thumb.transpose(2, 0, 1).ravel(),
            hist.ravel(),
            numpy.concatenate(edge),
            [
                float(magnitude.mean()),
                float(width),
                float(height),
                width / height,
            ],
        ]
    )
