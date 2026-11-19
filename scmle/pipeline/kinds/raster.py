"""The raster kind: labelled tiles on the shared stages.

A data block with `modality: raster` names a manifest table with a `path`
column (`.npy` arrays, or GeoTIFF through the optional `rasterio`) next to
the target and any other columns, or a directory of label-named
subdirectories. A manifest may declare each tile's `nodata` value, its
`crs` and its bounds (`west`, `south`, `east`, `north`); a GeoTIFF's own
are used where the manifest declares none. `Raster.read` describes every
tile in bounded strips (`raster.describe`) into numeric feature columns.

Tiles that are copies or near-copies, or lie in one spatial block, form
one cohort, which the shared splitter (`rows.Cohorts`) keeps on one side of
every split, together with any declared group (register rows MM-10 and
MM-11). Blocks are squares of `span` tiles anchored at the CRS origin, so
adding a tile never moves another tile's block.
"""

import math
import pathlib
from typing import Any

import numpy
import pandas

from scmle.contracts import tasks
from scmle.data.modalities.geospatial import raster
from scmle.pipeline.kinds import media

SIDES = ("west", "south", "east", "north")
"""Manifest columns of a tile's bounds in its CRS."""

NODATA = "nodata"
"""Manifest column of a tile's nodata value."""

CRS = "crs"
"""Manifest column of a tile's CRS identifier."""

CROSSING = 0.05
"""Largest expected share of adjacent tile pairs that straddle the locked
set and the development rows. Splitting tiles at random puts a pair on two
sides with probability 2p(1 - p) for a locked share p (0.32 at p = 0.2);
blocks of k tiles a side leave only the 1/k of adjacent pairs that cross a
block border exposed, so k = ceil(2p(1 - p) / CROSSING) brings the share to
at most 5%, the conventional size of a test."""

SECONDS = 0.02
"""Seconds to describe one tile, an upper bound set by observation: a
1,024 x 1,024 four-band float tile took 0.12 s here, and tiles are mostly
far smaller (256 x 256 took 0.008 s)."""


def span(fraction: float) -> int:
    """Tiles per side of a spatial block for a locked share.

    Args:
        fraction: The share of rows locked for the final test.

    Returns:
        ceil(2p(1 - p) / `CROSSING`), at least one.
    """
    return max(1, math.ceil(2.0 * fraction * (1.0 - fraction) / CROSSING))


def blocks(pictures: list[raster.Picture], fraction: float) -> list[object]:
    """The spatial block of each tile with bounds.

    A block is a square of `span` tiles a side, the tile size being the
    median tile width of the CRS; blocks are anchored at the CRS origin.

    Args:
        pictures: Each tile's description.
        fraction: The share of rows locked for the final test.

    Returns:
        `crs:column:row` per tile with bounds, None for one without.
    """
    widths: dict[str, list[float]] = {}
    for picture in pictures:
        if picture.bounds is not None:
            widths.setdefault(picture.crs, []).append(
                abs(picture.bounds[2] - picture.bounds[0])
            )
    sizes = {
        crs: span(fraction) * float(numpy.median(values))
        for crs, values in widths.items()
    }
    made: list[object] = []
    for picture in pictures:
        size = sizes.get(picture.crs, 0.0)
        if picture.bounds is None or size <= 0.0:
            made.append(None)
            continue
        west, south, east, north = picture.bounds
        made.append(
            f"{picture.crs}:{math.floor((west + east) / 2.0 / size)}:"
            f"{math.floor((south + north) / 2.0 / size)}"
        )
    return made


def declared(row: pandas.Series, name: str) -> object | None:
    """A manifest value, or None when the column or value is missing.

    Args:
        row: A manifest row.
        name: The column.

    Returns:
        The value, or None.
    """
    value = row.get(name)
    return None if value is None or pandas.isna(value) else value


PAIR = 2e-8
"""Seconds to compare two thumbnails: the pairing is the audio kind's
(`media.near`), observed at 0.8 s for 50 million pairs."""


class Raster(media.Media):
    """Labelled raster tiles, described by `raster.describe`."""

    name = "raster"
    suffixes = frozenset({".npy", ".tif", ".tiff"})
    radius = raster.RADIUS
    seconds = SECONDS
    pair = PAIR
    consumed = frozenset({*SIDES, NODATA, CRS})
    evidence = "tests/unit/test_kind_raster.py"

    def limits(self) -> tuple[str, ...]:
        """The boundaries the raster capability states.

        Returns:
            Plain statements of what the kind does not cover.
        """
        return (
            (
                "tiles are described into band statistics and a thumbnail "
                "before any model sees them; no learned representation"
            ),
            "GeoTIFF needs the optional rasterio library",
            "a served model predicts from described rows (`describe`)",
        )

    def trace(self, path: pathlib.Path, row: pandas.Series) -> raster.Picture:
        """Describes one tile in bounded strips.

        Args:
            path: The tile.
            row: Its manifest row, which may declare its nodata value, CRS
                and bounds.

        Returns:
            Its features, digest, thumbnail, bounds and CRS.
        """
        sides = [declared(row, side) for side in SIDES]
        nodata = declared(row, NODATA)
        return raster.describe(
            path,
            None if nodata is None else float(str(nodata)),
            None
            if any(side is None for side in sides)
            else (
                float(str(sides[0])),
                float(str(sides[1])),
                float(str(sides[2])),
                float(str(sides[3])),
            ),
            str(declared(row, CRS) or ""),
        )

    def keys(
        self,
        spec: tasks.Config,
        listed: pandas.DataFrame,  # noqa: ARG002 -- media.Media.keys; the bounds are in the traces
        traces: list[Any],
    ) -> list[list[object]]:
        """Ties the tiles of one spatial block into one cohort.

        Args:
            spec: The run's specification, for the locked share.
            listed: The manifest.
            traces: Each tile's description.

        Returns:
            The tiles' blocks.
        """
        return [blocks(traces, spec.validation.test_fraction)]


RASTER = Raster()
"""The raster kind."""
