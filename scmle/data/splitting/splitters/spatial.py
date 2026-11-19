"""The spatial blocked split: whole square blocks of space on one side.

Nearby rows share their label more than distant ones, so a random split
scores a model on neighbours of its training rows. `Tiling` cuts the plane
into square blocks whose side is derived from the label's semivariogram
(`scmle.data.modalities.geospatial.blocks`, Roberts et al. 2017) and keeps
every block wholly on one side of the test split and of every fold.
"""

import numpy
import pandas

from scmle.contracts import errors
from scmle.data.modalities.geospatial import blocks
from scmle.data.splitting.splitters.partitions import Partition
from scmle.data.splitting.splitters.partitions import Partitioner

NAMES = (
    ("x", "y"),
    ("longitude", "latitude"),
    ("lon", "lat"),
    ("lng", "lat"),
    ("long", "lat"),
    ("easting", "northing"),
)
"""Coordinate column pairs found by name (case ignored), east first: the
spellings of GeoPandas (`x`, `y`), of WGS 84 tables and of projected grids."""


class Tiling(Partitioner):
    """Spatial blocked cross-validation with a derived block side.

    The side starts at the label's practical range rounded up to the 1-2-5
    series and shrinks only until there are `folds + 1` blocks, one for the
    test set and one per fold (`blocks.design`). Test blocks are drawn with
    the validation seed until they hold `test_fraction` of the rows; the
    other blocks are dealt, largest first, to the fold holding fewest rows.
    """

    name = "spatial_block"

    def columns(self, frame: pandas.DataFrame) -> tuple[str, str]:
        """The east and north coordinate columns.

        Args:
            frame: The full dataset.

        Returns:
            The first pair of `NAMES` the frame holds, matched without case.
            The target and other role columns are never taken.

        Raises:
            BlockedError: If no coordinate pair is found.
        """
        roles = {
            self.data.target,
            self.data.group_column,
            self.data.time_column,
        }
        lookup = {
            str(c).lower(): str(c) for c in frame.columns if c not in roles
        }
        for east, north in NAMES:
            if east in lookup and north in lookup:
                return lookup[east], lookup[north]
        raise errors.BlockedError(
            "spatial_block validation needs coordinate columns: name them "
            "one of " + ", ".join(f"{e}/{n}" for e, n in NAMES)
        )

    def points(self, frame: pandas.DataFrame) -> numpy.ndarray:
        """The coordinates of each row.

        Args:
            frame: The full dataset.

        Returns:
            An `(rows, 2)` float array, east then north.

        Raises:
            BlockedError: If a coordinate is missing or not a number.
        """
        east, north = self.columns(frame)
        values = numpy.column_stack(
            [
                pandas.to_numeric(frame[c], errors="coerce").to_numpy(float)
                for c in (east, north)
            ]
        )
        if not numpy.isfinite(values).all():
            raise errors.BlockedError(
                f"coordinate columns '{east}' and '{north}' have missing or "
                "non-numeric values"
            )
        return values

    def design(
        self, frame: pandas.DataFrame
    ) -> tuple[float, float, numpy.ndarray]:
        """Derives the block side and names each row's block.

        Args:
            frame: The full dataset.

        Returns:
            The block side, the label's practical range and each row's
            block name.

        Raises:
            BlockedError: If the coordinates are unusable or fewer than
                `folds + 1` distinct places exist.
        """
        points = self.points(frame)
        label = blocks.encode(
            frame[self.data.target], self.task.is_classification
        )
        gram = blocks.variogram(points, label, self.validation.seed)
        side, places = blocks.design(points, gram, self.validation.folds + 1)
        return side, gram.reach, places

    def side(self, frame: pandas.DataFrame) -> float:
        """The derived block side.

        Args:
            frame: The full dataset.

        Returns:
            The side of the square blocks, in coordinate units.
        """
        return self.design(frame)[0]

    def partition(self, frame: pandas.DataFrame) -> Partition:
        """Spatial block holdout and block-disjoint folds.

        Args:
            frame: The full dataset.

        Returns:
            The development and test positions, one fold per
            `validation.folds` validating on its blocks alone, and notes on
            the side and range.

        Raises:
            BlockedError: If the coordinates are unusable or too few
                distinct places exist for the folds.
        """
        side, reach, places = self.design(frame)
        folds = self.validation.folds
        names = sorted(set(places.tolist()))
        rng = numpy.random.default_rng(self.validation.seed)
        order = [names[i] for i in rng.permutation(len(names))]
        sizes = pandas.Series(places).value_counts()
        goal = len(frame) * self.validation.test_fraction
        held: list[str] = []
        taken = 0
        for name in order:
            if taken >= goal or len(order) - len(held) <= folds:
                break
            held.append(name)
            taken += int(sizes[name])
        tested = pandas.Series(places).isin(held).to_numpy()
        dev_ids, test_ids = (
            numpy.flatnonzero(~tested),
            numpy.flatnonzero(tested),
        )
        chosen = set(held)
        rest = sorted(
            (n for n in order if n not in chosen), key=lambda n: -sizes[n]
        )
        loads = [0] * folds
        team: dict[str, int] = {}
        for name in rest:
            lightest = loads.index(min(loads))
            team[name] = lightest
            loads[lightest] += int(sizes[name])
        labels = numpy.array([team[p] for p in places[dev_ids]])
        windows = [
            (numpy.flatnonzero(labels != k), numpy.flatnonzero(labels == k))
            for k in range(folds)
        ]
        notes = [
            (
                f"spatial blocks of side {side:g}: {len(names)} blocks, "
                f"{len(held)} held out for the test set and {len(rest)} dealt "
                f"to {folds} folds; the label's practical range is {reach:g}"
            )
        ]
        if side < reach:
            notes.append(
                "the blocks are smaller than the label's range of spatial "
                "dependence, so rows at a block's edge resemble the next "
                "block's and scores may be optimistic; add data over a wider "
                "area or use fewer folds"
            )
        return Partition(dev_ids, test_ids, windows, notes)
