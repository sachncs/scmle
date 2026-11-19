"""Plants the standard leaks into a target, for any kind of target.

The leakage screen proves on each run's own rows that it finds these leaks
(`screen.screen_table`), and the qualification harness measures its recall
on the same plants (`benchmarks.qualification.screen_certify`), so the
evidence and the product use one definition of each leak type. `Classes`
and `Numbers` write each type for their kind of target:

* `leak_copy`: the target itself (class codes for classes).
* `leak_noisy_copy`: the target's labels with `FLIP` of them moved to
  another class, or with `NOISE` standard deviations of noise for a number.
* `leak_threshold`: a field derived from the target with noise: the class
  code, or the target cut into `BANDS` quantile bands (such as a price band).
* `leak_missing_for_class`: noise recorded only when the target is not the
  first row's class, or only when it is at or below its median.
* `leak_rare_record`: noise recorded for `RARE` of the rows of the first
  row's class, or for `TOP` of the rows in the top decile of a number; about
  9 percent of the rows either way, all from one class or band.
* `leak_target_mean`: a target encoding fitted on all rows: the mean of the
  target (or of the first row's class) over consecutive groups of `LEVEL`
  rows, each row's own outcome included.
* `leak_duplicates` (`copy`, rows rather than a column): `COPIES` of the
  rows copied once, so copies fall on both sides of a split.
"""

from collections.abc import Hashable
from collections.abc import Iterable

import numpy
import pandas

from scmle.runtime import randomness

DEFECTS = {
    "leak_copy": "copy",
    "leak_noisy_copy": "noisy",
    "leak_threshold": "threshold",
    "leak_missing_for_class": "missing",
    "leak_rare_record": "rare",
    "leak_target_mean": "mean",
}
DUPLICATES = "leak_duplicates"
KINDS = (*DEFECTS, DUPLICATES)
FLIP = 0.02
NOISE = 0.1
SPREAD = 0.25
BANDS = 10
RARE = 0.25
TOP = 0.9
LEVEL = 5
COPIES = 0.3


class Classes:
    """The five leak types for a target that holds classes.

    Attributes:
        target: The target.
        codes: Its class codes, as floats.
        first: Whether each row has the first row's class.
        noise: Standard normal noise, one value per row.
        draws: Uniform draws, one per row.
        rng: The plant's random generator.
    """

    def __init__(
        self, target: pandas.Series, rng: numpy.random.Generator
    ) -> None:
        """Draws the noise every leak type of this target uses.

        Args:
            target: The target.
            rng: Random generator for this plant.
        """
        self.target = target
        self.rng = rng
        self.noise = pandas.Series(rng.normal(size=len(target)), target.index)
        self.draws = pandas.Series(rng.random(len(target)), target.index)
        self.codes = pandas.Series(
            pandas.factorize(target)[0].astype(float), target.index
        )
        self.first = target == target.iloc[0]

    def copy(self) -> pandas.Series:
        """The class codes."""
        return self.codes

    def noisy(self) -> pandas.Series:
        """The labels with `FLIP` of them moved to another class.

        A field recorded after the outcome holds the label itself, so the
        plant spells the target's labels (not its codes, whose order
        follows the first row and can invert a two-class label).
        """
        labels = pandas.unique(self.target)
        classes = max(len(labels), 2)
        shift = self.rng.integers(1, classes, size=len(self.codes))
        moved = (self.codes.to_numpy("int64") + shift) % classes
        if len(labels) < classes:
            return self.codes.where(self.draws >= FLIP, moved)
        written = pandas.Series(labels[moved], self.target.index)
        return self.target.where(self.draws >= FLIP, written)

    def threshold(self) -> pandas.Series:
        """The class codes with noise."""
        return self.codes + SPREAD * self.noise

    def missing(self) -> pandas.Series:
        """Noise recorded only when the class is not the first row's."""
        return self.noise.where(~self.first)

    def rare(self) -> pandas.Series:
        """Noise recorded for `RARE` of the rows of the first row's class."""
        return self.noise.where(self.first & (self.draws < RARE))

    def mean(self) -> pandas.Series:
        """The share of the first row's class over groups of `LEVEL` rows."""
        return self.encode(self.first.astype(float))

    def encode(self, values: pandas.Series) -> pandas.Series:
        """Means of `values` over consecutive groups of `LEVEL` rows.

        Args:
            values: One number per row.

        Returns:
            Each row's group mean, its own value included.
        """
        groups = numpy.arange(len(values)) // LEVEL
        means = values.groupby(groups).transform("mean")
        return pandas.Series(means.to_numpy(), self.target.index)


class Numbers(Classes):
    """The five leak types for a target that is a number."""

    def __init__(
        self, target: pandas.Series, rng: numpy.random.Generator
    ) -> None:
        """Draws the noise every leak type of this target uses.

        Args:
            target: The target.
            rng: Random generator for this plant.
        """
        super().__init__(target, rng)
        self.values = target.astype(float)

    def copy(self) -> pandas.Series:
        """The target."""
        return self.values.copy()

    def noisy(self) -> pandas.Series:
        """The target with `NOISE` standard deviations of noise."""
        return self.values + NOISE * float(self.values.std()) * self.noise

    def threshold(self) -> pandas.Series:
        """The target cut into `BANDS` quantile bands, with noise."""
        band = pandas.qcut(self.values, BANDS, labels=False, duplicates="drop")
        return band.astype(float) + SPREAD * self.noise

    def missing(self) -> pandas.Series:
        """Noise recorded only when the target is at or below its median."""
        return self.noise.where(self.values <= float(self.values.median()))

    def rare(self) -> pandas.Series:
        """Noise recorded for `TOP` of the rows in the top decile."""
        top = self.values >= float(self.values.quantile(0.9))
        return self.noise.where(top & (self.draws < TOP))

    def mean(self) -> pandas.Series:
        """The target's mean over groups of `LEVEL` rows."""
        return self.encode(self.values)


def plant(
    target: pandas.Series, defect: str, seed: int, continuous: bool
) -> pandas.Series:
    """Builds the planted leak column for one leak type.

    Args:
        target: The target, one value per row.
        defect: A leak type in `DEFECTS`.
        seed: Run seed.
        continuous: Whether the target is a number (regression) rather than
            classes.

    Returns:
        The planted column, aligned to `target`.

    Raises:
        KeyError: For an unknown leak type.
    """
    method = DEFECTS[defect]
    rng = randomness.component_rng(seed, "plants", list(DEFECTS).index(defect))
    kind = (Numbers if continuous else Classes)(target, rng)
    column: pandas.Series = getattr(kind, method)()
    return column


def copy(frame: pandas.DataFrame, seed: int) -> pandas.DataFrame:
    """Copies `COPIES` of the rows once.

    Args:
        frame: Rows including the target.
        seed: Run seed.

    Returns:
        The rows and their copies, in a seeded random order, with a fresh
        index.
    """
    rng = randomness.component_rng(seed, "plants", len(DEFECTS))
    picked = rng.choice(len(frame), round(COPIES * len(frame)), replace=False)
    # Positions, not `pandas.concat`: concat compares the frames' `attrs`,
    # which fails when they hold a Series (an uplift task's treatment).
    rows = numpy.concatenate([numpy.arange(len(frame)), numpy.sort(picked)])
    order = rng.permutation(len(rows))
    return frame.iloc[rows[order]].reset_index(drop=True)


def names(columns: Iterable[Hashable]) -> dict[str, str]:
    """Names a planted column per leak type that no given column has.

    The probe adds its plants to a copy of the user's rows, which may hold a
    column of any name, a leak type's own among them. Each plant keeps its
    type's name when that is free, and otherwise takes the first free
    `<type> (<k>)`, `k` counting up from 1, so no plant replaces or shadows
    a user column.

    Args:
        columns: The columns the plants go beside, the target included.

    Returns:
        Leak type in `DEFECTS` to its column name, every name distinct and
        none in `columns`.
    """
    taken = {str(c) for c in columns}
    found = {}
    for defect in DEFECTS:
        name, count = defect, 0
        while name in taken:
            count += 1
            name = f"{defect} ({count})"
        taken.add(name)
        found[defect] = name
    return found


def table(
    frame: pandas.DataFrame, target: str, seed: int, continuous: bool
) -> pandas.DataFrame:
    """Adds one planted column per leak type to a copy of the rows.

    Args:
        frame: Rows including the target.
        target: Name of the target column.
        seed: Run seed.
        continuous: Whether the target is a number.

    Returns:
        A copy of `frame` with a column per leak type, in `DEFECTS` order,
        named by `names` so that none clashes with a column of `frame`.
    """
    labels = names(frame.columns)
    return frame.assign(
        **{
            labels[d]: plant(frame[target], d, seed, continuous)
            for d in DEFECTS
        }
    )
