"""Plants the standard leaks into a target, for any kind of target.

The leakage screen proves on each run's own rows that it finds these leaks
(`screen.screen_table`), and the qualification harness measures its recall
on the same plants (`benchmarks.qualification.screen_certify`), so the
evidence and the product use one definition of each leak type. `Classes`
and `Numbers` write each type for their kind of target:

* `leak_copy`: the target itself (class codes for classes).
* `leak_noisy_copy`: the target with `FLIP` of the labels moved to another
  class, or with `NOISE` standard deviations of noise for a number.
* `leak_threshold`: a field derived from the target with noise: the class
  code, or the target cut into `BANDS` quantile bands (such as a price band).
* `leak_missing_for_class`: noise recorded only when the target is not the
  first row's class, or only when it is at or below its median.
* `leak_rare_record`: noise recorded for `RARE` of the rows of the first
  row's class, or for `TOP` of the rows in the top decile of a number; about
  9 percent of the rows either way, all from one class or band.
"""

import numpy
import pandas

from scmle.core import randomness

DEFECTS = {
    "leak_copy": "copy",
    "leak_noisy_copy": "noisy",
    "leak_threshold": "threshold",
    "leak_missing_for_class": "missing",
    "leak_rare_record": "rare",
}
FLIP = 0.02
NOISE = 0.1
SPREAD = 0.25
BANDS = 10
RARE = 0.25
TOP = 0.9


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
        """The class codes with `FLIP` of them moved to another class."""
        classes = max(int(self.codes.max()) + 1, 2)
        shift = self.rng.integers(1, classes, size=len(self.codes))
        moved = (self.codes + shift) % classes
        return self.codes.where(self.draws >= FLIP, moved)

    def threshold(self) -> pandas.Series:
        """The class codes with noise."""
        return self.codes + SPREAD * self.noise

    def missing(self) -> pandas.Series:
        """Noise recorded only when the class is not the first row's."""
        return self.noise.where(~self.first)

    def rare(self) -> pandas.Series:
        """Noise recorded for `RARE` of the rows of the first row's class."""
        return self.noise.where(self.first & (self.draws < RARE))


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
        A copy of `frame` with a column per leak type, named after it.

    Raises:
        ValueError: If a column already has a leak type's name.
    """
    clashes = [d for d in DEFECTS if d in frame.columns]
    if clashes:
        raise ValueError(f"columns already named as plants: {clashes}")
    return frame.assign(
        **{d: plant(frame[target], d, seed, continuous) for d in DEFECTS}
    )
