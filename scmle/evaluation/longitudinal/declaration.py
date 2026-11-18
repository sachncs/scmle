"""The declared estimand, the estimation policy and one period's estimate."""

import dataclasses
import math
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records

PROPENSITY_CLIP_CEILING = 0.5


SHA256_HEX_LENGTH = 64


MATRIX_NDIM = 2


REQUIRED_ASSUMPTIONS = frozenset(
    {
        "consistency",
        "positivity",
        "conditional_exchangeability",
        "no_unmeasured_time_varying_confounding",
        "no_time_varying_covariates",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Estimand:
    """Declaration of the period-specific longitudinal causal quantity.

    Attributes:
        name: Non-empty label of the estimand.
        subject_identity: Column identifying each subject.
        time_column: Column giving the period of each row.
        treatment_column: Binary (0/1) treatment column.
        outcome: Outcome column.
        covariates: Adjustment columns; each must be constant within a
            subject.
        population: Non-empty description of the target population.
        assumptions: Declared identification assumptions; must include every
            member of `REQUIRED_ASSUMPTIONS`.
        migrations: The written layout's version steps, oldest first;
            version 2 writes the three columns as `subject`, `time` and
            `treatment`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={
                "subject_identity": "subject",
                "time_column": "time",
                "treatment_column": "treatment",
            }
        ),
    )

    __module__ = "scmle.evaluation.longitudinal"

    name: str
    subject_identity: str
    time_column: str
    treatment_column: str
    outcome: str
    covariates: tuple[str, ...]
    population: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validates the columns and required causal assumptions.

        Raises:
            ValueError: If the name or population is empty, any column name
                is empty or not a string, column names repeat, no
                assumptions are given, or a required assumption is missing.
        """
        name: object = self.name  # may be a non-string at runtime
        if not isinstance(name, str) or not name.strip():
            raise errors.InputError("estimand name must not be empty")
        names: tuple[object, ...] = (
            self.subject_identity,
            self.time_column,
            self.treatment_column,
            self.outcome,
            *self.covariates,
        )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise errors.InputError(
                "estimand column names must be non-empty strings"
            )
        if len(set(names)) != len(names):
            raise errors.InputError("estimand columns must be unique")
        population: object = self.population  # may be a non-string at runtime
        if not isinstance(population, str) or not population.strip():
            raise errors.InputError("estimand population must not be empty")
        if not self.assumptions:
            raise errors.InputError("estimand needs explicit assumptions")
        missing = REQUIRED_ASSUMPTIONS.difference(self.assumptions)
        if missing:
            raise errors.InputError(
                "estimand is missing required assumptions: "
                + ", ".join(sorted(missing))
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible estimand metadata.

        Returns:
            The estimand fields keyed by field name.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Bounded numerical and missingness policy for longitudinal IPW.

    Attributes:
        confidence_level: Two-sided confidence level of the intervals, in
            (0, 1).
        propensity_clip: Bound in (0, 0.5) to which fitted propensities are
            clipped before weighting.
        missingness: Either "reject" (any missing value is an error) or
            "complete_case" (rows with a missing value are dropped).
        seed: Non-negative integer recorded with the policy.
        tolerance: Convergence threshold on the largest absolute Newton step
            of the propensity fit.
        max_iterations: Maximum Newton iterations per propensity fit.
        minimum_rows_per_period: Minimum number of rows in each period and in
            the analysed frame; at least twice `minimum_rows_per_arm`.
        minimum_rows_per_arm: Minimum number of rows in each treatment arm
            of each period.
        minimum_effective_sample_size: Minimum Kish effective sample size of
            the weights of each arm in each period.
        reference_tolerance: Largest absolute difference allowed between the
            primary period estimates and their independent recomputation.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `confidence`, `clip`, `iterations`,
            `reference` and `minimums.{period, arm, ess}`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "confidence_level": "confidence",
                "propensity_clip": "clip",
                "max_iterations": "iterations",
                "reference_tolerance": "reference",
            },
            groups={
                "minimums": {
                    "minimum_rows_per_period": "period",
                    "minimum_rows_per_arm": "arm",
                    "minimum_effective_sample_size": "ess",
                }
            },
        ),
    )

    __module__ = "scmle.evaluation.longitudinal"

    confidence_level: float = 0.95
    propensity_clip: float = 0.01
    missingness: str = "reject"
    seed: int = 0
    tolerance: float = 1e-8
    max_iterations: int = 100
    minimum_rows_per_period: int = 8
    minimum_rows_per_arm: int = 2
    minimum_effective_sample_size: float = 2.0
    reference_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        """Rejects an unbounded or ambiguous estimation policy.

        Raises:
            ValueError: If a field lies outside the range given in the class
                attributes, `missingness` is not a recognised value, or
                `minimum_rows_per_period` is below twice
                `minimum_rows_per_arm`.
        """
        if not 0.0 < self.confidence_level < 1.0:
            raise errors.InputError(
                "confidence_level must lie strictly between 0 and 1"
            )
        if not 0.0 < self.propensity_clip < PROPENSITY_CLIP_CEILING:
            raise errors.InputError(
                "propensity_clip must lie strictly between 0 and 0.5"
            )
        if self.missingness not in ("reject", "complete_case"):
            raise errors.InputError(
                "missingness must be reject or complete_case"
            )
        seed: object = self.seed  # may be a non-integer at runtime
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise errors.InputError("seed must be a non-negative integer")
        self.check_limits()

    def check_limits(self) -> None:
        """Checks the tolerances, iteration limit and row minimums.

        Raises:
            ValueError: If a tolerance is not positive and finite, a count is
                not a positive integer, the effective sample size minimum is
                not positive, or `minimum_rows_per_period` is below twice
                `minimum_rows_per_arm`.
        """
        for name, value in (
            ("tolerance", self.tolerance),
            ("reference_tolerance", self.reference_tolerance),
        ):
            if not math.isfinite(float(value)) or value <= 0.0:
                raise errors.InputError(f"{name} must be positive and finite")
        counts: tuple[tuple[str, object], ...] = (
            ("max_iterations", self.max_iterations),
            ("minimum_rows_per_period", self.minimum_rows_per_period),
            ("minimum_rows_per_arm", self.minimum_rows_per_arm),
        )
        for name, count in counts:
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or count < 1
            ):
                raise errors.InputError(f"{name} must be a positive integer")
        if (
            not math.isfinite(self.minimum_effective_sample_size)
            or self.minimum_effective_sample_size <= 0.0
        ):
            raise errors.InputError(
                "minimum_effective_sample_size must be positive"
            )
        if self.minimum_rows_per_period < 2 * self.minimum_rows_per_arm:
            raise errors.InputError(
                "minimum_rows_per_period must cover both treatment arms"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible policy metadata.

        Returns:
            The policy fields keyed by field name.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )


@dataclasses.dataclass(frozen=True, slots=True)
class PeriodATE:
    """A period-specific Hájek IPW treatment-effect estimate.

    Attributes:
        period: Label of the period.
        estimate: Treated weighted mean minus control weighted mean.
        standard_error: Influence-function standard error of the estimate.
        interval: Lower and upper confidence limits of the estimate.
        rows: Number of rows in the period.
        treated_rows: Number of treated rows in the period.
        control_rows: Number of control rows in the period.
        propensity_min: Smallest fitted propensity before clipping.
        propensity_max: Largest fitted propensity before clipping.
        fraction_clipped: Share of rows whose propensity was changed by
            clipping, in [0, 1].
        effective_sample_size_treated: Kish effective sample size of the
            treated-arm weights.
        effective_sample_size_control: Kish effective sample size of the
            control-arm weights.
        converged: Whether the propensity fit converged; must be True.
        iterations: Newton iterations the propensity fit used.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `se`, `treated`, `control`, `clipped`,
            `propensity.{low, high}` and `ess.{treated, control}`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "standard_error": "se",
                "treated_rows": "treated",
                "control_rows": "control",
                "fraction_clipped": "clipped",
            },
            groups={
                "propensity": {
                    "propensity_min": "low",
                    "propensity_max": "high",
                },
                "ess": {
                    "effective_sample_size_treated": "treated",
                    "effective_sample_size_control": "control",
                },
            },
        ),
    )

    __module__ = "scmle.evaluation.longitudinal"

    period: str
    estimate: float
    standard_error: float
    interval: tuple[float, float]
    rows: int
    treated_rows: int
    control_rows: int
    propensity_min: float
    propensity_max: float
    fraction_clipped: float
    effective_sample_size_treated: float
    effective_sample_size_control: float
    converged: bool
    iterations: int

    def __post_init__(self) -> None:
        """Validates finite period-level evidence.

        Raises:
            ValueError: If the period label is empty, a row count is not
                positive, arm counts do not sum to `rows`, `iterations` is
                below 1, the fit did not converge, a value is not finite,
                the standard error is negative, the interval is reversed,
                the clipped fraction lies outside [0, 1], or a propensity
                lies outside [0, 1].
        """
        if not self.period:
            raise errors.InputError("period label must not be empty")
        if self.rows < 1 or self.treated_rows < 1 or self.control_rows < 1:
            raise errors.InputError("period row counts must be positive")
        if self.treated_rows + self.control_rows != self.rows:
            raise errors.InputError("period arm counts must reconcile to rows")
        if self.iterations < 1:
            raise errors.InputError("period iterations must be positive")
        if not self.converged:
            raise errors.InputError("a period result must be converged")
        self.check_finite()
        if self.standard_error < 0.0:
            raise errors.InputError("standard_error must not be negative")
        if self.interval[0] > self.interval[1]:
            raise errors.InputError("period interval must be ordered")
        if not 0.0 <= self.fraction_clipped <= 1.0:
            raise errors.InputError("fraction_clipped must lie between 0 and 1")
        if self.propensity_min < 0.0 or self.propensity_max > 1.0:
            raise errors.InputError("propensities must lie between 0 and 1")

    def check_finite(self) -> None:
        """Checks that every numeric field of the period is finite.

        Raises:
            ValueError: If a numeric field is not finite.
        """
        for name, value in (
            ("estimate", self.estimate),
            ("standard_error", self.standard_error),
            ("interval_lower", self.interval[0]),
            ("interval_upper", self.interval[1]),
            ("propensity_min", self.propensity_min),
            ("propensity_max", self.propensity_max),
            ("fraction_clipped", self.fraction_clipped),
            (
                "effective_sample_size_treated",
                self.effective_sample_size_treated,
            ),
            (
                "effective_sample_size_control",
                self.effective_sample_size_control,
            ),
        ):
            if not math.isfinite(float(value)):
                raise errors.InputError(f"{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible period evidence.

        Returns:
            The period fields keyed by field name.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )
