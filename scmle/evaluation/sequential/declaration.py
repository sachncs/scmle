"""The declared estimand and the bounded estimation policy."""

import dataclasses
import math
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records

PROPENSITY_CLIP_CEILING = 0.5


MIN_PERIODS = 2


REGIME_COUNT = 2


COVARIANCE_DIMENSION = 2


SHA256_HEX_LENGTH = 64


MATRIX_NDIM = 2


REQUIRED_ASSUMPTIONS = frozenset(
    {
        "consistency",
        "positivity",
        "sequential_exchangeability",
        "no_unmeasured_time_varying_confounding",
        "time_varying_covariates_are_pre_treatment",
        "complete_follow_up",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class Estimand:
    """Declaration of a sustained binary-treatment causal contrast.

    Attributes:
        name: Non-empty label of the estimand.
        subject_identity: Column identifying each subject.
        time_column: Column giving the period of each row.
        treatment_column: Binary (0/1) treatment column.
        outcome: Outcome column; the value at the last period is analysed.
        baseline_covariates: Columns that must be constant within a subject.
        time_varying_covariates: Columns measured before treatment in each
            period; they enter the denominator propensity model.
        population: Non-empty description of the target population.
        assumptions: Declared identification assumptions; must include every
            member of `REQUIRED_ASSUMPTIONS`.
        target_treatment: Treatment value (0 or 1) held in every period by
            the target regime.
        comparator_treatment: Treatment value (0 or 1) held in every period
            by the comparator regime; differs from `target_treatment`.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `subject`, `time`, `treatment`,
            `covariates.{baseline, varying}` and
            `regimes.{target, comparator}`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "subject_identity": "subject",
                "time_column": "time",
                "treatment_column": "treatment",
            },
            groups={
                "covariates": {
                    "baseline_covariates": "baseline",
                    "time_varying_covariates": "varying",
                },
                "regimes": {
                    "target_treatment": "target",
                    "comparator_treatment": "comparator",
                },
            },
        ),
    )

    __module__ = "scmle.evaluation.sequential"

    name: str
    subject_identity: str
    time_column: str
    treatment_column: str
    outcome: str
    baseline_covariates: tuple[str, ...]
    time_varying_covariates: tuple[str, ...]
    population: str
    assumptions: tuple[str, ...]
    target_treatment: int = 1
    comparator_treatment: int = 0

    def __post_init__(self) -> None:
        """Validates the declared columns, regimes, and assumptions.

        Raises:
            ValueError: If the name or population is empty, any column name
                is empty or not a string, column names repeat, no
                assumptions are given, a required assumption is missing, a
                treatment value is not 0 or 1, or the target and comparator
                treatments are equal.
        """
        name: object = self.name  # may be a non-string at runtime
        if not isinstance(name, str) or not name.strip():
            raise errors.InputError("estimand name must not be empty")
        names: tuple[object, ...] = (
            self.subject_identity,
            self.time_column,
            self.treatment_column,
            self.outcome,
            *self.baseline_covariates,
            *self.time_varying_covariates,
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
        if self.target_treatment not in (0, 1):
            raise errors.InputError("target_treatment must be 0 or 1")
        if self.comparator_treatment not in (0, 1):
            raise errors.InputError("comparator_treatment must be 0 or 1")
        if self.target_treatment == self.comparator_treatment:
            raise errors.InputError(
                "target and comparator treatments must differ"
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
    """Bounded numerical, support, and truncation policy.

    Attributes:
        confidence_level: Two-sided confidence level of the intervals, in
            (0, 1).
        propensity_clip: Positivity bound in (0, 0.5); fitted denominator
            propensities must lie within
            `[propensity_clip, 1 - propensity_clip]`.
        maximum_weight: Finite cap, greater than 1, applied to the
            cumulative stabilized weight.
        missingness: Either "reject" (any missing value is an error) or
            "complete_case" (subjects with a missing value are dropped).
        seed: Non-negative integer recorded with the policy.
        tolerance: Convergence threshold on the largest absolute Newton step
            of the propensity fit.
        max_iterations: Maximum Newton iterations per propensity fit.
        minimum_subjects: Minimum number of subjects in the analysed panel.
        minimum_arm_rows: Minimum number of subjects in each treatment arm
            per period and in each regime.
        minimum_effective_sample_size: Minimum Kish effective sample size of
            the truncated weights per period and per regime.
        reference_tolerance: Largest absolute difference between a primary
            regime mean and its independent recomputation that still counts
            as agreement; a larger difference is recorded in the reference
            record and does not raise.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `confidence`, `clip`, `weight`, `iterations`,
            `reference` and `minimums.{subjects, arm, ess}`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.nest(
            {
                "confidence_level": "confidence",
                "propensity_clip": "clip",
                "maximum_weight": "weight",
                "max_iterations": "iterations",
                "reference_tolerance": "reference",
            },
            groups={
                "minimums": {
                    "minimum_subjects": "subjects",
                    "minimum_arm_rows": "arm",
                    "minimum_effective_sample_size": "ess",
                }
            },
        ),
    )

    __module__ = "scmle.evaluation.sequential"

    confidence_level: float = 0.95
    propensity_clip: float = 0.01
    maximum_weight: float = 20.0
    missingness: str = "reject"
    seed: int = 0
    tolerance: float = 1e-8
    max_iterations: int = 100
    minimum_subjects: int = 30
    minimum_arm_rows: int = 2
    minimum_effective_sample_size: float = 5.0
    reference_tolerance: float = 1e-6

    def __post_init__(self) -> None:
        """Rejects an unbounded or ambiguous estimation policy.

        Raises:
            ValueError: If a field lies outside the range given in the class
                attributes, or `missingness` is not a recognised value.
        """
        if not 0.0 < self.confidence_level < 1.0:
            raise errors.InputError(
                "confidence_level must lie strictly between 0 and 1"
            )
        if not 0.0 < self.propensity_clip < PROPENSITY_CLIP_CEILING:
            raise errors.InputError(
                "propensity_clip must lie strictly between 0 and 0.5"
            )
        if not math.isfinite(self.maximum_weight) or self.maximum_weight <= 1.0:
            raise errors.InputError(
                "maximum_weight must be finite and greater than 1"
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
        """Checks the tolerances, counts and effective sample size minimum.

        Raises:
            ValueError: If a tolerance is not positive and finite, a count is
                not a positive integer, or the effective sample size minimum
                is not positive.
        """
        for name, value in (
            ("tolerance", self.tolerance),
            ("reference_tolerance", self.reference_tolerance),
        ):
            if not math.isfinite(float(value)) or value <= 0.0:
                raise errors.InputError(f"{name} must be positive and finite")
        counts: tuple[tuple[str, object], ...] = (
            ("max_iterations", self.max_iterations),
            ("minimum_subjects", self.minimum_subjects),
            ("minimum_arm_rows", self.minimum_arm_rows),
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

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible policy metadata.

        Returns:
            The policy fields keyed by field name.
        """
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )
