"""The auditable evidence of one sequential estimation run."""

from collections.abc import Mapping
import dataclasses
import functools
import math
from typing import Any
from typing import ClassVar

import numpy

from scmle.contracts import errors
from scmle.contracts import records
from scmle.evaluation.envelope import Counts
from scmle.evaluation.envelope import Envelope
from scmle.evaluation.envelope import fold
from scmle.evaluation.envelope import lowercase
from scmle.evaluation.envelope import unfold
from scmle.evaluation.sequential.declaration import Config
from scmle.evaluation.sequential.declaration import COVARIANCE_DIMENSION
from scmle.evaluation.sequential.declaration import Estimand
from scmle.evaluation.sequential.declaration import MIN_PERIODS
from scmle.evaluation.sequential.declaration import REGIME_COUNT

NESTED = {
    "contrast": {
        "contrast_estimate": "estimate",
        "contrast_standard_error": "se",
        "contrast_interval": "interval",
    },
    "rows": {
        "rows_input": "received",
        "rows_used": "used",
        "rows_dropped": "dropped",
    },
    "subjects": {
        "subjects_input": "received",
        "subjects_used": "used",
        "subjects_dropped": "dropped",
    },
    "digests": {"raw_input_fingerprint": "frame"},
}


@dataclasses.dataclass(frozen=True, slots=True)
class PeriodDiagnostics:
    """Auditable propensity and cumulative-weight evidence for one period.

    Attributes:
        period: Label of the period.
        rows: Number of subjects in the period.
        treated_rows: Number of subjects treated in the period.
        control_rows: Number of subjects not treated in the period.
        denominator_propensity_min: Smallest fitted denominator propensity.
        denominator_propensity_max: Largest fitted denominator propensity.
        numerator_propensity_min: Smallest fitted numerator propensity.
        numerator_propensity_max: Largest fitted numerator propensity.
        fraction_weight_truncated: Share of subjects whose cumulative weight
            exceeds the cap, in [0, 1].
        maximum_untruncated_weight: Largest cumulative weight before the cap.
        maximum_weight: Largest cumulative weight after the cap.
        effective_sample_size: Kish effective sample size of the capped
            cumulative weights.
    """

    __module__ = "scmle.evaluation.sequential"

    period: str
    rows: int
    treated_rows: int
    control_rows: int
    denominator_propensity_min: float
    denominator_propensity_max: float
    numerator_propensity_min: float
    numerator_propensity_max: float
    fraction_weight_truncated: float
    maximum_untruncated_weight: float
    maximum_weight: float
    effective_sample_size: float

    def __post_init__(self) -> None:
        """Validates finite, reconcilable period diagnostics.

        Raises:
            ValueError: If the period label is empty, a row count is not
                positive, arm counts do not sum to `rows`, a value is not
                finite, a propensity lies outside (0, 1), the truncated
                fraction lies outside [0, 1], or a weight bound or the
                effective sample size is not positive.
        """
        if not self.period:
            raise errors.InputError("period label must not be empty")
        if self.rows < 1 or self.treated_rows < 1 or self.control_rows < 1:
            raise errors.InputError("period row counts must be positive")
        if self.treated_rows + self.control_rows != self.rows:
            raise errors.InputError("period arm counts must reconcile to rows")
        self.check_finite()
        if not 0.0 <= self.fraction_weight_truncated <= 1.0:
            raise errors.InputError(
                "fraction_weight_truncated must lie between 0 and 1"
            )
        for name, value in (
            ("denominator_propensity_min", self.denominator_propensity_min),
            ("denominator_propensity_max", self.denominator_propensity_max),
            ("numerator_propensity_min", self.numerator_propensity_min),
            ("numerator_propensity_max", self.numerator_propensity_max),
        ):
            if not 0.0 < value < 1.0:
                raise errors.InputError(
                    f"{name} must lie strictly between 0 and 1"
                )
        if self.maximum_untruncated_weight <= 0.0 or self.maximum_weight <= 0.0:
            raise errors.InputError("weight bounds must be positive")
        if self.effective_sample_size <= 0.0:
            raise errors.InputError("effective_sample_size must be positive")

    def check_finite(self) -> None:
        """Checks that every numeric field of the period is finite.

        Raises:
            ValueError: If a numeric field is not finite.
        """
        for name, value in (
            ("denominator_propensity_min", self.denominator_propensity_min),
            ("denominator_propensity_max", self.denominator_propensity_max),
            ("numerator_propensity_min", self.numerator_propensity_min),
            ("numerator_propensity_max", self.numerator_propensity_max),
            ("fraction_weight_truncated", self.fraction_weight_truncated),
            ("maximum_untruncated_weight", self.maximum_untruncated_weight),
            ("maximum_weight", self.maximum_weight),
            ("effective_sample_size", self.effective_sample_size),
        ):
            if not math.isfinite(float(value)):
                raise errors.InputError(f"{name} must be finite")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible period diagnostics.

        Returns:
            The diagnostic fields keyed by field name.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class RegimeEstimate:
    """A weighted mean outcome for one sustained treatment regime.

    Attributes:
        regime: Name of the regime, "always_treated" or "always_control".
        treatment: Treatment value (0 or 1) held in every period.
        estimate: Normalized weighted mean of the final-period outcome.
        standard_error: Influence-function standard error of the estimate.
        interval: Lower and upper confidence limits of the estimate.
        rows: Number of subjects whose full history follows the regime.
        weight_sum: Sum of the capped cumulative weights of those subjects.
        effective_sample_size: Kish effective sample size of those weights.
    """

    __module__ = "scmle.evaluation.sequential"

    regime: str
    treatment: int
    estimate: float
    standard_error: float
    interval: tuple[float, float]
    rows: int
    weight_sum: float
    effective_sample_size: float

    def __post_init__(self) -> None:
        """Validates finite regime-level evidence.

        Raises:
            ValueError: If the regime name is empty, the treatment is not 0
                or 1, rows or weight sum are not positive, the effective
                sample size is not positive, a value is not finite, the
                standard error is negative, or the interval is reversed.
        """
        if not self.regime:
            raise errors.InputError("regime name must not be empty")
        if self.treatment not in (0, 1):
            raise errors.InputError("regime treatment must be 0 or 1")
        if self.rows < 1 or self.weight_sum <= 0.0:
            raise errors.InputError("regime needs positive rows and weight sum")
        if self.effective_sample_size <= 0.0:
            raise errors.InputError(
                "regime effective sample size must be positive"
            )
        for name, value in (
            ("estimate", self.estimate),
            ("standard_error", self.standard_error),
            ("interval_lower", self.interval[0]),
            ("interval_upper", self.interval[1]),
        ):
            if not math.isfinite(float(value)):
                raise errors.InputError(f"{name} must be finite")
        if self.standard_error < 0.0 or self.interval[0] > self.interval[1]:
            raise errors.InputError(
                "regime uncertainty must be ordered and non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible regime evidence.

        Returns:
            The regime fields keyed by field name.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Contrast:
    """The target regime mean minus the comparator regime mean.

    Attributes:
        estimate: Target mean minus comparator mean.
        se: Standard error of the contrast.
        interval: Lower and upper confidence limits of the contrast.
    """

    estimate: float
    se: float
    interval: tuple[float, float]

    def __post_init__(self) -> None:
        """Checks that the contrast, its error and its interval are valid.

        Raises:
            InputError: If a value is not finite, the standard error is
                negative, or the interval is reversed.
        """
        for name, value in (
            ("contrast estimate", self.estimate),
            ("contrast se", self.se),
            ("contrast interval lower", self.interval[0]),
            ("contrast interval upper", self.interval[1]),
        ):
            if not math.isfinite(float(value)):
                raise errors.InputError(f"{name} must be finite")
        if self.se < 0.0:
            raise errors.InputError("contrast se must not be negative")
        if self.interval[0] > self.interval[1]:
            raise errors.InputError("contrast interval must be ordered")


@dataclasses.dataclass(frozen=True, slots=True)
class Estimation(Envelope):
    """Complete auditable evidence for one sequential estimation run.

    Attributes:
        estimand: The declared estimand.
        policy: The policy the run used.
        periods: Per-period diagnostics, in time order; at least two.
        regimes: The target regime estimate followed by the comparator
            regime estimate.
        contrast: Target mean minus comparator mean, its standard error and
            its confidence interval.
        covariance: Symmetric 2 by 2 covariance of the two regime means,
            ordered target then comparator.
        rows: Rows in the supplied frame, analysed, and removed by the
            missingness policy.
        subjects: Distinct subjects in the supplied frame, analysed, and
            removed by the missingness policy.
        diagnostics: Method, model, and fit details.
        limitations: Statements of what the estimate does not cover.
        reference: Record of the independent recomputation of the regime
            means.
        digests: The lowercase SHA-256 digest of the supplied frame, under
            `frame`.
    """

    __module__ = "scmle.evaluation.sequential"
    label = "sequential"
    migration: ClassVar[records.Migration] = records.Migration(
        change=functools.partial(fold, groups=NESTED),
        revert=functools.partial(unfold, groups=NESTED),
    )

    estimand: Estimand
    policy: Config
    periods: tuple[PeriodDiagnostics, ...]
    regimes: tuple[RegimeEstimate, ...]
    contrast: Contrast
    covariance: tuple[tuple[float, ...], ...]
    rows: Counts
    subjects: Counts
    diagnostics: Mapping[str, Any]
    limitations: Mapping[str, Any]
    reference: Mapping[str, Any]
    digests: Mapping[str, str]

    def __post_init__(self) -> None:
        """Validates report shape before serialization.

        Raises:
            ValueError: If there are fewer than two periods, there are not
                exactly two regimes, the covariance is not a finite
                symmetric 2 by 2 matrix, or the digests are not exactly the
                frame's lowercase SHA-256 digest.
            TypeError: If `contrast` is not `Contrast`, `rows` or
                `subjects` is not `Counts`, or `diagnostics`,
                `limitations`, or `reference` is not a mapping.
        """
        if len(self.periods) < MIN_PERIODS:
            raise errors.InputError(
                "sequential result needs at least two periods"
            )
        if len(self.regimes) != REGIME_COUNT:
            raise errors.InputError(
                "sequential result needs target and comparator regimes"
            )
        self.invertible()
        if not isinstance(self.contrast, Contrast):
            raise errors.KindError("contrast must be Contrast")
        for name, counts in (("rows", self.rows), ("subjects", self.subjects)):
            if not isinstance(counts, Counts):
                raise errors.KindError(f"{name} must be Counts")
        for mapping_name, mapping_value in (
            ("diagnostics", self.diagnostics),
            ("limitations", self.limitations),
            ("reference", self.reference),
        ):
            if not isinstance(mapping_value, Mapping):
                raise errors.KindError(f"{mapping_name} must be a mapping")
        if set(self.digests) != {"frame"}:
            raise errors.InputError("digests must hold exactly the frame")
        lowercase(self.digests["frame"], "digests.frame")

    def invertible(self) -> None:
        """Checks that the covariance is a finite symmetric 2 by 2 matrix.

        Raises:
            ValueError: If the covariance is not 2 by 2, has a non-finite
                value, or is not symmetric.
        """
        if len(self.covariance) != COVARIANCE_DIMENSION or any(
            len(row) != COVARIANCE_DIMENSION for row in self.covariance
        ):
            raise errors.InputError("covariance must be a 2 by 2 matrix")
        covariance = numpy.asarray(self.covariance, dtype=float)
        if not numpy.isfinite(covariance).all() or not numpy.allclose(
            covariance, covariance.T, rtol=1e-10, atol=1e-12
        ):
            raise errors.InputError("covariance must be finite and symmetric")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible sequential report.

        Returns:
            All report fields keyed by field name. The estimand, policy,
            periods and regimes are the dictionaries from their own
            `to_dict` (tuples inside them stay tuples); the contrast, rows
            and subjects are objects; `covariance` is a list; `diagnostics`,
            `limitations`, `reference` and `digests` are shallow dictionary
            copies.
        """
        return {
            "estimand": self.estimand.to_dict(),
            "policy": self.policy.to_dict(),
            "periods": [item.to_dict() for item in self.periods],
            "regimes": [item.to_dict() for item in self.regimes],
            "contrast": {
                **dataclasses.asdict(self.contrast),
                "interval": list(self.contrast.interval),
            },
            "covariance": [list(row) for row in self.covariance],
            "rows": dataclasses.asdict(self.rows),
            "subjects": dataclasses.asdict(self.subjects),
            "diagnostics": dict(self.diagnostics),
            "limitations": dict(self.limitations),
            "reference": dict(self.reference),
            "digests": dict(self.digests),
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Estimation":
        """Rebuilds a result from its JSON-compatible representation.

        A missing key raises `KeyError`.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The result built from the record; field validation runs on
            construction.
        """
        estimand_values = dict(record["estimand"])
        estimand_values["baseline_covariates"] = tuple(
            estimand_values["baseline_covariates"]
        )
        estimand_values["time_varying_covariates"] = tuple(
            estimand_values["time_varying_covariates"]
        )
        estimand_values["assumptions"] = tuple(estimand_values["assumptions"])
        policy = Config(**dict(record["policy"]))
        periods = tuple(
            PeriodDiagnostics(**dict(item)) for item in record["periods"]
        )
        regimes = tuple(
            RegimeEstimate(
                regime=str(item["regime"]),
                treatment=int(item["treatment"]),
                estimate=float(item["estimate"]),
                standard_error=float(item["standard_error"]),
                interval=(
                    float(item["interval"][0]),
                    float(item["interval"][1]),
                ),
                rows=int(item["rows"]),
                weight_sum=float(item["weight_sum"]),
                effective_sample_size=float(item["effective_sample_size"]),
            )
            for item in record["regimes"]
        )
        return cls(
            estimand=Estimand(**estimand_values),
            policy=policy,
            periods=periods,
            regimes=regimes,
            contrast=Contrast(
                estimate=float(record["contrast"]["estimate"]),
                se=float(record["contrast"]["se"]),
                interval=(
                    float(record["contrast"]["interval"][0]),
                    float(record["contrast"]["interval"][1]),
                ),
            ),
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in record["covariance"]
            ),
            rows=Counts(**record["rows"]),
            subjects=Counts(**record["subjects"]),
            diagnostics=dict(record["diagnostics"]),
            limitations=dict(record["limitations"]),
            reference=dict(record["reference"]),
            digests={
                key: str(value) for key, value in record["digests"].items()
            },
        )
