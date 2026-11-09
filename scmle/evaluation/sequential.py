"""Bounded sequential causal estimation for binary treatment histories.

This module estimates a static sustained-treatment contrast from a balanced
long-format panel.  At each period it fits a denominator propensity model
conditioned on treatment history, baseline covariates, and declared
pre-treatment time-varying covariates.  A numerator model conditioned only on
treatment history stabilizes the cumulative inverse-probability weight.

The contract is deliberately bounded: it does not claim support for dynamic
regimes, intermittent missing visits, continuous treatment, g-computation,
or nuisance-model uncertainty propagation.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
from itertools import pairwise
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy
import pandas
from scipy import special
from scipy import stats

from scmle import durability

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
    """

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
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("estimand name must not be empty")
        names = (
            self.subject_identity,
            self.time_column,
            self.treatment_column,
            self.outcome,
            *self.baseline_covariates,
            *self.time_varying_covariates,
        )
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("estimand column names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ValueError("estimand columns must be unique")
        if not isinstance(self.population, str) or not self.population.strip():
            raise ValueError("estimand population must not be empty")
        if not self.assumptions:
            raise ValueError("estimand needs explicit assumptions")
        missing = REQUIRED_ASSUMPTIONS.difference(self.assumptions)
        if missing:
            raise ValueError(
                "estimand is missing required assumptions: "
                + ", ".join(sorted(missing))
            )
        if self.target_treatment not in (0, 1):
            raise ValueError("target_treatment must be 0 or 1")
        if self.comparator_treatment not in (0, 1):
            raise ValueError("comparator_treatment must be 0 or 1")
        if self.target_treatment == self.comparator_treatment:
            raise ValueError("target and comparator treatments must differ")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible estimand metadata.

        Returns:
            The estimand fields keyed by field name.
        """
        return dataclasses.asdict(self)


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
    """

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
            raise ValueError(
                "confidence_level must lie strictly between 0 and 1"
            )
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError(
                "propensity_clip must lie strictly between 0 and 0.5"
            )
        if not math.isfinite(self.maximum_weight) or self.maximum_weight <= 1.0:
            raise ValueError("maximum_weight must be finite and greater than 1")
        if self.missingness not in ("reject", "complete_case"):
            raise ValueError("missingness must be reject or complete_case")
        if (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise ValueError("seed must be a non-negative integer")
        for name, value in (
            ("tolerance", self.tolerance),
            ("reference_tolerance", self.reference_tolerance),
        ):
            if not math.isfinite(float(value)) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        for name, value in (
            ("max_iterations", self.max_iterations),
            ("minimum_subjects", self.minimum_subjects),
            ("minimum_arm_rows", self.minimum_arm_rows),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        if (
            not math.isfinite(self.minimum_effective_sample_size)
            or self.minimum_effective_sample_size <= 0.0
        ):
            raise ValueError("minimum_effective_sample_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible policy metadata.

        Returns:
            The policy fields keyed by field name.
        """
        return dataclasses.asdict(self)


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
            raise ValueError("period label must not be empty")
        if self.rows < 1 or self.treated_rows < 1 or self.control_rows < 1:
            raise ValueError("period row counts must be positive")
        if self.treated_rows + self.control_rows != self.rows:
            raise ValueError("period arm counts must reconcile to rows")
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
                raise ValueError(f"{name} must be finite")
        if not 0.0 <= self.fraction_weight_truncated <= 1.0:
            raise ValueError(
                "fraction_weight_truncated must lie between 0 and 1"
            )
        for name, value in (
            ("denominator_propensity_min", self.denominator_propensity_min),
            ("denominator_propensity_max", self.denominator_propensity_max),
            ("numerator_propensity_min", self.numerator_propensity_min),
            ("numerator_propensity_max", self.numerator_propensity_max),
        ):
            if not 0.0 < value < 1.0:
                raise ValueError(f"{name} must lie strictly between 0 and 1")
        if self.maximum_untruncated_weight <= 0.0 or self.maximum_weight <= 0.0:
            raise ValueError("weight bounds must be positive")
        if self.effective_sample_size <= 0.0:
            raise ValueError("effective_sample_size must be positive")

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
            raise ValueError("regime name must not be empty")
        if self.treatment not in (0, 1):
            raise ValueError("regime treatment must be 0 or 1")
        if self.rows < 1 or self.weight_sum <= 0.0:
            raise ValueError("regime needs positive rows and weight sum")
        if self.effective_sample_size <= 0.0:
            raise ValueError("regime effective sample size must be positive")
        for name, value in (
            ("estimate", self.estimate),
            ("standard_error", self.standard_error),
            ("interval_lower", self.interval[0]),
            ("interval_upper", self.interval[1]),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.standard_error < 0.0 or self.interval[0] > self.interval[1]:
            raise ValueError(
                "regime uncertainty must be ordered and non-negative"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible regime evidence.

        Returns:
            The regime fields keyed by field name.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Estimation:
    """Complete auditable evidence for one sequential estimation run.

    Attributes:
        estimand: The declared estimand.
        policy: The policy the run used.
        periods: Per-period diagnostics, in time order; at least two.
        regimes: The target regime estimate followed by the comparator
            regime estimate.
        contrast_estimate: Target mean minus comparator mean.
        contrast_standard_error: Standard error of the contrast.
        contrast_interval: Lower and upper confidence limits of the
            contrast.
        covariance: Symmetric 2 by 2 covariance of the two regime means,
            ordered target then comparator.
        rows_input: Number of rows in the supplied frame.
        rows_used: Number of rows analysed.
        rows_dropped: Number of rows removed by the missingness policy.
        subjects_input: Number of distinct subjects in the supplied frame.
        subjects_used: Number of subjects analysed.
        subjects_dropped: Number of subjects removed by the missingness
            policy.
        diagnostics: Method, model, and fit details.
        limitations: Statements of what the estimate does not cover.
        reference: Record of the independent recomputation of the regime
            means.
        raw_input_fingerprint: Lowercase SHA-256 digest of the supplied
            frame.
    """

    estimand: Estimand
    policy: Config
    periods: tuple[PeriodDiagnostics, ...]
    regimes: tuple[RegimeEstimate, ...]
    contrast_estimate: float
    contrast_standard_error: float
    contrast_interval: tuple[float, float]
    covariance: tuple[tuple[float, ...], ...]
    rows_input: int
    rows_used: int
    rows_dropped: int
    subjects_input: int
    subjects_used: int
    subjects_dropped: int
    diagnostics: Mapping[str, Any]
    limitations: Mapping[str, Any]
    reference: Mapping[str, Any]
    raw_input_fingerprint: str

    def __post_init__(self) -> None:
        """Validates report shape and accounting before serialization.

        Raises:
            ValueError: If there are fewer than two periods, there are not
                exactly two regimes, the covariance is not a finite
                symmetric 2 by 2 matrix, a count is not a non-negative
                integer, used plus dropped counts do not equal input
                counts, a contrast value is not finite, the standard error
                is negative, the interval is reversed, or the fingerprint
                is not a lowercase SHA-256 digest.
            TypeError: If `diagnostics`, `limitations`, or `reference` is
                not a mapping.
        """
        if len(self.periods) < 2:
            raise ValueError("sequential result needs at least two periods")
        if len(self.regimes) != 2:
            raise ValueError(
                "sequential result needs target and comparator regimes"
            )
        if len(self.covariance) != 2 or any(
            len(row) != 2 for row in self.covariance
        ):
            raise ValueError("covariance must be a 2 by 2 matrix")
        covariance = numpy.asarray(self.covariance, dtype=float)
        if not numpy.isfinite(covariance).all() or not numpy.allclose(
            covariance, covariance.T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("covariance must be finite and symmetric")
        for name, value in (
            ("rows_input", self.rows_input),
            ("rows_used", self.rows_used),
            ("rows_dropped", self.rows_dropped),
            ("subjects_input", self.subjects_input),
            ("subjects_used", self.subjects_used),
            ("subjects_dropped", self.subjects_dropped),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.rows_used + self.rows_dropped != self.rows_input:
            raise ValueError("row accounting must reconcile to rows_input")
        if self.subjects_used + self.subjects_dropped != self.subjects_input:
            raise ValueError(
                "subject accounting must reconcile to subjects_input"
            )
        for metric_name, metric_value in (
            ("contrast_estimate", self.contrast_estimate),
            ("contrast_standard_error", self.contrast_standard_error),
            ("contrast_interval_lower", self.contrast_interval[0]),
            ("contrast_interval_upper", self.contrast_interval[1]),
        ):
            if not math.isfinite(float(metric_value)):
                raise ValueError(f"{metric_name} must be finite")
        if self.contrast_standard_error < 0.0:
            raise ValueError("contrast_standard_error must not be negative")
        if self.contrast_interval[0] > self.contrast_interval[1]:
            raise ValueError("contrast interval must be ordered")
        for mapping_name, mapping_value in (
            ("diagnostics", self.diagnostics),
            ("limitations", self.limitations),
            ("reference", self.reference),
        ):
            if not isinstance(mapping_value, Mapping):
                raise TypeError(f"{mapping_name} must be a mapping")
        if (
            not isinstance(self.raw_input_fingerprint, str)
            or len(self.raw_input_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.raw_input_fingerprint
            )
        ):
            raise ValueError(
                "raw_input_fingerprint must be a lowercase SHA-256 digest"
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible sequential report.

        Returns:
            All report fields keyed by field name. The estimand, policy,
            periods and regimes are the dictionaries from their own
            `to_dict` (tuples inside them stay tuples); `contrast_interval`
            and `covariance` are lists; `diagnostics`, `limitations` and
            `reference` are shallow dictionary copies.
        """
        return {
            "estimand": self.estimand.to_dict(),
            "policy": self.policy.to_dict(),
            "periods": [item.to_dict() for item in self.periods],
            "regimes": [item.to_dict() for item in self.regimes],
            "contrast_estimate": self.contrast_estimate,
            "contrast_standard_error": self.contrast_standard_error,
            "contrast_interval": list(self.contrast_interval),
            "covariance": [list(row) for row in self.covariance],
            "rows_input": self.rows_input,
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "subjects_input": self.subjects_input,
            "subjects_used": self.subjects_used,
            "subjects_dropped": self.subjects_dropped,
            "diagnostics": dict(self.diagnostics),
            "limitations": dict(self.limitations),
            "reference": dict(self.reference),
            "raw_input_fingerprint": self.raw_input_fingerprint,
        }

    def digest(self) -> str:
        """Returns the digest of canonical report content.

        Returns:
            The hexadecimal SHA-256 digest of the report serialized as
            sorted-key compact JSON.
        """
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def write(self, path: str | Path) -> str:
        """Atomically writes a digest-bound report.

        The envelope holds `schema_version` 1, the `to_dict` report and its
        `digest`. It is written as sorted-key JSON to a temporary file in the
        destination directory, flushed, renamed over the destination, and the
        directory is then synced. On any failure the temporary file is
        removed and the exception propagates.

        Args:
            path: Destination file; missing parent directories are created.

        Returns:
            The report digest stored in the envelope.

        Raises:
            RuntimeError: If the temporary file path was not recorded
                (not reachable in practice, since it is set on entering
                the file context).
        """
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "schema_version": 1,
            "report": self.to_dict(),
            "digest": self.digest(),
        }
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                json.dump(envelope, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if temporary is None:
                raise RuntimeError("temporary report was not created")
            os.replace(temporary, destination)
            durability.sync_directory(destination.parent)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return self.digest()

    @classmethod
    def read(cls, path: str | Path) -> "Estimation":
        """Reads and verifies an atomic report envelope.

        An unreadable or malformed file raises the underlying OS or JSON
        error.

        Args:
            path: File previously written by `write`.

        Returns:
            The result rebuilt from the stored report.

        Raises:
            ValueError: If the schema version is not 1, the report is not a
                mapping, the stored digest does not match the report, or the
                rebuilt result serializes to a different digest.
        """
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if envelope.get("schema_version") != 1:
            raise ValueError("unsupported sequential report schema")
        record = envelope.get("report")
        if not isinstance(record, Mapping):
            raise ValueError("sequential report must contain a mapping")
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if envelope.get("digest") != digest:
            raise ValueError("sequential report digest mismatch")
        result = cls.from_dict(record)
        if result.digest() != digest:
            raise ValueError("sequential report canonicalization mismatch")
        return result

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
            contrast_estimate=float(record["contrast_estimate"]),
            contrast_standard_error=float(record["contrast_standard_error"]),
            contrast_interval=(
                float(record["contrast_interval"][0]),
                float(record["contrast_interval"][1]),
            ),
            covariance=tuple(
                tuple(float(value) for value in row)
                for row in record["covariance"]
            ),
            rows_input=int(record["rows_input"]),
            rows_used=int(record["rows_used"]),
            rows_dropped=int(record["rows_dropped"]),
            subjects_input=int(record["subjects_input"]),
            subjects_used=int(record["subjects_used"]),
            subjects_dropped=int(record["subjects_dropped"]),
            diagnostics=dict(record["diagnostics"]),
            limitations=dict(record["limitations"]),
            reference=dict(record["reference"]),
            raw_input_fingerprint=str(record["raw_input_fingerprint"]),
        )


class Estimator:
    """Estimates a sustained-treatment contrast with stabilized sequential IPW.

    Attributes:
        estimand: The declared estimand.
        policy: The estimation policy; the default policy when none was
            supplied.
    """

    def __init__(
        self,
        estimand: Estimand,
        policy: Config | None = None,
    ) -> None:
        """Stores a validated estimand and bounded policy.

        Args:
            estimand: The contrast to estimate.
            policy: Estimation policy; `Config()` is
                used when None.

        Raises:
            TypeError: If `estimand` is not a `Estimand` or
                `policy` is neither None nor a `Config`.
        """
        if not isinstance(estimand, Estimand):
            raise TypeError("estimand must be Estimand")
        if policy is not None and not isinstance(policy, Config):
            raise TypeError("policy must be Config")
        self.estimand = estimand
        self.policy = policy or Config()

    def fit(self, frame: pandas.DataFrame) -> Estimation:
        """Fits sequential stabilized weights and a sustained regime contrast.

        Args:
            frame: Balanced long-format panel with one row per subject and
                period.

        Returns:
            The regime means, their contrast, per-period diagnostics, and
            accounting for the run.

        Raises:
            ValueError: If a period has fewer than `minimum_arm_rows`
                subjects in either arm or its effective sample size is
                below the policy minimum.  Invalid panels, failed
                propensity fits, positivity violations, and unsupported
                regimes raise `ValueError` from the helper functions.
        """
        prepared, rows_dropped, subjects_input, subjects_dropped = (
            prepare_sequential_frame(frame, self.estimand, self.policy)
        )
        fingerprint = fingerprint_frame(frame)
        period_values = ordered_periods(prepared, self.estimand.time_column)
        subject_values = ordered_subjects(
            prepared, self.estimand.subject_identity
        )
        matrices = build_matrices(
            prepared, self.estimand, period_values, subject_values
        )
        treatment = matrices["treatment"]
        outcome = matrices["outcome"][:, -1]
        cumulative = numpy.ones(len(subject_values), dtype=float)
        period_diagnostics: list[PeriodDiagnostics] = []
        critical_value = float(
            stats.norm.ppf(0.5 + self.policy.confidence_level / 2.0)
        )
        for index, value in enumerate(period_values):
            current_treatment = treatment[:, index]
            treated_rows = int(numpy.sum(current_treatment == 1.0))
            control_rows = int(numpy.sum(current_treatment == 0.0))
            if (
                treated_rows < self.policy.minimum_arm_rows
                or control_rows < self.policy.minimum_arm_rows
            ):
                raise ValueError(
                    f"period {value!r} has insufficient treatment support"
                )
            history = treatment[:, :index]
            numerator_features = history
            denominator_features = numpy.column_stack(
                (
                    history,
                    matrices["baseline"],
                    matrices["time_varying"][:, index, :],
                )
            )
            numerator, numerator_iterations = fit_propensity(
                numerator_features, current_treatment, self.policy
            )
            denominator, denominator_iterations = fit_propensity(
                denominator_features, current_treatment, self.policy
            )
            check_propensity_support(
                denominator, value, self.policy.propensity_clip
            )
            increment = numpy.where(
                current_treatment == 1.0,
                numerator / denominator,
                (1.0 - numerator) / (1.0 - denominator),
            )
            cumulative *= increment
            truncated = numpy.minimum(cumulative, self.policy.maximum_weight)
            period_diagnostic = PeriodDiagnostics(
                period=period_label(value),
                rows=len(subject_values),
                treated_rows=treated_rows,
                control_rows=control_rows,
                denominator_propensity_min=float(numpy.min(denominator)),
                denominator_propensity_max=float(numpy.max(denominator)),
                numerator_propensity_min=float(numpy.min(numerator)),
                numerator_propensity_max=float(numpy.max(numerator)),
                fraction_weight_truncated=float(
                    numpy.mean(cumulative > self.policy.maximum_weight)
                ),
                maximum_untruncated_weight=float(numpy.max(cumulative)),
                maximum_weight=float(numpy.max(truncated)),
                effective_sample_size=effective_sample_size(truncated),
            )
            if (
                period_diagnostic.effective_sample_size
                < self.policy.minimum_effective_sample_size
            ):
                raise ValueError(
                    f"period {value!r} has insufficient effective sample size"
                )
            period_diagnostics.append(period_diagnostic)
            if index == len(period_values) - 1:
                iterations = {
                    "numerator": numerator_iterations,
                    "denominator": denominator_iterations,
                }
        final_weights = numpy.minimum(cumulative, self.policy.maximum_weight)
        target_mask = numpy.all(
            treatment == self.estimand.target_treatment, axis=1
        )
        comparator_mask = numpy.all(
            treatment == self.estimand.comparator_treatment, axis=1
        )
        target = build_regime_estimate(
            "always_treated"
            if self.estimand.target_treatment == 1
            else "always_control",
            self.estimand.target_treatment,
            target_mask,
            final_weights,
            outcome,
            self.policy,
            critical_value,
        )
        comparator = build_regime_estimate(
            "always_treated"
            if self.estimand.comparator_treatment == 1
            else "always_control",
            self.estimand.comparator_treatment,
            comparator_mask,
            final_weights,
            outcome,
            self.policy,
            critical_value,
        )
        regimes = (target, comparator)
        covariance = regime_covariance(
            target_mask,
            comparator_mask,
            final_weights,
            outcome,
            target,
            comparator,
        )
        contrast = target.estimate - comparator.estimate
        contrast_standard_error = math.sqrt(max(float(covariance.sum()), 0.0))
        reference = independent_reference(
            target_mask,
            comparator_mask,
            final_weights,
            outcome,
            target,
            comparator,
            self.policy.reference_tolerance,
        )
        return Estimation(
            estimand=self.estimand,
            policy=self.policy,
            periods=tuple(period_diagnostics),
            regimes=regimes,
            contrast_estimate=contrast,
            contrast_standard_error=contrast_standard_error,
            contrast_interval=(
                contrast - critical_value * contrast_standard_error,
                contrast + critical_value * contrast_standard_error,
            ),
            covariance=tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            rows_input=len(frame),
            rows_used=len(prepared),
            rows_dropped=rows_dropped,
            subjects_input=subjects_input,
            subjects_used=len(subject_values),
            subjects_dropped=subjects_dropped,
            diagnostics={
                "method": "stabilized_treatment_history_ipw",
                "propensity_model": "deterministic_newton_logistic_with_ridge",
                "period_count": len(period_values),
                "subject_count": len(subject_values),
                "regime_history": {
                    "target": [self.estimand.target_treatment]
                    * len(period_values),
                    "comparator": [self.estimand.comparator_treatment]
                    * len(period_values),
                },
                "weight_truncation": {
                    "maximum_weight": self.policy.maximum_weight,
                    "fraction_truncated_final": period_diagnostics[
                        -1
                    ].fraction_weight_truncated,
                },
                "periods": [item.to_dict() for item in period_diagnostics],
                "fit_iterations_final_period": iterations,
                "time_ordering": "strictly_increasing_with_complete_follow_up",
            },
            limitations={
                "regimes": (
                    "static always-treated versus always-control histories only"
                ),
                "missingness": {
                    "policy": self.policy.missingness,
                    "subjects_dropped": subjects_dropped,
                    "message": (
                        "Complete-case deletion is not a missing-at-random "
                        "proof; "
                        "the declared assumptions must cover missingness."
                    ),
                },
                "truncation": (
                    "Hard cumulative-weight truncation trades variance for "
                    "bias; "
                    "the cap is recorded in policy and diagnostics."
                ),
                "uncertainty": (
                    "Intervals include weighted outcome influence only; "
                    "propensity-model uncertainty is not propagated."
                ),
                "unsupported": (
                    "Dynamic regimes, intermittent follow-up, continuous "
                    "treatment, "
                    "g-computation, and marginal structural model "
                    "contrasts are "
                    "outside this bounded contract."
                ),
            },
            reference=reference,
            raw_input_fingerprint=fingerprint,
        )


def prepare_sequential_frame(
    frame: pandas.DataFrame,
    estimand: Estimand,
    policy: Config,
) -> tuple[pandas.DataFrame, int, int, int]:
    """Validates and prepares a balanced long-format panel.

    Args:
        frame: Long-format panel to check.
        estimand: Declares the columns that must be present.
        policy: Supplies the missingness rule and the minimum subject count.

    Returns:
        A tuple of the prepared frame (declared columns only, with the
        treatment, outcome and covariate columns converted to numeric and
        dropped subjects removed), the number of rows dropped, the number of
        distinct subjects in the input, and the number of subjects dropped.

    Raises:
        TypeError: If `frame` is not a pandas DataFrame.
        ValueError: If the frame is empty, declared columns are missing,
            subject or time values are missing or non-finite, subject-time
            pairs repeat, time values are invalid, values are missing under
            the "reject" policy, a column is non-numeric or non-finite,
            nothing remains after dropping, treatment is not binary,
            fewer than two periods remain, a subject lacks a row in some
            period, too few subjects remain, or a baseline covariate varies
            within a subject.
    """
    if not isinstance(frame, pandas.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("sequential frame must not be empty")
    required = [
        estimand.subject_identity,
        estimand.time_column,
        estimand.treatment_column,
        estimand.outcome,
        *estimand.baseline_covariates,
        *estimand.time_varying_covariates,
    ]
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"sequential frame is missing columns: {missing_columns}"
        )
    subject_name = estimand.subject_identity
    time_name = estimand.time_column
    if frame[subject_name].isna().any() or frame[time_name].isna().any():
        raise ValueError(
            "subject identities and time values must not be missing"
        )
    for name in (subject_name, time_name):
        if pandas.api.types.is_numeric_dtype(frame[name]):
            values = frame[name].to_numpy(dtype=float)
            if not numpy.isfinite(values).all():
                raise ValueError(f"column {name!r} contains non-finite values")
    if frame.duplicated([subject_name, time_name]).any():
        raise ValueError("duplicate subject-time rows are not allowed")
    subjects_input = int(frame[subject_name].nunique())
    validate_time_order(frame, estimand)
    value_columns = [
        estimand.treatment_column,
        estimand.outcome,
        *estimand.baseline_covariates,
        *estimand.time_varying_covariates,
    ]
    missing_mask = frame[value_columns].isna().any(axis=1)
    subjects_dropped = 0
    rows_dropped = 0
    prepared = frame[required].copy()
    if missing_mask.any() and policy.missingness == "reject":
        raise ValueError("missing treatment, outcome, or covariate values")
    if missing_mask.any():
        bad_subjects = set(frame.loc[missing_mask, subject_name].tolist())
        subjects_dropped = len(bad_subjects)
        keep_mask = ~prepared[subject_name].isin(bad_subjects)
        rows_dropped = int((~keep_mask).sum())
        prepared = prepared.loc[keep_mask].copy()
    for name in value_columns:
        try:
            prepared[name] = pandas.to_numeric(prepared[name], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"column {name!r} must be numeric") from error
        values = prepared[name].to_numpy(dtype=float)
        if not numpy.isfinite(values).all():
            raise ValueError(f"column {name!r} contains non-finite values")
    if prepared.empty:
        raise ValueError("missingness policy left no subjects")
    treatment_values = set(
        prepared[estimand.treatment_column].unique().tolist()
    )
    if not treatment_values.issubset({0, 1}):
        raise ValueError("treatment must be binary with values 0 and 1")
    period_values = ordered_periods(prepared, time_name)
    if len(period_values) < 2:
        raise ValueError("sequential estimation needs at least two periods")
    counts = prepared.groupby(subject_name, sort=False).size()
    if not (counts == len(period_values)).all():
        raise ValueError("each subject must have one row at every period")
    if len(prepared[subject_name].unique()) < policy.minimum_subjects:
        raise ValueError("sequential frame has insufficient subjects")
    validate_baseline_covariates(prepared, estimand)
    return prepared, rows_dropped, subjects_input, subjects_dropped


def validate_time_order(frame: pandas.DataFrame, estimand: Estimand) -> None:
    """Rejects repeated consecutive time values within a subject.

    Only equal neighbouring values (in row order) are rejected; descending
    values are not.

    Args:
        frame: Long-format panel.
        estimand: Names the subject and time columns.

    Raises:
        ValueError: If a subject has two equal consecutive time values, or
            its time values cannot be examined in order.
        TypeError: Re-raised unchanged only if its message contains
            "strictly increasing", which the comparisons here never
            produce; every other `TypeError` becomes `ValueError`.
    """
    try:
        for item in frame.groupby(
            estimand.subject_identity, sort=False, dropna=False
        ):
            group = item[1]
            values = group[estimand.time_column].tolist()
            if any(left == right for left, right in pairwise(values)):
                raise ValueError(
                    "time values must be strictly increasing per subject"
                )
    except (TypeError, ValueError) as error:
        if "strictly increasing" in str(error):
            raise
        raise ValueError("time values have invalid ordering") from error


def ordered_periods(frame: pandas.DataFrame, time_column: str) -> list[Any]:
    """Returns stable sorted period values.

    Args:
        frame: Long-format panel.
        time_column: Name of the time column.

    Returns:
        The distinct time values in ascending order.

    Raises:
        ValueError: If the time values cannot be sorted.
    """
    try:
        return frame[time_column].drop_duplicates().sort_values().tolist()
    except (TypeError, ValueError) as error:
        raise ValueError("time values have invalid ordering") from error


def ordered_subjects(frame: pandas.DataFrame, subject_column: str) -> list[Any]:
    """Returns subject identities in a deterministic, type-aware order.

    Args:
        frame: Long-format panel.
        subject_column: Name of the subject column.

    Returns:
        The distinct subject identities sorted by type name, then by
        `repr`.
    """
    values = frame[subject_column].drop_duplicates().tolist()
    return sorted(values, key=lambda value: (type(value).__name__, repr(value)))


def validate_baseline_covariates(
    frame: pandas.DataFrame, estimand: Estimand
) -> None:
    """Ensures declared baseline covariates are stable within subjects.

    Args:
        frame: Long-format panel.
        estimand: Names the subject column and the baseline covariates.

    Raises:
        ValueError: If a baseline covariate takes more than one value
            (counting missing as a value) within a subject.
    """
    grouped = frame.groupby(estimand.subject_identity, sort=False)
    for name in estimand.baseline_covariates:
        if (grouped[name].nunique(dropna=False) > 1).any():
            raise ValueError(
                f"baseline covariate {name!r} varies within a subject"
            )


def build_matrices(
    frame: pandas.DataFrame,
    estimand: Estimand,
    period_values: Sequence[Any],
    subject_values: Sequence[Any],
) -> dict[str, numpy.ndarray]:
    """Builds deterministic subject-by-period arrays from the long frame.

    Args:
        frame: Prepared long-format panel.
        estimand: Names the treatment, outcome, and covariate columns.
        period_values: Periods in time order; fixes the period axis.
        subject_values: Subjects in order; fixes the subject axis.

    Returns:
        A dictionary with "treatment" and "outcome" (subjects by periods),
        "baseline" (subjects by baseline covariates, read at the first
        period), and "time_varying" (subjects by periods by time-varying
        covariates).

    Raises:
        ValueError: If a subject's first-period subject-time pair maps to
            more than one row.
    """
    indexed = frame.set_index(
        [estimand.subject_identity, estimand.time_column], drop=False
    )
    treatment = numpy.empty(
        (len(subject_values), len(period_values)), dtype=float
    )
    outcome = numpy.empty_like(treatment)
    baseline = numpy.empty(
        (len(subject_values), len(estimand.baseline_covariates))
    )
    time_varying = numpy.empty(
        (
            len(subject_values),
            len(period_values),
            len(estimand.time_varying_covariates),
        )
    )
    for subject_index, subject in enumerate(subject_values):
        first = indexed.loc[(subject, period_values[0])]
        if isinstance(first, pandas.DataFrame):
            raise ValueError("subject-time index is not unique")
        baseline[subject_index] = first[
            list(estimand.baseline_covariates)
        ].to_numpy(dtype=float)
        for period_index, period in enumerate(period_values):
            row = indexed.loc[(subject, period)]
            treatment[subject_index, period_index] = float(
                row[estimand.treatment_column]
            )
            outcome[subject_index, period_index] = float(row[estimand.outcome])
            time_varying[subject_index, period_index] = row[
                list(estimand.time_varying_covariates)
            ].to_numpy(dtype=float)
    return {
        "treatment": treatment,
        "outcome": outcome,
        "baseline": baseline,
        "time_varying": time_varying,
    }


def fit_propensity(
    covariates: numpy.ndarray,
    treatment: numpy.ndarray,
    policy: Config,
) -> tuple[numpy.ndarray, int]:
    """Fits a deterministic ridge-logistic propensity by bounded Newton steps.

    Args:
        covariates: Two-dimensional matrix, subjects by features, without an
            intercept column; one is added.
        treatment: Treatment indicator per subject (0 or 1).
        policy: Supplies the iteration limit and convergence tolerance.

    Returns:
        A tuple of the fitted treatment probability for each subject and
        the number of iterations used.

    Raises:
        ValueError: If `covariates` is not two-dimensional, its rows do not
            match `treatment`, the Hessian is singular, a step is not
            finite, or the fit does not converge within
            `policy.max_iterations`.
    """
    if covariates.ndim != 2:
        raise ValueError("covariates must be a two-dimensional matrix")
    if len(covariates) != len(treatment):
        raise ValueError("covariates and treatment must have matching rows")
    design = numpy.column_stack((numpy.ones(len(treatment)), covariates))
    coefficients = numpy.zeros(design.shape[1], dtype=float)
    ridge = 1e-8
    ridge_matrix = ridge * numpy.eye(design.shape[1])
    for iteration in range(1, policy.max_iterations + 1):
        probabilities = special.expit(design @ coefficients)
        variance = numpy.maximum(probabilities * (1.0 - probabilities), 1e-12)
        gradient = design.T @ (probabilities - treatment) + ridge * coefficients
        hessian = design.T @ (variance[:, None] * design) + ridge_matrix
        try:
            step = numpy.linalg.solve(hessian, gradient)
        except numpy.linalg.LinAlgError as error:
            raise ValueError("propensity model is not identifiable") from error
        if not numpy.isfinite(step).all():
            raise ValueError(
                "propensity model produced non-finite coefficients"
            )
        coefficients -= step
        if float(numpy.max(numpy.abs(step))) <= policy.tolerance:
            return special.expit(design @ coefficients), iteration
    raise ValueError("propensity model did not converge within policy bound")


def check_propensity_support(
    propensity: numpy.ndarray, period: Any, minimum: float
) -> None:
    """Rejects fitted propensities outside the declared positivity support.

    Args:
        propensity: Fitted probabilities for one period.
        period: Period value, used in the error message.
        minimum: Smallest allowed probability; the largest allowed is
            `1 - minimum`.

    Raises:
        ValueError: If any probability is below `minimum` or above
            `1 - minimum`.
    """
    lower = float(numpy.min(propensity))
    upper = float(numpy.max(propensity))
    if lower < minimum or upper > 1.0 - minimum:
        raise ValueError(
            f"period {period!r} violates declared positivity support "
            f"[{minimum}, {1.0 - minimum}]"
        )


def effective_sample_size(weights: numpy.ndarray) -> float:
    """Returns the Kish effective sample size for positive weights.

    Args:
        weights: Non-negative weights.

    Returns:
        The squared sum of the weights divided by the sum of their squares,
        or 0.0 when the sum of squares is not positive.
    """
    total = float(numpy.sum(weights))
    squared = float(numpy.sum(weights**2))
    if squared <= 0.0:
        return 0.0
    return total**2 / squared


def build_regime_estimate(
    regime: str,
    treatment: int,
    mask: numpy.ndarray,
    weights: numpy.ndarray,
    outcome: numpy.ndarray,
    policy: Config,
    critical_value: float,
) -> RegimeEstimate:
    """Builds a normalized weighted regime mean and influence standard error.

    Args:
        regime: Name of the regime.
        treatment: Treatment value held in every period.
        mask: Boolean flag per subject for following the regime.
        weights: Capped cumulative weight per subject.
        outcome: Final-period outcome per subject.
        policy: Supplies the support and effective-sample-size minimums.
        critical_value: Normal quantile that scales the standard error into
            the interval half-width.

    Returns:
        The weighted mean over the selected subjects with its standard
        error, interval, row count, weight sum, and effective sample size.

    Raises:
        ValueError: If fewer than `policy.minimum_arm_rows` subjects follow
            the regime, their weights sum to zero or less, or their
            effective sample size is below the policy minimum.
    """
    selected_weights = weights[mask]
    selected_outcome = outcome[mask]
    if len(selected_weights) < policy.minimum_arm_rows:
        raise ValueError(
            f"regime {regime!r} has insufficient treatment-history support"
        )
    weight_sum = float(numpy.sum(selected_weights))
    if weight_sum <= 0.0:
        raise ValueError(f"regime {regime!r} has no weighted support")
    ess = effective_sample_size(selected_weights)
    if ess < policy.minimum_effective_sample_size:
        raise ValueError(
            f"regime {regime!r} has insufficient effective sample size"
        )
    estimate = float(
        numpy.sum(selected_weights * selected_outcome) / weight_sum
    )
    influence = selected_weights * (selected_outcome - estimate) / weight_sum
    standard_error = math.sqrt(max(float(numpy.sum(influence**2)), 0.0))
    return RegimeEstimate(
        regime=regime,
        treatment=treatment,
        estimate=estimate,
        standard_error=standard_error,
        interval=(
            estimate - critical_value * standard_error,
            estimate + critical_value * standard_error,
        ),
        rows=len(selected_weights),
        weight_sum=weight_sum,
        effective_sample_size=ess,
    )


def regime_covariance(
    target_mask: numpy.ndarray,
    comparator_mask: numpy.ndarray,
    weights: numpy.ndarray,
    outcome: numpy.ndarray,
    target: RegimeEstimate,
    comparator: RegimeEstimate,
) -> numpy.ndarray:
    """Estimates subject-level covariance of two normalized regime means.

    Args:
        target_mask: Boolean flag per subject for following the target
            regime.
        comparator_mask: Boolean flag per subject for following the
            comparator regime.
        weights: Capped cumulative weight per subject.
        outcome: Final-period outcome per subject.
        target: Target regime estimate.
        comparator: Comparator regime estimate.

    Returns:
        A symmetric 2 by 2 array ordered target then comparator, holding
        sums over subjects of the products of the influence values.
    """
    target_influence = numpy.zeros(len(outcome), dtype=float)
    comparator_influence = numpy.zeros(len(outcome), dtype=float)
    target_influence[target_mask] = (
        weights[target_mask]
        * (outcome[target_mask] - target.estimate)
        / target.weight_sum
    )
    comparator_influence[comparator_mask] = (
        weights[comparator_mask]
        * (outcome[comparator_mask] - comparator.estimate)
        / comparator.weight_sum
    )
    return numpy.asarray(
        [
            [
                numpy.sum(target_influence**2),
                numpy.sum(target_influence * comparator_influence),
            ],
            [
                numpy.sum(target_influence * comparator_influence),
                numpy.sum(comparator_influence**2),
            ],
        ],
        dtype=float,
    )


def independent_reference(
    target_mask: numpy.ndarray,
    comparator_mask: numpy.ndarray,
    weights: numpy.ndarray,
    outcome: numpy.ndarray,
    target: RegimeEstimate,
    comparator: RegimeEstimate,
    tolerance: float,
) -> dict[str, Any]:
    """Recalculates normalized regime means through an independent path.

    Args:
        target_mask: Boolean flag per subject for following the target
            regime.
        comparator_mask: Boolean flag per subject for following the
            comparator regime.
        weights: Capped cumulative weight per subject.
        outcome: Final-period outcome per subject.
        target: Target regime estimate to compare against.
        comparator: Comparator regime estimate to compare against.
        tolerance: Largest absolute difference counted as agreement.

    Returns:
        A dictionary with the method name, the tolerance, an overall
        "passed" flag (true when every regime is within `tolerance`), one
        record per regime in target then comparator order (regime name,
        recomputed estimate, primary estimate, absolute difference, pass
        flag, rows, method name), and the recomputed contrast of target
        minus comparator.
    """
    records: list[dict[str, Any]] = []
    for name, mask, primary in (
        (target.regime, target_mask, target),
        (comparator.regime, comparator_mask, comparator),
    ):
        selected_weights = [float(value) for value in weights[mask]]
        selected_outcome = [float(value) for value in outcome[mask]]
        weighted_total = sum(
            weight * value
            for weight, value in zip(
                selected_weights, selected_outcome, strict=True
            )
        )
        reference_estimate = weighted_total / sum(selected_weights)
        difference = abs(primary.estimate - reference_estimate)
        records.append(
            {
                "regime": name,
                "estimate": reference_estimate,
                "primary_estimate": primary.estimate,
                "absolute_difference": difference,
                "passed": difference <= tolerance,
                "rows": len(selected_weights),
                "independent_method": "direct_normalized_history_weight_mean",
            }
        )
    contrast = float(records[0]["estimate"]) - float(records[1]["estimate"])
    return {
        "method": "independent_direct_normalized_history_weight_means",
        "tolerance": tolerance,
        "passed": all(bool(record["passed"]) for record in records),
        "regimes": records,
        "contrast": contrast,
    }


def period_label(value: Any) -> str:
    """Converts a period value into a stable report label.

    Args:
        value: A period value from the time column.

    Returns:
        The ISO 8601 text for a timestamp, otherwise `str` of the value
        with NumPy scalars unwrapped first.
    """
    if isinstance(value, pandas.Timestamp):
        return value.isoformat()
    if isinstance(value, numpy.generic):
        value = value.item()
    return str(value)


def fingerprint_frame(frame: pandas.DataFrame) -> str:
    """Returns a deterministic SHA-256 digest of the supplied frame.

    Args:
        frame: Frame to fingerprint.

    Returns:
        The hexadecimal digest covering the column names, dtypes, shape,
        and hashed row values including the index.

    Raises:
        ValueError: If the frame values cannot be hashed.
    """
    try:
        values = (
            pandas.util.hash_pandas_object(frame, index=True)
            .to_numpy()
            .tobytes()
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "frame values must be hashable for fingerprinting"
        ) from error
    descriptor = json.dumps(
        {
            "columns": [str(column) for column in frame.columns],
            "dtypes": [str(dtype) for dtype in frame.dtypes],
            "shape": list(frame.shape),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    digest.update(values)
    return digest.hexdigest()
