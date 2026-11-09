"""Bounded period-specific causal estimation for longitudinal data.

The estimator in this module treats each time period as a cross-sectional
binary-treatment study.  It does not estimate a sequential g-formula,
marginal structural model, or other longitudinal intervention estimand.
Longitudinal structure is used for identity, ordering, and influence-function
covariance only; the propensity model is refit independently in each period.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy import special
from scipy import stats

from scmle import durability

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
class LongitudinalEstimand:
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
    """

    name: str
    subject_identity: str
    time_column: str
    treatment_column: str
    outcome: str
    covariates: tuple[str, ...]
    population: str
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate the columns and required causal assumptions.

        Raises:
            ValueError: If the name or population is empty, any column name
                is empty or not a string, column names repeat, no
                assumptions are given, or a required assumption is missing.
        """
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("estimand name must not be empty")
        names = (
            self.subject_identity,
            self.time_column,
            self.treatment_column,
            self.outcome,
            *self.covariates,
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

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible estimand metadata.

        Returns:
            The estimand fields keyed by field name.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class LongitudinalEstimationPolicy:
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
    """

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
        """Reject an unbounded or ambiguous estimation policy.

        Raises:
            ValueError: If a field lies outside the range given in the class
                attributes, `missingness` is not a recognised value, or
                `minimum_rows_per_period` is below twice
                `minimum_rows_per_arm`.
        """
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must lie strictly between 0 and 1"
            )
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError(
                "propensity_clip must lie strictly between 0 and 0.5"
            )
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
            ("minimum_rows_per_period", self.minimum_rows_per_period),
            ("minimum_rows_per_arm", self.minimum_rows_per_arm),
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
        if self.minimum_rows_per_period < 2 * self.minimum_rows_per_arm:
            raise ValueError(
                "minimum_rows_per_period must cover both treatment arms"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible policy metadata.

        Returns:
            The policy fields keyed by field name.
        """
        return dataclasses.asdict(self)


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
    """

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
        """Validate finite period-level evidence.

        Raises:
            ValueError: If the period label is empty, a row count is not
                positive, arm counts do not sum to `rows`, `iterations` is
                below 1, the fit did not converge, a value is not finite,
                the standard error is negative, the interval is reversed,
                the clipped fraction lies outside [0, 1], or a propensity
                lies outside [0, 1].
        """
        if not self.period:
            raise ValueError("period label must not be empty")
        if self.rows < 1 or self.treated_rows < 1 or self.control_rows < 1:
            raise ValueError("period row counts must be positive")
        if self.treated_rows + self.control_rows != self.rows:
            raise ValueError("period arm counts must reconcile to rows")
        if self.iterations < 1:
            raise ValueError("period iterations must be positive")
        if not self.converged:
            raise ValueError("a period result must be converged")
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
                raise ValueError(f"{name} must be finite")
        if self.standard_error < 0.0:
            raise ValueError("standard_error must not be negative")
        if self.interval[0] > self.interval[1]:
            raise ValueError("period interval must be ordered")
        if not 0.0 <= self.fraction_clipped <= 1.0:
            raise ValueError("fraction_clipped must lie between 0 and 1")
        if self.propensity_min < 0.0 or self.propensity_max > 1.0:
            raise ValueError("propensities must lie between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible period evidence.

        Returns:
            The period fields keyed by field name.
        """
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class LongitudinalEstimationResult:
    """Complete serializable evidence for one longitudinal estimation run.

    Attributes:
        estimand: The declared estimand.
        policy: The policy the run used.
        periods: Per-period estimates, in time order, with unique labels.
        covariance: Symmetric covariance matrix of the period estimates, one
            row and column per period in the order of `periods`.
        rows_input: Number of rows in the supplied frame.
        rows_used: Number of rows analysed.
        rows_dropped: Number of rows removed by the missingness policy.
        diagnostics: Method, overlap, and effective-sample-size details.
        limitations: Statements of what the estimates do not cover.
        reference: Record of the independent recomputation of the period
            estimates.
        raw_input_fingerprint: Lowercase SHA-256 digest of the supplied
            frame.
    """

    estimand: LongitudinalEstimand
    policy: LongitudinalEstimationPolicy
    periods: tuple[PeriodATE, ...]
    covariance: tuple[tuple[float, ...], ...]
    rows_input: int
    rows_used: int
    rows_dropped: int
    diagnostics: Mapping[str, Any]
    limitations: Mapping[str, Any]
    reference: Mapping[str, Any]
    raw_input_fingerprint: str

    def __post_init__(self) -> None:
        """Validate the report shape before serialization.

        Raises:
            ValueError: If there are no periods, period labels repeat, the
                covariance is not square over the periods, not finite, or
                not symmetric, a count is not a non-negative integer, used
                plus dropped rows do not equal input rows, or the
                fingerprint is not a lowercase SHA-256 digest.
            TypeError: If `diagnostics`, `limitations`, or `reference` is
                not a mapping.
        """
        if not self.periods:
            raise ValueError("longitudinal result needs at least one period")
        labels = tuple(item.period for item in self.periods)
        if len(set(labels)) != len(labels):
            raise ValueError("period labels must be unique")
        expected = len(self.periods)
        if len(self.covariance) != expected or any(
            len(row) != expected for row in self.covariance
        ):
            raise ValueError("covariance must be square over periods")
        covariance = np.asarray(self.covariance, dtype=float)
        if not np.isfinite(covariance).all():
            raise ValueError("covariance must contain finite values")
        if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric")
        for name, value in (
            ("rows_input", self.rows_input),
            ("rows_used", self.rows_used),
            ("rows_dropped", self.rows_dropped),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.rows_used + self.rows_dropped != self.rows_input:
            raise ValueError("row accounting must reconcile to rows_input")
        for name, item_value in (
            ("diagnostics", self.diagnostics),
            ("limitations", self.limitations),
            ("reference", self.reference),
        ):
            if not isinstance(item_value, Mapping):
                raise TypeError(f"{name} must be a mapping")
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
        """Return a JSON-compatible longitudinal report.

        Returns:
            All report fields keyed by field name, with nested objects
            converted to dictionaries and tuples to lists.
        """
        return {
            "estimand": self.estimand.to_dict(),
            "policy": self.policy.to_dict(),
            "periods": [item.to_dict() for item in self.periods],
            "covariance": [list(row) for row in self.covariance],
            "rows_input": self.rows_input,
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "diagnostics": dict(self.diagnostics),
            "limitations": dict(self.limitations),
            "reference": dict(self.reference),
            "raw_input_fingerprint": self.raw_input_fingerprint,
        }

    def digest(self) -> str:
        """Return the digest of the canonical report content.

        Returns:
            The hexadecimal SHA-256 digest of the report serialized as
            sorted-key compact JSON.
        """
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def write(self, path: str | Path) -> str:
        """Atomically write a digest-bound report.

        The report is written to a temporary file in the destination
        directory, flushed, and renamed over the destination.

        Args:
            path: Destination file; missing parent directories are created.

        Returns:
            The report digest stored in the envelope.

        Raises:
            RuntimeError: If the temporary file path was not recorded.
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
    def read(cls, path: str | Path) -> "LongitudinalEstimationResult":
        """Read and verify an atomic report envelope.

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
            raise ValueError("unsupported longitudinal report schema")
        record = envelope.get("report")
        if not isinstance(record, Mapping):
            raise ValueError("longitudinal report must contain a mapping")
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if envelope.get("digest") != digest:
            raise ValueError("longitudinal report digest mismatch")
        report = cls.from_dict(record)
        if report.digest() != digest:
            raise ValueError("longitudinal report canonicalization mismatch")
        return report

    @classmethod
    def from_dict(
        cls, record: Mapping[str, Any]
    ) -> "LongitudinalEstimationResult":
        """Rebuild a result from its JSON-compatible representation.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The result built from the record; field validation runs on
            construction.
        """
        estimand_values = dict(record["estimand"])
        estimand_values["covariates"] = tuple(estimand_values["covariates"])
        estimand_values["assumptions"] = tuple(estimand_values["assumptions"])
        periods = tuple(
            PeriodATE(
                period=str(item["period"]),
                estimate=float(item["estimate"]),
                standard_error=float(item["standard_error"]),
                interval=(
                    float(item["interval"][0]),
                    float(item["interval"][1]),
                ),
                rows=int(item["rows"]),
                treated_rows=int(item["treated_rows"]),
                control_rows=int(item["control_rows"]),
                propensity_min=float(item["propensity_min"]),
                propensity_max=float(item["propensity_max"]),
                fraction_clipped=float(item["fraction_clipped"]),
                effective_sample_size_treated=float(
                    item["effective_sample_size_treated"]
                ),
                effective_sample_size_control=float(
                    item["effective_sample_size_control"]
                ),
                converged=bool(item["converged"]),
                iterations=int(item["iterations"]),
            )
            for item in record["periods"]
        )
        policy = LongitudinalEstimationPolicy(**dict(record["policy"]))
        return cls(
            LongitudinalEstimand(**estimand_values),
            policy,
            periods,
            tuple(
                tuple(float(value) for value in row)
                for row in record["covariance"]
            ),
            int(record["rows_input"]),
            int(record["rows_used"]),
            int(record["rows_dropped"]),
            dict(record["diagnostics"]),
            dict(record["limitations"]),
            dict(record["reference"]),
            str(record["raw_input_fingerprint"]),
        )


class LongitudinalIPWEstimator:
    """Estimate period-specific binary-treatment ATEs with Hájek IPW.

    Attributes:
        estimand: The declared estimand.
        policy: The estimation policy; the default policy when none was
            supplied.
    """

    def __init__(
        self,
        estimand: LongitudinalEstimand,
        policy: LongitudinalEstimationPolicy | None = None,
    ) -> None:
        """Store a validated declaration and policy.

        Args:
            estimand: The causal quantity to estimate.
            policy: Estimation policy; `LongitudinalEstimationPolicy()` is
                used when None.

        Raises:
            TypeError: If `estimand` is not a `LongitudinalEstimand`.
        """
        if not isinstance(estimand, LongitudinalEstimand):
            raise TypeError("estimand must be LongitudinalEstimand")
        self.estimand = estimand
        self.policy = policy or LongitudinalEstimationPolicy()

    def fit(self, frame: pd.DataFrame) -> LongitudinalEstimationResult:
        """Fit one covariate-adjusted period-specific ATE per time value.

        Args:
            frame: Long-format panel with one row per subject and period.

        Returns:
            The period estimates in ascending time order, their covariance,
            row accounting, diagnostics, and the independent recomputation.

        Raises:
            ValueError: If the time values cannot be sorted.  Invalid
                panels and periods that fail the support, overlap, or
                effective-sample-size checks raise `ValueError` from the
                helper functions and `fit_period`.
        """
        prepared, rows_dropped = prepare_longitudinal_frame(
            frame, self.estimand, self.policy
        )
        fingerprint = fingerprint_frame(frame)
        try:
            period_series = prepared[
                self.estimand.time_column
            ].drop_duplicates()
            period_values: list[Any] = period_series.sort_values().tolist()
        except (TypeError, ValueError) as error:
            raise ValueError("time values have invalid ordering") from error
        fitted: list[dict[str, Any]] = []
        for value in period_values:
            period_frame = prepared[
                prepared[self.estimand.time_column] == value
            ]
            fitted.append(self.fit_period(period_frame, value))
        covariance = influence_covariance(fitted)
        critical_value = float(
            stats.norm.ppf(0.5 + self.policy.confidence_level / 2.0)
        )
        periods = tuple(
            build_period_result(item, covariance[index, index], critical_value)
            for index, item in enumerate(fitted)
        )
        reference = independent_reference(
            fitted, self.policy.reference_tolerance
        )
        return LongitudinalEstimationResult(
            estimand=self.estimand,
            policy=self.policy,
            periods=periods,
            covariance=tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            rows_input=len(frame),
            rows_used=len(prepared),
            rows_dropped=rows_dropped,
            diagnostics={
                "method": "period_specific_covariate_adjusted_hajek_ipw",
                "propensity_model": "deterministic_newton_logistic_with_ridge",
                "period_count": len(periods),
                "subject_count": int(
                    prepared[self.estimand.subject_identity].nunique()
                ),
                "overlap": {
                    item["period"]: {
                        "propensity_min": item["propensity_min"],
                        "propensity_max": item["propensity_max"],
                        "common_support_lower": item["common_support_lower"],
                        "common_support_upper": item["common_support_upper"],
                        "fraction_clipped": item["fraction_clipped"],
                    }
                    for item in fitted
                },
                "effective_sample_size": {
                    item["period"]: {
                        "treated": item["effective_sample_size_treated"],
                        "control": item["effective_sample_size_control"],
                    }
                    for item in fitted
                },
                "covariance_method": (
                    "subject-aligned_influence_function_sandwich"
                ),
                "time_ordering": "strictly_increasing_within_subject",
            },
            limitations={
                "prior": {
                    "status": "not_applicable",
                    "message": (
                        "This frequentist IPW estimator has no Bayesian prior."
                    ),
                },
                "missingness": {
                    "policy": self.policy.missingness,
                    "rows_dropped": rows_dropped,
                    "message": (
                        "Complete-case deletion is not a missing-at-random "
                        "proof; "
                        "the declared assumptions must cover missingness."
                    ),
                },
                "distribution_shift": {
                    "status": "not_evaluated",
                    "message": "No target-population shift data were supplied.",
                },
                "longitudinal_scope": (
                    "Period-specific cross-sectional weighting; not a full "
                    "sequential g-formula or marginal structural model."
                ),
            },
            reference=reference,
            raw_input_fingerprint=fingerprint,
        )

    def fit_period(
        self, period_frame: pd.DataFrame, period_value: Any
    ) -> dict[str, Any]:
        """Fit one period and retain influence values for covariance.

        Args:
            period_frame: Prepared rows of a single period.
            period_value: The period's time value, used in labels and error
                messages.

        Returns:
            A dictionary of the period label, estimate, row and arm counts,
            raw and clipped propensities, treatment and outcome arrays,
            clipped fraction, propensity range, common-support bounds,
            arm effective sample sizes, subject identities, per-row
            influence values, fit iterations, and the treated and control
            weighted means.

        Raises:
            ValueError: If the period has fewer rows than
                `minimum_rows_per_period`, either arm has fewer rows than
                `minimum_rows_per_arm`, the treated and control propensity
                ranges do not overlap, an arm has no weighted support, or an
                arm's effective sample size is below the policy minimum.
                A failed propensity fit also raises `ValueError`.
        """
        policy = self.policy
        if len(period_frame) < policy.minimum_rows_per_period:
            raise ValueError(
                f"period {period_value!r} has insufficient rows: "
                f"{len(period_frame)} < {policy.minimum_rows_per_period}"
            )
        treatment_name = self.estimand.treatment_column
        treatment = period_frame[treatment_name].to_numpy(dtype=float)
        treated_rows = int(np.sum(treatment == 1.0))
        control_rows = int(np.sum(treatment == 0.0))
        if (
            treated_rows < policy.minimum_rows_per_arm
            or control_rows < policy.minimum_rows_per_arm
        ):
            raise ValueError(
                f"period {period_value!r} has insufficient treatment support"
            )
        outcome = period_frame[self.estimand.outcome].to_numpy(dtype=float)
        covariates = period_frame[list(self.estimand.covariates)].to_numpy(
            dtype=float
        )
        propensity, iterations = fit_logistic_propensity(
            covariates, treatment, policy
        )
        clipped = np.clip(
            propensity, policy.propensity_clip, 1.0 - policy.propensity_clip
        )
        treated_propensity = propensity[treatment == 1.0]
        control_propensity = propensity[treatment == 0.0]
        common_support_lower = float(
            max(np.min(treated_propensity), np.min(control_propensity))
        )
        common_support_upper = float(
            min(np.max(treated_propensity), np.max(control_propensity))
        )
        if common_support_lower >= common_support_upper:
            raise ValueError(
                f"period {period_value!r} has insufficient propensity overlap"
            )
        treated_weights = treatment / clipped
        control_weights = (1.0 - treatment) / (1.0 - clipped)
        treated_weight_sum = float(np.sum(treated_weights))
        control_weight_sum = float(np.sum(control_weights))
        if treated_weight_sum <= 0.0 or control_weight_sum <= 0.0:
            raise ValueError(f"period {period_value!r} has no weighted support")
        treated_mean = float(
            np.sum(treated_weights * outcome) / treated_weight_sum
        )
        control_mean = float(
            np.sum(control_weights * outcome) / control_weight_sum
        )
        estimate = treated_mean - control_mean
        influence = treated_weights * (outcome - treated_mean) / (
            treated_weight_sum / len(outcome)
        ) - control_weights * (outcome - control_mean) / (
            control_weight_sum / len(outcome)
        )
        ess_treated = effective_sample_size(treated_weights)
        ess_control = effective_sample_size(control_weights)
        if (
            ess_treated < policy.minimum_effective_sample_size
            or ess_control < policy.minimum_effective_sample_size
        ):
            raise ValueError(
                f"period {period_value!r} has insufficient effective "
                "sample size"
            )
        subject_name = self.estimand.subject_identity
        subject_values = period_frame[subject_name].tolist()
        return {
            "period": period_label(period_value),
            "estimate": estimate,
            "rows": len(period_frame),
            "treated_rows": treated_rows,
            "control_rows": control_rows,
            "propensity": propensity,
            "clipped_propensity": clipped,
            "treatment": treatment,
            "outcome": outcome,
            "fraction_clipped": float(np.mean(propensity != clipped)),
            "propensity_min": float(np.min(propensity)),
            "propensity_max": float(np.max(propensity)),
            "common_support_lower": common_support_lower,
            "common_support_upper": common_support_upper,
            "effective_sample_size_treated": ess_treated,
            "effective_sample_size_control": ess_control,
            "subjects": subject_values,
            "influence": influence,
            "iterations": iterations,
            "reference_treated_mean": treated_mean,
            "reference_control_mean": control_mean,
        }


def prepare_longitudinal_frame(
    frame: pd.DataFrame,
    estimand: LongitudinalEstimand,
    policy: LongitudinalEstimationPolicy,
) -> tuple[pd.DataFrame, int]:
    """Validate and numerically prepare a long-format input frame.

    Args:
        frame: Long-format panel to check.
        estimand: Declares the columns that must be present.
        policy: Supplies the missingness rule and the minimum row count.

    Returns:
        A tuple of the prepared frame (declared columns only, numeric, rows
        with missing values removed) and the number of rows removed.

    Raises:
        TypeError: If `frame` is not a pandas DataFrame.
        ValueError: If the frame is empty, declared columns are missing,
            subject identities or time values are missing or non-finite,
            subject-time pairs repeat, time values are invalid, values are
            missing under the "reject" policy, a column is non-numeric or
            non-finite, treatment is not binary, no rows remain, too few
            rows remain, or a covariate varies within a subject.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if frame.empty:
        raise ValueError("longitudinal frame must not be empty")
    required = [
        estimand.subject_identity,
        estimand.time_column,
        estimand.treatment_column,
        estimand.outcome,
        *estimand.covariates,
    ]
    missing_columns = [name for name in required if name not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"longitudinal frame is missing columns: {missing_columns}"
        )
    if frame[estimand.subject_identity].isna().any():
        raise ValueError("missing subject identities are not allowed")
    subject_labels = frame[estimand.subject_identity].astype("string")
    if subject_labels.fillna("").str.strip().eq("").any():
        raise ValueError("missing subject identities are not allowed")
    if frame[estimand.time_column].isna().any():
        raise ValueError("missing time values are not allowed")
    for name in (estimand.subject_identity, estimand.time_column):
        if pd.api.types.is_numeric_dtype(frame[name]):
            values = frame[name].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"column {name!r} contains non-finite values")
    if frame.duplicated(
        [estimand.subject_identity, estimand.time_column]
    ).any():
        raise ValueError("duplicate subject-time rows are not allowed")
    validate_time_order(frame, estimand)
    prepared = frame[required].copy()
    value_columns = [
        estimand.treatment_column,
        estimand.outcome,
        *estimand.covariates,
    ]
    missing_mask = prepared[value_columns].isna().any(axis=1)
    if missing_mask.any() and policy.missingness == "reject":
        raise ValueError("missing treatment, outcome, or covariate values")
    rows_dropped = int(missing_mask.sum())
    if rows_dropped:
        prepared = prepared.loc[~missing_mask].copy()
    for name in value_columns:
        try:
            prepared[name] = pd.to_numeric(prepared[name], errors="raise")
        except (TypeError, ValueError) as error:
            raise ValueError(f"column {name!r} must be numeric") from error
        values = prepared[name].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"column {name!r} contains non-finite values")
    treatment_values = set(
        prepared[estimand.treatment_column].unique().tolist()
    )
    if not treatment_values.issubset({0, 1}):
        raise ValueError("treatment must be binary with values 0 and 1")
    if prepared.empty:
        raise ValueError("missingness policy left no rows")
    if len(prepared) < policy.minimum_rows_per_period:
        raise ValueError("longitudinal frame has insufficient rows")
    validate_baseline_covariates(prepared, estimand)
    return prepared, rows_dropped


def validate_time_order(
    frame: pd.DataFrame, estimand: LongitudinalEstimand
) -> None:
    """Require strictly increasing time values within each subject.

    Args:
        frame: Long-format panel.
        estimand: Names the subject and time columns.

    Raises:
        ValueError: If a subject's time values, in row order, are not
            strictly increasing or cannot be compared.
        TypeError: Only if a `TypeError` raised while examining the values
            has a message containing "strictly increasing", which the
            comparisons used do not produce; every other `TypeError` is
            converted to `ValueError`.
    """
    subject_name = estimand.subject_identity
    time_name = estimand.time_column
    try:
        for subject_group in frame.groupby(
            subject_name, sort=False, dropna=False
        ):
            group = subject_group[1]
            if not group[time_name].is_monotonic_increasing:
                raise ValueError(
                    "time values must be strictly increasing per subject"
                )
            values = group[time_name].tolist()
            if any(left == right for left, right in itertools.pairwise(values)):
                raise ValueError(
                    "time values must be strictly increasing per subject"
                )
    except (TypeError, ValueError) as error:
        if "strictly increasing" in str(error):
            raise
        raise ValueError("time values have invalid ordering") from error


def validate_baseline_covariates(
    frame: pd.DataFrame, estimand: LongitudinalEstimand
) -> None:
    """Check the declared no-time-varying-covariates condition in the data.

    Args:
        frame: Long-format panel.
        estimand: Names the subject column and the covariates.

    Raises:
        ValueError: If a covariate takes more than one value (counting
            missing as a value) within a subject.
    """
    if not estimand.covariates:
        return
    grouped = frame.groupby(estimand.subject_identity, sort=False, dropna=False)
    for name in estimand.covariates:
        if (grouped[name].nunique(dropna=False) > 1).any():
            raise ValueError(
                f"covariate {name!r} varies within a subject despite the "
                "no_time_varying_covariates assumption"
            )


def fit_logistic_propensity(
    covariates: np.ndarray,
    treatment: np.ndarray,
    policy: LongitudinalEstimationPolicy,
) -> tuple[np.ndarray, int]:
    """Fit a deterministic ridge-logistic propensity model by Newton steps.

    Args:
        covariates: Two-dimensional matrix, rows by features, without an
            intercept column; one is added.
        treatment: Treatment indicator per row (0 or 1).
        policy: Supplies the iteration limit and convergence tolerance.

    Returns:
        A tuple of the fitted treatment probability for each row and the
        number of iterations used.

    Raises:
        ValueError: If `covariates` is not two-dimensional, the Hessian is
            singular, a step is not finite, or the fit does not converge
            within `policy.max_iterations`.
    """
    if covariates.ndim != 2:
        raise ValueError("covariates must be a two-dimensional matrix")
    design = np.column_stack((np.ones(len(treatment)), covariates))
    coefficients = np.zeros(design.shape[1], dtype=float)
    ridge = 1e-8
    ridge_matrix = ridge * np.eye(design.shape[1])
    for iteration in range(1, policy.max_iterations + 1):
        probabilities = special.expit(design @ coefficients)
        variance = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        gradient = design.T @ (probabilities - treatment) + ridge * coefficients
        hessian = design.T @ (variance[:, None] * design) + ridge_matrix
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise ValueError("propensity model is not identifiable") from error
        if not np.isfinite(step).all():
            raise ValueError(
                "propensity model produced non-finite coefficients"
            )
        coefficients -= step
        if float(np.max(np.abs(step))) <= policy.tolerance:
            probabilities = special.expit(design @ coefficients)
            return probabilities, iteration
    raise ValueError(
        "propensity model did not converge within the policy bound"
    )


def effective_sample_size(weights: np.ndarray) -> float:
    """Return the Kish effective sample size for positive arm weights.

    Args:
        weights: Non-negative weights of one arm.

    Returns:
        The squared sum of the weights divided by the sum of their squares,
        or 0.0 when the sum of squares is not positive.
    """
    total = float(np.sum(weights))
    squared = float(np.sum(weights**2))
    if squared <= 0.0:
        return 0.0
    return total**2 / squared


def influence_covariance(fitted: Sequence[Mapping[str, Any]]) -> np.ndarray:
    """Estimate covariance from subject-aligned influence values.

    Args:
        fitted: One `fit_period` record per period, in period order.

    Returns:
        A symmetric matrix with one row and column per period.  Each entry
        sums, over subjects present in both periods, the product of their
        centred influence values divided by the period row counts; a
        diagonal entry with more than one subject is scaled by
        `n / (n - 1)`.  Entries are 0.0 when no subject is shared.
    """
    periods = len(fitted)
    covariance = np.zeros((periods, periods), dtype=float)
    influence_by_period: list[dict[Any, float]] = []
    for item in fitted:
        subjects = item["subjects"]
        influence = np.asarray(item["influence"], dtype=float)
        count = int(item["rows"])
        values = {
            subject: float(value / count)
            for subject, value in zip(subjects, influence, strict=True)
        }
        mean = float(np.mean(list(values.values())))
        influence_by_period.append(
            {subject: value - mean for subject, value in values.items()}
        )
    for first in range(periods):
        for second in range(first, periods):
            first_values = influence_by_period[first]
            second_values = influence_by_period[second]
            shared = set(first_values).intersection(second_values)
            if not shared:
                value = 0.0
            else:
                value = sum(
                    first_values[subject] * second_values[subject]
                    for subject in shared
                )
                if first == second and len(shared) > 1:
                    value *= len(shared) / (len(shared) - 1)
            covariance[first, second] = value
            covariance[second, first] = value
    return (covariance + covariance.T) / 2.0


def build_period_result(
    fitted: Mapping[str, Any], variance: float, critical_value: float
) -> PeriodATE:
    """Build a serializable period result after covariance is available.

    Args:
        fitted: The `fit_period` record of the period.
        variance: Variance of the period estimate; negative values are
            treated as zero.
        critical_value: Normal quantile that scales the standard error into
            the interval half-width.

    Returns:
        The period estimate with its standard error, interval, and fit
        diagnostics, marked as converged.
    """
    standard_error = math.sqrt(max(float(variance), 0.0))
    estimate = float(fitted["estimate"])
    return PeriodATE(
        period=str(fitted["period"]),
        estimate=estimate,
        standard_error=standard_error,
        interval=(
            estimate - critical_value * standard_error,
            estimate + critical_value * standard_error,
        ),
        rows=int(fitted["rows"]),
        treated_rows=int(fitted["treated_rows"]),
        control_rows=int(fitted["control_rows"]),
        propensity_min=float(fitted["propensity_min"]),
        propensity_max=float(fitted["propensity_max"]),
        fraction_clipped=float(fitted["fraction_clipped"]),
        effective_sample_size_treated=float(
            fitted["effective_sample_size_treated"]
        ),
        effective_sample_size_control=float(
            fitted["effective_sample_size_control"]
        ),
        converged=True,
        iterations=int(fitted["iterations"]),
    )


def independent_reference(
    fitted: Sequence[Mapping[str, Any]], tolerance: float
) -> dict[str, Any]:
    """Recalculate normalized weighted means through an independent path.

    Args:
        fitted: One `fit_period` record per period.
        tolerance: Largest absolute difference counted as agreement.

    Returns:
        A dictionary with the method name, the tolerance, an overall
        "passed" flag, and one record per period (recomputed estimate,
        primary estimate, absolute difference, pass flag, arm row counts,
        propensity range, and the recomputed arm means).
    """
    records = []
    for item in fitted:
        propensity = np.asarray(item["propensity"], dtype=float)
        clipped = np.asarray(item["clipped_propensity"], dtype=float)
        treatment = np.asarray(item["treatment"], dtype=float)
        outcome = np.asarray(item["outcome"], dtype=float)
        treated_count = int(item["treated_rows"])
        control_count = int(item["control_rows"])
        treated_weights = np.where(treatment == 1.0, 1.0 / clipped, 0.0)
        control_weights = np.where(treatment == 0.0, 1.0 / (1.0 - clipped), 0.0)
        treated_mean = float(
            np.sum(treated_weights * outcome) / np.sum(treated_weights)
        )
        control_mean = float(
            np.sum(control_weights * outcome) / np.sum(control_weights)
        )
        primary = float(item["estimate"])
        reference_estimate = treated_mean - control_mean
        difference = abs(primary - reference_estimate)
        records.append(
            {
                "period": str(item["period"]),
                "estimate": reference_estimate,
                "primary_estimate": primary,
                "absolute_difference": difference,
                "passed": difference <= tolerance,
                "treated_rows": treated_count,
                "control_rows": control_count,
                "propensity_range": [
                    float(np.min(propensity)),
                    float(np.max(propensity)),
                ],
                "independent_method": "direct_normalized_weighted_means",
                "treated_mean": treated_mean,
                "control_mean": control_mean,
            }
        )
    return {
        "method": "independent_direct_hajek_recalculation",
        "tolerance": tolerance,
        "passed": all(bool(record["passed"]) for record in records),
        "periods": records,
    }


def period_label(value: Any) -> str:
    """Convert a pandas/numpy time value to a stable JSON label.

    Args:
        value: A time value from the time column.

    Returns:
        The ISO 8601 text for a timestamp, date, or datetime, otherwise
        `str` of the value with NumPy scalars unwrapped first.
    """
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def fingerprint_frame(frame: pd.DataFrame) -> str:
    """Return a deterministic digest of the supplied frame.

    Args:
        frame: Frame to fingerprint.

    Returns:
        The hexadecimal SHA-256 digest covering the column names, dtypes,
        shape, and hashed row values including the index.

    Raises:
        ValueError: If the frame values cannot be hashed.
    """
    try:
        values = (
            pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
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
