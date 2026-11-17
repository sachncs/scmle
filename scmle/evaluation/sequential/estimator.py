"""Fits stabilized sequential weights and a sustained regime contrast."""

from collections.abc import Mapping
import math
from typing import Any

import numpy
import pandas
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.sequential.declaration import Config
from scmle.evaluation.sequential.declaration import Estimand
from scmle.evaluation.sequential.panel import order
from scmle.evaluation.sequential.panel import prepare
from scmle.evaluation.sequential.panel import tabulate
from scmle.evaluation.sequential.report import Estimation
from scmle.evaluation.sequential.report import PeriodDiagnostics
from scmle.evaluation.sequential.report import RegimeEstimate
from scmle.evaluation.sequential.weighting import average
from scmle.evaluation.sequential.weighting import covary
from scmle.evaluation.sequential.weighting import effective_sample_size
from scmle.evaluation.sequential.weighting import imprint
from scmle.evaluation.sequential.weighting import overlap
from scmle.evaluation.sequential.weighting import periodize
from scmle.evaluation.sequential.weighting import referee
from scmle.evaluation.sequential.weighting import regress


class Estimator:
    """Estimates a sustained-treatment contrast with stabilized sequential IPW.

    Attributes:
        estimand: The declared estimand.
        policy: The estimation policy; the default policy when none was
            supplied.
    """

    __module__ = "scmle.evaluation.sequential"

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
            raise errors.KindError("estimand must be Estimand")
        if policy is not None and not isinstance(policy, Config):
            raise errors.KindError("policy must be Config")
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
        prepared, rows_dropped, subjects_input, subjects_dropped = prepare(
            frame, self.estimand, self.policy
        )
        fingerprint = imprint(frame)
        period_values = order(prepared, self.estimand.time_column)
        # Subjects of mixed types sort by type name, then by `repr`.
        subject_values = sorted(
            prepared[self.estimand.subject_identity].drop_duplicates().tolist(),
            key=lambda value: (type(value).__name__, repr(value)),
        )
        matrices = tabulate(
            prepared, self.estimand, period_values, subject_values
        )
        critical_value = float(
            stats.norm.ppf(0.5 + self.policy.confidence_level / 2.0)
        )
        cumulative, period_diagnostics, iterations = accumulate(
            matrices, period_values, len(subject_values), self.policy
        )
        regimes, covariance, reference = compare(
            self.estimand, self.policy, matrices, cumulative, critical_value
        )
        target, comparator = regimes
        contrast = target.estimate - comparator.estimate
        contrast_standard_error = math.sqrt(max(float(covariance.sum()), 0.0))
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
            diagnostics=summarize(
                self.estimand,
                self.policy,
                period_diagnostics,
                len(subject_values),
                iterations,
            ),
            limitations=qualify(self.policy, subjects_dropped),
            reference=reference,
            raw_input_fingerprint=fingerprint,
        )


def record(
    value: Any,
    rows: int,
    arms: tuple[int, int],
    numerator: numpy.ndarray,
    denominator: numpy.ndarray,
    cumulative: numpy.ndarray,
    policy: Config,
) -> PeriodDiagnostics:
    """Builds one period's diagnostics and checks its effective sample size.

    Args:
        value: The period's time value.
        rows: Number of subjects.
        arms: Treated and control subjects in the period.
        numerator: Fitted numerator propensities of the period.
        denominator: Fitted denominator propensities of the period.
        cumulative: Untruncated cumulative weights through the period.
        policy: Supplies the weight cap and the effective-sample-size floor.

    Returns:
        The period's diagnostics.

    Raises:
        ValueError: If the effective sample size of the truncated weights is
            below the policy minimum.
    """
    truncated = numpy.minimum(cumulative, policy.maximum_weight)
    period_diagnostic = PeriodDiagnostics(
        period=periodize(value),
        rows=rows,
        treated_rows=arms[0],
        control_rows=arms[1],
        denominator_propensity_min=float(numpy.min(denominator)),
        denominator_propensity_max=float(numpy.max(denominator)),
        numerator_propensity_min=float(numpy.min(numerator)),
        numerator_propensity_max=float(numpy.max(numerator)),
        fraction_weight_truncated=float(
            numpy.mean(cumulative > policy.maximum_weight)
        ),
        maximum_untruncated_weight=float(numpy.max(cumulative)),
        maximum_weight=float(numpy.max(truncated)),
        effective_sample_size=effective_sample_size(truncated),
    )
    if (
        period_diagnostic.effective_sample_size
        < policy.minimum_effective_sample_size
    ):
        raise errors.InputError(
            f"period {value!r} has insufficient effective sample size"
        )
    return period_diagnostic


def accumulate(
    matrices: dict[str, numpy.ndarray],
    period_values: list[Any],
    rows: int,
    policy: Config,
) -> tuple[numpy.ndarray, list[PeriodDiagnostics], dict[str, int]]:
    """Multiplies the stabilized weight increments of every period.

    Args:
        matrices: The panel as arrays, from `tabulate`.
        period_values: Periods in time order.
        rows: Number of subjects.
        policy: The estimation policy.

    Returns:
        The untruncated cumulative weight per subject, the diagnostics of
        each period, and the numerator and denominator fit iterations of the
        final period.

    Raises:
        ValueError: If a period has fewer than `minimum_arm_rows` subjects in
            either arm; the propensity fits, the positivity check and
            `record` raise their own.
    """
    treatment = matrices["treatment"]
    cumulative = numpy.ones(rows, dtype=float)
    period_diagnostics: list[PeriodDiagnostics] = []
    iterations: dict[str, int] = {}
    for index, value in enumerate(period_values):
        current_treatment = treatment[:, index]
        treated_rows = int(numpy.sum(current_treatment == 1.0))
        control_rows = int(numpy.sum(current_treatment == 0.0))
        if (
            treated_rows < policy.minimum_arm_rows
            or control_rows < policy.minimum_arm_rows
        ):
            raise errors.InputError(
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
        numerator, numerator_iterations = regress(
            numerator_features, current_treatment, policy
        )
        denominator, denominator_iterations = regress(
            denominator_features, current_treatment, policy
        )
        overlap(denominator, value, policy.propensity_clip)
        increment = numpy.where(
            current_treatment == 1.0,
            numerator / denominator,
            (1.0 - numerator) / (1.0 - denominator),
        )
        cumulative *= increment
        period_diagnostics.append(
            record(
                value,
                rows,
                (treated_rows, control_rows),
                numerator,
                denominator,
                cumulative,
                policy,
            )
        )
        if index == len(period_values) - 1:
            iterations = {
                "numerator": numerator_iterations,
                "denominator": denominator_iterations,
            }
    return cumulative, period_diagnostics, iterations


def compare(
    estimand: Estimand,
    policy: Config,
    matrices: dict[str, numpy.ndarray],
    cumulative: numpy.ndarray,
    critical_value: float,
) -> tuple[
    tuple[RegimeEstimate, RegimeEstimate], numpy.ndarray, Mapping[str, Any]
]:
    """Estimates both regime means with the truncated final weights.

    Args:
        estimand: Names the target and comparator treatments.
        policy: The estimation policy.
        matrices: The panel as arrays, from `tabulate`.
        cumulative: Untruncated cumulative weight per subject.
        critical_value: Normal quantile of the confidence level.

    Returns:
        The target and comparator regime estimates, the 2 by 2 covariance of
        their means, and the independent recomputation.
    """
    treatment = matrices["treatment"]
    outcome = matrices["outcome"][:, -1]
    final_weights = numpy.minimum(cumulative, policy.maximum_weight)
    target_mask = numpy.all(treatment == estimand.target_treatment, axis=1)
    comparator_mask = numpy.all(
        treatment == estimand.comparator_treatment, axis=1
    )
    target = average(
        "always_treated"
        if estimand.target_treatment == 1
        else "always_control",
        estimand.target_treatment,
        target_mask,
        final_weights,
        outcome,
        policy,
        critical_value,
    )
    comparator = average(
        "always_treated"
        if estimand.comparator_treatment == 1
        else "always_control",
        estimand.comparator_treatment,
        comparator_mask,
        final_weights,
        outcome,
        policy,
        critical_value,
    )
    covariance = covary(
        target_mask,
        comparator_mask,
        final_weights,
        outcome,
        target,
        comparator,
    )
    reference = referee(
        target_mask,
        comparator_mask,
        final_weights,
        outcome,
        target,
        comparator,
        policy.reference_tolerance,
    )
    return (target, comparator), covariance, reference


def summarize(
    estimand: Estimand,
    policy: Config,
    period_diagnostics: list[PeriodDiagnostics],
    subjects: int,
    iterations: dict[str, int],
) -> Mapping[str, Any]:
    """Describes the method, the fit and the weights of a run.

    Args:
        estimand: Names the target and comparator treatments.
        policy: Supplies the weight cap.
        period_diagnostics: The diagnostics of each period, in time order.
        subjects: Number of subjects analysed.
        iterations: Fit iterations of the final period.

    Returns:
        The diagnostics document of the estimation.
    """
    periods = len(period_diagnostics)
    return {
        "method": "stabilized_treatment_history_ipw",
        "propensity_model": "deterministic_newton_logistic_with_ridge",
        "period_count": periods,
        "subject_count": subjects,
        "regime_history": {
            "target": [estimand.target_treatment] * periods,
            "comparator": [estimand.comparator_treatment] * periods,
        },
        "weight_truncation": {
            "maximum_weight": policy.maximum_weight,
            "fraction_truncated_final": period_diagnostics[
                -1
            ].fraction_weight_truncated,
        },
        "periods": [item.to_dict() for item in period_diagnostics],
        "fit_iterations_final_period": iterations,
        "time_ordering": "strictly_increasing_with_complete_follow_up",
    }


def qualify(policy: Config, subjects_dropped: int) -> Mapping[str, Any]:
    """States what the sequential estimate does not cover.

    Args:
        policy: Supplies the missingness policy.
        subjects_dropped: Subjects removed by the missingness policy.

    Returns:
        The limitations document of the estimation.
    """
    return {
        "regimes": (
            "static always-treated versus always-control histories only"
        ),
        "missingness": {
            "policy": policy.missingness,
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
    }
