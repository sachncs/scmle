"""Fits one covariate-adjusted Hajek IPW estimate per period."""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

import numpy
import pandas
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation.longitudinal.declaration import Config
from scmle.evaluation.longitudinal.declaration import Estimand
from scmle.evaluation.longitudinal.panel import prepare
from scmle.evaluation.longitudinal.report import Estimation
from scmle.evaluation.longitudinal.weighting import imprint
from scmle.evaluation.longitudinal.weighting import periodise
from scmle.evaluation.longitudinal.weighting import periodize
from scmle.evaluation.longitudinal.weighting import referee
from scmle.evaluation.longitudinal.weighting import regress
from scmle.evaluation.longitudinal.weighting import sandwich
from scmle.evaluation.longitudinal.weighting import weigh


class Estimator:
    """Estimates period-specific binary-treatment ATEs with Hájek IPW.

    Attributes:
        estimand: The declared estimand.
        policy: The estimation policy; the default policy when none was
            supplied.
    """

    __module__ = "scmle.evaluation.longitudinal"

    def __init__(
        self,
        estimand: Estimand,
        policy: Config | None = None,
    ) -> None:
        """Stores a validated declaration and policy.

        Args:
            estimand: The causal quantity to estimate.
            policy: Estimation policy; `Config()` is
                used when None.

        Raises:
            TypeError: If `estimand` is not a `Estimand`.
        """
        if not isinstance(estimand, Estimand):
            raise errors.KindError("estimand must be Estimand")
        self.estimand = estimand
        self.policy = policy or Config()

    def fit(self, frame: pandas.DataFrame) -> Estimation:
        """Fits one covariate-adjusted period-specific ATE per time value.

        Args:
            frame: Long-format panel with one row per subject and period.

        Returns:
            The period estimates in ascending time order, their covariance,
            row accounting, diagnostics, and the independent recomputation.

        Raises:
            ValueError: If the time values cannot be sorted.  Invalid
                panels and periods that fail the support, overlap, or
                effective-sample-size checks raise `ValueError` from the
                helper functions and `wave`.
        """
        prepared, rows_dropped = prepare(frame, self.estimand, self.policy)
        fingerprint = imprint(frame)
        try:
            period_series = prepared[
                self.estimand.time_column
            ].drop_duplicates()
            period_values: list[Any] = period_series.sort_values().tolist()
        except (TypeError, ValueError) as error:
            raise errors.InputError(
                "time values have invalid ordering"
            ) from error
        fitted: list[dict[str, Any]] = []
        for value in period_values:
            period_frame = prepared[
                prepared[self.estimand.time_column] == value
            ]
            fitted.append(self.wave(period_frame, value))
        covariance = sandwich(fitted)
        critical_value = float(
            stats.norm.ppf(0.5 + self.policy.confidence_level / 2.0)
        )
        periods = tuple(
            periodise(item, covariance[index, index], critical_value)
            for index, item in enumerate(fitted)
        )
        reference = referee(fitted, self.policy.reference_tolerance)
        return Estimation(
            estimand=self.estimand,
            policy=self.policy,
            periods=periods,
            covariance=tuple(
                tuple(float(value) for value in row) for row in covariance
            ),
            rows_input=len(frame),
            rows_used=len(prepared),
            rows_dropped=rows_dropped,
            diagnostics=summarize(
                self.estimand, prepared, fitted, len(periods)
            ),
            limitations=qualify(self.policy, rows_dropped),
            reference=reference,
            raw_input_fingerprint=fingerprint,
        )

    def wave(
        self, period_frame: pandas.DataFrame, period_value: Any
    ) -> dict[str, Any]:
        """Fits one period and retains influence values for covariance.

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
            raise errors.InputError(
                f"period {period_value!r} has insufficient rows: "
                f"{len(period_frame)} < {policy.minimum_rows_per_period}"
            )
        treatment_name = self.estimand.treatment_column
        treatment = period_frame[treatment_name].to_numpy(dtype=float)
        treated_rows = int(numpy.sum(treatment == 1.0))
        control_rows = int(numpy.sum(treatment == 0.0))
        if (
            treated_rows < policy.minimum_rows_per_arm
            or control_rows < policy.minimum_rows_per_arm
        ):
            raise errors.InputError(
                f"period {period_value!r} has insufficient treatment support"
            )
        outcome = period_frame[self.estimand.outcome].to_numpy(dtype=float)
        covariates = period_frame[list(self.estimand.covariates)].to_numpy(
            dtype=float
        )
        propensity, iterations = regress(covariates, treatment, policy)
        bound = policy.propensity_clip
        clipped = numpy.clip(propensity, bound, 1.0 - bound)
        weighted = weigh(
            propensity, clipped, treatment, outcome, policy, period_value
        )
        return {
            "period": periodize(period_value),
            "estimate": weighted["estimate"],
            "rows": len(period_frame),
            "treated_rows": treated_rows,
            "control_rows": control_rows,
            "propensity": propensity,
            "clipped_propensity": clipped,
            "treatment": treatment,
            "outcome": outcome,
            "fraction_clipped": float(numpy.mean(propensity != clipped)),
            "propensity_min": float(numpy.min(propensity)),
            "propensity_max": float(numpy.max(propensity)),
            "common_support_lower": weighted["common_support_lower"],
            "common_support_upper": weighted["common_support_upper"],
            "effective_sample_size_treated": weighted[
                "effective_sample_size_treated"
            ],
            "effective_sample_size_control": weighted[
                "effective_sample_size_control"
            ],
            "subjects": period_frame[self.estimand.subject_identity].tolist(),
            "influence": weighted["influence"],
            "iterations": iterations,
            "reference_treated_mean": weighted["reference_treated_mean"],
            "reference_control_mean": weighted["reference_control_mean"],
        }


def summarize(
    estimand: Estimand,
    prepared: pandas.DataFrame,
    fitted: Sequence[Mapping[str, Any]],
    periods: int,
) -> Mapping[str, Any]:
    """Describes the method, the overlap and the weights of a run.

    Args:
        estimand: Names the subject column.
        prepared: The analysed rows.
        fitted: One `wave` record per period, in period order.
        periods: Number of periods.

    Returns:
        The diagnostics document of the estimation.
    """
    return {
        "method": "period_specific_covariate_adjusted_hajek_ipw",
        "propensity_model": "deterministic_newton_logistic_with_ridge",
        "period_count": periods,
        "subject_count": int(prepared[estimand.subject_identity].nunique()),
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
        "covariance_method": "subject-aligned_influence_function_sandwich",
        "time_ordering": "strictly_increasing_within_subject",
    }


def qualify(policy: Config, rows_dropped: int) -> Mapping[str, Any]:
    """States what the period estimates do not cover.

    Args:
        policy: Supplies the missingness policy.
        rows_dropped: Rows removed by the missingness policy.

    Returns:
        The limitations document of the estimation.
    """
    return {
        "prior": {
            "status": "not_applicable",
            "message": (
                "This frequentist IPW estimator has no Bayesian prior."
            ),
        },
        "missingness": {
            "policy": policy.missingness,
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
    }
