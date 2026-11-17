"""Range and applicability checks shared by every estimation policy."""

import math

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.declaration import CAUSAL_ESTIMATORS
from scmle.evaluation.parameter_estimation.declaration import CausalAssumption
from scmle.evaluation.parameter_estimation.declaration import (
    ContinuousTreatmentAssumption,
)
from scmle.evaluation.parameter_estimation.declaration import CovarianceKind
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_CONTINUOUS_WEIGHT_CAP,
)
from scmle.evaluation.parameter_estimation.declaration import (
    DEFAULT_PROPENSITY_CLIP,
)
from scmle.evaluation.parameter_estimation.declaration import EstimatorKind
from scmle.evaluation.parameter_estimation.declaration import (
    HierarchicalAssumption,
)
from scmle.evaluation.parameter_estimation.declaration import HierarchicalMethod
from scmle.evaluation.parameter_estimation.declaration import MissingPolicy
from scmle.evaluation.parameter_estimation.declaration import (
    PROPENSITY_CLIP_CEILING,
)
from scmle.evaluation.parameter_estimation.declaration import (
    PROPENSITY_ESTIMATORS,
)


class Rules:
    """The policy checks that do not depend on the estimator family.

    `Config` inherits these so its own body keeps the per-estimator checks.
    The class holds no state: the annotations below name the `Config` fields
    the checks read, and the empty `__slots__` keeps `Config` free of an
    instance dictionary.
    """

    __slots__ = ()

    estimator: EstimatorKind
    covariance: CovarianceKind
    confidence_level: float
    missing: MissingPolicy
    cluster_name: str | None
    bootstrap_replicates: int
    max_iterations: int
    tolerance: float
    reference_tolerance: float
    treatment_position: int | None
    propensity_clip: float
    causal_assumptions: tuple[CausalAssumption, ...]
    continuous_treatment_assumptions: tuple[ContinuousTreatmentAssumption, ...]
    continuous_weight_cap: float
    hierarchical_assumptions: tuple[HierarchicalAssumption, ...]
    cross_fitting_folds: int
    hierarchical_method: HierarchicalMethod
    reml_variance_bounds: tuple[float, float]
    random_slope_position: int | None

    def vet(self) -> None:
        """Validates the named choices and the first scalar limits.

        Raises:
            ValueError: If a choice is not one of its options or a limit
                is out of range.
        """
        if self.estimator not in (
            "linear",
            "logistic",
            "bayesian_linear",
            "bayesian_ate",
            "ipw_ate",
            "aipw_ate",
            "continuous_ipw_ate",
            "hierarchical_linear",
        ):
            raise errors.InputError(
                "estimator must be linear, logistic, bayesian_linear, "
                "bayesian_ate, ipw_ate, aipw_ate, continuous_ipw_ate, or "
                "hierarchical_linear"
            )
        if self.covariance not in ("iid", "hc3", "clustered"):
            raise errors.InputError("covariance must be iid, hc3, or clustered")
        if self.missing not in ("reject", "complete_case"):
            raise errors.InputError("missing must be reject or complete_case")
        if not 0.0 < self.confidence_level < 1.0:
            raise errors.InputError(
                "confidence_level must lie strictly between 0 and 1"
            )
        if self.covariance == "clustered" and not self.cluster_name:
            raise errors.InputError("clustered covariance needs cluster_name")
        if self.bootstrap_replicates < 0:
            raise errors.InputError("bootstrap_replicates must not be negative")
        if self.max_iterations < 1:
            raise errors.InputError("max_iterations must be positive")
        if self.tolerance <= 0.0 or self.reference_tolerance <= 0.0:
            raise errors.InputError("estimation tolerances must be positive")

    def check_ranges(self) -> None:
        """Validates the numeric ranges and the hierarchical options.

        Raises:
            ValueError: If a value is outside its range.
        """
        if not 0.0 < self.propensity_clip < PROPENSITY_CLIP_CEILING:
            raise errors.InputError(
                "propensity_clip must lie strictly between 0 and 0.5"
            )
        if (
            not math.isfinite(self.continuous_weight_cap)
            or self.continuous_weight_cap <= 1.0
        ):
            raise errors.InputError(
                "continuous_weight_cap must be greater than 1"
            )
        folds: object = self.cross_fitting_folds  # may be a non-integer
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 1:
            raise errors.InputError(
                "cross_fitting_folds must be a positive integer"
            )
        if self.hierarchical_method not in ("moment_gls", "reml"):
            raise errors.InputError(
                "hierarchical_method must be moment_gls or reml"
            )
        slope_position: object = self.random_slope_position  # may be non-int
        if slope_position is not None and (
            isinstance(slope_position, bool)
            or not isinstance(slope_position, int)
            or slope_position < 0
        ):
            raise errors.InputError(
                "random_slope_position must be a non-negative integer or None"
            )

    def fence(self) -> None:
        """Validates the variance bounds of the REML search.

        Raises:
            ValueError: If the bounds are not two finite positive
                increasing values.
        """
        try:
            lower_variance_bound, upper_variance_bound = (
                float(value) for value in self.reml_variance_bounds
            )
        except (TypeError, ValueError) as error:
            raise errors.InputError(
                "reml_variance_bounds must contain two finite positive "
                "values in increasing order"
            ) from error
        if (
            not math.isfinite(lower_variance_bound)
            or not math.isfinite(upper_variance_bound)
            or lower_variance_bound <= 0.0
            or lower_variance_bound >= upper_variance_bound
        ):
            raise errors.InputError(
                "reml_variance_bounds must contain two finite positive "
                "values in increasing order"
            )

    def relevant(self) -> None:
        """Rejects fields set for an estimator that ignores them.

        Raises:
            ValueError: If a field is set for an estimator that does not
                support it.
        """
        if (
            self.estimator not in CAUSAL_ESTIMATORS
            and self.treatment_position is not None
        ):
            raise errors.InputError(
                "treatment_position is supported only by bayesian_ate, "
                "ipw_ate, aipw_ate, or continuous_ipw_ate"
            )
        if (
            self.estimator not in PROPENSITY_ESTIMATORS
            and self.propensity_clip != DEFAULT_PROPENSITY_CLIP
        ):
            raise errors.InputError(
                "propensity_clip is supported only by ipw_ate or aipw_ate"
            )
        if self.estimator not in CAUSAL_ESTIMATORS and self.causal_assumptions:
            raise errors.InputError(
                "causal assumptions are supported only by bayesian_ate, "
                "ipw_ate, or aipw_ate"
            )
        if (
            self.estimator != "continuous_ipw_ate"
            and self.continuous_treatment_assumptions
        ):
            raise errors.InputError(
                "continuous treatment assumptions are supported only by "
                "continuous_ipw_ate"
            )
        if (
            self.estimator != "continuous_ipw_ate"
            and self.continuous_weight_cap != DEFAULT_CONTINUOUS_WEIGHT_CAP
        ):
            raise errors.InputError(
                "continuous_weight_cap is supported only by continuous_ipw_ate"
            )
        if self.estimator != "aipw_ate" and self.cross_fitting_folds != 1:
            raise errors.InputError(
                "cross_fitting_folds is supported only by aipw_ate"
            )
        if (
            self.estimator != "hierarchical_linear"
            and self.hierarchical_assumptions
        ):
            raise errors.InputError(
                "hierarchical assumptions are supported only by "
                "hierarchical_linear"
            )
