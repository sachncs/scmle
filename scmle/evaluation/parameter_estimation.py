"""Reproducible frequentist parameter estimation with uncertainty evidence.

This module answers a different question from model selection: what is the
estimated value of a declared population parameter, and how uncertain is that
estimate under an explicit likelihood and covariance policy?  It provides
transparent frequentist linear and logistic likelihoods plus a bounded,
conjugate Bayesian Gaussian estimator.  More specialised estimators can
implement the same result contract without changing the artifact or review
surface.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import cast
from typing import Literal

import numpy
import pandas
from scipy import linalg
from scipy import optimize
from scipy import special
from scipy import stats

EstimatorKind = Literal[
    "linear",
    "logistic",
    "bayesian_linear",
    "bayesian_ate",
    "ipw_ate",
    "aipw_ate",
    "continuous_ipw_ate",
    "hierarchical_linear",
]
CovarianceKind = Literal["iid", "hc3", "clustered"]
MissingPolicy = Literal["reject", "complete_case"]
CausalAssumption = Literal[
    "linearity",
    "consistency",
    "conditional_exchangeability",
    "positivity",
]
ContinuousTreatmentAssumption = Literal[
    "consistency",
    "conditional_exchangeability",
    "continuous_treatment_positivity",
    "linear_dose_response",
    "normal_conditional_treatment_density",
]
HierarchicalAssumption = Literal[
    "random_intercept",
    "random_slope",
    "cluster_independence",
    "random_effect_exogeneity",
    "constant_residual_variance",
]
HierarchicalMethod = Literal["moment_gls", "reml"]

DEFAULT_REML_VARIANCE_BOUNDS = (1e-12, 1e12)
COVARIANCE_CONDITION_THRESHOLD = 1e12
CAUSAL_ESTIMATORS = frozenset(
    {"bayesian_ate", "ipw_ate", "aipw_ate", "continuous_ipw_ate"}
)
PROPENSITY_ESTIMATORS = frozenset({"ipw_ate", "aipw_ate"})
SCALE_FLOOR = 1e-12
CLIP_TOLERANCE = 1e-12
PROPENSITY_CLIP_CEILING = 0.5
DEFAULT_PROPENSITY_CLIP = 0.01
DEFAULT_CONTINUOUS_WEIGHT_CAP = 20.0
SHA256_HEX_LENGTH = 64
MATRIX_NDIM = 2
MIN_ROWS_PER_ARM = 2
MIN_TREATMENT_LEVELS = 4
MIN_EFFECTIVE_SAMPLE_SIZE = 2.0
MIN_VARIANCE_DEGREES = 2.0
MIN_CLUSTERS = 2
SMALL_CLUSTER_COUNT = 30
MIN_FOLDS = 2
MIN_INFLUENCE_ROWS = 2
MIN_REPLICATES = 2
RANDOM_EFFECT_COLUMNS = 2

REQUIRED_CAUSAL_ASSUMPTIONS = frozenset(
    {"consistency", "conditional_exchangeability", "positivity"}
)
REQUIRED_BAYESIAN_ATE_ASSUMPTIONS = frozenset(
    {"linearity", *REQUIRED_CAUSAL_ASSUMPTIONS}
)
REQUIRED_CONTINUOUS_TREATMENT_ASSUMPTIONS = frozenset(
    {
        "consistency",
        "conditional_exchangeability",
        "continuous_treatment_positivity",
        "linear_dose_response",
        "normal_conditional_treatment_density",
    }
)
REQUIRED_HIERARCHICAL_ASSUMPTIONS = frozenset(
    {
        "random_intercept",
        "cluster_independence",
        "random_effect_exogeneity",
        "constant_residual_variance",
    }
)
REQUIRED_HIERARCHICAL_RANDOM_SLOPE_ASSUMPTIONS = frozenset(
    {*REQUIRED_HIERARCHICAL_ASSUMPTIONS, "random_slope"}
)


@dataclasses.dataclass(frozen=True, slots=True)
class Estimand:
    """The scientific quantity and assumptions a result claims to estimate.

    Attributes:
        name: Stable identifier for the estimand.
        outcome: Human-readable outcome variable name.
        population: Population and observation window represented by the data.
        likelihood: Declared likelihood, such as ``gaussian`` or ``bernoulli``.
        feature_names: Names matching the columns supplied to the estimator.
        assumptions: Explicit identification and sampling assumptions.
        description: Optional plain-language interpretation.
    """

    name: str
    outcome: str
    population: str
    likelihood: str
    feature_names: tuple[str, ...]
    assumptions: tuple[str, ...]
    description: str = ""

    def __post_init__(self) -> None:
        """Rejects ambiguous or incomplete scientific declarations.

        Raises:
            ValueError: If name, outcome, population or likelihood is blank,
                if there are no feature names or they are not unique, or if
                no assumptions are declared.
        """
        if not self.name.strip():
            raise ValueError("estimand name must not be empty")
        if not self.outcome.strip():
            raise ValueError("estimand outcome must not be empty")
        if not self.population.strip():
            raise ValueError("estimand population must not be empty")
        if not self.likelihood.strip():
            raise ValueError("estimand likelihood must not be empty")
        if not self.feature_names:
            raise ValueError("estimand needs at least one feature")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("estimand feature names must be unique")
        if not self.assumptions:
            raise ValueError("estimand needs explicit assumptions")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible scientific metadata."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianPrior:
    """Conjugate normal-inverse-gamma prior for a Gaussian design.

    The coefficient order is ``("intercept", *estimand.feature_names)``.
    ``precision`` is the positive-definite prior precision matrix for the
    coefficients conditional on the residual variance.  ``shape`` and
    ``scale`` parameterize the inverse-gamma prior on that variance.

    Attributes:
        mean: Prior mean of the coefficients, intercept first.
        precision: Symmetric positive-definite prior precision matrix of the
            coefficients, conditional on the residual variance.
        shape: Positive shape of the inverse-gamma prior on the variance.
        scale: Positive scale of the inverse-gamma prior on the variance.
    """

    mean: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    shape: float = 2.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Rejects a prior that cannot define a proper posterior.

        Raises:
            ValueError: If the mean is empty, not one-dimensional or not
                finite; if the precision is not square over the mean, not
                finite, not symmetric or not positive definite; or if shape
                or scale is not a positive finite number.
        """
        if not self.mean:
            raise ValueError("Bayesian prior needs a coefficient mean")
        prior_mean = numpy.asarray(self.mean, dtype=float)
        prior_precision = numpy.asarray(self.precision, dtype=float)
        if prior_mean.ndim != 1:
            raise ValueError("Bayesian prior mean must be one-dimensional")
        if prior_precision.shape != (len(self.mean), len(self.mean)):
            raise ValueError("Bayesian prior precision must be square")
        if not numpy.isfinite(prior_mean).all():
            raise ValueError("Bayesian prior mean must be finite")
        if not numpy.isfinite(prior_precision).all():
            raise ValueError("Bayesian prior precision must be finite")
        if not numpy.allclose(
            prior_precision, prior_precision.T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Bayesian prior precision must be symmetric")
        try:
            numpy.linalg.cholesky(prior_precision)
        except numpy.linalg.LinAlgError as error:
            raise ValueError(
                "Bayesian prior precision must be positive definite"
            ) from error
        if not math.isfinite(self.shape) or self.shape <= 0.0:
            raise ValueError("Bayesian prior shape must be positive")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("Bayesian prior scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible prior metadata."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Config:
    """Numerical and uncertainty policy for an estimation run.

    Attributes:
        estimator: Estimator to run.
        covariance: Covariance estimator for the coefficients.
        confidence_level: Confidence level of the reported intervals, strictly
            between 0 and 1.
        missing: Whether missing values are rejected or rows with them are
            dropped.
        cluster_name: Name of the cluster column, or None when the data are
            not clustered.
        bootstrap_replicates: Number of bootstrap replicates for the
            sensitivity report; 0 disables the bootstrap.
        seed: Seed of the random number generator.
        max_iterations: Iteration limit of iterative fits.
        tolerance: Convergence tolerance of iterative fits.
        reference_tolerance: Tolerance for agreement with the independent
            reference computation.
        prior: Conjugate prior for the Bayesian estimators, else None.
        treatment_position: Index of the treatment column among the features
            for the causal estimators, else None.
        propensity_clip: Propensity scores are clipped to
            ``[propensity_clip, 1 - propensity_clip]``; strictly between 0
            and 0.5.
        causal_assumptions: Declared assumptions for the binary-treatment
            causal estimators.
        continuous_treatment_assumptions: Declared assumptions for
            ``continuous_ipw_ate``.
        continuous_weight_cap: Upper cap on the continuous-treatment weights;
            greater than 1.
        hierarchical_assumptions: Declared assumptions for
            ``hierarchical_linear``.
        cross_fitting_folds: Number of cross-fitting folds for ``aipw_ate``;
            a positive integer.
        hierarchical_method: Variance-component method for
            ``hierarchical_linear``.
        reml_variance_bounds: Lower and upper bound of the variance
            components in the REML search.
        random_slope_position: Index of the feature carrying a random slope
            in ``hierarchical_linear``, or None for a random intercept only.
    """

    estimator: EstimatorKind
    covariance: CovarianceKind = "hc3"
    confidence_level: float = 0.95
    missing: MissingPolicy = "reject"
    cluster_name: str | None = None
    bootstrap_replicates: int = 0
    seed: int = 0
    max_iterations: int = 100
    tolerance: float = 1e-8
    reference_tolerance: float = 1e-6
    prior: BayesianPrior | None = None
    treatment_position: int | None = None
    propensity_clip: float = DEFAULT_PROPENSITY_CLIP
    causal_assumptions: tuple[CausalAssumption, ...] = ()
    continuous_treatment_assumptions: tuple[
        ContinuousTreatmentAssumption, ...
    ] = ()
    continuous_weight_cap: float = DEFAULT_CONTINUOUS_WEIGHT_CAP
    hierarchical_assumptions: tuple[HierarchicalAssumption, ...] = ()
    cross_fitting_folds: int = 1
    hierarchical_method: HierarchicalMethod = "moment_gls"
    reml_variance_bounds: tuple[float, float] = DEFAULT_REML_VARIANCE_BOUNDS
    random_slope_position: int | None = None

    def __post_init__(self) -> None:
        """Validates the policy before data is touched.

        Raises:
            ValueError: If a field is outside its range, or if a field is
                set for an estimator that does not support it, or if the
                estimator's required prior, treatment position, cluster name,
                covariance kind or exact assumption set is missing or wrong.
        """
        self.check_choices()
        self.check_ranges()
        self.check_reml_bounds()
        self.check_field_support()
        if self.estimator in {"bayesian_linear", "bayesian_ate"}:
            self.check_bayesian_policy()
        elif self.estimator in CAUSAL_ESTIMATORS:
            self.check_causal_policy()
        elif self.estimator == "hierarchical_linear":
            self.check_hierarchical_policy()
        elif self.prior is not None:
            raise ValueError(
                "a Bayesian prior is supported only by bayesian_linear or "
                "bayesian_ate"
            )
        self.check_hierarchical_only_fields()

    def check_choices(self) -> None:
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
            raise ValueError(
                "estimator must be linear, logistic, bayesian_linear, "
                "bayesian_ate, ipw_ate, aipw_ate, continuous_ipw_ate, or "
                "hierarchical_linear"
            )
        if self.covariance not in ("iid", "hc3", "clustered"):
            raise ValueError("covariance must be iid, hc3, or clustered")
        if self.missing not in ("reject", "complete_case"):
            raise ValueError("missing must be reject or complete_case")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError(
                "confidence_level must lie strictly between 0 and 1"
            )
        if self.covariance == "clustered" and not self.cluster_name:
            raise ValueError("clustered covariance needs cluster_name")
        if self.bootstrap_replicates < 0:
            raise ValueError("bootstrap_replicates must not be negative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0.0 or self.reference_tolerance <= 0.0:
            raise ValueError("estimation tolerances must be positive")

    def check_ranges(self) -> None:
        """Validates the numeric ranges and the hierarchical options.

        Raises:
            ValueError: If a value is outside its range.
        """
        if not 0.0 < self.propensity_clip < PROPENSITY_CLIP_CEILING:
            raise ValueError(
                "propensity_clip must lie strictly between 0 and 0.5"
            )
        if (
            not math.isfinite(self.continuous_weight_cap)
            or self.continuous_weight_cap <= 1.0
        ):
            raise ValueError("continuous_weight_cap must be greater than 1")
        folds: object = self.cross_fitting_folds  # may be a non-integer
        if isinstance(folds, bool) or not isinstance(folds, int) or folds < 1:
            raise ValueError("cross_fitting_folds must be a positive integer")
        if self.hierarchical_method not in ("moment_gls", "reml"):
            raise ValueError("hierarchical_method must be moment_gls or reml")
        slope_position: object = self.random_slope_position  # may be non-int
        if slope_position is not None and (
            isinstance(slope_position, bool)
            or not isinstance(slope_position, int)
            or slope_position < 0
        ):
            raise ValueError(
                "random_slope_position must be a non-negative integer or None"
            )

    def check_reml_bounds(self) -> None:
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
            raise ValueError(
                "reml_variance_bounds must contain two finite positive "
                "values in increasing order"
            ) from error
        if (
            not math.isfinite(lower_variance_bound)
            or not math.isfinite(upper_variance_bound)
            or lower_variance_bound <= 0.0
            or lower_variance_bound >= upper_variance_bound
        ):
            raise ValueError(
                "reml_variance_bounds must contain two finite positive "
                "values in increasing order"
            )

    def check_field_support(self) -> None:
        """Rejects fields set for an estimator that ignores them.

        Raises:
            ValueError: If a field is set for an estimator that does not
                support it.
        """
        if (
            self.estimator not in CAUSAL_ESTIMATORS
            and self.treatment_position is not None
        ):
            raise ValueError(
                "treatment_position is supported only by bayesian_ate, "
                "ipw_ate, aipw_ate, or continuous_ipw_ate"
            )
        if (
            self.estimator not in PROPENSITY_ESTIMATORS
            and self.propensity_clip != DEFAULT_PROPENSITY_CLIP
        ):
            raise ValueError(
                "propensity_clip is supported only by ipw_ate or aipw_ate"
            )
        if self.estimator not in CAUSAL_ESTIMATORS and self.causal_assumptions:
            raise ValueError(
                "causal assumptions are supported only by bayesian_ate, "
                "ipw_ate, or aipw_ate"
            )
        if (
            self.estimator != "continuous_ipw_ate"
            and self.continuous_treatment_assumptions
        ):
            raise ValueError(
                "continuous treatment assumptions are supported only by "
                "continuous_ipw_ate"
            )
        if (
            self.estimator != "continuous_ipw_ate"
            and self.continuous_weight_cap != DEFAULT_CONTINUOUS_WEIGHT_CAP
        ):
            raise ValueError(
                "continuous_weight_cap is supported only by continuous_ipw_ate"
            )
        if self.estimator != "aipw_ate" and self.cross_fitting_folds != 1:
            raise ValueError(
                "cross_fitting_folds is supported only by aipw_ate"
            )
        if (
            self.estimator != "hierarchical_linear"
            and self.hierarchical_assumptions
        ):
            raise ValueError(
                "hierarchical assumptions are supported only by "
                "hierarchical_linear"
            )

    def check_bayesian_policy(self) -> None:
        """Validates the policy of a Bayesian estimator.

        Raises:
            ValueError: If the prior, covariance, bootstrap, treatment
                position, assumptions or cluster name do not suit the
                estimator.
        """
        if self.prior is None:
            raise ValueError(
                f"{self.estimator} estimation requires an explicit prior"
            )
        if self.covariance != "iid":
            raise ValueError(
                f"{self.estimator} estimation supports only iid covariance"
            )
        if self.bootstrap_replicates:
            raise ValueError(
                f"{self.estimator} bootstrap sensitivity is unsupported"
            )
        if self.estimator == "bayesian_ate":
            if self.treatment_position is None:
                raise ValueError("bayesian_ate requires treatment_position")
            if (
                len(self.causal_assumptions)
                != len(set(self.causal_assumptions))
                or frozenset(self.causal_assumptions)
                != REQUIRED_BAYESIAN_ATE_ASSUMPTIONS
            ):
                raise ValueError(
                    "bayesian_ate requires consistency, "
                    "conditional_exchangeability, positivity, and "
                    "linearity assumptions"
                )
            if self.cluster_name is not None:
                raise ValueError("bayesian_ate does not support clustered data")

    def check_causal_policy(self) -> None:
        """Validates the policy of a frequentist causal estimator.

        Raises:
            ValueError: If the prior, covariance, treatment position or
                assumption set does not suit the estimator.
        """
        if self.prior is not None:
            raise ValueError(
                f"a Bayesian prior is not valid for {self.estimator}"
            )
        if self.covariance == "hc3":
            raise ValueError(
                f"{self.estimator} supports only iid or clustered "
                "covariance; hc3 is unsupported"
            )
        if self.treatment_position is None:
            raise ValueError(f"{self.estimator} requires treatment_position")
        if self.estimator == "continuous_ipw_ate":
            if self.causal_assumptions:
                raise ValueError(
                    "continuous_ipw_ate requires "
                    "continuous_treatment_assumptions"
                )
            if (
                len(self.continuous_treatment_assumptions)
                != len(set(self.continuous_treatment_assumptions))
                or frozenset(self.continuous_treatment_assumptions)
                != REQUIRED_CONTINUOUS_TREATMENT_ASSUMPTIONS
            ):
                raise ValueError(
                    "continuous_ipw_ate requires consistency, "
                    "conditional_exchangeability, "
                    "continuous_treatment_positivity, "
                    "linear_dose_response, and "
                    "normal_conditional_treatment_density assumptions"
                )
        elif (
            len(self.causal_assumptions) != len(set(self.causal_assumptions))
            or frozenset(self.causal_assumptions) != REQUIRED_CAUSAL_ASSUMPTIONS
        ):
            raise ValueError(
                f"{self.estimator} requires consistency, "
                "conditional_exchangeability, "
                "and positivity assumptions"
            )

    def check_hierarchical_policy(self) -> None:
        """Validates the policy of the hierarchical estimator.

        Raises:
            ValueError: If the prior, covariance, cluster name or assumption
                set does not suit the estimator.
        """
        if self.prior is not None:
            raise ValueError(
                "a Bayesian prior is not valid for hierarchical_linear"
            )
        if self.covariance != "iid":
            raise ValueError(
                "hierarchical_linear supplies its own covariance model"
            )
        if not self.cluster_name:
            raise ValueError("hierarchical_linear requires cluster_name")
        required_hierarchical_assumptions = (
            REQUIRED_HIERARCHICAL_RANDOM_SLOPE_ASSUMPTIONS
            if self.random_slope_position is not None
            else REQUIRED_HIERARCHICAL_ASSUMPTIONS
        )
        if (
            len(self.hierarchical_assumptions)
            != len(set(self.hierarchical_assumptions))
            or frozenset(self.hierarchical_assumptions)
            != required_hierarchical_assumptions
        ):
            raise ValueError(
                "hierarchical_linear requires all assumptions for its "
                "declared random-effects structure"
            )

    def check_hierarchical_only_fields(self) -> None:
        """Rejects hierarchical-only fields set for another estimator.

        Raises:
            ValueError: If a hierarchical-only field is set for another
                estimator.
        """
        if self.estimator != "hierarchical_linear":
            if self.random_slope_position is not None:
                raise ValueError(
                    "random_slope_position is supported only by "
                    "hierarchical_linear"
                )
            if self.hierarchical_method != "moment_gls":
                raise ValueError(
                    "hierarchical_method is supported only by "
                    "hierarchical_linear"
                )
            if self.reml_variance_bounds != DEFAULT_REML_VARIANCE_BOUNDS:
                raise ValueError(
                    "reml_variance_bounds is supported only by "
                    "hierarchical_linear"
                )

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible policy metadata."""
        values = dataclasses.asdict(self)
        if self.prior is None:
            values.pop("prior")
        if self.estimator not in {
            "bayesian_ate",
            "ipw_ate",
            "aipw_ate",
            "continuous_ipw_ate",
        }:
            values.pop("treatment_position")
        if self.estimator not in {"bayesian_ate", "ipw_ate", "aipw_ate"}:
            values.pop("causal_assumptions")
        if self.estimator != "continuous_ipw_ate":
            values.pop("continuous_treatment_assumptions")
            values.pop("continuous_weight_cap")
        if self.estimator not in {"ipw_ate", "aipw_ate"}:
            values.pop("propensity_clip")
        if self.estimator != "aipw_ate":
            values.pop("cross_fitting_folds")
        if self.estimator != "hierarchical_linear":
            values.pop("hierarchical_assumptions")
            values.pop("hierarchical_method")
            values.pop("reml_variance_bounds")
            values.pop("random_slope_position")
        if self.prior is not None:
            values["prior"] = self.prior.to_dict()
        return values


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """One coefficient estimate and its declared uncertainty.

    Attributes:
        name: Coefficient name.
        estimate: Point estimate.
        standard_error: Non-negative standard error of the estimate.
        interval: Lower and upper end of the confidence (or credible)
            interval.
        p_value: Two-sided p-value in ``[0, 1]``.
    """

    name: str
    estimate: float
    standard_error: float
    interval: tuple[float, float]
    p_value: float

    def __post_init__(self) -> None:
        """Rejects coefficient evidence that cannot be interpreted safely.

        Raises:
            ValueError: If the name is blank, any number is not finite, the
                standard error is negative, the p-value is outside
                ``[0, 1]``, or the interval is not ordered.
        """
        if not self.name.strip():
            raise ValueError("parameter name must not be empty")
        for name, value in (
            ("estimate", self.estimate),
            ("standard_error", self.standard_error),
            ("p_value", self.p_value),
            ("interval_lower", self.interval[0]),
            ("interval_upper", self.interval[1]),
        ):
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.standard_error < 0.0:
            raise ValueError("standard_error must not be negative")
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError("p_value must lie between 0 and 1")
        if self.interval[0] > self.interval[1]:
            raise ValueError("parameter interval must be ordered")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible coefficient evidence."""
        return dataclasses.asdict(self)


def created_temporary(temporary: Path | None) -> Path:
    """Returns the temporary report file, which must have been created.

    Args:
        temporary: The path of the temporary file, or None if none was made.

    Returns:
        `temporary`.

    Raises:
        RuntimeError: If no temporary file was created.
    """
    if temporary is None:
        raise RuntimeError("temporary estimation report was not created")
    return temporary


@dataclasses.dataclass(frozen=True, slots=True)
class Estimation:
    """Complete, serializable evidence for one parameter-estimation run.

    Attributes:
        estimand: The quantity the run claims to estimate.
        policy: The policy the run used.
        parameters: Coefficient estimates, in coefficient order.
        covariance: Symmetric covariance matrix of the coefficients, in the
            same order as ``parameters``.
        rows_input: Number of rows supplied.
        rows_used: Number of rows the fit used.
        rows_dropped: Number of rows removed by the missing-value policy.
        diagnostics: Numerical and model diagnostics of the fit.
        sensitivity: Bootstrap and other sensitivity evidence.
        reference: Result of the independent reference computation.
        raw_input_fingerprint: Lowercase SHA-256 hex digest of the raw input.
    """

    estimand: Estimand
    policy: Config
    parameters: tuple[ParameterEstimate, ...]
    covariance: tuple[tuple[float, ...], ...]
    rows_input: int
    rows_used: int
    rows_dropped: int
    diagnostics: Mapping[str, Any]
    sensitivity: Mapping[str, Any]
    reference: Mapping[str, Any]
    raw_input_fingerprint: str

    def __post_init__(self) -> None:
        """Validates the report shape before it can be persisted or used.

        Raises:
            TypeError: If the estimand or policy has the wrong type, or if
                diagnostics, sensitivity or reference is not a mapping.
            ValueError: If there are no parameters or their names repeat, if
                the covariance is not square, finite and symmetric over the
                parameters, if a row count is negative or not an integer, if
                used plus dropped rows differ from input rows, or if the
                fingerprint is not a lowercase SHA-256 digest.
        """
        if not isinstance(self.estimand, Estimand):
            raise TypeError("estimand must be Estimand")
        if not isinstance(self.policy, Config):
            raise TypeError("policy must be Config")
        if not self.parameters:
            raise ValueError("estimation result needs parameters")
        names = tuple(item.name for item in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        self.check_covariance()
        self.check_row_accounting()
        for mapping_name, mapping_value in (
            ("diagnostics", self.diagnostics),
            ("sensitivity", self.sensitivity),
            ("reference", self.reference),
        ):
            if not isinstance(mapping_value, Mapping):
                raise TypeError(f"{mapping_name} must be a mapping")
        fingerprint: object = self.raw_input_fingerprint  # may be non-text
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != SHA256_HEX_LENGTH
            or any(
                character not in "0123456789abcdef" for character in fingerprint
            )
        ):
            raise ValueError(
                "raw_input_fingerprint must be a lowercase SHA-256 digest"
            )

    def check_row_accounting(self) -> None:
        """Checks the row counts and that used plus dropped rows add up.

        Raises:
            ValueError: If a row count is negative or not an integer, or the
                used and dropped rows do not sum to the input rows.
        """
        counts: tuple[tuple[str, object], ...] = (
            ("rows_input", self.rows_input),
            ("rows_used", self.rows_used),
            ("rows_dropped", self.rows_dropped),
        )
        for name, count in counts:
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.rows_used + self.rows_dropped != self.rows_input:
            raise ValueError("row accounting must reconcile to rows_input")

    def check_covariance(self) -> None:
        """Checks that the covariance is square, finite and symmetric.

        Raises:
            ValueError: If the covariance is not square over all parameters,
                has a non-finite value, or is not symmetric.
        """
        expected = len(self.parameters)
        if len(self.covariance) != expected or any(
            len(row) != expected for row in self.covariance
        ):
            raise ValueError("covariance must be square over all parameters")
        covariance = numpy.asarray(self.covariance, dtype=float)
        if not numpy.isfinite(covariance).all():
            raise ValueError("covariance must contain finite values")
        if not numpy.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
            raise ValueError("covariance must be symmetric")

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible estimation report.

        Returns:
            A dictionary holding every field, with nested objects converted
            to dictionaries and matrices to lists.
        """
        return {
            "estimand": self.estimand.to_dict(),
            "policy": self.policy.to_dict(),
            "parameters": [item.to_dict() for item in self.parameters],
            "covariance": [list(row) for row in self.covariance],
            "rows_input": self.rows_input,
            "rows_used": self.rows_used,
            "rows_dropped": self.rows_dropped,
            "diagnostics": dict(self.diagnostics),
            "sensitivity": dict(self.sensitivity),
            "reference": dict(self.reference),
            "raw_input_fingerprint": self.raw_input_fingerprint,
        }

    def digest(self) -> str:
        """Returns the content digest of the canonical estimation report."""
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def write(self, path: str | Path) -> str:
        """Atomically writes a digest-bound estimation report.

        The report is written to a temporary file in the destination
        directory, flushed to disk and renamed over the destination. The file
        holds an envelope with `schema_version` 1, the report and its digest.
        On any failure the temporary file is removed and the error
        propagates.

        Args:
            path: Destination file; missing parent directories are created.

        Returns:
            The SHA-256 hex digest of the canonical report.

        Raises:
            RuntimeError: If the temporary file path was not recorded; the
                check cannot trigger in practice.
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
            created_temporary(temporary).replace(destination)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return self.digest()

    @classmethod
    def read(cls, path: str | Path) -> "Estimation":
        """Reads a report and rejects a modified or malformed envelope.

        Args:
            path: File written by `write`.

        Returns:
            The rebuilt estimation result.

        Raises:
            ValueError: If the schema version is not 1, the report is not a
                mapping, the stored digest does not match the report, or the
                rebuilt result does not reproduce the digest. A file that is
                not valid JSON raises `json.JSONDecodeError`, a subclass.
        """
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        if envelope.get("schema_version") != 1:
            raise ValueError("unsupported estimation report schema")
        record = envelope.get("report")
        if not isinstance(record, Mapping):
            raise ValueError("estimation report must contain a mapping")
        canonical = json.dumps(
            record, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        digest = hashlib.sha256(canonical).hexdigest()
        if envelope.get("digest") != digest:
            raise ValueError("estimation report digest mismatch")
        report = cls.from_dict(record)
        if report.digest() != digest:
            raise ValueError("estimation report canonicalization mismatch")
        return report

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "Estimation":
        """Rebuilds a result from its JSON-compatible report.

        Args:
            record: Dictionary in the form produced by `to_dict`.

        Returns:
            The estimation result described by the record.  A missing key
            or a component that fails its own validation propagates the
            error of the lookup or of that component's constructor.
        """
        estimand_record = record["estimand"]
        policy_record = record["policy"]
        parameters = tuple(
            ParameterEstimate(
                item["name"],
                float(item["estimate"]),
                float(item["standard_error"]),
                (float(item["interval"][0]), float(item["interval"][1])),
                float(item["p_value"]),
            )
            for item in record["parameters"]
        )
        estimand_values = dict(estimand_record)
        estimand_values["feature_names"] = tuple(
            estimand_record["feature_names"]
        )
        estimand_values["assumptions"] = tuple(estimand_record["assumptions"])
        policy_values = dict(policy_record)
        for policy_field in (
            "causal_assumptions",
            "continuous_treatment_assumptions",
            "hierarchical_assumptions",
        ):
            if policy_field in policy_values:
                policy_values[policy_field] = tuple(policy_values[policy_field])
        if "reml_variance_bounds" in policy_values:
            policy_values["reml_variance_bounds"] = tuple(
                float(value) for value in policy_values["reml_variance_bounds"]
            )
        if policy_values.get("prior") is not None:
            prior_values = dict(policy_values["prior"])
            prior_values["mean"] = tuple(prior_values["mean"])
            prior_values["precision"] = tuple(
                tuple(row) for row in prior_values["precision"]
            )
            policy_values["prior"] = BayesianPrior(**prior_values)
        return cls(
            Estimand(**estimand_values),
            Config(**policy_values),
            parameters,
            tuple(
                tuple(float(value) for value in row)
                for row in record["covariance"]
            ),
            int(record["rows_input"]),
            int(record["rows_used"]),
            int(record["rows_dropped"]),
            dict(record["diagnostics"]),
            dict(record["sensitivity"]),
            dict(record["reference"]),
            str(record["raw_input_fingerprint"]),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class EstimatorFit:
    """The fitted quantities of one estimator, before they are reported.

    Attributes:
        names: Parameter names, one per coefficient.
        coefficients: Point estimates.
        covariance: Covariance matrix of the estimates.
        diagnostics: Estimator diagnostics, including the likelihood.
        reference: Independent reference calculation.
        likelihood: Likelihood family of the fitted model.
        pivot: Critical value that scales the interval half-widths.
        p_values: Two-sided p-value of each coefficient.
        standard_errors: Standard errors when the estimator sets them (the
            Bayesian estimators), else None to derive them from the
            covariance.
        interval_errors: Scales of the interval half-widths when the
            estimator sets them, else None to use the standard errors.
    """

    names: tuple[str, ...]
    coefficients: numpy.ndarray
    covariance: numpy.ndarray
    diagnostics: dict[str, Any]
    reference: dict[str, Any]
    likelihood: str
    pivot: Any
    p_values: numpy.ndarray
    standard_errors: numpy.ndarray | None = None
    interval_errors: numpy.ndarray | None = None


def check_estimation_request(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    input_clusters: numpy.ndarray | None,
    policy: Config,
    estimand: Estimand,
) -> None:
    """Checks the inputs against the policy before any row is dropped.

    Args:
        matrix: Complete feature matrix.
        values: Complete outcome vector.
        input_clusters: Validated cluster labels, or None.
        policy: Estimator policy.
        estimand: Scientific estimand.

    Raises:
        ValueError: If the row counts differ, cluster labels are missing or
            not allowed, the random slope lies outside the features, or the
            estimand likelihood does not match the estimator.
    """
    if matrix.shape[0] != len(values):
        raise ValueError("features and outcome must have the same row count")
    if policy.covariance == "clustered" and input_clusters is None:
        raise ValueError("clustered covariance needs cluster labels")
    if policy.estimator == "hierarchical_linear" and input_clusters is None:
        raise ValueError("hierarchical_linear requires cluster labels")
    if (
        policy.random_slope_position is not None
        and policy.random_slope_position >= matrix.shape[1]
    ):
        raise ValueError("random_slope_position is outside the feature matrix")
    if (
        policy.estimator in {"ipw_ate", "aipw_ate", "continuous_ipw_ate"}
        and input_clusters is not None
        and policy.covariance != "clustered"
    ):
        raise ValueError(
            f"{policy.estimator} cluster labels require covariance='clustered'"
        )
    expected_likelihood = {
        "linear": "gaussian",
        "logistic": "bernoulli",
        "bayesian_linear": "gaussian",
        "bayesian_ate": "gaussian",
        "ipw_ate": "gaussian",
        "aipw_ate": "gaussian",
        "continuous_ipw_ate": "gaussian",
        "hierarchical_linear": "gaussian",
    }[policy.estimator]
    if estimand.likelihood != expected_likelihood:
        raise ValueError(
            "estimand likelihood does not match estimator: "
            f"expected {expected_likelihood}, got {estimand.likelihood}"
        )


def check_prepared_rows(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    clusters: Sequence[Any] | None,
    policy: Config,
) -> None:
    """Checks the rows that remain after missing values were handled.

    Args:
        matrix: Feature matrix of the kept rows.
        values: Outcome vector of the kept rows.
        clusters: The cluster labels as the caller supplied them.
        policy: Estimator policy.

    Raises:
        ValueError: If a logistic outcome is not 0 or 1, a Bayesian estimator
            was given clusters, or the treatment position of a causal
            estimator is missing or outside the feature matrix.
    """
    if policy.estimator == "logistic" and not set(
        numpy.unique(values)
    ).issubset({0.0, 1.0}):
        raise ValueError("logistic outcome must contain only 0 and 1")
    if (
        policy.estimator in {"bayesian_linear", "bayesian_ate"}
        and clusters is not None
    ):
        raise ValueError(
            f"{policy.estimator} estimation does not accept clusters"
        )
    if policy.estimator in CAUSAL_ESTIMATORS:
        if policy.treatment_position is None:
            raise ValueError(f"{policy.estimator} requires treatment_position")
        if not 0 <= policy.treatment_position < matrix.shape[1]:
            raise ValueError("treatment_position is outside the feature matrix")


def fit_linear_estimator(
    design: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    cluster_values: numpy.ndarray | None,
    names: tuple[str, ...],
) -> EstimatorFit:
    """Fits ordinary least squares and its reference calculation.

    Args:
        design: Design matrix with an intercept column.
        values: Outcome vector.
        policy: Estimator policy.
        cluster_values: Cluster labels of the kept rows, or None.
        names: Parameter names.

    Returns:
        The fitted quantities.
    """
    coefficients, diagnostics = fit_linear(
        design, values, policy.max_iterations, policy.tolerance
    )
    fitted = design @ coefficients
    residuals = values - fitted
    covariance = covariance_matrix(
        design, residuals, policy.covariance, cluster_values, None
    )
    reference = reference_linear(design, values, coefficients, policy)
    likelihood = "gaussian"
    degrees_of_freedom = max(len(values) - design.shape[1], 1)
    pivot = stats.t.ppf(0.5 + policy.confidence_level / 2.0, degrees_of_freedom)
    p_values = 2.0 * stats.t.sf(
        numpy.abs(
            coefficients
            / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
        ),
        degrees_of_freedom,
    )
    diagnostics = {
        **diagnostics,
        "likelihood": likelihood,
        "residual_sum_of_squares": float(numpy.sum(residuals**2)),
        "rmse": float(numpy.sqrt(numpy.mean(residuals**2))),
        "degrees_of_freedom": degrees_of_freedom,
    }
    return EstimatorFit(
        names,
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def fit_logistic_estimator(
    design: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    cluster_values: numpy.ndarray | None,
    names: tuple[str, ...],
) -> EstimatorFit:
    """Fits logistic regression and its reference calculation.

    Args:
        design: Design matrix with an intercept column.
        values: Binary outcome vector.
        policy: Estimator policy.
        cluster_values: Cluster labels of the kept rows, or None.
        names: Parameter names.

    Returns:
        The fitted quantities.
    """
    coefficients, probabilities, diagnostics = fit_logistic(
        design, values, policy.max_iterations, policy.tolerance
    )
    residuals = values - probabilities
    covariance = covariance_matrix(
        design, residuals, policy.covariance, cluster_values, probabilities
    )
    reference = reference_logistic(design, values, coefficients, policy)
    likelihood = "bernoulli"
    pivot = stats.norm.ppf(0.5 + policy.confidence_level / 2.0)
    p_values = 2.0 * stats.norm.sf(
        numpy.abs(
            coefficients
            / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
        )
    )
    diagnostics = {**diagnostics, "likelihood": likelihood}
    return EstimatorFit(
        names,
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def check_bayesian_treatment(
    matrix: numpy.ndarray, policy: Config
) -> tuple[int, int, int]:
    """Checks the binary treatment of the Bayesian average treatment effect.

    Args:
        matrix: Feature matrix of the kept rows.
        policy: Estimator policy with a treatment position.

    Returns:
        The treatment position, the number of treated rows and the number of
        control rows.

    Raises:
        ValueError: If the treatment position is missing, the treatment is
            not 0 or 1, or an arm has fewer than two rows.
    """
    treatment_position = policy.treatment_position
    if treatment_position is None:
        raise ValueError("bayesian_ate requires treatment_position")
    treatment_position = int(treatment_position)
    treatment = matrix[:, treatment_position]
    if not set(numpy.unique(treatment)).issubset({0.0, 1.0}):
        raise ValueError("bayesian_ate treatment must contain only 0 and 1")
    treated_rows = int(numpy.sum(treatment == 1.0))
    control_rows = int(numpy.sum(treatment == 0.0))
    if treated_rows < MIN_ROWS_PER_ARM or control_rows < MIN_ROWS_PER_ARM:
        raise ValueError(
            "bayesian_ate needs at least two rows per treatment arm"
        )
    return treatment_position, treated_rows, control_rows


def fit_bayesian_estimator(
    matrix: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    estimand: Estimand,
    names: tuple[str, ...],
) -> EstimatorFit:
    """Fits the conjugate Bayesian linear model or its causal variant.

    Args:
        matrix: Feature matrix of the kept rows.
        design: Design matrix with an intercept column.
        values: Outcome vector.
        policy: Estimator policy with an explicit prior.
        estimand: Scientific estimand.
        names: Parameter names.

    Returns:
        The fitted quantities, with posterior standard errors and interval
        scales.

    Raises:
        ValueError: If the policy has no prior, or the treatment of a
            Bayesian average treatment effect is invalid.
    """
    if policy.prior is None:
        raise ValueError(
            f"{policy.estimator} estimation requires an explicit prior"
        )
    treatment_position, treated_rows, control_rows = 0, 0, 0
    if policy.estimator == "bayesian_ate":
        treatment_position, treated_rows, control_rows = (
            check_bayesian_treatment(matrix, policy)
        )
    coefficients, covariance, diagnostics = fit_bayesian_linear(
        design, values, policy.prior
    )
    if policy.estimator == "bayesian_ate":
        reference = reference_bayesian_ate(
            design,
            values,
            policy.prior,
            coefficients,
            policy,
        )
        diagnostics = {
            **diagnostics,
            "estimator_family": "causal_bayesian_gaussian",
            "method": "normal_inverse_gamma_outcome_regression",
            "treatment_position": treatment_position,
            "treatment_name": estimand.feature_names[treatment_position],
            "treated_rows": treated_rows,
            "control_rows": control_rows,
            "identification": (
                "the treatment coefficient is the ATE under the "
                "declared linearity, consistency, conditional "
                "exchangeability, and positivity assumptions"
            ),
            "limitations": [
                "time-varying treatment effects are unsupported",
                (
                    "propensity or other nuisance-model uncertainty is not "
                    "propagated"
                ),
                "clustered covariance is unsupported",
                "fully Bayesian multilevel models are unsupported",
            ],
        }
    else:
        reference = reference_bayesian_linear(
            design, values, policy.prior, coefficients, policy
        )
    likelihood = "gaussian"
    posterior_scale = numpy.asarray(diagnostics["posterior_scale"])
    posterior_degrees = diagnostics["posterior_degrees_of_freedom"]
    pivot = stats.t.ppf(0.5 + policy.confidence_level / 2.0, posterior_degrees)
    p_values = 2.0 * stats.t.sf(
        numpy.abs(coefficients)
        / numpy.maximum(numpy.sqrt(numpy.diag(posterior_scale)), 1e-15),
        posterior_degrees,
    )
    standard_errors = numpy.sqrt(numpy.maximum(numpy.diag(covariance), 0.0))
    interval_errors = numpy.sqrt(
        numpy.maximum(numpy.diag(posterior_scale), 0.0)
    )
    diagnostics = {
        **diagnostics,
        "likelihood": likelihood,
        "uncertainty": "posterior_credible_interval",
        "p_value_semantics": "two-sided posterior tail probability",
        "posterior_covariance": covariance.tolist(),
        "credible_interval": {
            "level": policy.confidence_level,
            "distribution": "student_t",
            "degrees_of_freedom": posterior_degrees,
            "scale_matrix": posterior_scale.tolist(),
        },
    }
    return EstimatorFit(
        names,
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
        standard_errors,
        interval_errors,
    )


def fit_propensity_estimator(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    cluster_values: numpy.ndarray | None,
) -> EstimatorFit:
    """Fits the inverse-probability-weighted or doubly robust effect.

    Args:
        matrix: Feature matrix of the kept rows.
        values: Outcome vector.
        policy: Estimator policy, `ipw_ate` or `aipw_ate`.
        cluster_values: Cluster labels of the kept rows, or None.

    Returns:
        The fitted quantities for the single parameter `ate`.
    """
    if policy.estimator == "ipw_ate":
        coefficients, covariance, diagnostics, propensity_design = fit_ipw_ate(
            matrix,
            values,
            policy.treatment_position,
            policy,
            cluster_values,
        )
        reference = reference_ipw_ate(
            propensity_design,
            values,
            matrix[:, policy.treatment_position],
            coefficients,
            policy,
            cluster_values,
        )
    else:
        (
            coefficients,
            covariance,
            diagnostics,
            aipw_inputs,
        ) = fit_aipw_ate(
            matrix,
            values,
            policy.treatment_position,
            policy,
            cluster_values,
        )
        reference = reference_aipw_ate(
            aipw_inputs,
            values,
            coefficients,
            policy,
            cluster_values,
        )
    likelihood = "gaussian"
    pivot = stats.norm.ppf(0.5 + policy.confidence_level / 2.0)
    p_values = 2.0 * stats.norm.sf(
        numpy.abs(coefficients)
        / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
    )
    diagnostics = {**diagnostics, "likelihood": likelihood}
    return EstimatorFit(
        ("ate",),
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def fit_continuous_estimator(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    estimand: Estimand,
    cluster_values: numpy.ndarray | None,
) -> EstimatorFit:
    """Fits the continuous-treatment inverse-probability-weighted effect.

    Args:
        matrix: Feature matrix of the kept rows.
        values: Outcome vector.
        policy: Estimator policy with a treatment position.
        estimand: Scientific estimand.
        cluster_values: Cluster labels of the kept rows, or None.

    Returns:
        The fitted quantities for the intercept and the dose-response slope.

    Raises:
        ValueError: If the policy has no treatment position.
    """
    if policy.treatment_position is None:
        raise ValueError("continuous_ipw_ate requires treatment_position")
    coefficients, covariance, diagnostics = fit_continuous_ipw_effect(
        matrix,
        values,
        policy.treatment_position,
        policy,
        cluster_values,
    )
    reference = reference_continuous_ipw_effect(
        matrix,
        values,
        policy.treatment_position,
        policy,
        coefficients,
        cluster_values,
    )
    treatment_name = estimand.feature_names[policy.treatment_position]
    names = ("intercept", f"{treatment_name}_dose_response_slope")
    likelihood = "gaussian"
    pivot = stats.norm.ppf(0.5 + policy.confidence_level / 2.0)
    p_values = 2.0 * stats.norm.sf(
        numpy.abs(coefficients)
        / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
    )
    diagnostics = {
        **diagnostics,
        "likelihood": likelihood,
        "treatment_name": treatment_name,
        "treatment_position": policy.treatment_position,
        "estimand_interpretation": (
            "the slope is the marginal outcome change for a one-unit "
            "increase in continuous treatment under the declared "
            "linear dose-response and generalized-propensity-density "
            "assumptions"
        ),
    }
    return EstimatorFit(
        names,
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def hierarchical_reference(
    design: numpy.ndarray,
    values: numpy.ndarray,
    cluster_values: numpy.ndarray,
    policy: Config,
    fitted: tuple[Any, ...],
) -> dict[str, Any]:
    """Runs the independent reference for the hierarchical model.

    Args:
        design: Design matrix with an intercept column.
        values: Outcome vector.
        cluster_values: Cluster labels of the kept rows.
        policy: Estimator policy, selecting method and random slope.
        fitted: The coefficients, diagnostics, residual variance, random
            intercept variance and random-effect covariance of the fit.

    Returns:
        The reference record matching the declared random-effects structure.
    """
    (
        coefficients,
        diagnostics,
        residual_variance,
        random_intercept_variance,
        random_effect_covariance,
    ) = fitted
    if policy.random_slope_position is not None:
        if policy.hierarchical_method == "reml":
            return reference_hierarchical_random_slope_reml(
                design,
                values,
                cluster_values,
                policy.random_slope_position,
                residual_variance,
                random_effect_covariance,
                coefficients,
                diagnostics["reml_objective"],
                policy,
            )
        return reference_hierarchical_random_slope(
            design,
            values,
            cluster_values,
            policy.random_slope_position,
            residual_variance,
            random_effect_covariance,
            coefficients,
            policy,
        )
    if policy.hierarchical_method == "reml":
        return reference_hierarchical_reml(
            design,
            values,
            cluster_values,
            residual_variance,
            random_intercept_variance,
            coefficients,
            diagnostics["reml_objective"],
            policy,
        )
    return reference_hierarchical_linear(
        design,
        values,
        cluster_values,
        residual_variance,
        random_intercept_variance,
        coefficients,
        policy,
    )


def fit_hierarchical_estimator(
    design: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    cluster_values: numpy.ndarray | None,
    names: tuple[str, ...],
) -> EstimatorFit:
    """Fits the hierarchical linear model and its reference calculation.

    Args:
        design: Design matrix with an intercept column.
        values: Outcome vector.
        policy: Estimator policy.
        cluster_values: Cluster labels of the kept rows.
        names: Parameter names.

    Returns:
        The fitted quantities.

    Raises:
        ValueError: If there are no cluster labels.
    """
    if cluster_values is None:
        raise ValueError("hierarchical_linear requires cluster labels")
    (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        random_intercept_variance,
        random_effect_covariance,
    ) = fit_hierarchical_linear(design, values, cluster_values, policy)
    reference = hierarchical_reference(
        design,
        values,
        cluster_values,
        policy,
        (
            coefficients,
            diagnostics,
            residual_variance,
            random_intercept_variance,
            random_effect_covariance,
        ),
    )
    likelihood = "gaussian"
    degrees_of_freedom = max(len(dict.fromkeys(cluster_values.tolist())) - 1, 1)
    pivot = stats.t.ppf(0.5 + policy.confidence_level / 2.0, degrees_of_freedom)
    p_values = 2.0 * stats.t.sf(
        numpy.abs(
            coefficients
            / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
        ),
        degrees_of_freedom,
    )
    diagnostics = {
        **diagnostics,
        "likelihood": likelihood,
        "degrees_of_freedom": degrees_of_freedom,
    }
    return EstimatorFit(
        names,
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def fit_estimator(
    matrix: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    estimand: Estimand,
    policy: Config,
    cluster_values: numpy.ndarray | None,
) -> EstimatorFit:
    """Fits the estimator the policy names.

    Args:
        matrix: Feature matrix of the kept rows.
        design: Design matrix with an intercept column.
        values: Outcome vector.
        estimand: Scientific estimand.
        policy: Estimator policy.
        cluster_values: Cluster labels of the kept rows, or None.

    Returns:
        The fitted quantities of the chosen estimator.
    """
    names = ("intercept", *estimand.feature_names)
    if policy.estimator == "linear":
        return fit_linear_estimator(
            design, values, policy, cluster_values, names
        )
    if policy.estimator == "logistic":
        return fit_logistic_estimator(
            design, values, policy, cluster_values, names
        )
    if policy.estimator in {"bayesian_linear", "bayesian_ate"}:
        return fit_bayesian_estimator(
            matrix, design, values, policy, estimand, names
        )
    if policy.estimator in {"ipw_ate", "aipw_ate"}:
        return fit_propensity_estimator(matrix, values, policy, cluster_values)
    if policy.estimator == "continuous_ipw_ate":
        return fit_continuous_estimator(
            matrix, values, policy, estimand, cluster_values
        )
    return fit_hierarchical_estimator(
        design, values, policy, cluster_values, names
    )


def estimate_parameters(
    features: Any,
    outcome: Any,
    estimand: Estimand,
    policy: Config,
    clusters: Sequence[Any] | None = None,
    shift_reference: Any | None = None,
) -> Estimation:
    """Estimates declared parameters with auditable uncertainty evidence.

    Args:
        features: Numeric matrix with one column per ``estimand.feature_names``.
        outcome: Numeric response, or binary labels for logistic estimation.
        estimand: Scientific definition and identification assumptions.
        policy: Estimator, covariance, missingness, and uncertainty policy.
            ``bayesian_linear`` and ``bayesian_ate`` require an explicit
            ``BayesianPrior`` and report posterior credible intervals.
        clusters: Optional cluster labels, one per input row. Required for
            clustered covariance and ``hierarchical_linear``; rejected by
            the Bayesian estimators; for ``ipw_ate``, ``aipw_ate`` and
            ``continuous_ipw_ate`` allowed only with clustered covariance.
        shift_reference: Optional feature matrix from the target population,
            with the same columns as `features`, for standardized mean-shift
            diagnostics.

    Returns:
        A result containing estimates, covariance, intervals, diagnostics,
        sensitivity records, and an independent reference calculation.

    Raises:
        ValueError: If the data, policy, or estimator assumptions are invalid.
    """
    matrix = numpy.asarray(features, dtype=float)
    values = numpy.asarray(outcome, dtype=float)
    validate_matrix(matrix, len(estimand.feature_names), "features")
    validate_vector(values, "outcome")
    input_clusters = validate_clusters(clusters, len(values))
    raw_input_fingerprint_value = raw_input_fingerprint(
        matrix, values, input_clusters
    )
    input_rows = matrix.shape[0]
    check_estimation_request(matrix, values, input_clusters, policy, estimand)
    matrix, values, kept, dropped = prepare_rows(
        matrix, values, policy.missing, input_clusters
    )
    check_prepared_rows(matrix, values, clusters, policy)
    cluster_values = kept if input_clusters is not None else None
    design = numpy.column_stack((numpy.ones(matrix.shape[0]), matrix))
    fit = fit_estimator(
        matrix, design, values, estimand, policy, cluster_values
    )
    names = fit.names
    coefficients = fit.coefficients
    covariance = fit.covariance
    diagnostics = fit.diagnostics

    covariance_conditioning = validate_covariance(covariance, len(names))
    diagnostics = {
        **diagnostics,
        "covariance_conditioning": covariance_conditioning,
    }
    if policy.covariance == "clustered":
        if cluster_values is None:
            raise ValueError(
                "cluster labels are required for cluster diagnostics"
            )
        diagnostics["cluster_uncertainty"] = cluster_uncertainty_diagnostics(
            cluster_values
        )

    if fit.standard_errors is None:
        standard_errors = numpy.sqrt(numpy.maximum(numpy.diag(covariance), 0.0))
        interval_errors = standard_errors
    else:
        standard_errors = fit.standard_errors
        interval_errors = (
            standard_errors
            if fit.interval_errors is None
            else fit.interval_errors
        )

    parameters = tuple(
        ParameterEstimate(
            name,
            float(value),
            float(error),
            (
                float(value - fit.pivot * interval_error),
                float(value + fit.pivot * interval_error),
            ),
            float(p_value),
        )
        for name, value, error, interval_error, p_value in zip(
            names,
            coefficients,
            standard_errors,
            interval_errors,
            fit.p_values,
            strict=True,
        )
    )
    sensitivity = sensitivity_report(
        matrix,
        values,
        policy,
        design,
        coefficients,
        shift_reference,
        estimand,
        cluster_values,
        prior=(
            policy.prior
            if policy.estimator in {"bayesian_linear", "bayesian_ate"}
            else None
        ),
    )
    return Estimation(
        estimand=dataclasses.replace(estimand, likelihood=fit.likelihood),
        policy=policy,
        parameters=parameters,
        covariance=tuple(
            tuple(float(value) for value in row) for row in covariance
        ),
        rows_input=int(input_rows),
        rows_used=len(values),
        rows_dropped=int(dropped),
        diagnostics=diagnostics,
        sensitivity=sensitivity,
        reference=fit.reference,
        raw_input_fingerprint=raw_input_fingerprint_value,
    )


def raw_input_fingerprint(
    features: Any,
    outcome: Any,
    clusters: Any | None = None,
) -> str:
    """Returns a deterministic fingerprint of the estimator's raw inputs.

    Numeric values are hashed after the same float conversion used by the
    estimator. Cluster labels retain their value types and row order. The
    fingerprint covers inputs before missing-row filtering so it can identify
    the complete evidence source independently of the fitted result.

    Args:
        features: Feature matrix, converted to float.
        outcome: Response values, flattened after conversion to float.
        clusters: Optional cluster labels, hashed as objects.

    Returns:
        The SHA-256 hex digest over the shape, dtype and values of each
        input.
    """
    matrix = numpy.asarray(features, dtype=float)
    values = numpy.asarray(outcome, dtype=float).reshape(-1)
    digest = hashlib.sha256()
    update_input_digest(digest, "features", matrix)
    update_input_digest(digest, "outcome", values)
    if clusters is None:
        digest.update(b"clusters:none")
    else:
        cluster_values = numpy.asarray(clusters, dtype=object)
        update_input_digest(digest, "clusters", cluster_values)
    return digest.hexdigest()


def update_input_digest(digest: Any, name: str, values: numpy.ndarray) -> None:
    """Adds one shape-, dtype-, and value-bound input to a digest.

    Args:
        digest: Hash object updated in place.
        name: Label bound into the descriptor of this input.
        values: Array whose shape, dtype and contents are hashed.
    """
    array = numpy.asarray(values)
    descriptor = json.dumps(
        {
            "name": name,
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest.update(len(descriptor).to_bytes(8, "big"))
    digest.update(descriptor)
    if array.dtype.kind in "biufc":
        canonical = numpy.ascontiguousarray(
            array.astype(array.dtype.newbyteorder("<"), copy=False)
        )
        values_bytes = canonical.tobytes()
    else:
        value_hashes = pandas.util.hash_pandas_object(
            pandas.Series(array.reshape(-1)), index=False
        ).to_numpy(dtype="<u8")
        type_names = [
            f"{type(value).__module__}.{type(value).__qualname__}"
            for value in array.reshape(-1).tolist()
        ]
        values_bytes = json.dumps(
            {
                "types": type_names,
                "hashes": value_hashes.tolist(),
            },
            separators=(",", ":"),
        ).encode("utf-8")
    digest.update(len(values_bytes).to_bytes(8, "big"))
    digest.update(values_bytes)


def validate_matrix(matrix: numpy.ndarray, columns: int, name: str) -> None:
    """Validates a numeric two-dimensional matrix.

    Args:
        matrix: Matrix to check.
        columns: Required number of columns.
        name: Matrix name used in the error message.

    Raises:
        ValueError: If the matrix is not two-dimensional with `columns`
            columns, or does not have more rows than `columns`.
    """
    if matrix.ndim != MATRIX_NDIM or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape (rows, {columns})")
    if matrix.shape[0] <= columns:
        raise ValueError("estimation needs more rows than parameters")


def validate_vector(values: numpy.ndarray, name: str) -> None:
    """Checks that a vector is one-dimensional without altering it.

    Args:
        values: Vector to check.
        name: Vector name used in the error message.

    Raises:
        ValueError: If the array is not one-dimensional.
    """
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (rows,)")


def validate_clusters(
    clusters: Sequence[Any] | None, rows: int
) -> numpy.ndarray | None:
    """Validates cluster-label shape before missingness processing.

    Args:
        clusters: Cluster labels, or None when the data are not clustered.
        rows: Number of rows the labels must cover.

    Returns:
        The labels as a one-dimensional object array, or None when
        `clusters` is None.

    Raises:
        ValueError: If the labels are not one-dimensional or their length
            differs from `rows`.
    """
    if clusters is None:
        return None
    cluster_values = numpy.asarray(clusters, dtype=object)
    if cluster_values.ndim != 1:
        raise ValueError("clusters must have shape (rows,)")
    if len(cluster_values) != rows:
        raise ValueError("clusters must have the same row count as outcome")
    return cluster_values


def validate_covariance(
    covariance: numpy.ndarray, dimension: int
) -> dict[str, Any]:
    """Validates covariance shape and reports rank and numerical conditioning.

    Args:
        covariance: Coefficient covariance matrix.
        dimension: Required number of rows and columns.

    Returns:
        A dictionary with the identification ``status`` (``identified``,
        ``ill_conditioned`` or ``rank_deficient``), ``dimension``, numerical
        ``rank``, ``condition_number`` (None when the smallest singular value
        is zero) and the ``condition_number_threshold`` used.

    Raises:
        ValueError: If the matrix is not `dimension` by `dimension`, has a
            non-finite entry, or is not symmetric.
    """
    matrix = numpy.asarray(covariance, dtype=float)
    expected_shape = (dimension, dimension)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"covariance must have shape ({dimension}, {dimension})"
        )
    if not numpy.isfinite(matrix).all():
        raise ValueError("covariance must contain finite values")
    if not numpy.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    singular_values = numpy.linalg.svd(matrix, compute_uv=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = numpy.finfo(float).eps * max(expected_shape) * scale
    rank = int(numpy.sum(singular_values > rank_tolerance))
    smallest_singular_value = (
        float(singular_values[-1]) if singular_values.size else 0.0
    )
    condition_number = (
        float(scale / smallest_singular_value)
        if smallest_singular_value > 0.0
        else None
    )
    if rank < dimension:
        status = "rank_deficient"
    elif (
        condition_number is None
        or condition_number > COVARIANCE_CONDITION_THRESHOLD
    ):
        status = "ill_conditioned"
    else:
        status = "identified"
    return {
        "status": status,
        "dimension": dimension,
        "rank": rank,
        "condition_number": condition_number,
        "condition_number_threshold": COVARIANCE_CONDITION_THRESHOLD,
    }


def prepare_rows(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    missing: MissingPolicy,
    clusters: Any | None,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None, int]:
    """Applies the declared missingness policy and keeps cluster labels aligned.

    Args:
        matrix: Feature matrix.
        values: Outcome vector.
        missing: ``reject`` raises when any row is incomplete;
            ``complete_case`` drops incomplete rows.
        clusters: Optional cluster labels, one per row.

    Returns:
        The kept feature rows, the kept outcomes, the kept cluster labels
        (None when `clusters` is None) and the number of rows dropped.  A row
        is dropped when any feature or the outcome is not finite or its
        cluster label is missing.

    Raises:
        ValueError: If cluster labels are not scalar, if rows are incomplete
            and the policy is ``reject``, or if no more than the number of
            columns plus one complete rows remain.
    """
    valid = numpy.isfinite(matrix).all(axis=1) & numpy.isfinite(values)
    if clusters is not None:
        cluster_array = numpy.asarray(clusters, dtype=object)
        cluster_missing = numpy.asarray(pandas.isna(cluster_array))
        if cluster_missing.ndim != 1:
            raise ValueError("cluster labels must be scalar values")
        valid &= ~cluster_missing.astype(bool)
    else:
        cluster_array = None
    dropped = int((~valid).sum())
    if dropped and missing == "reject":
        raise ValueError(
            "missing or non-finite values require an explicit policy"
        )
    if valid.sum() <= matrix.shape[1] + 1:
        raise ValueError("complete rows are insufficient for estimation")
    kept_clusters = None
    if cluster_array is not None:
        kept_clusters = cluster_array[valid]
    return matrix[valid], values[valid], kept_clusters, dropped


def fit_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[numpy.ndarray, dict[str, Any]]:
    """Solves the Gaussian likelihood using a full-rank normal equation.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        max_iterations: Unused; the solution is direct.
        tolerance: Unused; the solution is direct.

    Returns:
        The least-squares coefficients and a diagnostics dictionary reporting
        ``converged`` as True and one iteration.

    Raises:
        ValueError: If the Gram matrix of the design is rank deficient.
    """
    del max_iterations, tolerance
    gram = design.T @ design
    if numpy.linalg.matrix_rank(gram) != gram.shape[0]:
        raise ValueError("linear design matrix is rank deficient")
    coefficients = numpy.linalg.solve(gram, design.T @ values)
    return coefficients, {"converged": True, "iterations": 1}


def fit_treatment_density(
    treatment: numpy.ndarray, confounders: numpy.ndarray
) -> tuple[numpy.ndarray, numpy.ndarray, float]:
    """Fits the conditional Gaussian density of a continuous treatment.

    Args:
        treatment: The continuous treatment values.
        confounders: The non-treatment feature columns.

    Returns:
        The conditional mean, the residuals and the residual scale of the
        linear treatment model.

    Raises:
        ValueError: If the density design is rank deficient, the model has no
            residual degrees of freedom, or the residual scale is not
            positive.
    """
    density_design = numpy.column_stack(
        (numpy.ones(len(confounders)), confounders)
    )
    if numpy.linalg.matrix_rank(density_design) != density_design.shape[1]:
        raise ValueError(
            "continuous treatment density design matrix is rank deficient"
        )
    density_coefficients, residual_sum, rank, singular_values = (
        numpy.linalg.lstsq(density_design, treatment, rcond=None)
    )
    del residual_sum, singular_values
    if rank != density_design.shape[1]:
        raise ValueError(
            "continuous treatment density model is not identifiable"
        )
    conditional_mean = density_design @ density_coefficients
    density_residuals = treatment - conditional_mean
    residual_degrees = len(treatment) - density_design.shape[1]
    if residual_degrees < 1:
        raise ValueError(
            "continuous treatment density model needs residual degrees of "
            "freedom"
        )
    conditional_scale = float(
        numpy.sqrt(numpy.sum(density_residuals**2) / residual_degrees)
    )
    if not math.isfinite(conditional_scale) or conditional_scale <= SCALE_FLOOR:
        raise ValueError(
            "continuous treatment density has non-positive residual scale"
        )
    return conditional_mean, density_residuals, conditional_scale


def fit_continuous_ipw_effect(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Estimates a linear continuous-treatment effect with GPS weights.

    The generalized propensity score is a conditional Gaussian treatment
    density estimated from the declared non-treatment features. Stabilized
    density-ratio weights then define a weighted marginal dose-response
    regression. The reported treatment slope is a one-unit marginal effect,
    not a claim about an arbitrary nonlinear dose-response curve. The weights
    are capped at `policy.continuous_weight_cap`.

    Args:
        matrix: Complete numeric feature matrix.
        values: Complete numeric outcomes.
        treatment_position: Column containing the continuous treatment.
        policy: Supplies the weight cap and the covariance kind.
        clusters: Optional independent sampling-unit labels.

    Returns:
        The weighted intercept and dose-response slope, their covariance
        matrix, and a diagnostics dictionary.

    Raises:
        ValueError: If treatment support, density fitting, or weighted
            regression is not identifiable.
    """
    treatment = matrix[:, treatment_position]
    unique_treatment = numpy.unique(treatment)
    if len(unique_treatment) < MIN_TREATMENT_LEVELS:
        raise ValueError(
            "continuous_ipw_ate needs at least four distinct treatment values"
        )
    marginal_scale = float(numpy.std(treatment, ddof=1))
    if not math.isfinite(marginal_scale) or marginal_scale <= SCALE_FLOOR:
        raise ValueError("continuous_ipw_ate needs positive treatment variance")

    confounders = numpy.delete(matrix, treatment_position, axis=1)
    conditional_mean, density_residuals, conditional_scale = (
        fit_treatment_density(treatment, confounders)
    )
    residual_degrees = len(treatment) - (1 + confounders.shape[1])
    log_marginal_density = stats.norm.logpdf(
        treatment, loc=float(numpy.mean(treatment)), scale=marginal_scale
    )
    log_conditional_density = stats.norm.logpdf(
        treatment, loc=conditional_mean, scale=conditional_scale
    )
    log_weight = log_marginal_density - log_conditional_density
    raw_weights = numpy.exp(numpy.clip(log_weight, -745.0, 709.0))
    if not numpy.isfinite(raw_weights).all() or numpy.any(raw_weights <= 0.0):
        raise ValueError("continuous treatment density weights are invalid")
    weights = numpy.minimum(raw_weights, policy.continuous_weight_cap)
    capped = raw_weights > policy.continuous_weight_cap
    effective_sample_size = float(
        numpy.sum(weights) ** 2 / numpy.sum(weights**2)
    )
    if effective_sample_size < MIN_EFFECTIVE_SAMPLE_SIZE:
        raise ValueError(
            "continuous treatment weights leave fewer than two effective "
            "observations"
        )

    dose_design = numpy.column_stack((numpy.ones(len(treatment)), treatment))
    weighted_design = dose_design * numpy.sqrt(weights)[:, None]
    weighted_outcome = values * numpy.sqrt(weights)
    if numpy.linalg.matrix_rank(weighted_design) != weighted_design.shape[1]:
        raise ValueError("weighted dose-response design is rank deficient")
    coefficients = numpy.linalg.solve(
        weighted_design.T @ weighted_design,
        weighted_design.T @ weighted_outcome,
    )
    weighted_residuals = (values - dose_design @ coefficients) * numpy.sqrt(
        weights
    )
    covariance = covariance_matrix(
        weighted_design,
        weighted_residuals,
        policy.covariance,
        clusters,
        None,
    )
    overlap = {
        "status": "diagnosed_not_proven",
        "treatment_min": float(numpy.min(treatment)),
        "treatment_max": float(numpy.max(treatment)),
        "unique_treatment_values": len(unique_treatment),
        "marginal_treatment_mean": float(numpy.mean(treatment)),
        "marginal_treatment_sd": marginal_scale,
        "conditional_treatment_residual_sd": conditional_scale,
        "conditional_standardized_residual_quantiles": [
            float(item)
            for item in numpy.quantile(
                numpy.abs(density_residuals) / conditional_scale,
                (0.5, 0.9, 0.99, 1.0),
            )
        ],
        "conditional_density_min": float(
            numpy.min(numpy.exp(log_conditional_density))
        ),
        "conditional_density_max": float(
            numpy.max(numpy.exp(log_conditional_density))
        ),
    }
    weight_diagnostics = {
        "raw_min": float(numpy.min(raw_weights)),
        "raw_max": float(numpy.max(raw_weights)),
        "capped_min": float(numpy.min(weights)),
        "capped_max": float(numpy.max(weights)),
        "cap": policy.continuous_weight_cap,
        "fraction_capped": float(numpy.mean(capped)),
        "quantiles": [
            float(item)
            for item in numpy.quantile(weights, (0.0, 0.5, 0.9, 0.99, 1.0))
        ],
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": effective_sample_size / len(weights),
        "normalization": "stabilized_marginal_over_conditional_density",
    }
    diagnostics: dict[str, Any] = {
        "converged": True,
        "iterations": 1,
        "estimator_family": "causal_continuous_ipw",
        "method": "stabilized_generalized_propensity_weighted_regression",
        "uncertainty_method": (
            "weighted_gaussian_regression"
            if policy.covariance == "iid"
            else "clustered_weighted_gaussian_regression"
        ),
        "nuisance_model_uncertainty": "not_propagated",
        "treatment_density_model": {
            "method": "ordinary_least_squares_gaussian_density",
            "conditional_mean_features": "all_declared_non_treatment_features",
            "residual_degrees_of_freedom": residual_degrees,
            "residual_scale": conditional_scale,
        },
        "overlap": overlap,
        "weights": weight_diagnostics,
        "capability": {
            "status": "experimental",
            "name": "continuous_treatment_effect",
            "contract": "linear_marginal_dose_response",
            "assumptions": list(policy.continuous_treatment_assumptions),
            "evidence": [
                "generalized_propensity_density_diagnostics",
                "weight_effective_sample_size",
                "independent_reference_calculation",
            ],
            "limitations": [
                "normal conditional treatment density is required",
                "linear dose-response is required",
                "nuisance-model uncertainty is not propagated",
                (
                    "nonlinear dose-response and continuous-treatment AIPW are "
                    "unsupported"
                ),
            ],
        },
        "limitations": [
            "positivity is diagnosed but not proven",
            "normal conditional treatment density is imposed",
            "linear dose-response is imposed",
            "nuisance-model uncertainty is not propagated",
            (
                "nonlinear dose-response and continuous-treatment AIPW are "
                "unsupported"
            ),
        ],
    }
    if policy.covariance == "clustered":
        diagnostics["cluster_uncertainty"] = cluster_uncertainty_diagnostics(
            clusters
        )
    return coefficients, covariance, diagnostics


def reference_continuous_ipw_effect(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int,
    policy: Config,
    coefficients: numpy.ndarray,
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Recomputes the GPS-weighted dose-response with dense linear algebra.

    This reference intentionally uses a separate least-squares and normal
    equation calculation from :func:`fit_continuous_ipw_effect`. It verifies
    the point estimate, while the primary fit separately records covariance
    and overlap evidence.

    Args:
        matrix: Complete numeric feature matrix.
        values: Complete numeric outcomes.
        treatment_position: Column containing the continuous treatment.
        policy: Supplies the weight cap and the reference tolerance.
        coefficients: Intercept and slope from the primary fit.
        clusters: Optional cluster labels; only their presence is recorded.

    Returns:
        A dictionary with the method name, the largest absolute difference
        from `coefficients`, the reference slope, whether clusters were
        supplied, and ``passed``, True when the difference is within ten
        times the policy's reference tolerance.  The dense solver raises
        its own error if the weighted system is singular or not positive
        definite.
    """
    treatment = matrix[:, treatment_position]
    confounders = numpy.delete(matrix, treatment_position, axis=1)
    density_design = numpy.column_stack(
        (numpy.ones(len(confounders)), confounders)
    )
    density_coefficients = numpy.linalg.lstsq(
        density_design, treatment, rcond=None
    )[0]
    conditional_mean = density_design @ density_coefficients
    residual_degrees = len(treatment) - density_design.shape[1]
    conditional_scale = float(
        numpy.sqrt(
            numpy.sum((treatment - conditional_mean) ** 2)
            / max(residual_degrees, 1)
        )
    )
    marginal_scale = float(numpy.std(treatment, ddof=1))
    log_weights = stats.norm.logpdf(
        treatment, numpy.mean(treatment), marginal_scale
    ) - stats.norm.logpdf(treatment, conditional_mean, conditional_scale)
    weights = numpy.minimum(
        numpy.exp(numpy.clip(log_weights, -745.0, 709.0)),
        policy.continuous_weight_cap,
    )
    dose_design = numpy.column_stack((numpy.ones(len(treatment)), treatment))
    information = dose_design.T @ (weights[:, None] * dose_design)
    right_hand_side = dose_design.T @ (weights * values)
    reference = linalg.solve(information, right_hand_side, assume_a="pos")
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:dense_stabilized_gps_regression",
        "max_absolute_difference": difference,
        "reference_dose_response_slope": float(reference[1]),
        "clustered_reference": clusters is not None,
        "passed": difference <= policy.reference_tolerance * 10,
    }


def cluster_uncertainty_diagnostics(
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Returns explicit sampling-unit diagnostics for clustered covariance.

    Args:
        clusters: Cluster label of each row, or None.

    Returns:
        A dictionary with the method name, cluster count, cluster sizes in
        first-appearance order, smallest and largest size, fraction of
        singleton clusters, the sampling-unit declaration, and
        ``small_cluster_caution``, True when there are fewer than 30
        clusters.

    Raises:
        ValueError: If `clusters` is None, the labels are not hashable, or
            there are fewer than two clusters.
    """
    if clusters is None:
        raise ValueError("cluster labels are required for cluster diagnostics")
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "cluster labels must be hashable scalar values"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise ValueError("clustered covariance needs at least two clusters")
    sizes = [int(numpy.sum(clusters == label)) for label in labels]
    return {
        "method": "finite_sample_cluster_robust_sandwich",
        "cluster_count": len(labels),
        "cluster_sizes": sizes,
        "minimum_cluster_size": min(sizes),
        "maximum_cluster_size": max(sizes),
        "singleton_cluster_fraction": float(
            numpy.mean(numpy.asarray(sizes) == 1)
        ),
        "independent_sampling_unit": "declared_cluster_label",
        "small_cluster_caution": len(labels) < SMALL_CLUSTER_COUNT,
    }


def fit_ipw_ate(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int | None,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, Any], numpy.ndarray]:
    """Estimates a binary-treatment Hájek inverse-probability ATE.

    Args:
        matrix: Complete numeric feature matrix, including the treatment.
        values: Complete numeric outcomes.
        treatment_position: Column of the 0/1 treatment.
        policy: Supplies the propensity clip, iteration limit, tolerance and
            covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        The one-element ATE vector, its 1x1 variance matrix, a diagnostics
        dictionary, and the propensity design matrix (intercept plus the
        non-treatment features).

    Raises:
        ValueError: If `treatment_position` is None, the treatment is not
            0/1, or either arm has fewer than two rows.  Errors from the
            propensity fit and variance calculation also propagate.
    """
    if treatment_position is None:
        raise ValueError("ipw_ate requires treatment_position")
    treatment = matrix[:, treatment_position]
    if not set(numpy.unique(treatment)).issubset({0.0, 1.0}):
        raise ValueError("ipw_ate treatment must contain only 0 and 1")
    treated_count = int(numpy.sum(treatment == 1.0))
    control_count = int(numpy.sum(treatment == 0.0))
    if treated_count < MIN_ROWS_PER_ARM or control_count < MIN_ROWS_PER_ARM:
        raise ValueError("ipw_ate needs at least two rows per treatment arm")
    confounders = numpy.delete(matrix, treatment_position, axis=1)
    propensity_design = numpy.column_stack(
        (numpy.ones(confounders.shape[0]), confounders)
    )
    propensity_coefficients, raw_propensity, propensity_diagnostics = (
        fit_logistic(
            propensity_design,
            treatment,
            policy.max_iterations,
            policy.tolerance,
        )
    )
    del propensity_coefficients
    propensity = numpy.clip(
        raw_propensity, policy.propensity_clip, 1.0 - policy.propensity_clip
    )
    ate, variance, evidence = hajek_ipw_statistics(
        treatment,
        values,
        propensity,
        clusters,
        policy.covariance,
    )
    clipped = numpy.abs(raw_propensity - propensity) > CLIP_TOLERANCE
    diagnostics = {
        "converged": True,
        "iterations": propensity_diagnostics["iterations"],
        "estimator_family": "causal_ipw",
        "method": "hajek_ipw",
        "uncertainty_method": (
            "influence_function_with_estimated_propensity"
            if policy.covariance == "iid"
            else evidence["uncertainty_method"]
        ),
        "propensity_estimation_uncertainty": "not_propagated",
        "treatment_position": treatment_position,
        "treated_rows": treated_count,
        "control_rows": control_count,
        "propensity_model": {
            "method": "logistic_mle",
            "converged": propensity_diagnostics["converged"],
            "iterations": propensity_diagnostics["iterations"],
            "log_likelihood": propensity_diagnostics["log_likelihood"],
        },
        "positivity": {
            "status": "diagnosed_not_proven",
            "raw_min": float(numpy.min(raw_propensity)),
            "raw_max": float(numpy.max(raw_propensity)),
            "clipped_min": float(numpy.min(propensity)),
            "clipped_max": float(numpy.max(propensity)),
            "clip_threshold": policy.propensity_clip,
            "fraction_clipped": float(numpy.mean(clipped)),
        },
        "effective_sample_size": evidence["effective_sample_size"],
        "treatment_means": evidence["treatment_means"],
        "influence_function": {
            "method": "centered_ipw_influence_function",
            "variance": variance,
            "variance_method": evidence["uncertainty_method"],
        },
    }
    if policy.covariance == "clustered":
        diagnostics.update(evidence["uncertainty"]["cluster_diagnostics"])
    return (
        numpy.asarray([ate]),
        numpy.asarray([[variance]]),
        diagnostics,
        propensity_design,
    )


def fit_aipw_ate(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    treatment_position: int | None,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[
    numpy.ndarray, numpy.ndarray, dict[str, Any], dict[str, numpy.ndarray]
]:
    """Estimates a binary-treatment ATE with an augmented IPW score.

    With one fold, nuisance models are fitted on the full sample and scored
    on the same rows. With multiple folds, each nuisance
    model is fitted only on the corresponding training rows and its score is
    evaluated on held-out rows.

    Args:
        matrix: Complete numeric feature matrix, including the treatment.
        values: Complete numeric outcomes.
        treatment_position: Column of the 0/1 treatment.
        policy: Supplies the number of cross-fitting folds, the propensity
            clip, iteration limit, tolerance and covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        The one-element ATE vector, its 1x1 variance matrix, a diagnostics
        dictionary, and a dictionary of the arrays the reference calculation
        needs (nuisance design, treatment, predicted control and treated
        outcomes, and fold assignments).

    Raises:
        ValueError: If `treatment_position` is None, the treatment is not
            0/1, or either arm has fewer than two rows.  Errors from the fold
            assignment, support check, nuisance fits and variance calculation
            also propagate.
    """
    if treatment_position is None:
        raise ValueError("aipw_ate requires treatment_position")
    treatment = matrix[:, treatment_position]
    if not set(numpy.unique(treatment)).issubset({0.0, 1.0}):
        raise ValueError("aipw_ate treatment must contain only 0 and 1")
    treated_count = int(numpy.sum(treatment == 1.0))
    control_count = int(numpy.sum(treatment == 0.0))
    if treated_count < MIN_ROWS_PER_ARM or control_count < MIN_ROWS_PER_ARM:
        raise ValueError("aipw_ate needs at least two rows per treatment arm")

    confounders = numpy.delete(matrix, treatment_position, axis=1)
    nuisance_design = numpy.column_stack(
        (numpy.ones(confounders.shape[0]), confounders)
    )
    fold_count = policy.cross_fitting_folds
    if fold_count == 1:
        (
            propensity,
            raw_propensity,
            outcome_control,
            outcome_treated,
            nuisance_diagnostics,
        ) = fit_aipw_nuisance_models(
            nuisance_design,
            treatment,
            values,
            nuisance_design,
            policy,
        )
        fold_assignments = numpy.zeros(len(values), dtype=int)
        cross_fitting = {
            "status": "disabled",
            "fold_count": 1,
            "assignment_method": "not_applicable",
            "limitations": aipw_limitations(cross_fitted=False),
        }
    else:
        fold_assignments = deterministic_fold_assignments(
            len(values), fold_count
        )
        support = validate_cross_fitting_support(
            treatment, fold_assignments, fold_count
        )
        propensity = numpy.empty(len(values), dtype=float)
        raw_propensity = numpy.empty(len(values), dtype=float)
        outcome_control = numpy.empty(len(values), dtype=float)
        outcome_treated = numpy.empty(len(values), dtype=float)
        fold_diagnostics: list[dict[str, Any]] = []
        for fold in range(fold_count):
            training = fold_assignments != fold
            validation = ~training
            (
                fold_propensity,
                fold_raw_propensity,
                fold_outcome_control,
                fold_outcome_treated,
                fold_nuisance_diagnostics,
            ) = fit_aipw_nuisance_models(
                nuisance_design[training],
                treatment[training],
                values[training],
                nuisance_design[validation],
                policy,
            )
            propensity[validation] = fold_propensity
            raw_propensity[validation] = fold_raw_propensity
            outcome_control[validation] = fold_outcome_control
            outcome_treated[validation] = fold_outcome_treated
            fold_diagnostics.append(
                {
                    "fold": fold,
                    "training_rows": int(numpy.sum(training)),
                    "validation_rows": int(numpy.sum(validation)),
                    "propensity_iterations": fold_nuisance_diagnostics[
                        "propensity_iterations"
                    ],
                    "propensity_log_likelihood": fold_nuisance_diagnostics[
                        "propensity_log_likelihood"
                    ],
                    "treated_rank": fold_nuisance_diagnostics["treated_rank"],
                    "control_rank": fold_nuisance_diagnostics["control_rank"],
                }
            )
        nuisance_diagnostics = {
            "propensity_iterations": max(
                item["propensity_iterations"] for item in fold_diagnostics
            ),
            "folds": fold_diagnostics,
        }
        cross_fitting = {
            "status": "enabled",
            "fold_count": fold_count,
            "assignment_method": "row_index_modulo",
            "fold_assignments": fold_assignments.tolist(),
            "support": support,
            "training_rows_by_fold": {
                str(fold): int(numpy.sum(fold_assignments != fold))
                for fold in range(fold_count)
            },
            "validation_rows_by_fold": {
                str(fold): int(numpy.sum(fold_assignments == fold))
                for fold in range(fold_count)
            },
            "nuisance_fits": fold_diagnostics,
            "limitations": aipw_limitations(cross_fitted=True),
        }
    score = (
        outcome_treated
        - outcome_control
        + treatment * (values - outcome_treated) / propensity
        - (1.0 - treatment) * (values - outcome_control) / (1.0 - propensity)
    )
    ate = float(numpy.mean(score))
    influence = score - ate
    variance, uncertainty = influence_variance(
        influence, policy.covariance, clusters
    )
    clipped = numpy.abs(raw_propensity - propensity) > CLIP_TOLERANCE
    diagnostics = {
        "converged": True,
        "iterations": nuisance_diagnostics["propensity_iterations"],
        "estimator_family": "causal_augmented_ipw",
        "method": "augmented_ipw",
        "uncertainty_method": (
            (
                "influence_function_with_cross_fitted_nuisance"
                if fold_count > 1
                else "influence_function_with_fitted_nuisance"
            )
            if policy.covariance == "iid"
            else uncertainty["method"]
        ),
        "cross_fitting": cross_fitting,
        "propensity_estimation_uncertainty": "not_propagated",
        "outcome_model_uncertainty": "not_propagated",
        "treatment_position": treatment_position,
        "treated_rows": treated_count,
        "control_rows": control_count,
        "propensity_model": {
            "method": "logistic_mle",
            "converged": True,
            "iterations": nuisance_diagnostics["propensity_iterations"],
            "log_likelihood": nuisance_diagnostics.get(
                "propensity_log_likelihood", "recorded_per_fold"
            ),
        },
        "outcome_models": {
            "method": "separate_ordinary_least_squares",
            "treated_rank": nuisance_diagnostics["treated_rank"]
            if fold_count == 1
            else "recorded_per_fold",
            "control_rank": nuisance_diagnostics["control_rank"]
            if fold_count == 1
            else "recorded_per_fold",
        },
        "positivity": {
            "status": "diagnosed_not_proven",
            "raw_min": float(numpy.min(raw_propensity)),
            "raw_max": float(numpy.max(raw_propensity)),
            "clipped_min": float(numpy.min(propensity)),
            "clipped_max": float(numpy.max(propensity)),
            "clip_threshold": policy.propensity_clip,
            "fraction_clipped": float(numpy.mean(clipped)),
        },
        "influence_function": {
            "method": "centered_augmented_ipw_score",
            "variance": variance,
            "variance_method": uncertainty["method"],
        },
    }
    if policy.covariance == "clustered":
        diagnostics.update(uncertainty["cluster_diagnostics"])
    return (
        numpy.asarray([ate]),
        numpy.asarray([[variance]]),
        diagnostics,
        {
            "nuisance_design": nuisance_design,
            "treatment": treatment,
            "outcome_control": outcome_control,
            "outcome_treated": outcome_treated,
            "fold_assignments": fold_assignments,
        },
    )


def fit_aipw_nuisance_models(
    training_design: numpy.ndarray,
    training_treatment: numpy.ndarray,
    training_values: numpy.ndarray,
    evaluation_design: numpy.ndarray,
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
]:
    """Fits AIPW nuisance models on training rows and predicts evaluation rows.

    Args:
        training_design: Design matrix of the training rows, with intercept.
        training_treatment: 0/1 treatment of the training rows.
        training_values: Outcomes of the training rows.
        evaluation_design: Design matrix of the rows to predict.
        policy: Supplies the propensity clip, iteration limit and tolerance.

    Returns:
        For the evaluation rows: the clipped propensity, the unclipped
        propensity, the predicted outcome under control, the predicted
        outcome under treatment; and a diagnostics dictionary with the
        propensity iterations and log-likelihood and the ranks of the two
        outcome designs.

    Raises:
        ValueError: If a treatment arm has fewer than two training rows or
            an outcome design is rank deficient.
    """
    treated = training_treatment == 1.0
    control = training_treatment == 0.0
    if (
        int(numpy.sum(treated)) < MIN_ROWS_PER_ARM
        or int(numpy.sum(control)) < MIN_ROWS_PER_ARM
    ):
        raise ValueError(
            "aipw_ate each training fold needs at least two rows per "
            "treatment arm"
        )
    (
        propensity_coefficients,
        propensity_probabilities,
        propensity_diagnostics,
    ) = fit_logistic(
        training_design,
        training_treatment,
        policy.max_iterations,
        policy.tolerance,
    )
    del propensity_probabilities
    evaluation_raw_propensity = special.expit(
        evaluation_design @ propensity_coefficients
    )
    propensity = numpy.clip(
        evaluation_raw_propensity,
        policy.propensity_clip,
        1.0 - policy.propensity_clip,
    )
    treated_design = training_design[treated]
    control_design = training_design[control]
    if (
        numpy.linalg.matrix_rank(treated_design) != treated_design.shape[1]
        or numpy.linalg.matrix_rank(control_design) != control_design.shape[1]
    ):
        raise ValueError("aipw_ate outcome design is not identifiable")
    treated_coefficients = numpy.linalg.lstsq(
        treated_design, training_values[treated], rcond=None
    )[0]
    control_coefficients = numpy.linalg.lstsq(
        control_design, training_values[control], rcond=None
    )[0]
    outcome_control = evaluation_design @ control_coefficients
    outcome_treated = evaluation_design @ treated_coefficients
    return (
        propensity,
        evaluation_raw_propensity,
        outcome_control,
        outcome_treated,
        {
            "propensity_iterations": propensity_diagnostics["iterations"],
            "propensity_log_likelihood": propensity_diagnostics[
                "log_likelihood"
            ],
            "treated_rank": int(numpy.linalg.matrix_rank(treated_design)),
            "control_rank": int(numpy.linalg.matrix_rank(control_design)),
        },
    )


def deterministic_fold_assignments(rows: int, fold_count: int) -> numpy.ndarray:
    """Assigns rows to balanced folds using a deterministic row-order rule.

    Args:
        rows: Number of rows.
        fold_count: Number of folds.

    Returns:
        An integer array of length `rows` giving each row's fold, the row
        index modulo `fold_count`.

    Raises:
        ValueError: If `fold_count` is below 2 or exceeds `rows`.
    """
    if fold_count < MIN_FOLDS:
        raise ValueError("cross-fitting needs at least two folds")
    if fold_count > rows:
        raise ValueError("cross_fitting_folds must not exceed rows")
    return numpy.arange(rows, dtype=int) % fold_count


def validate_cross_fitting_support(
    treatment: numpy.ndarray,
    fold_assignments: numpy.ndarray,
    fold_count: int,
) -> list[dict[str, int]]:
    """Validates treatment support in every training and validation fold.

    Args:
        treatment: 0/1 treatment of every row.
        fold_assignments: Fold index of every row.
        fold_count: Number of folds.

    Returns:
        One dictionary per fold with the fold index and the treated and
        control row counts of its training and validation parts.

    Raises:
        ValueError: If a validation fold lacks either arm or a training fold
            has fewer than two rows in either arm.
    """
    support: list[dict[str, int]] = []
    for fold in range(fold_count):
        validation = fold_assignments == fold
        training = ~validation
        validation_treated = int(numpy.sum(treatment[validation] == 1.0))
        validation_control = int(numpy.sum(treatment[validation] == 0.0))
        training_treated = int(numpy.sum(treatment[training] == 1.0))
        training_control = int(numpy.sum(treatment[training] == 0.0))
        if validation_treated < 1 or validation_control < 1:
            raise ValueError(
                "aipw_ate cross-fitting needs both treatment arms in every "
                "validation fold"
            )
        if (
            training_treated < MIN_ROWS_PER_ARM
            or training_control < MIN_ROWS_PER_ARM
        ):
            raise ValueError(
                "aipw_ate cross-fitting needs at least two rows per "
                "treatment arm in every training fold"
            )
        support.append(
            {
                "fold": fold,
                "training_treated_rows": training_treated,
                "training_control_rows": training_control,
                "validation_treated_rows": validation_treated,
                "validation_control_rows": validation_control,
            }
        )
    return support


def aipw_limitations(cross_fitted: bool) -> list[str]:
    """Returns limitations of the AIPW estimator.

    Args:
        cross_fitted: Whether the nuisance models were cross-fitted.

    Returns:
        Plain-language limitation statements; one extra statement is added
        when the nuisance models were fitted on the rows they score.
    """
    limitations = [
        "propensity and outcome nuisance uncertainty is not propagated",
        "positivity is diagnosed but not proven",
        "outcome nuisance models are separate ordinary least squares fits",
        (
            "this estimator is not Bayesian and does not support time-varying "
            "treatment effects"
        ),
    ]
    if not cross_fitted:
        limitations.append(
            "nuisance models are fitted on the same rows used for scoring"
        )
    return limitations


def center_influence(influence: numpy.ndarray) -> numpy.ndarray:
    """Validates observation-level influence scores and centers them.

    Args:
        influence: Observation-level influence scores.

    Returns:
        The scores minus their mean.

    Raises:
        ValueError: If the scores are not one-dimensional, are fewer than
            two, or are not finite.
    """
    centered_influence = numpy.asarray(influence, dtype=float)
    if centered_influence.ndim != 1:
        raise ValueError("influence scores must have shape (rows,)")
    if len(centered_influence) < MIN_INFLUENCE_ROWS:
        raise ValueError("influence uncertainty needs at least two rows")
    if not numpy.isfinite(centered_influence).all():
        raise ValueError("influence scores must contain only finite values")
    centered: numpy.ndarray = centered_influence - numpy.mean(
        centered_influence
    )
    return centered


def influence_variance(
    influence: numpy.ndarray,
    covariance: CovarianceKind,
    clusters: numpy.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    """Computes IID or finite-sample clustered influence uncertainty.

    The clustered calculation treats each declared cluster as an independent
    sampling unit. It centers the observation-level influence scores before
    summing them within clusters and applies the finite-sample correction
    ``G / (G - 1) * (n - 1) / n``. This is intentionally separate from the
    HC3 and regression sandwich paths: causal ATE uncertainty is defined over
    an influence function, not over a model residual score.

    Args:
        influence: Observation-level influence scores.
        covariance: ``iid`` or ``clustered`` uncertainty.
        clusters: Cluster label of each row; needed for ``clustered``.

    Returns:
        The variance of the estimate and a dictionary describing the method
        (with cluster diagnostics for clustered uncertainty).

    Raises:
        ValueError: If the scores are not one-dimensional, are fewer than
            two, or are not finite; if `covariance` is neither ``iid`` nor
            ``clustered``; if clustered uncertainty has no labels, labels
            that are unhashable or of the wrong length, or fewer than two
            clusters; or if the clustered variance is not finite and
            non-negative.
    """
    centered_influence = center_influence(influence)
    if covariance == "iid":
        variance = float(
            numpy.sum(centered_influence**2)
            / (len(centered_influence) - 1)
            / len(centered_influence)
        )
        return variance, {
            "method": "iid_centered_influence_function",
            "centered": True,
        }
    if covariance != "clustered":
        raise ValueError(
            "causal influence uncertainty supports only iid or clustered "
            "covariance"
        )
    if clusters is None:
        raise ValueError("clustered covariance needs cluster labels")
    cluster_values = validate_clusters(
        clusters.tolist(), len(centered_influence)
    )
    if cluster_values is None:
        raise ValueError("clustered covariance needs cluster labels")
    try:
        labels = tuple(dict.fromkeys(cluster_values.tolist()))
    except TypeError as error:
        raise ValueError(
            "cluster labels must be hashable scalar values"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise ValueError("clustered covariance needs at least two clusters")
    cluster_sums = []
    cluster_sizes = []
    for label in labels:
        membership = cluster_values == label
        cluster_sums.append(float(numpy.sum(centered_influence[membership])))
        cluster_sizes.append(int(numpy.sum(membership)))
    cluster_sum_array = numpy.asarray(cluster_sums, dtype=float)
    row_count = len(centered_influence)
    correction = (len(labels) / (len(labels) - 1)) * (
        (row_count - 1) / row_count
    )
    variance = float(
        correction * numpy.sum(cluster_sum_array**2) / row_count**2
    )
    if not math.isfinite(variance) or variance < 0.0:
        raise ValueError("clustered influence variance must be finite")
    return variance, {
        "method": "clustered_centered_influence_function",
        "centered": True,
        "cluster_diagnostics": {
            "cluster_count": len(labels),
            "cluster_sizes": cluster_sizes,
            "uncertainty_method": "clustered_centered_influence_function",
        },
    }


def hajek_ipw_statistics(
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    propensity: numpy.ndarray,
    clusters: numpy.ndarray | None = None,
    covariance: CovarianceKind = "iid",
) -> tuple[float, float, dict[str, Any]]:
    """Computes normalized IPW means and an influence-function variance.

    Args:
        treatment: 0/1 treatment of every row.
        values: Outcome of every row.
        propensity: Clipped propensity score of every row.
        clusters: Cluster label of every row, for clustered uncertainty.
        covariance: ``iid`` or ``clustered`` uncertainty.

    Returns:
        The difference of the normalized treated and control means (the
        ATE), its variance, and a dictionary with the effective sample size
        and mean of each arm and the uncertainty description.

    Raises:
        ValueError: If either arm's weights sum to a non-positive value, or
            if the variance calculation rejects its inputs.
    """
    treated_weights = treatment / propensity
    control_weights = (1.0 - treatment) / (1.0 - propensity)
    treated_denominator = float(numpy.sum(treated_weights))
    control_denominator = float(numpy.sum(control_weights))
    if treated_denominator <= 0.0 or control_denominator <= 0.0:
        raise ValueError("ipw_ate treatment weights must have positive support")
    treated_mean = float(
        numpy.sum(treated_weights * values) / treated_denominator
    )
    control_mean = float(
        numpy.sum(control_weights * values) / control_denominator
    )
    influence = treated_weights * (values - treated_mean) - control_weights * (
        values - control_mean
    )
    variance, uncertainty = influence_variance(influence, covariance, clusters)
    effective_sample_size = {
        "treated": float(
            treated_denominator**2 / numpy.sum(treated_weights**2)
        ),
        "control": float(
            control_denominator**2 / numpy.sum(control_weights**2)
        ),
    }
    return (
        treated_mean - control_mean,
        variance,
        {
            "effective_sample_size": effective_sample_size,
            "treatment_means": {
                "treated": treated_mean,
                "control": control_mean,
            },
            "uncertainty_method": uncertainty["method"],
            "uncertainty": uncertainty,
        },
    )


def fit_bayesian_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    prior: BayesianPrior,
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Computes a conjugate Gaussian posterior from an explicit prior.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        prior: Normal-inverse-gamma prior over the coefficients.

    Returns:
        The posterior mean of the coefficients, their posterior covariance
        (the covariance of the marginal Student-t), and a diagnostics
        dictionary that includes the posterior scale matrix and degrees of
        freedom.

    Raises:
        ValueError: If the prior dimension does not match the design, the
            posterior precision is not positive definite, the posterior
            variance scale is invalid, or the posterior has two or fewer
            degrees of freedom.
    """
    validate_bayesian_prior_shape(prior, design.shape[1])
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    try:
        factor = linalg.cho_factor(posterior_precision, check_finite=True)
        posterior_mean = linalg.cho_solve(
            factor, prior_precision @ prior_mean + design.T @ values
        )
        posterior_precision_inverse = linalg.cho_solve(
            factor, numpy.eye(design.shape[1])
        )
    except numpy.linalg.LinAlgError as error:
        raise ValueError(
            "Bayesian posterior precision is not identifiable"
        ) from error
    posterior_shape = prior.shape + len(values) / 2.0
    posterior_scale = prior.scale + 0.5 * (
        values @ values
        + prior_mean @ prior_precision @ prior_mean
        - posterior_mean @ posterior_precision @ posterior_mean
    )
    if not math.isfinite(posterior_scale) or posterior_scale <= 0.0:
        raise ValueError("Bayesian posterior variance scale is invalid")
    posterior_degrees = 2.0 * posterior_shape
    if posterior_degrees <= MIN_VARIANCE_DEGREES:
        raise ValueError(
            "Bayesian posterior needs more than two degrees of freedom"
        )
    posterior_covariance = (
        posterior_scale
        / posterior_shape
        * (posterior_degrees / (posterior_degrees - 2.0))
        * posterior_precision_inverse
    )
    posterior_scale_matrix = (
        posterior_scale / posterior_shape
    ) * posterior_precision_inverse
    return (
        posterior_mean,
        posterior_covariance,
        {
            "converged": True,
            "iterations": 1,
            "estimator_family": "bayesian_conjugate",
            "prior_shape": prior.shape,
            "prior_scale": prior.scale,
            "posterior_shape": posterior_shape,
            "posterior_scale": posterior_scale_matrix.tolist(),
            "posterior_degrees_of_freedom": posterior_degrees,
        },
    )


def validate_bayesian_prior_shape(prior: BayesianPrior, dimension: int) -> None:
    """Validates prior vector and matrix dimensions against a design.

    Args:
        prior: Prior to check.
        dimension: Number of design columns, including the intercept.

    Raises:
        ValueError: If the prior mean is not of length `dimension` or the
            precision is not `dimension` by `dimension`.
    """
    mean_shape = numpy.asarray(prior.mean, dtype=float).shape
    precision_shape = numpy.asarray(prior.precision, dtype=float).shape
    expected_shape = (dimension, dimension)
    if mean_shape != (dimension,) or precision_shape != expected_shape:
        raise ValueError(
            "Bayesian prior dimension must match intercept and features"
        )


def fit_hierarchical_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits a bounded two-stage hierarchical linear model.

    The default estimates variance components by moments from an initial
    fixed-effect fit.  The bounded REML policy uses a deterministic profile
    likelihood over the residual and random-intercept variances, followed by
    generalized least squares for the fixed effects.  When a random slope is
    declared, the same methods use one random-intercept/random-slope block.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        policy: Supplies the variance-component method, the random-slope
            position, the REML bounds, iteration limit and tolerance.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary, the residual variance, the random-intercept variance,
        and the 2x2 random-effect covariance matrix (intercept and slope; the
        slope entries are zero for a random intercept only).

    Raises:
        ValueError: If the labels are unhashable, there are fewer than two
            clusters, the design is rank deficient, the within-cluster
            variance is not positive, or the GLS system is not identifiable.
            Errors from the random-slope and REML fits also propagate.
    """
    if policy.random_slope_position is not None:
        return fit_hierarchical_random_slope(design, values, clusters, policy)
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise ValueError("hierarchical_linear needs at least two clusters")
    if numpy.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("hierarchical design matrix is rank deficient")
    initial_coefficients = numpy.linalg.lstsq(design, values, rcond=None)[0]
    initial_residuals = values - design @ initial_coefficients
    cluster_means = []
    cluster_sizes = []
    within_sum_of_squares = 0.0
    for label in labels:
        group = clusters == label
        group_residuals = initial_residuals[group]
        cluster_means.append(float(numpy.mean(group_residuals)))
        cluster_sizes.append(int(numpy.sum(group)))
        centered = group_residuals - numpy.mean(group_residuals)
        within_sum_of_squares += float(numpy.sum(centered**2))
    residual_variance = within_sum_of_squares / max(
        len(values) - len(labels), 1
    )
    if not math.isfinite(residual_variance) or residual_variance <= SCALE_FLOOR:
        raise ValueError(
            "hierarchical_linear needs positive within-cluster variance"
        )
    between_variance = float(numpy.var(cluster_means, ddof=1))
    finite_sample_adjustment = residual_variance * float(
        numpy.mean(1.0 / numpy.asarray(cluster_sizes, dtype=float))
    )
    random_intercept_variance = max(
        0.0, between_variance - finite_sample_adjustment
    )
    if policy.hierarchical_method == "reml":
        return fit_hierarchical_reml(
            design,
            values,
            clusters,
            labels,
            residual_variance,
            random_intercept_variance,
            policy,
        )
    information, right_hand_side = hierarchical_gls_system(
        design,
        values,
        clusters,
        labels,
        residual_variance,
        random_intercept_variance,
    )
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
        covariance = numpy.linalg.solve(information, numpy.eye(design.shape[1]))
    except numpy.linalg.LinAlgError as error:
        raise ValueError(
            "hierarchical GLS system is not identifiable"
        ) from error
    random_effects = []
    fitted_residuals = values - design @ coefficients
    for label in labels:
        group = clusters == label
        count = int(numpy.sum(group))
        group_sum = float(numpy.sum(fitted_residuals[group]))
        shrinkage = random_intercept_variance / (
            residual_variance + count * random_intercept_variance
        )
        random_effects.append(shrinkage * group_sum)
    total_variance = residual_variance + random_intercept_variance
    diagnostics = {
        "converged": True,
        "iterations": 2,
        "estimator_family": "hierarchical_random_intercept",
        "method": "two_stage_moment_gls",
        "cluster_count": len(labels),
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "residual_variance": residual_variance,
        "random_intercept_variance": random_intercept_variance,
        "intraclass_correlation": random_intercept_variance / total_variance,
        "random_intercept_status": (
            "estimated" if random_intercept_variance > 0.0 else "boundary_zero"
        ),
        "random_intercept_effect": {
            "mean_absolute_blup": float(numpy.mean(numpy.abs(random_effects))),
            "max_absolute_blup": float(numpy.max(numpy.abs(random_effects))),
        },
        "variance_component_method": "moment_estimate_from_initial_ols",
        "uncertainty_method": ("conditional_on_moment_variance_components"),
        "variance_component_uncertainty": "not_propagated",
    }
    return (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        random_intercept_variance,
        numpy.array(
            [[random_intercept_variance, 0.0], [0.0, 0.0]], dtype=float
        ),
    )


def fit_hierarchical_random_slope(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits one random slope with a random intercept by moments or REML.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        policy: Supplies the random-slope position and the variance-component
            method; the REML settings are used when the method is ``reml``.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary, the residual variance, the random-intercept variance,
        and the 2x2 random-effect covariance matrix.

    Raises:
        ValueError: If no random slope position is declared, the labels are
            unhashable, there are fewer than two clusters, the design is rank
            deficient, a cluster lacks variation in the slope feature, the
            within-cluster variance is not positive, or the GLS system is not
            identifiable.
    """
    random_slope_position = policy.random_slope_position
    if random_slope_position is None:
        raise ValueError("random slope position must be declared")
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise ValueError("hierarchical_linear needs at least two clusters")
    if numpy.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("hierarchical design matrix is rank deficient")
    initial_coefficients = numpy.linalg.lstsq(design, values, rcond=None)[0]
    initial_residuals = values - design @ initial_coefficients
    random_effect_estimates = []
    random_effect_inverse_matrices = []
    cluster_sizes = []
    within_sum_of_squares = 0.0
    for label in labels:
        group = clusters == label
        group_design = design[group]
        random_effect_design = numpy.column_stack(
            (
                numpy.ones(int(numpy.sum(group))),
                group_design[:, random_slope_position + 1],
            )
        )
        if (
            numpy.linalg.matrix_rank(random_effect_design)
            != RANDOM_EFFECT_COLUMNS
        ):
            raise ValueError(
                "each hierarchical cluster needs within-cluster variation "
                "for the declared random slope"
            )
        inverse_matrix = numpy.linalg.solve(
            random_effect_design.T @ random_effect_design,
            numpy.eye(2),
        )
        group_residuals = initial_residuals[group]
        random_effect_estimates.append(
            inverse_matrix @ random_effect_design.T @ group_residuals
        )
        random_effect_inverse_matrices.append(inverse_matrix)
        cluster_sizes.append(int(numpy.sum(group)))
        centered = group_residuals - numpy.mean(group_residuals)
        within_sum_of_squares += float(numpy.sum(centered**2))
    residual_variance = within_sum_of_squares / max(
        len(values) - len(labels), 1
    )
    if not math.isfinite(residual_variance) or residual_variance <= SCALE_FLOOR:
        raise ValueError(
            "hierarchical_linear needs positive within-cluster variance"
        )
    effects = numpy.asarray(random_effect_estimates, dtype=float)
    centered_effects = effects - effects.mean(axis=0)
    effect_covariance = (
        centered_effects.T @ centered_effects / (len(labels) - 1)
    )
    effect_covariance -= residual_variance * numpy.mean(
        numpy.asarray(random_effect_inverse_matrices), axis=0
    )
    random_effect_covariance = project_random_effect_covariance(
        effect_covariance
    )
    if policy.hierarchical_method == "reml":
        return fit_hierarchical_random_slope_reml(
            design,
            values,
            clusters,
            labels,
            random_slope_position,
            residual_variance,
            random_effect_covariance,
            cluster_sizes,
            policy,
        )
    information, right_hand_side = hierarchical_random_slope_gls_system(
        design,
        values,
        clusters,
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
    )
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
        covariance = numpy.linalg.solve(information, numpy.eye(design.shape[1]))
    except numpy.linalg.LinAlgError as error:
        raise ValueError(
            "hierarchical GLS system is not identifiable"
        ) from error
    diagnostics = hierarchical_random_slope_diagnostics(
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
        cluster_sizes,
        method="two_stage_moment_gls",
    )
    return (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        float(random_effect_covariance[0, 0]),
        random_effect_covariance,
    )


def project_random_effect_covariance(
    covariance: numpy.ndarray,
) -> numpy.ndarray:
    """Projects a moment covariance to the positive-semidefinite cone.

    Args:
        covariance: Square moment estimate of the random-effect covariance.

    Returns:
        The symmetrized matrix with negative eigenvalues replaced by zero.
    """
    symmetric = (covariance + covariance.T) / 2.0
    eigenvalues, eigenvectors = numpy.linalg.eigh(symmetric)
    projected = eigenvectors @ numpy.diag(numpy.maximum(eigenvalues, 0.0))
    projected = projected @ eigenvectors.T
    symmetrized: numpy.ndarray = (projected + projected.T) / 2.0
    return symmetrized


def hierarchical_random_slope_gls_system(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Builds the GLS system for one random slope and intercept.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: Residual variance.
        random_effect_covariance: 2x2 covariance of the random intercept and
            slope.

    Returns:
        The GLS information matrix and right-hand-side vector summed over
        clusters.  A cluster whose marginal covariance is not positive
        definite raises the ValueError of
        `hierarchical_random_effect_precision`.
    """
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        group_design = design[group]
        random_effect_design = numpy.column_stack(
            (
                numpy.ones(group_design.shape[0]),
                group_design[:, random_slope_position + 1],
            )
        )
        precision = hierarchical_random_effect_precision(
            random_effect_design,
            residual_variance,
            random_effect_covariance,
        )
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ values[group]
    return information, right_hand_side


def hierarchical_random_effect_precision(
    random_effect_design: numpy.ndarray,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
) -> numpy.ndarray:
    """Returns dense precision for one random-effect covariance block.

    Args:
        random_effect_design: Intercept and slope columns for one cluster.
        residual_variance: Residual variance.
        random_effect_covariance: 2x2 covariance of the random effects.

    Returns:
        The inverse of the cluster's marginal covariance matrix.

    Raises:
        ValueError: If the marginal covariance is not positive definite.
    """
    count = random_effect_design.shape[0]
    covariance = (
        residual_variance * numpy.eye(count)
        + random_effect_design
        @ random_effect_covariance
        @ random_effect_design.T
    )
    try:
        precision: numpy.ndarray = linalg.solve(
            covariance, numpy.eye(count), assume_a="pos"
        )
    except numpy.linalg.LinAlgError as error:
        raise ValueError(
            "hierarchical random-effect covariance is not positive definite"
        ) from error
    return precision


def hierarchical_random_slope_diagnostics(
    labels: tuple[Any, ...],
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
    cluster_sizes: list[int],
    method: str,
) -> dict[str, Any]:
    """Builds auditable diagnostics for the one-random-slope model.

    Args:
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: Residual variance.
        random_effect_covariance: 2x2 covariance of the random intercept and
            slope.
        cluster_sizes: Number of rows in each cluster.
        method: ``two_stage_moment_gls`` or the REML method name; it selects
            the recorded variance-component method.

    Returns:
        A dictionary of variance components, correlation, cluster-size
        range, status flags and stated limitations.
    """
    random_intercept_variance = float(random_effect_covariance[0, 0])
    random_slope_variance = float(random_effect_covariance[1, 1])
    covariance = float(random_effect_covariance[0, 1])
    denominator = math.sqrt(
        max(random_intercept_variance * random_slope_variance, 0.0)
    )
    correlation = covariance / denominator if denominator > 0.0 else 0.0
    total_variance = residual_variance + random_intercept_variance
    return {
        "converged": True,
        "iterations": 2,
        "estimator_family": "hierarchical_random_intercept_slope",
        "method": method,
        "cluster_count": len(labels),
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "random_slope_position": random_slope_position,
        "residual_variance": residual_variance,
        "random_intercept_variance": random_intercept_variance,
        "random_slope_variance": random_slope_variance,
        "random_intercept_slope_covariance": covariance,
        "random_effect_correlation": correlation,
        "random_effect_covariance": random_effect_covariance.tolist(),
        "intraclass_correlation": random_intercept_variance / total_variance,
        "random_intercept_status": (
            "estimated" if random_intercept_variance > 0.0 else "boundary_zero"
        ),
        "random_slope_status": (
            "estimated" if random_slope_variance > 0.0 else "boundary_zero"
        ),
        "variance_component_method": (
            "moment_estimate_from_initial_ols_random_effects"
            if method == "two_stage_moment_gls"
            else "profile_reml_with_l_bfgs_b"
        ),
        "uncertainty_method": "conditional_on_variance_components",
        "variance_component_uncertainty": "not_propagated",
        "limitations": [
            "only one random slope is supported",
            "nested effects are unsupported",
            "crossed effects are unsupported",
            "fully Bayesian multilevel inference is unsupported",
            "variance-component uncertainty is not propagated",
        ],
        "observed_random_effect_structure": "intercept_and_declared_slope",
        "random_effect_covariance_evidence": {
            "positive_semidefinite_projection": True,
            "covariance_is_conditional": True,
        },
    }


def fit_hierarchical_random_slope_reml(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    random_slope_position: int,
    initial_residual_variance: float,
    initial_random_effect_covariance: numpy.ndarray,
    cluster_sizes: list[int],
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits the one-random-slope model with bounded profile REML.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.
        initial_residual_variance: Moment estimate of the residual variance,
            the starting point of the search.
        initial_random_effect_covariance: Moment estimate of the 2x2
            random-effect covariance, the starting point of the search.
        cluster_sizes: Number of rows in each cluster.
        policy: Supplies the variance bounds, iteration limit and tolerance.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary, the residual variance, the random-intercept variance,
        and the 2x2 random-effect covariance matrix at the REML optimum.

    Raises:
        ValueError: If the optimizer result is rejected (see
            `validate_reml_optimizer_result`), the GLS system is not
            identifiable, or the objective rejects a covariance block that
            is not positive definite.
    """
    lower_bound, upper_bound = policy.reml_variance_bounds
    intercept_variance = max(float(initial_random_effect_covariance[0, 0]), 0.0)
    slope_variance = max(float(initial_random_effect_covariance[1, 1]), 0.0)
    if intercept_variance <= 0.0:
        intercept_variance = initial_residual_variance
    if slope_variance <= 0.0:
        slope_variance = initial_residual_variance
    initial_variances = numpy.clip(
        numpy.asarray(
            [
                initial_residual_variance,
                intercept_variance,
                slope_variance,
            ],
            dtype=float,
        ),
        lower_bound,
        upper_bound,
    )
    scale = math.sqrt(intercept_variance * slope_variance)
    initial_correlation = (
        float(initial_random_effect_covariance[0, 1]) / scale
        if scale > 0.0
        else 0.0
    )
    initial_correlation = float(numpy.clip(initial_correlation, -0.999, 0.999))
    initial_parameters = numpy.concatenate(
        (numpy.log(initial_variances), [numpy.arctanh(initial_correlation)])
    )
    log_bounds = (
        (math.log(lower_bound), math.log(upper_bound)),
        (math.log(lower_bound), math.log(upper_bound)),
        (math.log(lower_bound), math.log(upper_bound)),
        (-8.0, 8.0),
    )
    optimizer_result = optimize.minimize(
        random_slope_reml_profile_objective,
        initial_parameters,
        args=(design, values, clusters, labels, random_slope_position),
        method="L-BFGS-B",
        bounds=log_bounds,
        options={
            "maxiter": policy.max_iterations,
            "ftol": policy.tolerance,
            "gtol": policy.tolerance,
        },
    )
    validate_reml_optimizer_result(optimizer_result, log_bounds, dimension=4)
    point = numpy.asarray(optimizer_result.x, dtype=float)
    variance_components = numpy.exp(point[:3])
    residual_variance = float(variance_components[0])
    random_effect_covariance = random_slope_covariance_from_reml_point(
        variance_components, float(point[3])
    )
    information, right_hand_side = hierarchical_random_slope_gls_system(
        design,
        values,
        clusters,
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
    )
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
        covariance = numpy.linalg.solve(information, numpy.eye(design.shape[1]))
    except numpy.linalg.LinAlgError as error:
        raise ValueError(
            "hierarchical REML GLS system is not identifiable"
        ) from error
    diagnostics = hierarchical_random_slope_diagnostics(
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
        cluster_sizes,
        method="bounded_reml_gls",
    )
    jacobian = numpy.asarray(optimizer_result.jac, dtype=float)
    diagnostics.update(
        {
            "reml_objective": float(optimizer_result.fun),
            "optimizer": {
                "method": "L-BFGS-B",
                "success": bool(optimizer_result.success),
                "status": int(optimizer_result.status),
                "message": str(optimizer_result.message),
                "function_evaluations": int(optimizer_result.nfev),
                "gradient_norm": float(numpy.linalg.norm(jacobian)),
                "variance_bounds": [lower_bound, upper_bound],
                "correlation_bounds": [-0.999999, 0.999999],
            },
        }
    )
    return (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        float(random_effect_covariance[0, 0]),
        random_effect_covariance,
    )


def random_slope_covariance_from_reml_point(
    variance_components: numpy.ndarray,
    correlation_parameter: float,
) -> numpy.ndarray:
    """Constructs a random-effect covariance from a REML parameter point.

    Args:
        variance_components: Array of the residual, random-intercept and
            random-slope variances, in that order.
        correlation_parameter: Unbounded parameter whose scaled hyperbolic
            tangent gives the intercept-slope correlation (at most 0.999999
            in absolute value).

    Returns:
        The 2x2 covariance matrix of the random intercept and slope.
    """
    correlation = 0.999999 * math.tanh(correlation_parameter)
    intercept_variance, slope_variance = variance_components[1:]
    covariance = correlation * math.sqrt(intercept_variance * slope_variance)
    return numpy.asarray(
        [
            [intercept_variance, covariance],
            [covariance, slope_variance],
        ],
        dtype=float,
    )


def random_slope_reml_profile_objective(
    parameters: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    random_slope_position: int,
) -> float:
    """Returns the negative profile REML objective for one random slope.

    Args:
        parameters: Four values: the logs of the residual, random-intercept
            and random-slope variances, then the correlation parameter.
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        random_slope_position: Feature index carrying the random slope.

    Returns:
        The objective value to minimize, or infinity when the parameters
        give a variance that is not positive and finite, the information
        matrix or a cluster covariance has a non-positive determinant, or the
        GLS system is singular.

    Raises:
        ValueError: If `parameters` does not have exactly four values, or if
            the GLS system building rejects a covariance block (see
            `hierarchical_random_effect_precision`).
    """
    point = numpy.asarray(parameters, dtype=float)
    if point.shape != (4,):
        raise ValueError(
            "random-slope REML needs four bounded covariance parameters"
        )
    variance_components = numpy.exp(point[:3])
    if not numpy.isfinite(variance_components).all() or numpy.any(
        variance_components <= 0.0
    ):
        return float("inf")
    random_effect_covariance = random_slope_covariance_from_reml_point(
        variance_components, float(point[3])
    )
    residual_variance = float(variance_components[0])
    information, right_hand_side = hierarchical_random_slope_gls_system(
        design,
        values,
        clusters,
        labels,
        random_slope_position,
        residual_variance,
        random_effect_covariance,
    )
    sign, logdet_information = numpy.linalg.slogdet(information)
    if sign <= 0.0 or not math.isfinite(float(logdet_information)):
        return float("inf")
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
    except numpy.linalg.LinAlgError:
        return float("inf")
    residuals = values - design @ coefficients
    quadratic = 0.0
    logdet_covariance = 0.0
    for label in labels:
        group = clusters == label
        group_design = design[group]
        random_effect_design = numpy.column_stack(
            (
                numpy.ones(group_design.shape[0]),
                group_design[:, random_slope_position + 1],
            )
        )
        covariance = (
            residual_variance * numpy.eye(int(numpy.sum(group)))
            + random_effect_design
            @ random_effect_covariance
            @ random_effect_design.T
        )
        sign_covariance, logdet = numpy.linalg.slogdet(covariance)
        if sign_covariance <= 0.0:
            return float("inf")
        precision = linalg.solve(covariance, numpy.eye(covariance.shape[0]))
        group_residuals = residuals[group]
        quadratic += float(group_residuals @ precision @ group_residuals)
        logdet_covariance += float(logdet)
    observations = len(values)
    parameters_count = design.shape[1]
    return float(
        0.5
        * (
            logdet_covariance
            + logdet_information
            + quadratic
            + (observations - parameters_count) * math.log(2.0 * math.pi)
        )
    )


def fit_hierarchical_reml(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    initial_residual_variance: float,
    initial_random_intercept_variance: float,
    policy: Config,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    dict[str, Any],
    float,
    float,
    numpy.ndarray,
]:
    """Fits a bounded random-intercept model with profile REML.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        initial_residual_variance: Moment estimate of the residual variance,
            the starting point of the search.
        initial_random_intercept_variance: Moment estimate of the
            random-intercept variance, the starting point of the search.
        policy: Supplies the variance bounds, iteration limit and tolerance.

    Returns:
        The fixed-effect coefficients, their covariance, a diagnostics
        dictionary including the optimizer record, the residual variance, the
        random-intercept variance, and the 2x2 random-effect covariance
        (random intercept only; other entries zero).

    Raises:
        ValueError: If the optimizer result is rejected (see
            `validate_reml_optimizer_result`) or the GLS system is not
            identifiable.
    """
    lower_bound, upper_bound = policy.reml_variance_bounds
    initial_variances = numpy.clip(
        numpy.asarray(
            [initial_residual_variance, initial_random_intercept_variance],
            dtype=float,
        ),
        lower_bound,
        upper_bound,
    )
    log_bounds = (
        (math.log(lower_bound), math.log(upper_bound)),
        (math.log(lower_bound), math.log(upper_bound)),
    )
    optimizer_result = optimize.minimize(
        reml_profile_objective,
        numpy.log(initial_variances),
        args=(design, values, clusters, labels),
        method="L-BFGS-B",
        bounds=log_bounds,
        options={
            "maxiter": policy.max_iterations,
            "ftol": policy.tolerance,
            "gtol": policy.tolerance,
        },
    )
    validate_reml_optimizer_result(optimizer_result, log_bounds)
    variance_components = numpy.exp(
        numpy.asarray(optimizer_result.x, dtype=float)
    )
    residual_variance = float(variance_components[0])
    random_intercept_variance = float(variance_components[1])
    information, right_hand_side = hierarchical_gls_system(
        design,
        values,
        clusters,
        labels,
        residual_variance,
        random_intercept_variance,
    )
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
        covariance = numpy.linalg.solve(information, numpy.eye(design.shape[1]))
    except numpy.linalg.LinAlgError as error:
        raise ValueError(
            "hierarchical REML GLS system is not identifiable"
        ) from error
    fitted_residuals = values - design @ coefficients
    random_effects = []
    cluster_sizes = []
    for label in labels:
        group = clusters == label
        count = int(numpy.sum(group))
        cluster_sizes.append(count)
        group_sum = float(numpy.sum(fitted_residuals[group]))
        shrinkage = random_intercept_variance / (
            residual_variance + count * random_intercept_variance
        )
        random_effects.append(shrinkage * group_sum)
    total_variance = residual_variance + random_intercept_variance
    jacobian = numpy.asarray(optimizer_result.jac, dtype=float)
    diagnostics = {
        "converged": True,
        "iterations": int(optimizer_result.nit),
        "estimator_family": "hierarchical_random_intercept",
        "method": "bounded_reml_gls",
        "cluster_count": len(labels),
        "cluster_size_min": min(cluster_sizes),
        "cluster_size_max": max(cluster_sizes),
        "residual_variance": residual_variance,
        "random_intercept_variance": random_intercept_variance,
        "intraclass_correlation": random_intercept_variance / total_variance,
        "random_intercept_status": (
            "estimated" if random_intercept_variance > 0.0 else "boundary_zero"
        ),
        "random_intercept_effect": {
            "mean_absolute_blup": float(numpy.mean(numpy.abs(random_effects))),
            "max_absolute_blup": float(numpy.max(numpy.abs(random_effects))),
        },
        "variance_component_method": "profile_reml_with_l_bfgs_b",
        "variance_component_uncertainty": "not_propagated",
        "limitations": [
            "variance-component uncertainty is not propagated",
            "random slopes are unsupported",
            "nested effects are unsupported",
            "crossed effects are unsupported",
        ],
        "reml_objective": float(optimizer_result.fun),
        "optimizer": {
            "method": "L-BFGS-B",
            "success": bool(optimizer_result.success),
            "status": int(optimizer_result.status),
            "message": str(optimizer_result.message),
            "function_evaluations": int(optimizer_result.nfev),
            "gradient_norm": float(numpy.linalg.norm(jacobian)),
            "variance_bounds": [lower_bound, upper_bound],
        },
    }
    return (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        random_intercept_variance,
        numpy.array(
            [[random_intercept_variance, 0.0], [0.0, 0.0]], dtype=float
        ),
    )


def reml_profile_objective(
    log_variances: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
) -> float:
    """Returns the negative profile REML objective in log-variance space.

    Args:
        log_variances: The logs of the residual and random-intercept
            variances.
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.

    Returns:
        The objective value to minimize, or infinity when a variance is not
        positive and finite, the information matrix has a non-positive
        determinant, or the GLS system is singular.

    Raises:
        ValueError: If `log_variances` does not have exactly two values.
    """
    if numpy.asarray(log_variances).shape != (2,):
        raise ValueError("REML needs residual and random-intercept variances")
    variances = numpy.exp(numpy.asarray(log_variances, dtype=float))
    if not numpy.isfinite(variances).all() or numpy.any(variances <= 0.0):
        return float("inf")
    residual_variance, random_intercept_variance = variances
    information, right_hand_side = hierarchical_gls_system(
        design,
        values,
        clusters,
        labels,
        float(residual_variance),
        float(random_intercept_variance),
    )
    sign, logdet_information = numpy.linalg.slogdet(information)
    if sign <= 0.0 or not math.isfinite(float(logdet_information)):
        return float("inf")
    try:
        coefficients = numpy.linalg.solve(information, right_hand_side)
    except numpy.linalg.LinAlgError:
        return float("inf")
    residuals = values - design @ coefficients
    quadratic = 0.0
    logdet_covariance = 0.0
    for label in labels:
        group = clusters == label
        count = int(numpy.sum(group))
        group_residuals = residuals[group]
        group_sum = float(numpy.sum(group_residuals))
        quadratic += float(
            numpy.sum(group_residuals**2) / residual_variance
            - random_intercept_variance
            * group_sum**2
            / (
                residual_variance
                * (residual_variance + count * random_intercept_variance)
            )
        )
        logdet_covariance += (count - 1) * math.log(
            residual_variance
        ) + math.log(residual_variance + count * random_intercept_variance)
    observations = len(values)
    parameters = design.shape[1]
    return float(
        0.5
        * (
            logdet_covariance
            + logdet_information
            + quadratic
            + (observations - parameters) * math.log(2.0 * math.pi)
        )
    )


def validate_reml_optimizer_result(
    optimizer_result: optimize.OptimizeResult,
    log_bounds: tuple[tuple[float, float], ...],
    dimension: int = 2,
) -> None:
    """Rejects an optimizer result that cannot support a release artifact.

    Args:
        optimizer_result: Result returned by the optimizer.
        log_bounds: Lower and upper bound of each optimized parameter.
        dimension: Expected number of optimized parameters.

    Raises:
        ValueError: If the optimizer did not report success; the status,
            iteration count or evaluation count is invalid; the solution is
            not finite, has the wrong length or lies outside the bounds; the
            objective is not finite; or the gradient is not finite or has
            the wrong length.
    """
    if not bool(getattr(optimizer_result, "success", False)):
        raise ValueError(
            "REML optimizer did not converge: "
            f"{getattr(optimizer_result, 'message', 'unknown failure')}"
        )
    status = getattr(optimizer_result, "status", None)
    if isinstance(status, bool) or not isinstance(status, (int, numpy.integer)):
        raise ValueError("REML optimizer returned an invalid status")
    iterations = getattr(optimizer_result, "nit", None)
    evaluations = getattr(optimizer_result, "nfev", None)
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, numpy.integer))
        or iterations < 0
        or isinstance(evaluations, bool)
        or not isinstance(evaluations, (int, numpy.integer))
        or evaluations < 0
    ):
        raise ValueError(
            "REML optimizer returned invalid iteration diagnostics"
        )
    point = numpy.asarray(getattr(optimizer_result, "x", ()), dtype=float)
    if point.shape != (dimension,) or not numpy.isfinite(point).all():
        raise ValueError("REML optimizer returned invalid variance components")
    for value, bounds in zip(point, log_bounds, strict=True):
        if value < bounds[0] or value > bounds[1]:
            raise ValueError("REML optimizer returned out-of-bounds components")
    objective = float(getattr(optimizer_result, "fun", float("nan")))
    if not math.isfinite(objective):
        raise ValueError("REML optimizer returned an invalid objective")
    gradient = numpy.asarray(getattr(optimizer_result, "jac", ()), dtype=float)
    if gradient.shape != (dimension,) or not numpy.isfinite(gradient).all():
        raise ValueError("REML optimizer returned invalid gradient diagnostics")


def hierarchical_gls_system(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    labels: tuple[Any, ...],
    residual_variance: float,
    random_intercept_variance: float,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Builds the block-diagonal precision system for random intercepts.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        labels: Distinct cluster labels.
        residual_variance: Residual variance.
        random_intercept_variance: Random-intercept variance; zero gives
            independent rows.

    Returns:
        The GLS information matrix and right-hand-side vector summed over
        clusters.
    """
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        group_design = design[group]
        group_values = values[group]
        count = group_design.shape[0]
        precision = numpy.eye(count) / residual_variance
        if random_intercept_variance > 0.0:
            precision -= (
                random_intercept_variance
                / (
                    residual_variance
                    * (residual_variance + count * random_intercept_variance)
                )
            ) * numpy.ones((count, count))
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ group_values
    return information, right_hand_side


def fit_logistic(
    design: numpy.ndarray,
    values: numpy.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Fits a Bernoulli likelihood with deterministic Newton iterations.

    Args:
        design: Design matrix including the intercept column.
        values: Binary outcomes (0 or 1).
        max_iterations: Maximum number of Newton steps.
        tolerance: Relative step-size threshold for convergence.

    Returns:
        The maximum-likelihood coefficients, the fitted probabilities, and a
        diagnostics dictionary with ``converged``, the iteration count and
        the log-likelihood.

    Raises:
        ValueError: If the Hessian is singular or the iteration does not
            converge within `max_iterations`.
    """
    coefficients = numpy.zeros(design.shape[1], dtype=float)
    for iteration in range(1, max_iterations + 1):
        probabilities = special.expit(design @ coefficients)
        weights = numpy.maximum(probabilities * (1.0 - probabilities), 1e-12)
        hessian = design.T @ (weights[:, None] * design)
        try:
            step = numpy.linalg.solve(
                hessian, design.T @ (values - probabilities)
            )
        except numpy.linalg.LinAlgError as error:
            raise ValueError(
                "logistic design matrix is not identifiable"
            ) from error
        coefficients += step
        if numpy.linalg.norm(step) <= tolerance * (
            1.0 + numpy.linalg.norm(coefficients)
        ):
            probabilities = special.expit(design @ coefficients)
            log_likelihood = bernoulli_log_likelihood(values, probabilities)
            return (
                coefficients,
                probabilities,
                {
                    "converged": True,
                    "iterations": iteration,
                    "log_likelihood": log_likelihood,
                },
            )
    raise ValueError("logistic estimation did not converge")


def covariance_matrix(
    design: numpy.ndarray,
    residuals: numpy.ndarray,
    covariance: CovarianceKind,
    clusters: numpy.ndarray | None,
    probabilities: numpy.ndarray | None,
) -> numpy.ndarray:
    """Computes iid, HC3, or cluster-robust sandwich covariance.

    A singular design makes the matrix inversion raise numpy's
    `LinAlgError`.

    Args:
        design: Design matrix including the intercept column.
        residuals: Residuals of the fit (outcome minus fitted probability for
            logistic fits).
        covariance: ``iid``, ``hc3`` or ``clustered``.
        clusters: Cluster label of every row; required for ``clustered``.
        probabilities: Fitted probabilities for a logistic fit, or None for
            a linear fit.

    Returns:
        The coefficient covariance matrix.

    Raises:
        ValueError: If clustered covariance is requested without labels or
            with fewer than two clusters.
    """
    if probabilities is None:
        bread = numpy.linalg.inv(design.T @ design)
        scale = float(
            numpy.sum(residuals**2) / max(len(residuals) - design.shape[1], 1)
        )
        if covariance == "iid":
            return scale * bread
        leverage = numpy.sum((design @ bread) * design, axis=1)
        adjusted = residuals / numpy.maximum(1.0 - leverage, 1e-12)
    else:
        weights = numpy.maximum(probabilities * (1.0 - probabilities), 1e-12)
        bread = numpy.linalg.inv(design.T @ (weights[:, None] * design))
        if covariance == "iid":
            return bread
        leverage = numpy.sum(
            (design @ bread) * (weights[:, None] * design), axis=1
        )
        adjusted = residuals / numpy.maximum(1.0 - leverage, 1e-12)
    if covariance == "hc3":
        scores = design * adjusted[:, None]
        return numpy.asarray(bread @ (scores.T @ scores) @ bread)
    if clusters is None:
        raise ValueError("cluster labels are required")
    adjusted = residuals
    groups = []
    for label in dict.fromkeys(clusters.tolist()):
        group = clusters == label
        groups.append(design[group].T @ adjusted[group])
    if len(groups) < MIN_CLUSTERS:
        raise ValueError("clustered covariance needs at least two clusters")
    meat = sum(numpy.outer(score, score) for score in groups)
    correction = (len(groups) / (len(groups) - 1)) * (
        (len(design) - 1) / max(len(design) - design.shape[1], 1)
    )
    sandwich: numpy.ndarray = correction * bread @ meat @ bread
    return sandwich


def reference_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Independently verifies the linear solution with least squares.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        coefficients: Coefficients from the primary fit.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute coefficient
        difference, and ``passed``, True when it is within the tolerance.
    """
    reference = numpy.linalg.lstsq(design, values, rcond=None)[0]
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "numpy.linalg.lstsq",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_ipw_ate(
    propensity_design: numpy.ndarray,
    values: numpy.ndarray,
    treatment: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Recomputes IPW propensities with an independent BFGS calculation.

    Args:
        propensity_design: Design matrix of the propensity model.
        values: Outcome vector.
        treatment: 0/1 treatment of every row.
        coefficients: ATE vector from the primary fit.
        policy: Supplies the reference tolerance, propensity clip and
            covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        A dictionary with the method, whether the optimizer converged, the
        reference ATE, its absolute difference from the primary ATE, the
        reference variance and its status, ``passed`` (converged and within
        ten times the tolerance), and the optimizer message.
    """

    def objective(candidate: numpy.ndarray) -> tuple[float, numpy.ndarray]:
        """Returns the negative Bernoulli log-likelihood and its gradient.

        Args:
            candidate: Propensity-model coefficients.

        Returns:
            The objective value and its gradient with respect to
            `candidate`.
        """
        logits = propensity_design @ candidate
        probabilities = special.expit(logits)
        value = -float(
            numpy.sum(treatment * logits - numpy.logaddexp(0.0, logits))
        )
        gradient = propensity_design.T @ (probabilities - treatment)
        return value, gradient

    result = optimize.minimize(
        lambda candidate: objective(candidate)[0],
        numpy.zeros(propensity_design.shape[1]),
        jac=lambda candidate: objective(candidate)[1],
        method="BFGS",
        options={"gtol": policy.reference_tolerance, "maxiter": 1000},
    )
    reference_propensity = numpy.clip(
        special.expit(propensity_design @ result.x),
        policy.propensity_clip,
        1.0 - policy.propensity_clip,
    )
    (
        reference_ate,
        reference_variance,
        reference_evidence,
    ) = hajek_ipw_statistics(
        treatment,
        values,
        reference_propensity,
        clusters,
        policy.covariance,
    )
    difference = float(abs(reference_ate - coefficients[0]))
    return {
        "method": "scipy.optimize.minimize:BFGS_propensity",
        "converged": bool(result.success),
        "reference_ate": reference_ate,
        "max_absolute_difference": difference,
        "variance": reference_variance,
        "variance_method": reference_evidence["uncertainty_method"],
        "variance_status": (
            "declared_influence_calculation_not_independently_verified"
        ),
        "passed": bool(
            result.success and difference <= policy.reference_tolerance * 10
        ),
        "message": str(result.message),
    }


def reference_aipw_ate(
    inputs: Mapping[str, numpy.ndarray],
    values: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Recomputes the augmented IPW score with an independent reference.

    Args:
        inputs: Arrays from the primary fit. Without cross-fitting only the
            `nuisance_design` and `treatment` entries are used; the
            reference refits the nuisance models itself.
        values: Outcome vector.
        coefficients: ATE vector from the primary fit.
        policy: Supplies the cross-fitting folds, reference tolerance,
            propensity clip and covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        A dictionary with the method, whether the propensity optimizer
        converged, the reference ATE, its absolute difference from the
        primary ATE, the reference variance and its status, ``passed``, and
        the optimizer message.  With cross-fitting it is the result of
        `reference_cross_fitted_aipw_ate`.
    """
    if policy.cross_fitting_folds > 1:
        return reference_cross_fitted_aipw_ate(
            inputs, values, coefficients, policy, clusters
        )
    nuisance_design = inputs["nuisance_design"]
    treatment = inputs["treatment"]
    confounder_design = nuisance_design

    def objective(candidate: numpy.ndarray) -> tuple[float, numpy.ndarray]:
        """Returns the negative Bernoulli log-likelihood and its gradient.

        Args:
            candidate: Propensity-model coefficients.

        Returns:
            The objective value and its gradient with respect to
            `candidate`.
        """
        logits = confounder_design @ candidate
        probabilities = special.expit(logits)
        value = -float(
            numpy.sum(treatment * logits - numpy.logaddexp(0.0, logits))
        )
        gradient = confounder_design.T @ (probabilities - treatment)
        return value, gradient

    propensity_fit = optimize.minimize(
        lambda candidate: objective(candidate)[0],
        numpy.zeros(confounder_design.shape[1]),
        jac=lambda candidate: objective(candidate)[1],
        method="BFGS",
        options={"gtol": policy.reference_tolerance, "maxiter": 1000},
    )
    propensity = numpy.clip(
        special.expit(confounder_design @ propensity_fit.x),
        policy.propensity_clip,
        1.0 - policy.propensity_clip,
    )
    treated = treatment == 1.0
    control = ~treated
    treated_coefficients = numpy.linalg.lstsq(
        nuisance_design[treated], values[treated], rcond=None
    )[0]
    control_coefficients = numpy.linalg.lstsq(
        nuisance_design[control], values[control], rcond=None
    )[0]
    outcome_treated = nuisance_design @ treated_coefficients
    outcome_control = nuisance_design @ control_coefficients
    score = (
        outcome_treated
        - outcome_control
        + treatment * (values - outcome_treated) / propensity
        - (1.0 - treatment) * (values - outcome_control) / (1.0 - propensity)
    )
    reference_ate = float(numpy.mean(score))
    reference_variance, reference_uncertainty = influence_variance(
        score - reference_ate,
        policy.covariance,
        clusters,
    )
    difference = float(abs(reference_ate - coefficients[0]))
    return {
        "method": "scipy.optimize.minimize:BFGS_plus_lstsq",
        "converged": bool(propensity_fit.success),
        "reference_ate": reference_ate,
        "max_absolute_difference": difference,
        "variance": reference_variance,
        "variance_method": reference_uncertainty["method"],
        "variance_status": (
            "declared_influence_calculation_not_independently_verified"
        ),
        "passed": bool(
            propensity_fit.success
            and difference <= policy.reference_tolerance * 10
        ),
        "message": str(propensity_fit.message),
    }


def reference_cross_fitted_aipw_ate(
    inputs: Mapping[str, numpy.ndarray],
    values: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Independently recomputes each cross-fitted AIPW validation score.

    Args:
        inputs: Arrays from the primary fit: the nuisance design, the
            treatment and the fold assignments.
        values: Outcome vector.
        coefficients: ATE vector from the primary fit.
        policy: Supplies the number of folds, reference tolerance,
            propensity clip and covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        A dictionary comparing the reference cross-fitted ATE with the
        primary ATE, with per-fold records and ``passed``.

    Raises:
        ValueError: If the fold assignments do not cover every row, or an
            outcome design is rank deficient.
    """
    nuisance_design = inputs["nuisance_design"]
    treatment = inputs["treatment"]
    fold_assignments = numpy.asarray(inputs["fold_assignments"], dtype=int)
    if len(fold_assignments) != len(values):
        raise ValueError("cross-fitting reference fold assignments are invalid")
    reference_scores = numpy.empty(len(values), dtype=float)
    fold_records: list[dict[str, Any]] = []
    for fold in range(policy.cross_fitting_folds):
        training = fold_assignments != fold
        validation = ~training
        training_design = nuisance_design[training]
        training_treatment = treatment[training]
        training_values = values[training]
        validation_design = nuisance_design[validation]
        treated = training_treatment == 1.0
        control = training_treatment == 0.0

        def objective(
            candidate: numpy.ndarray,
            design: numpy.ndarray = training_design,
            observed: numpy.ndarray = training_treatment,
        ) -> tuple[float, numpy.ndarray]:
            """Returns the negative Bernoulli log-likelihood and gradient.

            Args:
                candidate: Propensity-model coefficients.
                design: Training design matrix of this fold.
                observed: Training treatment of this fold.

            Returns:
                The objective value and its gradient with respect to
                `candidate`.
            """
            logits = design @ candidate
            probabilities = special.expit(logits)
            value = -float(
                numpy.sum(observed * logits - numpy.logaddexp(0.0, logits))
            )
            gradient = design.T @ (probabilities - observed)
            return value, gradient

        propensity_fit = optimize.minimize(
            lambda candidate: objective(candidate)[0],
            numpy.zeros(training_design.shape[1]),
            jac=lambda candidate: objective(candidate)[1],
            method="BFGS",
            options={"gtol": policy.reference_tolerance, "maxiter": 1000},
        )
        propensity = numpy.clip(
            special.expit(validation_design @ propensity_fit.x),
            policy.propensity_clip,
            1.0 - policy.propensity_clip,
        )
        treated_design = training_design[treated]
        control_design = training_design[control]
        if (
            numpy.linalg.matrix_rank(treated_design) != treated_design.shape[1]
            or numpy.linalg.matrix_rank(control_design)
            != control_design.shape[1]
        ):
            raise ValueError(
                "aipw_ate reference outcome design is not identifiable"
            )
        treated_coefficients = numpy.linalg.lstsq(
            treated_design, training_values[treated], rcond=None
        )[0]
        control_coefficients = numpy.linalg.lstsq(
            control_design, training_values[control], rcond=None
        )[0]
        outcome_treated = validation_design @ treated_coefficients
        outcome_control = validation_design @ control_coefficients
        reference_scores[validation] = (
            outcome_treated
            - outcome_control
            + treatment[validation]
            * (values[validation] - outcome_treated)
            / propensity
            - (1.0 - treatment[validation])
            * (values[validation] - outcome_control)
            / (1.0 - propensity)
        )
        fold_records.append(
            {
                "fold": fold,
                "converged": bool(propensity_fit.success),
                "message": str(propensity_fit.message),
            }
        )
    reference_ate = float(numpy.mean(reference_scores))
    reference_variance, reference_uncertainty = influence_variance(
        reference_scores - reference_ate,
        policy.covariance,
        clusters,
    )
    difference = float(abs(reference_ate - coefficients[0]))
    converged = all(record["converged"] for record in fold_records)
    return {
        "method": "scipy.optimize.minimize:BFGS_plus_lstsq_per_training_fold",
        "converged": converged,
        "reference_ate": reference_ate,
        "max_absolute_difference": difference,
        "variance": reference_variance,
        "variance_method": reference_uncertainty["method"],
        "variance_status": (
            "declared_influence_calculation_not_independently_verified"
        ),
        "folds": fold_records,
        "passed": bool(
            converged and difference <= policy.reference_tolerance * 10
        ),
    }


def reference_hierarchical_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    residual_variance: float,
    random_intercept_variance: float,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Verifies hierarchical GLS with explicit group covariance matrices.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        residual_variance: Residual variance of the primary fit.
        random_intercept_variance: Random-intercept variance of the primary
            fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute coefficient
        difference, and ``passed``, True when it is within the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable. A dense system
            that is not positive definite raises `LinAlgError`, a subclass.
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        count = int(numpy.sum(group))
        covariance = residual_variance * numpy.eye(count) + (
            random_intercept_variance * numpy.ones((count, count))
        )
        precision = linalg.solve(covariance, numpy.eye(count), assume_a="pos")
        group_design = design[group]
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ values[group]
    reference = linalg.solve(information, right_hand_side, assume_a="pos")
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:explicit_random_intercept_covariance",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_hierarchical_reml(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    residual_variance: float,
    random_intercept_variance: float,
    coefficients: numpy.ndarray,
    objective: float,
    policy: Config,
) -> dict[str, Any]:
    """Verifies REML GLS with an independent dense covariance calculation.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        residual_variance: REML residual variance of the primary fit.
        random_intercept_variance: REML random-intercept variance of the
            primary fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        objective: Negative REML objective at the primary optimum.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute coefficient
        difference, the absolute objective difference, and ``passed``, True
        when the coefficients agree within the tolerance and the objectives
        within ten times the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable, or if the dense
            covariance or information matrix is not positive definite
            (scipy raises `LinAlgError`, a subclass, before the
            determinant sign check can).
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    covariance_blocks = []
    for label in labels:
        count = int(numpy.sum(clusters == label))
        covariance_blocks.append(
            residual_variance * numpy.eye(count)
            + random_intercept_variance * numpy.ones((count, count))
        )
    covariance = cast("numpy.ndarray", linalg.block_diag(*covariance_blocks))
    precision = linalg.solve(covariance, numpy.eye(len(values)), assume_a="pos")
    # scipy.linalg.solve is typed as returning a generic array; the product
    # is an ndarray at runtime.
    information = cast("numpy.ndarray", design.T @ precision @ design)
    right_hand_side = design.T @ precision @ values
    reference_coefficients = linalg.solve(
        information, right_hand_side, assume_a="pos"
    )
    residuals = values - design @ reference_coefficients
    sign_covariance, logdet_covariance = numpy.linalg.slogdet(covariance)
    sign_information, logdet_information = numpy.linalg.slogdet(information)
    if sign_covariance <= 0.0 or sign_information <= 0.0:
        raise ValueError("reference REML covariance is not positive definite")
    reference_objective = 0.5 * (
        logdet_covariance
        + logdet_information
        + float(residuals @ precision @ residuals)
        + (len(values) - design.shape[1]) * math.log(2.0 * math.pi)
    )
    coefficient_difference = float(
        numpy.max(numpy.abs(reference_coefficients - coefficients))
    )
    objective_difference = float(abs(reference_objective - objective))
    return {
        "method": "scipy.linalg.solve:dense_random_intercept_covariance",
        "max_absolute_difference": coefficient_difference,
        "objective_difference": objective_difference,
        "passed": bool(
            coefficient_difference <= policy.reference_tolerance
            and objective_difference <= policy.reference_tolerance * 10
        ),
    }


def reference_hierarchical_random_slope(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Verifies one-random-slope GLS with an independent dense calculation.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: Residual variance of the primary fit.
        random_effect_covariance: 2x2 covariance of the random intercept and
            slope from the primary fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the slope position, the largest
        absolute coefficient difference, the random-effect covariance, and
        ``passed``, True when the difference is within the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable. A dense system
            that is not positive definite raises `LinAlgError`, a subclass.
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    information = numpy.zeros((design.shape[1], design.shape[1]))
    right_hand_side = numpy.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        group_design = design[group]
        count = group_design.shape[0]
        random_effect_design = numpy.column_stack(
            (
                numpy.ones(count),
                group_design[:, random_slope_position + 1],
            )
        )
        covariance = (
            residual_variance * numpy.eye(count)
            + random_effect_design
            @ random_effect_covariance
            @ random_effect_design.T
        )
        precision = linalg.solve(covariance, numpy.eye(count), assume_a="pos")
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ values[group]
    reference = linalg.solve(information, right_hand_side, assume_a="pos")
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:dense_random_intercept_slope_covariance",
        "random_slope_position": random_slope_position,
        "max_absolute_difference": difference,
        "random_effect_covariance": random_effect_covariance.tolist(),
        "passed": difference <= policy.reference_tolerance,
    }


def reference_hierarchical_random_slope_reml(
    design: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    random_slope_position: int,
    residual_variance: float,
    random_effect_covariance: numpy.ndarray,
    coefficients: numpy.ndarray,
    objective: float,
    policy: Config,
) -> dict[str, Any]:
    """Verifies random-slope REML with independent dense block algebra.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        clusters: Cluster label of every row.
        random_slope_position: Feature index carrying the random slope.
        residual_variance: REML residual variance of the primary fit.
        random_effect_covariance: 2x2 REML covariance of the random
            intercept and slope from the primary fit.
        coefficients: Fixed-effect coefficients from the primary fit.
        objective: Negative REML objective at the primary optimum.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the slope position, the largest
        absolute coefficient difference, the absolute objective difference,
        the random-effect covariance, and ``passed``, True when the
        coefficients agree within the tolerance and the objectives within
        ten times the tolerance.

    Raises:
        ValueError: If the cluster labels are not hashable, or if the dense
            covariance or information matrix is not positive definite
            (scipy raises `LinAlgError`, a subclass, before the
            determinant sign check can).
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    covariance_blocks = []
    for label in labels:
        group = clusters == label
        group_design = design[group]
        count = group_design.shape[0]
        random_effect_design = numpy.column_stack(
            (numpy.ones(count), group_design[:, random_slope_position + 1])
        )
        covariance_blocks.append(
            residual_variance * numpy.eye(count)
            + random_effect_design
            @ random_effect_covariance
            @ random_effect_design.T
        )
    covariance = cast("numpy.ndarray", linalg.block_diag(*covariance_blocks))
    precision = linalg.solve(covariance, numpy.eye(len(values)), assume_a="pos")
    # scipy.linalg.solve is typed as returning a generic array; the product
    # is an ndarray at runtime.
    information = cast("numpy.ndarray", design.T @ precision @ design)
    right_hand_side = design.T @ precision @ values
    reference_coefficients = linalg.solve(
        information, right_hand_side, assume_a="pos"
    )
    residuals = values - design @ reference_coefficients
    sign_covariance, logdet_covariance = numpy.linalg.slogdet(covariance)
    sign_information, logdet_information = numpy.linalg.slogdet(information)
    if sign_covariance <= 0.0 or sign_information <= 0.0:
        raise ValueError("reference REML covariance is not positive definite")
    reference_objective = 0.5 * (
        logdet_covariance
        + logdet_information
        + float(residuals @ precision @ residuals)
        + (len(values) - design.shape[1]) * math.log(2.0 * math.pi)
    )
    coefficient_difference = float(
        numpy.max(numpy.abs(reference_coefficients - coefficients))
    )
    objective_difference = float(abs(reference_objective - objective))
    return {
        "method": "scipy.linalg.solve:dense_random_intercept_slope_covariance",
        "random_slope_position": random_slope_position,
        "max_absolute_difference": coefficient_difference,
        "objective_difference": objective_difference,
        "random_effect_covariance": random_effect_covariance.tolist(),
        "passed": bool(
            coefficient_difference <= policy.reference_tolerance
            and objective_difference <= policy.reference_tolerance * 10
        ),
    }


def reference_bayesian_linear(
    design: numpy.ndarray,
    values: numpy.ndarray,
    prior: BayesianPrior,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Verifies the conjugate posterior mean with an independent solver.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        prior: Normal-inverse-gamma prior used by the primary fit.
        coefficients: Posterior mean from the primary fit.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute difference
        from `coefficients`, and ``passed``, True when it is within the
        tolerance.
    """
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    reference = linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + design.T @ values,
        assume_a="pos",
    )
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:positive_definite",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_bayesian_ate(
    design: numpy.ndarray,
    values: numpy.ndarray,
    prior: BayesianPrior,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Independently verifies the causal posterior mean and ATE coefficient.

    Args:
        design: Design matrix including the intercept column.
        values: Outcome vector.
        prior: Normal-inverse-gamma prior used by the primary fit.
        coefficients: Posterior mean from the primary fit.
        policy: Supplies the treatment position and reference tolerance.

    Returns:
        A dictionary with the method name, the largest absolute difference
        from `coefficients`, the reference treatment coefficient, the
        treatment position, and ``passed``, True when the difference is within
        the tolerance.

    Raises:
        ValueError: If the prior dimension does not match the design or the
            policy has no treatment position.
    """
    validate_bayesian_prior_shape(prior, design.shape[1])
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    reference = linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + design.T @ values,
        assume_a="pos",
    )
    difference = float(numpy.max(numpy.abs(reference - coefficients)))
    treatment_index = policy.treatment_position
    if treatment_index is None:
        raise ValueError("bayesian_ate requires treatment_position")
    coefficient_index = treatment_index + 1
    return {
        "method": "scipy.linalg.solve:positive_definite_causal_gaussian",
        "max_absolute_difference": difference,
        "reference_treatment_coefficient": float(reference[coefficient_index]),
        "treatment_position": treatment_index,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_logistic(
    design: numpy.ndarray,
    values: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Independently verifies logistic coefficients with scipy optimization.

    Args:
        design: Design matrix including the intercept column.
        values: Binary outcomes (0 or 1).
        coefficients: Coefficients from the primary fit, also the starting
            point of the reference optimization.
        policy: Supplies the reference tolerance.

    Returns:
        A dictionary with the method name, whether the optimizer converged,
        the largest absolute difference from `coefficients`, ``passed``
        (converged and within ten times the tolerance), and the optimizer
        message.
    """

    def objective(candidate: numpy.ndarray) -> tuple[float, numpy.ndarray]:
        """Returns the negative Bernoulli log-likelihood and its gradient.

        Args:
            candidate: Logistic coefficients.

        Returns:
            The objective value and its gradient with respect to
            `candidate`.
        """
        probabilities = special.expit(design @ candidate)
        value = -bernoulli_log_likelihood(values, probabilities)
        gradient = design.T @ (probabilities - values)
        return value, gradient

    result = optimize.minimize(
        lambda candidate: objective(candidate)[0],
        coefficients,
        jac=lambda candidate: objective(candidate)[1],
        method="BFGS",
        options={"gtol": policy.reference_tolerance, "maxiter": 1000},
    )
    difference = float(numpy.max(numpy.abs(result.x - coefficients)))
    return {
        "method": "scipy.optimize.minimize:BFGS",
        "converged": bool(result.success),
        "max_absolute_difference": difference,
        "passed": bool(
            result.success and difference <= policy.reference_tolerance * 10
        ),
        "message": str(result.message),
    }


def bernoulli_log_likelihood(
    values: numpy.ndarray, probabilities: numpy.ndarray
) -> float:
    """Returns a numerically stable Bernoulli log likelihood.

    Args:
        values: Binary outcomes (0 or 1).
        probabilities: Predicted probability of outcome 1 for each row;
            clipped away from 0 and 1 at 1e-15 before taking logs.

    Returns:
        The summed log-likelihood.
    """
    logits = numpy.log(numpy.maximum(probabilities, 1e-15))
    complements = numpy.log(numpy.maximum(1.0 - probabilities, 1e-15))
    return float(numpy.sum(values * logits + (1.0 - values) * complements))


def prior_sensitivity(
    prior: BayesianPrior,
    design: numpy.ndarray,
    values: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
) -> dict[str, Any]:
    """Refits a Bayesian model under weaker and stronger prior precision.

    Args:
        prior: Bayesian prior of the fit.
        design: Design matrix including the intercept column.
        values: Complete outcomes used by the fit.
        coefficients: Point-estimate coefficient vector.
        policy: Supplies the estimator and treatment position.

    Returns:
        The evaluated prior section with the coefficient change under each
        alternative precision.

    Raises:
        ValueError: If the Bayesian causal estimator has no treatment
            position.
    """
    validate_bayesian_prior_shape(prior, design.shape[1])
    baseline = numpy.asarray(coefficients, dtype=float)
    alternatives = []
    prior_mean = numpy.asarray(prior.mean, dtype=float)
    prior_precision = numpy.asarray(prior.precision, dtype=float)
    for label, precision_factor in (
        ("weakened_precision", 0.01),
        ("strengthened_precision", 100.0),
    ):
        alternative_precision = prior_precision * precision_factor
        alternative = numpy.linalg.solve(
            alternative_precision + design.T @ design,
            alternative_precision @ prior_mean + design.T @ values,
        )
        alternative_record = {
            "name": label,
            "max_absolute_coefficient_change": float(
                numpy.max(numpy.abs(alternative - baseline))
            ),
        }
        if policy.estimator == "bayesian_ate":
            if policy.treatment_position is None:
                raise ValueError("bayesian_ate requires treatment_position")
            target_index = policy.treatment_position + 1
            alternative_record["treatment_coefficient_change"] = float(
                alternative[target_index] - baseline[target_index]
            )
        alternatives.append(alternative_record)
    return {
        "status": "evaluated",
        "method": "precision_scale_sensitivity",
        "alternatives": alternatives,
    }


def shift_sensitivity(
    matrix: numpy.ndarray, shift_reference: Any
) -> dict[str, Any]:
    """Compares the fitted rows with a target-population feature matrix.

    Args:
        matrix: Complete feature rows used by the fit.
        shift_reference: Feature matrix from the target population.

    Returns:
        The evaluated distribution-shift section with standardized mean
        differences.

    Raises:
        ValueError: If `shift_reference` has the wrong shape or holds
            non-finite values.
    """
    reference = numpy.asarray(shift_reference, dtype=float)
    validate_matrix(reference, matrix.shape[1], "shift_reference")
    if not numpy.isfinite(reference).all():
        raise ValueError("shift_reference must contain only finite values")
    pooled = numpy.sqrt((matrix.var(axis=0) + reference.var(axis=0)) / 2.0)
    difference = numpy.abs(reference.mean(axis=0) - matrix.mean(axis=0)) / (
        numpy.maximum(pooled, 1e-12)
    )
    return {
        "status": "evaluated",
        "standardized_mean_difference": [float(item) for item in difference],
        "max_standardized_mean_difference": float(numpy.max(difference)),
    }


def sensitivity_report(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    design: numpy.ndarray,
    coefficients: numpy.ndarray,
    shift_reference: Any | None,
    estimand: Estimand,
    clusters: numpy.ndarray | None,
    prior: BayesianPrior | None = None,
) -> dict[str, Any]:
    """Records declared sensitivity coverage without inventing causal claims.

    Args:
        matrix: Complete feature rows used by the fit.
        values: Complete outcomes used by the fit.
        policy: Supplies the estimator, missing policy, seed, confidence
            level, bootstrap replicate count and treatment position.
        design: Design matrix including the intercept column.
        coefficients: Point-estimate coefficient vector.
        shift_reference: Optional feature matrix from a target population.
        estimand: Scientific definition of the estimand.
        clusters: Optional cluster labels of the rows used.
        prior: Bayesian prior, when the estimator is Bayesian.

    Returns:
        A dictionary with the sections ``prior``, ``missingness``,
        ``sampling`` and ``distribution_shift``, each recording whether it
        was evaluated and its evidence.

    Raises:
        ValueError: If the Bayesian causal estimator has no treatment
            position, if `shift_reference` has the wrong shape or holds
            non-finite values, if a hierarchical bootstrap is requested
            without cluster labels, or if a causal or hierarchical bootstrap
            yields fewer than two valid replicates.
    """
    report: dict[str, Any] = {
        "prior": {
            "status": "not_applicable",
            "reason": "current estimators are frequentist and use no prior",
        },
        "missingness": {
            "status": "evaluated",
            "policy": policy.missing,
            "rows": len(values),
        },
        "sampling": {
            "status": (
                "available" if policy.bootstrap_replicates else "not_requested"
            ),
            "bootstrap_replicates": policy.bootstrap_replicates,
        },
        "distribution_shift": {
            "status": (
                "not_available" if shift_reference is None else "evaluated"
            )
        },
    }
    if prior is not None:
        report["prior"] = prior_sensitivity(
            prior, design, values, coefficients, policy
        )
    if shift_reference is not None:
        report["distribution_shift"] = shift_sensitivity(
            matrix, shift_reference
        )
    if policy.bootstrap_replicates and policy.estimator in {
        "ipw_ate",
        "aipw_ate",
        "continuous_ipw_ate",
    }:
        report["sampling"] = causal_bootstrap_sensitivity(
            matrix,
            values,
            estimand,
            policy,
            coefficients,
            clusters,
        )
    elif (
        policy.bootstrap_replicates
        and policy.estimator == "hierarchical_linear"
    ):
        if clusters is None:
            raise ValueError("hierarchical_linear requires cluster labels")
        report["sampling"] = hierarchical_bootstrap_sensitivity(
            matrix,
            values,
            estimand,
            policy,
            coefficients,
            clusters,
        )
    elif policy.bootstrap_replicates:
        rng = numpy.random.default_rng(policy.seed)
        estimates: list[numpy.ndarray] = []
        bootstrap_replicate = 0
        while bootstrap_replicate < policy.bootstrap_replicates:
            sample = rng.integers(0, len(values), size=len(values))
            if policy.estimator == "linear":
                linear_fit = fit_linear(design[sample], values[sample], 1, 1e-8)
                estimate = linear_fit[0]
            else:
                logistic_fit = fit_logistic(
                    design[sample],
                    values[sample],
                    policy.max_iterations,
                    policy.tolerance,
                )
                estimate = logistic_fit[0]
            estimates.append(estimate)
            bootstrap_replicate += 1
        bootstrap = numpy.asarray(estimates)
        alpha = (1.0 - policy.confidence_level) / 2.0
        report["sampling"]["bootstrap_intervals"] = [
            [float(item) for item in numpy.quantile(bootstrap, alpha, axis=0)],
            [
                float(item)
                for item in numpy.quantile(bootstrap, 1.0 - alpha, axis=0)
            ],
        ]
    return report


def bootstrap_replicate_estimate(
    sampled_matrix: numpy.ndarray,
    sampled_values: numpy.ndarray,
    estimand: Estimand,
    bootstrap_policy: Config,
    clusters: Sequence[Any] | None,
    coefficients: numpy.ndarray,
) -> numpy.ndarray:
    """Estimates the parameters on one bootstrap draw and checks the result.

    Args:
        sampled_matrix: Feature rows of the draw.
        sampled_values: Outcomes of the draw.
        estimand: Scientific estimand and its explicit assumptions.
        bootstrap_policy: The estimator policy with bootstrapping turned off.
        clusters: Cluster labels of the draw, or None for row-level draws.
        coefficients: Point-estimate coefficient vector for shape checking.

    Returns:
        The parameter estimates of the draw.

    Raises:
        ValueError: If the estimate has an unexpected shape, is not finite,
            or fails its independent reference check.
    """
    result = estimate_parameters(
        sampled_matrix,
        sampled_values,
        estimand,
        bootstrap_policy,
        clusters=clusters,
    )
    estimate = numpy.asarray(
        [parameter.estimate for parameter in result.parameters],
        dtype=float,
    )
    if estimate.shape != numpy.asarray(coefficients).shape:
        raise ValueError(
            "bootstrap replicate returned an unexpected parameter shape"
        )
    if not numpy.isfinite(estimate).all():
        raise ValueError("bootstrap replicate returned non-finite estimates")
    if result.reference.get("passed") is not True:
        raise ValueError(
            "bootstrap replicate independent reference check failed"
        )
    return estimate


def hierarchical_bootstrap_sensitivity(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    estimand: Estimand,
    policy: Config,
    coefficients: numpy.ndarray,
    clusters: numpy.ndarray,
) -> dict[str, Any]:
    """Estimates grouped-parameter sensitivity by resampling whole clusters.

    Each bootstrap draw samples the source clusters with replacement and
    gives every selected copy a fresh identity. The declared moment-GLS or
    bounded-REML policy is then rerun on that draw. A replicate is accepted
    only when its independent reference calculation passes, so percentile
    intervals never combine estimates that failed the scientific check.

    Args:
        matrix: Complete feature rows used by the point estimate.
        values: Complete outcomes used by the point estimate.
        estimand: Scientific estimand and its explicit assumptions.
        policy: Hierarchical estimator policy with a positive replicate count.
        coefficients: Point-estimate coefficient vector for shape checking.
        clusters: Complete source cluster labels.

    Returns:
        Serializable bootstrap evidence with draw digests, replicate status,
        percentile intervals, independent reference counts, and limitations.

    Raises:
        ValueError: If fewer than two reference-checked replicates succeed,
            or the cluster labels are unhashable or form fewer than two
            clusters.
    """
    rng = numpy.random.default_rng(policy.seed)
    bootstrap_policy = dataclasses.replace(policy, bootstrap_replicates=0)
    estimates: list[numpy.ndarray] = []
    failures: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    if policy.random_slope_position is None:
        limitations = [
            "variance-component uncertainty is not propagated",
            "random slopes are unsupported",
            "nested effects are unsupported",
            "crossed effects are unsupported",
        ]
    else:
        limitations = [
            "variance-component uncertainty is not propagated",
            "only one random slope is supported",
            "nested effects are unsupported",
            "crossed effects are unsupported",
            "fully Bayesian multilevel inference is unsupported",
        ]
    source_labels = tuple(dict.fromkeys(clusters.tolist()))
    source_unit_count = len(source_labels)

    for replicate in range(policy.bootstrap_replicates):
        sampled_matrix, sampled_values, sampled_clusters, draw = (
            draw_hierarchical_bootstrap_sample(matrix, values, clusters, rng)
        )
        draws.append(
            {
                "replicate": replicate,
                "resampling_unit": "cluster",
                "estimator_method": policy.hierarchical_method,
                **draw,
            }
        )
        try:
            estimates.append(
                bootstrap_replicate_estimate(
                    sampled_matrix,
                    sampled_values,
                    estimand,
                    bootstrap_policy,
                    sampled_clusters.tolist(),
                    coefficients,
                )
            )
        except (
            FloatingPointError,
            numpy.linalg.LinAlgError,
            OverflowError,
            ValueError,
        ) as error:
            failures.append(
                {
                    "replicate": replicate,
                    "resampling_unit": "cluster",
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    if len(estimates) < MIN_REPLICATES:
        raise ValueError(
            "hierarchical bootstrap needs at least two successful replicates; "
            f"received {len(estimates)} of {policy.bootstrap_replicates}"
        )

    bootstrap = numpy.asarray(estimates)
    alpha = (1.0 - policy.confidence_level) / 2.0
    return {
        "status": "partial" if failures else "available",
        "method": "nonparametric_percentile_bootstrap",
        "bootstrap_replicates": policy.bootstrap_replicates,
        "successful_replicates": len(estimates),
        "failed_replicates": len(failures),
        "confidence_level": policy.confidence_level,
        "seed": policy.seed,
        "resampling_unit": "cluster",
        "estimator_method": policy.hierarchical_method,
        "source_unit_count": source_unit_count,
        "source_row_count": len(values),
        "cluster_identity_policy": "fresh_identity_per_selected_cluster",
        "draws": draws,
        "failures": failures,
        "reference_checks": {
            "successful": len(estimates),
            "failed": len(failures),
            "requirement": "every accepted replicate passes its independent "
            "point-estimate reference check",
        },
        "limitations": limitations,
        "bootstrap_intervals": [
            [float(item) for item in numpy.quantile(bootstrap, alpha, axis=0)],
            [
                float(item)
                for item in numpy.quantile(bootstrap, 1.0 - alpha, axis=0)
            ],
        ],
    }


def draw_hierarchical_bootstrap_sample(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray,
    rng: numpy.random.Generator,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, dict[str, Any]]:
    """Draws whole clusters and assigns fresh identities to selected copies.

    Args:
        matrix: Complete feature rows from the source estimate.
        values: Complete outcomes from the source estimate.
        clusters: Source cluster labels.
        rng: Seeded random generator owned by the sensitivity report.

    Returns:
        Resampled rows, outcomes, fresh cluster labels, and digest evidence.

    Raises:
        ValueError: If the cluster labels are not hashable or the source
            data contain fewer than two clusters.
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable for bootstrap"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise ValueError(
            "hierarchical bootstrap needs at least two source clusters"
        )
    source_indices = rng.integers(0, len(labels), size=len(labels))
    row_groups = [
        numpy.flatnonzero(clusters == labels[index]) for index in source_indices
    ]
    indices = numpy.concatenate(row_groups)
    sampled_clusters = numpy.repeat(
        numpy.arange(len(row_groups), dtype=object),
        [len(group) for group in row_groups],
    )
    return (
        matrix[indices],
        values[indices],
        sampled_clusters,
        {
            "sampled_rows": len(indices),
            "sampled_units": len(row_groups),
            "source_unit_count": len(labels),
            "sample_digest": bootstrap_sample_digest(source_indices),
            "fresh_cluster_identity_digest": bootstrap_sample_digest(
                numpy.arange(len(row_groups), dtype=numpy.int64)
            ),
        },
    )


def causal_bootstrap_sensitivity(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    estimand: Estimand,
    policy: Config,
    coefficients: numpy.ndarray,
    clusters: numpy.ndarray | None,
) -> dict[str, Any]:
    """Estimates causal sampling sensitivity with nonparametric bootstrap.

    Row-level causal policies resample observations independently. Clustered
    causal policies resample complete declared clusters, preserving each
    cluster as a sampling unit and assigning every selected copy a distinct
    bootstrap cluster label for the influence-function covariance.

    Args:
        matrix: Complete rows used by the point estimate.
        values: Complete outcomes used by the point estimate.
        estimand: Scientific estimand and its explicit causal assumptions.
        policy: Causal estimator policy with a positive replicate count.
        coefficients: Point-estimate coefficient vector for shape checking.
        clusters: Cluster labels for a clustered policy, otherwise ``None``.

    Returns:
        Serializable sampling-sensitivity evidence, including percentile
        intervals, resampling-unit evidence, and every failed replicate.

    Raises:
        ValueError: If the bootstrap cannot produce at least two valid,
            independently reference-checked replicates, or clustered labels
            are unhashable or form fewer than two clusters.
    """
    rng = numpy.random.default_rng(policy.seed)
    bootstrap_policy = dataclasses.replace(policy, bootstrap_replicates=0)
    estimates: list[numpy.ndarray] = []
    failures: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    resampling_unit = "cluster" if clusters is not None else "row"
    source_unit_count = (
        len(tuple(dict.fromkeys(clusters.tolist())))
        if clusters is not None
        else len(values)
    )
    source_row_count = len(values)

    for replicate in range(policy.bootstrap_replicates):
        sampled_matrix, sampled_values, sampled_clusters, draw = (
            draw_causal_bootstrap_sample(matrix, values, clusters, rng)
        )
        bootstrap_clusters: Sequence[Any] | None = (
            sampled_clusters.tolist() if sampled_clusters is not None else None
        )
        draw_record = {
            "replicate": replicate,
            "resampling_unit": resampling_unit,
            **draw,
        }
        draws.append(draw_record)
        try:
            estimates.append(
                bootstrap_replicate_estimate(
                    sampled_matrix,
                    sampled_values,
                    estimand,
                    bootstrap_policy,
                    bootstrap_clusters,
                    coefficients,
                )
            )
        except (
            FloatingPointError,
            numpy.linalg.LinAlgError,
            OverflowError,
            ValueError,
        ) as error:
            failures.append(
                {
                    "replicate": replicate,
                    "resampling_unit": resampling_unit,
                    "error_type": type(error).__name__,
                    "message": str(error),
                }
            )

    if len(estimates) < MIN_REPLICATES:
        raise ValueError(
            "causal bootstrap needs at least two successful replicates; "
            f"received {len(estimates)} of {policy.bootstrap_replicates}"
        )

    bootstrap = numpy.asarray(estimates)
    alpha = (1.0 - policy.confidence_level) / 2.0
    status = "partial" if failures else "available"
    return {
        "status": status,
        "method": "nonparametric_percentile_bootstrap",
        "bootstrap_replicates": policy.bootstrap_replicates,
        "successful_replicates": len(estimates),
        "failed_replicates": len(failures),
        "confidence_level": policy.confidence_level,
        "seed": policy.seed,
        "resampling_unit": resampling_unit,
        "source_unit_count": source_unit_count,
        "source_row_count": source_row_count,
        "draws": draws,
        "failures": failures,
        "reference_checks": {
            "successful": len(estimates),
            "failed": len(failures),
            "requirement": "every accepted replicate passes its independent "
            "point-estimate reference check",
        },
        "bootstrap_intervals": [
            [float(item) for item in numpy.quantile(bootstrap, alpha, axis=0)],
            [
                float(item)
                for item in numpy.quantile(bootstrap, 1.0 - alpha, axis=0)
            ],
        ],
    }


def draw_causal_bootstrap_sample(
    matrix: numpy.ndarray,
    values: numpy.ndarray,
    clusters: numpy.ndarray | None,
    rng: numpy.random.Generator,
) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray | None, dict[str, Any]]:
    """Draws one causal bootstrap sample and its auditable unit evidence.

    Args:
        matrix: Complete feature rows from the source estimate.
        values: Complete outcomes from the source estimate.
        clusters: Source cluster labels, or None to resample rows
            independently.
        rng: Seeded random generator owned by the sensitivity report.

    Returns:
        The resampled rows, the resampled outcomes, fresh cluster labels
        (None when `clusters` is None), and a dictionary of sampling
        evidence including a digest of the drawn indices.

    Raises:
        ValueError: If the cluster labels are not hashable or the source
            data contain fewer than two clusters.
    """
    if clusters is None:
        indices = rng.integers(0, len(values), size=len(values))
        return (
            matrix[indices],
            values[indices],
            None,
            {
                "sampled_rows": len(indices),
                "source_unit_count": len(values),
                "sample_digest": bootstrap_sample_digest(indices),
            },
        )

    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "cluster labels must be hashable for clustered bootstrap"
        ) from error
    if len(labels) < MIN_CLUSTERS:
        raise ValueError(
            "clustered bootstrap needs at least two source clusters"
        )
    source_indices = rng.integers(0, len(labels), size=len(labels))
    row_groups = [
        numpy.flatnonzero(clusters == labels[index]) for index in source_indices
    ]
    indices = numpy.concatenate(row_groups)
    sampled_clusters = numpy.repeat(
        numpy.arange(len(row_groups), dtype=object),
        [len(group) for group in row_groups],
    )
    return (
        matrix[indices],
        values[indices],
        sampled_clusters,
        {
            "sampled_rows": len(indices),
            "sampled_units": len(row_groups),
            "source_unit_count": len(labels),
            "sample_digest": bootstrap_sample_digest(source_indices),
        },
    )


def bootstrap_sample_digest(indices: numpy.ndarray) -> str:
    """Returns a digest of bootstrap draw indices without exposing raw rows.

    Args:
        indices: Integer indices of the draw.

    Returns:
        The SHA-256 hex digest of the indices as little-endian 64-bit
        integers.
    """
    canonical = numpy.asarray(indices, dtype="<i8").tobytes()
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "BayesianPrior",
    "CausalAssumption",
    "Config",
    "CovarianceKind",
    "Estimand",
    "Estimation",
    "EstimatorKind",
    "HierarchicalAssumption",
    "MissingPolicy",
    "ParameterEstimate",
    "estimate_parameters",
]
