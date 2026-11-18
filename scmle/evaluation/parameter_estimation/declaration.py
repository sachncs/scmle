"""The estimand, prior and coefficient records and the shared limits."""

import dataclasses
import math
from typing import Any
from typing import ClassVar
from typing import Literal

import numpy

from scmle.contracts import errors
from scmle.contracts import records

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
INTERCEPT_LIMITS = (
    "variance-component uncertainty is not propagated",
    "random slopes are unsupported",
    "nested effects are unsupported",
    "crossed effects are unsupported",
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
        migrations: The written layout's version steps, oldest first;
            version 2 writes `feature_names` as `features`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"feature_names": "features"}),
    )

    __module__ = "scmle.evaluation.parameter_estimation"

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
            raise errors.InputError("estimand name must not be empty")
        if not self.outcome.strip():
            raise errors.InputError("estimand outcome must not be empty")
        if not self.population.strip():
            raise errors.InputError("estimand population must not be empty")
        if not self.likelihood.strip():
            raise errors.InputError("estimand likelihood must not be empty")
        if not self.feature_names:
            raise errors.InputError("estimand needs at least one feature")
        if len(set(self.feature_names)) != len(self.feature_names):
            raise errors.InputError("estimand feature names must be unique")
        if not self.assumptions:
            raise errors.InputError("estimand needs explicit assumptions")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible scientific metadata."""
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )


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

    __module__ = "scmle.evaluation.parameter_estimation"

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
            raise errors.InputError("Bayesian prior needs a coefficient mean")
        prior_mean = numpy.asarray(self.mean, dtype=float)
        prior_precision = numpy.asarray(self.precision, dtype=float)
        if prior_mean.ndim != 1:
            raise errors.InputError(
                "Bayesian prior mean must be one-dimensional"
            )
        if prior_precision.shape != (len(self.mean), len(self.mean)):
            raise errors.InputError("Bayesian prior precision must be square")
        if not numpy.isfinite(prior_mean).all():
            raise errors.InputError("Bayesian prior mean must be finite")
        if not numpy.isfinite(prior_precision).all():
            raise errors.InputError("Bayesian prior precision must be finite")
        if not numpy.allclose(
            prior_precision, prior_precision.T, rtol=1e-10, atol=1e-12
        ):
            raise errors.InputError(
                "Bayesian prior precision must be symmetric"
            )
        try:
            numpy.linalg.cholesky(prior_precision)
        except numpy.linalg.LinAlgError as error:
            raise errors.InputError(
                "Bayesian prior precision must be positive definite"
            ) from error
        if not math.isfinite(self.shape) or self.shape <= 0.0:
            raise errors.InputError("Bayesian prior shape must be positive")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise errors.InputError("Bayesian prior scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible prior metadata."""
        return dataclasses.asdict(self)


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
        migrations: The written layout's version steps, oldest first;
            version 2 writes `standard_error` as `se`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"standard_error": "se"}),
    )

    __module__ = "scmle.evaluation.parameter_estimation"

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
            raise errors.InputError("parameter name must not be empty")
        for name, value in (
            ("estimate", self.estimate),
            ("standard_error", self.standard_error),
            ("p_value", self.p_value),
            ("interval_lower", self.interval[0]),
            ("interval_upper", self.interval[1]),
        ):
            if not math.isfinite(float(value)):
                raise errors.InputError(f"{name} must be finite")
        if self.standard_error < 0.0:
            raise errors.InputError("standard_error must not be negative")
        if not 0.0 <= self.p_value <= 1.0:
            raise errors.InputError("p_value must lie between 0 and 1")
        if self.interval[0] > self.interval[1]:
            raise errors.InputError("parameter interval must be ordered")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible coefficient evidence."""
        return records.upgrade(
            self.migrations, dataclasses.asdict(self), records.ORIGIN
        )
