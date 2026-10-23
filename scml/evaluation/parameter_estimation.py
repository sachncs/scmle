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

import numpy as np
import pandas as pd
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
HierarchicalAssumption = Literal[
    "random_intercept",
    "cluster_independence",
    "random_effect_exogeneity",
    "constant_residual_variance",
]
HierarchicalMethod = Literal["moment_gls", "reml"]

DEFAULT_REML_VARIANCE_BOUNDS = (1e-12, 1e12)
COVARIANCE_CONDITION_THRESHOLD = 1e12

REQUIRED_CAUSAL_ASSUMPTIONS = frozenset(
    {"consistency", "conditional_exchangeability", "positivity"}
)
REQUIRED_BAYESIAN_ATE_ASSUMPTIONS = frozenset(
    {"linearity", *REQUIRED_CAUSAL_ASSUMPTIONS}
)
REQUIRED_HIERARCHICAL_ASSUMPTIONS = frozenset(
    {
        "random_intercept",
        "cluster_independence",
        "random_effect_exogeneity",
        "constant_residual_variance",
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class EstimandSpec:
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
        """Reject ambiguous or incomplete scientific declarations."""
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
        """Return JSON-compatible scientific metadata."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class BayesianPrior:
    """Conjugate normal-inverse-gamma prior for a Gaussian design.

    The coefficient order is ``("intercept", *estimand.feature_names)``.
    ``precision`` is the positive-definite prior precision matrix for the
    coefficients conditional on the residual variance.  ``shape`` and
    ``scale`` parameterize the inverse-gamma prior on that variance.
    """

    mean: tuple[float, ...]
    precision: tuple[tuple[float, ...], ...]
    shape: float = 2.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        """Reject a prior that cannot define a proper posterior."""
        if not self.mean:
            raise ValueError("Bayesian prior needs a coefficient mean")
        prior_mean = np.asarray(self.mean, dtype=float)
        prior_precision = np.asarray(self.precision, dtype=float)
        if prior_mean.ndim != 1:
            raise ValueError("Bayesian prior mean must be one-dimensional")
        if prior_precision.shape != (len(self.mean), len(self.mean)):
            raise ValueError("Bayesian prior precision must be square")
        if not np.isfinite(prior_mean).all():
            raise ValueError("Bayesian prior mean must be finite")
        if not np.isfinite(prior_precision).all():
            raise ValueError("Bayesian prior precision must be finite")
        if not np.allclose(
            prior_precision, prior_precision.T, rtol=1e-10, atol=1e-12
        ):
            raise ValueError("Bayesian prior precision must be symmetric")
        try:
            np.linalg.cholesky(prior_precision)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "Bayesian prior precision must be positive definite"
            ) from error
        if not math.isfinite(self.shape) or self.shape <= 0.0:
            raise ValueError("Bayesian prior shape must be positive")
        if not math.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError("Bayesian prior scale must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible prior metadata."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class EstimationPolicy:
    """Numerical and uncertainty policy for an estimation run."""

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
    propensity_clip: float = 0.01
    causal_assumptions: tuple[CausalAssumption, ...] = ()
    hierarchical_assumptions: tuple[HierarchicalAssumption, ...] = ()
    cross_fitting_folds: int = 1
    hierarchical_method: HierarchicalMethod = "moment_gls"
    reml_variance_bounds: tuple[float, float] = DEFAULT_REML_VARIANCE_BOUNDS

    def __post_init__(self) -> None:
        """Validate the policy before data is touched."""
        if self.estimator not in (
            "linear",
            "logistic",
            "bayesian_linear",
            "bayesian_ate",
            "ipw_ate",
            "aipw_ate",
            "hierarchical_linear",
        ):
            raise ValueError(
                "estimator must be linear, logistic, bayesian_linear, "
                "bayesian_ate, ipw_ate, aipw_ate, or hierarchical_linear"
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
        if not 0.0 < self.propensity_clip < 0.5:
            raise ValueError(
                "propensity_clip must lie strictly between 0 and 0.5"
            )
        if (
            isinstance(self.cross_fitting_folds, bool)
            or not isinstance(self.cross_fitting_folds, int)
            or self.cross_fitting_folds < 1
        ):
            raise ValueError("cross_fitting_folds must be a positive integer")
        if self.hierarchical_method not in ("moment_gls", "reml"):
            raise ValueError("hierarchical_method must be moment_gls or reml")
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
        causal_estimators = {"bayesian_ate", "ipw_ate", "aipw_ate"}
        propensity_estimators = {"ipw_ate", "aipw_ate"}
        if (
            self.estimator not in causal_estimators
            and self.treatment_position is not None
        ):
            raise ValueError(
                "treatment_position is supported only by bayesian_ate, "
                "ipw_ate, or aipw_ate"
            )
        if (
            self.estimator not in propensity_estimators
            and self.propensity_clip != 0.01
        ):
            raise ValueError(
                "propensity_clip is supported only by ipw_ate or aipw_ate"
            )
        if self.estimator not in causal_estimators and self.causal_assumptions:
            raise ValueError(
                "causal assumptions are supported only by bayesian_ate, "
                "ipw_ate, or aipw_ate"
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
        if self.estimator in {"bayesian_linear", "bayesian_ate"}:
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
                    or set(self.causal_assumptions)
                    != REQUIRED_BAYESIAN_ATE_ASSUMPTIONS
                ):
                    raise ValueError(
                        "bayesian_ate requires consistency, "
                        "conditional_exchangeability, positivity, and "
                        "linearity assumptions"
                    )
                if self.cluster_name is not None:
                    raise ValueError(
                        "bayesian_ate does not support clustered data"
                    )
        elif self.estimator in causal_estimators:
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
                raise ValueError(
                    f"{self.estimator} requires treatment_position"
                )
            if (
                len(self.causal_assumptions)
                != len(set(self.causal_assumptions))
                or set(self.causal_assumptions) != REQUIRED_CAUSAL_ASSUMPTIONS
            ):
                raise ValueError(
                    f"{self.estimator} requires consistency, "
                    "conditional_exchangeability, "
                    "and positivity assumptions"
                )
        elif self.estimator == "hierarchical_linear":
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
            if (
                len(self.hierarchical_assumptions)
                != len(set(self.hierarchical_assumptions))
                or set(self.hierarchical_assumptions)
                != REQUIRED_HIERARCHICAL_ASSUMPTIONS
            ):
                raise ValueError(
                    "hierarchical_linear requires all random-intercept "
                    "assumptions"
                )
        elif self.prior is not None:
            raise ValueError(
                "a Bayesian prior is supported only by bayesian_linear or "
                "bayesian_ate"
            )
        if self.estimator != "hierarchical_linear":
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
        """Return JSON-compatible policy metadata."""
        values = dataclasses.asdict(self)
        if self.prior is None:
            values.pop("prior")
        if self.estimator not in {"bayesian_ate", "ipw_ate", "aipw_ate"}:
            values.pop("treatment_position")
            values.pop("causal_assumptions")
        if self.estimator not in {"ipw_ate", "aipw_ate"}:
            values.pop("propensity_clip")
        if self.estimator != "aipw_ate":
            values.pop("cross_fitting_folds")
        if self.estimator != "hierarchical_linear":
            values.pop("hierarchical_assumptions")
            values.pop("hierarchical_method")
            values.pop("reml_variance_bounds")
        if self.prior is not None:
            values["prior"] = self.prior.to_dict()
        return values


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """One coefficient estimate and its declared uncertainty."""

    name: str
    estimate: float
    standard_error: float
    interval: tuple[float, float]
    p_value: float

    def __post_init__(self) -> None:
        """Reject coefficient evidence that cannot be interpreted safely."""
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
        """Return JSON-compatible coefficient evidence."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class EstimationResult:
    """Complete, serializable evidence for one parameter-estimation run."""

    estimand: EstimandSpec
    policy: EstimationPolicy
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
        """Validate the report shape before it can be persisted or used."""
        if not isinstance(self.estimand, EstimandSpec):
            raise TypeError("estimand must be EstimandSpec")
        if not isinstance(self.policy, EstimationPolicy):
            raise TypeError("policy must be EstimationPolicy")
        if not self.parameters:
            raise ValueError("estimation result needs parameters")
        names = tuple(item.name for item in self.parameters)
        if len(set(names)) != len(names):
            raise ValueError("parameter names must be unique")
        expected = len(self.parameters)
        if len(self.covariance) != expected or any(
            len(row) != expected for row in self.covariance
        ):
            raise ValueError("covariance must be square over all parameters")
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
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.rows_used + self.rows_dropped != self.rows_input:
            raise ValueError("row accounting must reconcile to rows_input")
        for mapping_name, mapping_value in (
            ("diagnostics", self.diagnostics),
            ("sensitivity", self.sensitivity),
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
        """Return a JSON-compatible estimation report."""
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
        """Return the content digest of the canonical estimation report."""
        content = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(content).hexdigest()

    def write(self, path: str | Path) -> str:
        """Atomically write and return a digest-bound estimation report."""
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
                raise RuntimeError(
                    "temporary estimation report was not created"
                )
            os.replace(temporary, destination)
        except BaseException:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            raise
        return self.digest()

    @classmethod
    def read(cls, path: str | Path) -> "EstimationResult":
        """Read a report and reject a modified or malformed envelope."""
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
    def from_dict(cls, record: Mapping[str, Any]) -> "EstimationResult":
        """Rebuild a result from its JSON-compatible report."""
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
            EstimandSpec(**estimand_values),
            EstimationPolicy(**policy_values),
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


def estimate_parameters(
    features: Any,
    outcome: Any,
    estimand: EstimandSpec,
    policy: EstimationPolicy,
    clusters: Sequence[Any] | None = None,
    shift_reference: Any | None = None,
) -> EstimationResult:
    """Estimate declared parameters with auditable uncertainty evidence.

    Args:
        features: Numeric matrix with one column per ``estimand.feature_names``.
        outcome: Numeric response, or binary labels for logistic estimation.
        estimand: Scientific definition and identification assumptions.
        policy: Estimator, covariance, missingness, and uncertainty policy.
            ``bayesian_linear`` and ``bayesian_ate`` require an explicit
            ``BayesianPrior`` and report posterior credible intervals.
        clusters: Optional cluster labels. Required for clustered covariance.
        shift_reference: Optional feature matrix from the target population for
            standardized mean-shift diagnostics.

    Returns:
        A result containing estimates, covariance, intervals, diagnostics,
        sensitivity records, and an independent reference calculation.

    Raises:
        ValueError: If the data, policy, or estimator assumptions are invalid.
    """
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(outcome, dtype=float)
    validate_matrix(matrix, len(estimand.feature_names), "features")
    validate_vector(values, "outcome")
    input_clusters = validate_clusters(clusters, len(values))
    raw_input_fingerprint_value = raw_input_fingerprint(
        matrix, values, input_clusters
    )
    input_rows = matrix.shape[0]
    if matrix.shape[0] != len(values):
        raise ValueError("features and outcome must have the same row count")
    if policy.covariance == "clustered" and input_clusters is None:
        raise ValueError("clustered covariance needs cluster labels")
    if policy.estimator in ("hierarchical_linear",) and input_clusters is None:
        raise ValueError("hierarchical_linear requires cluster labels")
    if (
        policy.estimator in {"ipw_ate", "aipw_ate"}
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
        "hierarchical_linear": "gaussian",
    }[policy.estimator]
    if estimand.likelihood != expected_likelihood:
        raise ValueError(
            "estimand likelihood does not match estimator: "
            f"expected {expected_likelihood}, got {estimand.likelihood}"
        )
    matrix, values, kept, dropped = prepare_rows(
        matrix, values, policy.missing, input_clusters
    )
    if policy.estimator == "logistic" and not set(np.unique(values)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("logistic outcome must contain only 0 and 1")
    if (
        policy.estimator in {"bayesian_linear", "bayesian_ate"}
        and clusters is not None
    ):
        raise ValueError(
            f"{policy.estimator} estimation does not accept clusters"
        )
    if policy.estimator in {"bayesian_ate", "ipw_ate", "aipw_ate"}:
        if policy.treatment_position is None:
            raise ValueError(f"{policy.estimator} requires treatment_position")
        if not 0 <= policy.treatment_position < matrix.shape[1]:
            raise ValueError("treatment_position is outside the feature matrix")
    cluster_values = kept if input_clusters is not None else None
    design = np.column_stack((np.ones(matrix.shape[0]), matrix))
    names = ("intercept", *estimand.feature_names)
    if policy.estimator == "linear":
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
        pivot = stats.t.ppf(
            0.5 + policy.confidence_level / 2.0, degrees_of_freedom
        )
        p_values = 2.0 * stats.t.sf(
            np.abs(
                coefficients / np.maximum(np.sqrt(np.diag(covariance)), 1e-15)
            ),
            degrees_of_freedom,
        )
        diagnostics = {
            **diagnostics,
            "likelihood": likelihood,
            "residual_sum_of_squares": float(np.sum(residuals**2)),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "degrees_of_freedom": degrees_of_freedom,
        }
    elif policy.estimator == "logistic":
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
            np.abs(
                coefficients / np.maximum(np.sqrt(np.diag(covariance)), 1e-15)
            )
        )
        diagnostics = {**diagnostics, "likelihood": likelihood}
    elif policy.estimator in {"bayesian_linear", "bayesian_ate"}:
        if policy.prior is None:
            raise ValueError(
                f"{policy.estimator} estimation requires an explicit prior"
            )
        if policy.estimator == "bayesian_ate":
            treatment_position = policy.treatment_position
            if treatment_position is None:
                raise ValueError("bayesian_ate requires treatment_position")
            treatment_position = int(treatment_position)
            treatment = matrix[:, treatment_position]
            if not set(np.unique(treatment)).issubset({0.0, 1.0}):
                raise ValueError(
                    "bayesian_ate treatment must contain only 0 and 1"
                )
            treated_rows = int(np.sum(treatment == 1.0))
            control_rows = int(np.sum(treatment == 0.0))
            if treated_rows < 2 or control_rows < 2:
                raise ValueError(
                    "bayesian_ate needs at least two rows per treatment arm"
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
                "treatment_name": estimand.feature_names[
                    cast(int, treatment_position)
                ],
                "treated_rows": treated_rows,
                "control_rows": control_rows,
                "identification": (
                    "the treatment coefficient is the ATE under the "
                    "declared linearity, consistency, conditional "
                    "exchangeability, and positivity assumptions"
                ),
                "limitations": [
                    "time-varying treatment effects are unsupported",
                    "propensity or other nuisance-model uncertainty is not "
                    "propagated",
                    "clustered covariance is unsupported",
                    "fully Bayesian multilevel models are unsupported",
                ],
            }
        else:
            reference = reference_bayesian_linear(
                design, values, policy.prior, coefficients, policy
            )
        likelihood = "gaussian"
        posterior_scale = np.asarray(diagnostics["posterior_scale"])
        posterior_degrees = diagnostics["posterior_degrees_of_freedom"]
        pivot = stats.t.ppf(
            0.5 + policy.confidence_level / 2.0, posterior_degrees
        )
        p_values = 2.0 * stats.t.sf(
            np.abs(coefficients)
            / np.maximum(np.sqrt(np.diag(posterior_scale)), 1e-15),
            posterior_degrees,
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        interval_errors = np.sqrt(np.maximum(np.diag(posterior_scale), 0.0))
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
    elif policy.estimator in {"ipw_ate", "aipw_ate"}:
        if policy.estimator == "ipw_ate":
            coefficients, covariance, diagnostics, propensity_design = (
                fit_ipw_ate(
                    matrix,
                    values,
                    policy.treatment_position,
                    policy,
                    cluster_values,
                )
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
        names = ("ate",)
        pivot = stats.norm.ppf(0.5 + policy.confidence_level / 2.0)
        p_values = 2.0 * stats.norm.sf(
            np.abs(coefficients)
            / np.maximum(np.sqrt(np.diag(covariance)), 1e-15)
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        interval_errors = standard_errors
        diagnostics = {**diagnostics, "likelihood": likelihood}
    elif policy.estimator == "hierarchical_linear":
        if cluster_values is None:
            raise ValueError("hierarchical_linear requires cluster labels")
        (
            coefficients,
            covariance,
            diagnostics,
            residual_variance,
            random_intercept_variance,
        ) = fit_hierarchical_linear(design, values, cluster_values, policy)
        if policy.hierarchical_method == "reml":
            reference = reference_hierarchical_reml(
                design,
                values,
                cluster_values,
                residual_variance,
                random_intercept_variance,
                coefficients,
                diagnostics["reml_objective"],
                policy,
            )
        else:
            reference = reference_hierarchical_linear(
                design,
                values,
                cluster_values,
                residual_variance,
                random_intercept_variance,
                coefficients,
                policy,
            )
        likelihood = "gaussian"
        degrees_of_freedom = max(
            len(dict.fromkeys(cluster_values.tolist())) - 1, 1
        )
        pivot = stats.t.ppf(
            0.5 + policy.confidence_level / 2.0, degrees_of_freedom
        )
        p_values = 2.0 * stats.t.sf(
            np.abs(
                coefficients / np.maximum(np.sqrt(np.diag(covariance)), 1e-15)
            ),
            degrees_of_freedom,
        )
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        interval_errors = standard_errors
        diagnostics = {
            **diagnostics,
            "likelihood": likelihood,
            "degrees_of_freedom": degrees_of_freedom,
        }

    covariance_conditioning = validate_covariance(covariance, len(names))
    diagnostics = {
        **diagnostics,
        "covariance_conditioning": covariance_conditioning,
    }

    if policy.estimator not in {"bayesian_linear", "bayesian_ate"}:
        standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
        interval_errors = standard_errors

    parameters = tuple(
        ParameterEstimate(
            name,
            float(value),
            float(error),
            (
                float(value - pivot * interval_error),
                float(value + pivot * interval_error),
            ),
            float(p_value),
        )
        for name, value, error, interval_error, p_value in zip(
            names,
            coefficients,
            standard_errors,
            interval_errors,
            p_values,
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
    return EstimationResult(
        estimand=dataclasses.replace(estimand, likelihood=likelihood),
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
        reference=reference,
        raw_input_fingerprint=raw_input_fingerprint_value,
    )


def raw_input_fingerprint(
    features: Any,
    outcome: Any,
    clusters: Any | None = None,
) -> str:
    """Return a deterministic fingerprint of the estimator's raw inputs.

    Numeric values are hashed after the same float conversion used by the
    estimator. Cluster labels retain their value types and row order. The
    fingerprint covers inputs before missing-row filtering so it can identify
    the complete evidence source independently of the fitted result.
    """
    matrix = np.asarray(features, dtype=float)
    values = np.asarray(outcome, dtype=float).reshape(-1)
    digest = hashlib.sha256()
    update_input_digest(digest, "features", matrix)
    update_input_digest(digest, "outcome", values)
    if clusters is None:
        digest.update(b"clusters:none")
    else:
        cluster_values = np.asarray(clusters, dtype=object)
        update_input_digest(digest, "clusters", cluster_values)
    return digest.hexdigest()


def update_input_digest(digest: Any, name: str, values: np.ndarray) -> None:
    """Add one shape-, dtype-, and value-bound input to a digest."""
    array = np.asarray(values)
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
        canonical = np.ascontiguousarray(
            array.astype(array.dtype.newbyteorder("<"), copy=False)
        )
        values_bytes = canonical.tobytes()
    else:
        value_hashes = pd.util.hash_pandas_object(
            pd.Series(array.reshape(-1)), index=False
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


def validate_matrix(matrix: np.ndarray, columns: int, name: str) -> None:
    """Validate a numeric two-dimensional matrix."""
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape (rows, {columns})")
    if matrix.shape[0] <= columns:
        raise ValueError("estimation needs more rows than parameters")


def validate_vector(values: np.ndarray, name: str) -> None:
    """Validate a numeric one-dimensional vector without dropping values."""
    if values.ndim != 1:
        raise ValueError(f"{name} must have shape (rows,)")


def validate_clusters(
    clusters: Sequence[Any] | None, rows: int
) -> np.ndarray | None:
    """Validate cluster-label shape before missingness processing."""
    if clusters is None:
        return None
    cluster_values = np.asarray(clusters, dtype=object)
    if cluster_values.ndim != 1:
        raise ValueError("clusters must have shape (rows,)")
    if len(cluster_values) != rows:
        raise ValueError("clusters must have the same row count as outcome")
    return cluster_values


def validate_covariance(
    covariance: np.ndarray, dimension: int
) -> dict[str, Any]:
    """Validate covariance shape and report rank and numerical conditioning."""
    matrix = np.asarray(covariance, dtype=float)
    expected_shape = (dimension, dimension)
    if matrix.shape != expected_shape:
        raise ValueError(
            f"covariance must have shape ({dimension}, {dimension})"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("covariance must contain finite values")
    if not np.allclose(matrix, matrix.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    scale = float(singular_values[0]) if singular_values.size else 0.0
    rank_tolerance = np.finfo(float).eps * max(expected_shape) * scale
    rank = int(np.sum(singular_values > rank_tolerance))
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
    matrix: np.ndarray,
    values: np.ndarray,
    missing: MissingPolicy,
    clusters: Any | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Apply the declared missingness policy and retain cluster labels."""
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(values)
    if clusters is not None:
        cluster_array = np.asarray(clusters, dtype=object)
        cluster_missing = pd.isna(cluster_array)
        if not isinstance(cluster_missing, np.ndarray):
            cluster_missing = np.asarray(cluster_missing)
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
    design: np.ndarray,
    values: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Solve the Gaussian likelihood using a full-rank normal equation."""
    del max_iterations, tolerance
    gram = design.T @ design
    if np.linalg.matrix_rank(gram) != gram.shape[0]:
        raise ValueError("linear design matrix is rank deficient")
    coefficients = np.linalg.solve(gram, design.T @ values)
    return coefficients, {"converged": True, "iterations": 1}


def fit_ipw_ate(
    matrix: np.ndarray,
    values: np.ndarray,
    treatment_position: int | None,
    policy: EstimationPolicy,
    clusters: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], np.ndarray]:
    """Estimate a binary-treatment Hájek inverse-probability ATE."""
    if treatment_position is None:
        raise ValueError("ipw_ate requires treatment_position")
    treatment = matrix[:, treatment_position]
    if not set(np.unique(treatment)).issubset({0.0, 1.0}):
        raise ValueError("ipw_ate treatment must contain only 0 and 1")
    treated_count = int(np.sum(treatment == 1.0))
    control_count = int(np.sum(treatment == 0.0))
    if treated_count < 2 or control_count < 2:
        raise ValueError("ipw_ate needs at least two rows per treatment arm")
    confounders = np.delete(matrix, treatment_position, axis=1)
    propensity_design = np.column_stack(
        (np.ones(confounders.shape[0]), confounders)
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
    propensity = np.clip(
        raw_propensity, policy.propensity_clip, 1.0 - policy.propensity_clip
    )
    ate, variance, evidence = hajek_ipw_statistics(
        treatment,
        values,
        propensity,
        clusters,
        policy.covariance,
    )
    clipped = np.abs(raw_propensity - propensity) > 1e-12
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
            "raw_min": float(np.min(raw_propensity)),
            "raw_max": float(np.max(raw_propensity)),
            "clipped_min": float(np.min(propensity)),
            "clipped_max": float(np.max(propensity)),
            "clip_threshold": policy.propensity_clip,
            "fraction_clipped": float(np.mean(clipped)),
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
        np.asarray([ate]),
        np.asarray([[variance]]),
        diagnostics,
        propensity_design,
    )


def fit_aipw_ate(
    matrix: np.ndarray,
    values: np.ndarray,
    treatment_position: int | None,
    policy: EstimationPolicy,
    clusters: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Estimate a binary-treatment ATE with an augmented IPW score.

    With one fold, nuisance models are fitted on the full sample, preserving
    the historical estimator behavior. With multiple folds, each nuisance
    model is fitted only on the corresponding training rows and its score is
    evaluated on held-out rows.
    """
    if treatment_position is None:
        raise ValueError("aipw_ate requires treatment_position")
    treatment = matrix[:, treatment_position]
    if not set(np.unique(treatment)).issubset({0.0, 1.0}):
        raise ValueError("aipw_ate treatment must contain only 0 and 1")
    treated_count = int(np.sum(treatment == 1.0))
    control_count = int(np.sum(treatment == 0.0))
    if treated_count < 2 or control_count < 2:
        raise ValueError("aipw_ate needs at least two rows per treatment arm")

    confounders = np.delete(matrix, treatment_position, axis=1)
    nuisance_design = np.column_stack(
        (np.ones(confounders.shape[0]), confounders)
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
        fold_assignments = np.zeros(len(values), dtype=int)
        cross_fitting = {
            "status": "disabled",
            "fold_count": 1,
            "assignment_method": "not_applicable",
            "limitations": aipw_limitations(False),
        }
    else:
        fold_assignments = deterministic_fold_assignments(
            len(values), fold_count
        )
        support = validate_cross_fitting_support(
            treatment, fold_assignments, fold_count
        )
        propensity = np.empty(len(values), dtype=float)
        raw_propensity = np.empty(len(values), dtype=float)
        outcome_control = np.empty(len(values), dtype=float)
        outcome_treated = np.empty(len(values), dtype=float)
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
                    "training_rows": int(np.sum(training)),
                    "validation_rows": int(np.sum(validation)),
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
                str(fold): int(np.sum(fold_assignments != fold))
                for fold in range(fold_count)
            },
            "validation_rows_by_fold": {
                str(fold): int(np.sum(fold_assignments == fold))
                for fold in range(fold_count)
            },
            "nuisance_fits": fold_diagnostics,
            "limitations": aipw_limitations(True),
        }
    score = (
        outcome_treated
        - outcome_control
        + treatment * (values - outcome_treated) / propensity
        - (1.0 - treatment) * (values - outcome_control) / (1.0 - propensity)
    )
    ate = float(np.mean(score))
    influence = score - ate
    variance, uncertainty = influence_variance(
        influence, policy.covariance, clusters
    )
    clipped = np.abs(raw_propensity - propensity) > 1e-12
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
            "raw_min": float(np.min(raw_propensity)),
            "raw_max": float(np.max(raw_propensity)),
            "clipped_min": float(np.min(propensity)),
            "clipped_max": float(np.max(propensity)),
            "clip_threshold": policy.propensity_clip,
            "fraction_clipped": float(np.mean(clipped)),
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
        np.asarray([ate]),
        np.asarray([[variance]]),
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
    training_design: np.ndarray,
    training_treatment: np.ndarray,
    training_values: np.ndarray,
    evaluation_design: np.ndarray,
    policy: EstimationPolicy,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Fit AIPW nuisance models on training rows and predict evaluation rows."""
    treated = training_treatment == 1.0
    control = training_treatment == 0.0
    if int(np.sum(treated)) < 2 or int(np.sum(control)) < 2:
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
    propensity = np.clip(
        evaluation_raw_propensity,
        policy.propensity_clip,
        1.0 - policy.propensity_clip,
    )
    treated_design = training_design[treated]
    control_design = training_design[control]
    if (
        np.linalg.matrix_rank(treated_design) != treated_design.shape[1]
        or np.linalg.matrix_rank(control_design) != control_design.shape[1]
    ):
        raise ValueError("aipw_ate outcome design is not identifiable")
    treated_coefficients = np.linalg.lstsq(
        treated_design, training_values[treated], rcond=None
    )[0]
    control_coefficients = np.linalg.lstsq(
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
            "treated_rank": int(np.linalg.matrix_rank(treated_design)),
            "control_rank": int(np.linalg.matrix_rank(control_design)),
        },
    )


def deterministic_fold_assignments(rows: int, fold_count: int) -> np.ndarray:
    """Assign rows to balanced folds using a deterministic row-order rule."""
    if fold_count < 2:
        raise ValueError("cross-fitting needs at least two folds")
    if fold_count > rows:
        raise ValueError("cross_fitting_folds must not exceed rows")
    return np.arange(rows, dtype=int) % fold_count


def validate_cross_fitting_support(
    treatment: np.ndarray,
    fold_assignments: np.ndarray,
    fold_count: int,
) -> list[dict[str, int]]:
    """Validate treatment support in every training and validation fold."""
    support: list[dict[str, int]] = []
    for fold in range(fold_count):
        validation = fold_assignments == fold
        training = ~validation
        validation_treated = int(np.sum(treatment[validation] == 1.0))
        validation_control = int(np.sum(treatment[validation] == 0.0))
        training_treated = int(np.sum(treatment[training] == 1.0))
        training_control = int(np.sum(treatment[training] == 0.0))
        if validation_treated < 1 or validation_control < 1:
            raise ValueError(
                "aipw_ate cross-fitting needs both treatment arms in every "
                "validation fold"
            )
        if training_treated < 2 or training_control < 2:
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
    """Return limitations that apply to both bounded AIPW modes."""
    limitations = [
        "propensity and outcome nuisance uncertainty is not propagated",
        "positivity is diagnosed but not proven",
        "outcome nuisance models are separate ordinary least squares fits",
        "this estimator is not Bayesian and does not support time-varying "
        "treatment effects",
    ]
    if not cross_fitted:
        limitations.append(
            "nuisance models are fitted on the same rows used for scoring"
        )
    return limitations


def influence_variance(
    influence: np.ndarray,
    covariance: CovarianceKind,
    clusters: np.ndarray | None,
) -> tuple[float, dict[str, Any]]:
    """Compute IID or finite-sample clustered influence uncertainty.

    The clustered calculation treats each declared cluster as an independent
    sampling unit. It centers the observation-level influence scores before
    summing them within clusters and applies the finite-sample correction
    ``G / (G - 1) * (n - 1) / n``. This is intentionally separate from the
    HC3 and regression sandwich paths: causal ATE uncertainty is defined over
    an influence function, not over a model residual score.
    """
    centered_influence = np.asarray(influence, dtype=float)
    if centered_influence.ndim != 1:
        raise ValueError("influence scores must have shape (rows,)")
    if len(centered_influence) < 2:
        raise ValueError("influence uncertainty needs at least two rows")
    if not np.isfinite(centered_influence).all():
        raise ValueError("influence scores must contain only finite values")
    centered_influence = centered_influence - np.mean(centered_influence)
    if covariance == "iid":
        variance = float(
            np.sum(centered_influence**2)
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
    if len(labels) < 2:
        raise ValueError("clustered covariance needs at least two clusters")
    cluster_sums = []
    cluster_sizes = []
    for label in labels:
        membership = cluster_values == label
        cluster_sums.append(float(np.sum(centered_influence[membership])))
        cluster_sizes.append(int(np.sum(membership)))
    cluster_sum_array = np.asarray(cluster_sums, dtype=float)
    row_count = len(centered_influence)
    correction = (len(labels) / (len(labels) - 1)) * (
        (row_count - 1) / row_count
    )
    variance = float(correction * np.sum(cluster_sum_array**2) / row_count**2)
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
    treatment: np.ndarray,
    values: np.ndarray,
    propensity: np.ndarray,
    clusters: np.ndarray | None = None,
    covariance: CovarianceKind = "iid",
) -> tuple[float, float, dict[str, Any]]:
    """Compute normalized IPW means and an influence-function variance."""
    treated_weights = treatment / propensity
    control_weights = (1.0 - treatment) / (1.0 - propensity)
    treated_denominator = float(np.sum(treated_weights))
    control_denominator = float(np.sum(control_weights))
    if treated_denominator <= 0.0 or control_denominator <= 0.0:
        raise ValueError("ipw_ate treatment weights must have positive support")
    treated_mean = float(np.sum(treated_weights * values) / treated_denominator)
    control_mean = float(np.sum(control_weights * values) / control_denominator)
    influence = treated_weights * (values - treated_mean) - control_weights * (
        values - control_mean
    )
    variance, uncertainty = influence_variance(influence, covariance, clusters)
    effective_sample_size = {
        "treated": float(treated_denominator**2 / np.sum(treated_weights**2)),
        "control": float(control_denominator**2 / np.sum(control_weights**2)),
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
    design: np.ndarray,
    values: np.ndarray,
    prior: BayesianPrior,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Compute a conjugate Gaussian posterior from an explicit prior."""
    validate_bayesian_prior_shape(prior, design.shape[1])
    prior_mean = np.asarray(prior.mean, dtype=float)
    prior_precision = np.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    try:
        factor = linalg.cho_factor(posterior_precision, check_finite=True)
        posterior_mean = linalg.cho_solve(
            factor, prior_precision @ prior_mean + design.T @ values
        )
        posterior_precision_inverse = linalg.cho_solve(
            factor, np.eye(design.shape[1])
        )
    except np.linalg.LinAlgError as error:
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
    if posterior_degrees <= 2.0:
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
    """Validate prior vector and matrix dimensions against a design."""
    mean_shape = np.asarray(prior.mean, dtype=float).shape
    precision_shape = np.asarray(prior.precision, dtype=float).shape
    expected_shape = (dimension, dimension)
    if mean_shape != (dimension,) or precision_shape != expected_shape:
        raise ValueError(
            "Bayesian prior dimension must match intercept and features"
        )


def fit_hierarchical_linear(
    design: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    policy: EstimationPolicy,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float, float]:
    """Fit a bounded two-stage random-intercept linear model.

    The default estimates variance components by moments from an initial
    fixed-effect fit.  The bounded REML policy uses a deterministic profile
    likelihood over the residual and random-intercept variances, followed by
    generalized least squares for the fixed effects.
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    if len(labels) < 2:
        raise ValueError("hierarchical_linear needs at least two clusters")
    if np.linalg.matrix_rank(design) != design.shape[1]:
        raise ValueError("hierarchical design matrix is rank deficient")
    initial_coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    initial_residuals = values - design @ initial_coefficients
    cluster_means = []
    cluster_sizes = []
    within_sum_of_squares = 0.0
    for label in labels:
        group = clusters == label
        group_residuals = initial_residuals[group]
        cluster_means.append(float(np.mean(group_residuals)))
        cluster_sizes.append(int(np.sum(group)))
        centered = group_residuals - np.mean(group_residuals)
        within_sum_of_squares += float(np.sum(centered**2))
    residual_variance = within_sum_of_squares / max(
        len(values) - len(labels), 1
    )
    if not math.isfinite(residual_variance) or residual_variance <= 1e-12:
        raise ValueError(
            "hierarchical_linear needs positive within-cluster variance"
        )
    between_variance = float(np.var(cluster_means, ddof=1))
    finite_sample_adjustment = residual_variance * float(
        np.mean(1.0 / np.asarray(cluster_sizes, dtype=float))
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
        coefficients = np.linalg.solve(information, right_hand_side)
        covariance = np.linalg.solve(information, np.eye(design.shape[1]))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "hierarchical GLS system is not identifiable"
        ) from error
    random_effects = []
    fitted_residuals = values - design @ coefficients
    for label in labels:
        group = clusters == label
        count = int(np.sum(group))
        group_sum = float(np.sum(fitted_residuals[group]))
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
            "mean_absolute_blup": float(np.mean(np.abs(random_effects))),
            "max_absolute_blup": float(np.max(np.abs(random_effects))),
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
    )


def fit_hierarchical_reml(
    design: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    labels: tuple[Any, ...],
    initial_residual_variance: float,
    initial_random_intercept_variance: float,
    policy: EstimationPolicy,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], float, float]:
    """Fit a bounded random-intercept model with profile REML."""
    lower_bound, upper_bound = policy.reml_variance_bounds
    initial_variances = np.clip(
        np.asarray(
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
        np.log(initial_variances),
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
    variance_components = np.exp(np.asarray(optimizer_result.x, dtype=float))
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
        coefficients = np.linalg.solve(information, right_hand_side)
        covariance = np.linalg.solve(information, np.eye(design.shape[1]))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "hierarchical REML GLS system is not identifiable"
        ) from error
    fitted_residuals = values - design @ coefficients
    random_effects = []
    cluster_sizes = []
    for label in labels:
        group = clusters == label
        count = int(np.sum(group))
        cluster_sizes.append(count)
        group_sum = float(np.sum(fitted_residuals[group]))
        shrinkage = random_intercept_variance / (
            residual_variance + count * random_intercept_variance
        )
        random_effects.append(shrinkage * group_sum)
    total_variance = residual_variance + random_intercept_variance
    jacobian = np.asarray(optimizer_result.jac, dtype=float)
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
            "mean_absolute_blup": float(np.mean(np.abs(random_effects))),
            "max_absolute_blup": float(np.max(np.abs(random_effects))),
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
            "gradient_norm": float(np.linalg.norm(jacobian)),
            "variance_bounds": [lower_bound, upper_bound],
        },
    }
    return (
        coefficients,
        covariance,
        diagnostics,
        residual_variance,
        random_intercept_variance,
    )


def reml_profile_objective(
    log_variances: np.ndarray,
    design: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    labels: tuple[Any, ...],
) -> float:
    """Return the negative profile REML objective in log-variance space."""
    if np.asarray(log_variances).shape != (2,):
        raise ValueError("REML needs residual and random-intercept variances")
    variances = np.exp(np.asarray(log_variances, dtype=float))
    if not np.isfinite(variances).all() or np.any(variances <= 0.0):
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
    sign, logdet_information = np.linalg.slogdet(information)
    if sign <= 0.0 or not math.isfinite(float(logdet_information)):
        return float("inf")
    try:
        coefficients = np.linalg.solve(information, right_hand_side)
    except np.linalg.LinAlgError:
        return float("inf")
    residuals = values - design @ coefficients
    quadratic = 0.0
    logdet_covariance = 0.0
    for label in labels:
        group = clusters == label
        count = int(np.sum(group))
        group_residuals = residuals[group]
        group_sum = float(np.sum(group_residuals))
        quadratic += float(
            np.sum(group_residuals**2) / residual_variance
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
    log_bounds: tuple[tuple[float, float], tuple[float, float]],
) -> None:
    """Reject an optimizer result that cannot support a release artifact."""
    if not bool(getattr(optimizer_result, "success", False)):
        raise ValueError(
            "REML optimizer did not converge: "
            f"{getattr(optimizer_result, 'message', 'unknown failure')}"
        )
    status = getattr(optimizer_result, "status", None)
    if isinstance(status, bool) or not isinstance(status, (int, np.integer)):
        raise ValueError("REML optimizer returned an invalid status")
    iterations = getattr(optimizer_result, "nit", None)
    evaluations = getattr(optimizer_result, "nfev", None)
    if (
        isinstance(iterations, bool)
        or not isinstance(iterations, (int, np.integer))
        or iterations < 0
        or isinstance(evaluations, bool)
        or not isinstance(evaluations, (int, np.integer))
        or evaluations < 0
    ):
        raise ValueError(
            "REML optimizer returned invalid iteration diagnostics"
        )
    point = np.asarray(getattr(optimizer_result, "x", ()), dtype=float)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError("REML optimizer returned invalid variance components")
    for value, bounds in zip(point, log_bounds, strict=True):
        if value < bounds[0] or value > bounds[1]:
            raise ValueError("REML optimizer returned out-of-bounds components")
    objective = float(getattr(optimizer_result, "fun", float("nan")))
    if not math.isfinite(objective):
        raise ValueError("REML optimizer returned an invalid objective")
    gradient = np.asarray(getattr(optimizer_result, "jac", ()), dtype=float)
    if gradient.shape != (2,) or not np.isfinite(gradient).all():
        raise ValueError("REML optimizer returned invalid gradient diagnostics")


def hierarchical_gls_system(
    design: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    labels: tuple[Any, ...],
    residual_variance: float,
    random_intercept_variance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the block-diagonal precision system for random intercepts."""
    information = np.zeros((design.shape[1], design.shape[1]))
    right_hand_side = np.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        group_design = design[group]
        group_values = values[group]
        count = group_design.shape[0]
        precision = np.eye(count) / residual_variance
        if random_intercept_variance > 0.0:
            precision -= (
                random_intercept_variance
                / (
                    residual_variance
                    * (residual_variance + count * random_intercept_variance)
                )
            ) * np.ones((count, count))
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ group_values
    return information, right_hand_side


def fit_logistic(
    design: np.ndarray,
    values: np.ndarray,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a Bernoulli likelihood with deterministic Newton iterations."""
    coefficients = np.zeros(design.shape[1], dtype=float)
    for iteration in range(1, max_iterations + 1):
        probabilities = special.expit(design @ coefficients)
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        hessian = design.T @ (weights[:, None] * design)
        try:
            step = np.linalg.solve(hessian, design.T @ (values - probabilities))
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "logistic design matrix is not identifiable"
            ) from error
        coefficients += step
        if np.linalg.norm(step) <= tolerance * (
            1.0 + np.linalg.norm(coefficients)
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
    design: np.ndarray,
    residuals: np.ndarray,
    covariance: CovarianceKind,
    clusters: np.ndarray | None,
    probabilities: np.ndarray | None,
) -> np.ndarray:
    """Compute iid, HC3, or cluster-robust sandwich covariance."""
    if probabilities is None:
        bread = np.linalg.inv(design.T @ design)
        scale = float(
            np.sum(residuals**2) / max(len(residuals) - design.shape[1], 1)
        )
        if covariance == "iid":
            return scale * bread
        leverage = np.sum((design @ bread) * design, axis=1)
        adjusted = residuals / np.maximum(1.0 - leverage, 1e-12)
    else:
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-12)
        bread = np.linalg.inv(design.T @ (weights[:, None] * design))
        if covariance == "iid":
            return bread
        leverage = np.sum(
            (design @ bread) * (weights[:, None] * design), axis=1
        )
        adjusted = residuals / np.maximum(1.0 - leverage, 1e-12)
    if covariance == "hc3":
        scores = design * adjusted[:, None]
        return bread @ (scores.T @ scores) @ bread
    if clusters is None:
        raise ValueError("cluster labels are required")
    adjusted = residuals
    groups = []
    for label in dict.fromkeys(clusters.tolist()):
        group = clusters == label
        groups.append(design[group].T @ adjusted[group])
    if len(groups) < 2:
        raise ValueError("clustered covariance needs at least two clusters")
    meat = sum(np.outer(score, score) for score in groups)
    correction = (len(groups) / (len(groups) - 1)) * (
        (len(design) - 1) / max(len(design) - design.shape[1], 1)
    )
    return correction * bread @ meat @ bread


def reference_linear(
    design: np.ndarray,
    values: np.ndarray,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
) -> dict[str, Any]:
    """Independently verify the linear solution with least squares."""
    reference = np.linalg.lstsq(design, values, rcond=None)[0]
    difference = float(np.max(np.abs(reference - coefficients)))
    return {
        "method": "numpy.linalg.lstsq",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_ipw_ate(
    propensity_design: np.ndarray,
    values: np.ndarray,
    treatment: np.ndarray,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
    clusters: np.ndarray | None,
) -> dict[str, Any]:
    """Recompute IPW propensities with an independent BFGS calculation."""

    def objective(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        logits = propensity_design @ candidate
        probabilities = special.expit(logits)
        value = -float(np.sum(treatment * logits - np.logaddexp(0.0, logits)))
        gradient = propensity_design.T @ (probabilities - treatment)
        return value, gradient

    result = optimize.minimize(
        lambda candidate: objective(candidate)[0],
        np.zeros(propensity_design.shape[1]),
        jac=lambda candidate: objective(candidate)[1],
        method="BFGS",
        options={"gtol": policy.reference_tolerance, "maxiter": 1000},
    )
    reference_propensity = np.clip(
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
    inputs: Mapping[str, np.ndarray],
    values: np.ndarray,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
    clusters: np.ndarray | None,
) -> dict[str, Any]:
    """Recompute the augmented IPW score with an independent reference."""
    if policy.cross_fitting_folds > 1:
        return reference_cross_fitted_aipw_ate(
            inputs, values, coefficients, policy, clusters
        )
    nuisance_design = inputs["nuisance_design"]
    treatment = inputs["treatment"]
    confounder_design = nuisance_design

    def objective(candidate: np.ndarray) -> tuple[float, np.ndarray]:
        logits = confounder_design @ candidate
        probabilities = special.expit(logits)
        value = -float(np.sum(treatment * logits - np.logaddexp(0.0, logits)))
        gradient = confounder_design.T @ (probabilities - treatment)
        return value, gradient

    propensity_fit = optimize.minimize(
        lambda candidate: objective(candidate)[0],
        np.zeros(confounder_design.shape[1]),
        jac=lambda candidate: objective(candidate)[1],
        method="BFGS",
        options={"gtol": policy.reference_tolerance, "maxiter": 1000},
    )
    propensity = np.clip(
        special.expit(confounder_design @ propensity_fit.x),
        policy.propensity_clip,
        1.0 - policy.propensity_clip,
    )
    treated = treatment == 1.0
    control = ~treated
    treated_coefficients = np.linalg.lstsq(
        nuisance_design[treated], values[treated], rcond=None
    )[0]
    control_coefficients = np.linalg.lstsq(
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
    reference_ate = float(np.mean(score))
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
    inputs: Mapping[str, np.ndarray],
    values: np.ndarray,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
    clusters: np.ndarray | None,
) -> dict[str, Any]:
    """Independently recompute each cross-fitted AIPW validation score."""
    nuisance_design = inputs["nuisance_design"]
    treatment = inputs["treatment"]
    fold_assignments = np.asarray(inputs["fold_assignments"], dtype=int)
    if len(fold_assignments) != len(values):
        raise ValueError("cross-fitting reference fold assignments are invalid")
    reference_scores = np.empty(len(values), dtype=float)
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
            candidate: np.ndarray,
            design: np.ndarray = training_design,
            observed: np.ndarray = training_treatment,
        ) -> tuple[float, np.ndarray]:
            logits = design @ candidate
            probabilities = special.expit(logits)
            value = -float(
                np.sum(observed * logits - np.logaddexp(0.0, logits))
            )
            gradient = design.T @ (probabilities - observed)
            return value, gradient

        propensity_fit = optimize.minimize(
            lambda candidate: objective(candidate)[0],
            np.zeros(training_design.shape[1]),
            jac=lambda candidate: objective(candidate)[1],
            method="BFGS",
            options={"gtol": policy.reference_tolerance, "maxiter": 1000},
        )
        propensity = np.clip(
            special.expit(validation_design @ propensity_fit.x),
            policy.propensity_clip,
            1.0 - policy.propensity_clip,
        )
        treated_design = training_design[treated]
        control_design = training_design[control]
        if (
            np.linalg.matrix_rank(treated_design) != treated_design.shape[1]
            or np.linalg.matrix_rank(control_design) != control_design.shape[1]
        ):
            raise ValueError(
                "aipw_ate reference outcome design is not identifiable"
            )
        treated_coefficients = np.linalg.lstsq(
            treated_design, training_values[treated], rcond=None
        )[0]
        control_coefficients = np.linalg.lstsq(
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
    reference_ate = float(np.mean(reference_scores))
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
    design: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    residual_variance: float,
    random_intercept_variance: float,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
) -> dict[str, Any]:
    """Verify hierarchical GLS with explicit group covariance matrices."""
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    information = np.zeros((design.shape[1], design.shape[1]))
    right_hand_side = np.zeros(design.shape[1])
    for label in labels:
        group = clusters == label
        count = int(np.sum(group))
        covariance = residual_variance * np.eye(count) + (
            random_intercept_variance * np.ones((count, count))
        )
        precision = linalg.solve(covariance, np.eye(count), assume_a="pos")
        group_design = design[group]
        information += group_design.T @ precision @ group_design
        right_hand_side += group_design.T @ precision @ values[group]
    reference = linalg.solve(information, right_hand_side, assume_a="pos")
    difference = float(np.max(np.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:explicit_random_intercept_covariance",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_hierarchical_reml(
    design: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    residual_variance: float,
    random_intercept_variance: float,
    coefficients: np.ndarray,
    objective: float,
    policy: EstimationPolicy,
) -> dict[str, Any]:
    """Verify REML GLS with an independent dense covariance calculation."""
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable"
        ) from error
    covariance_blocks = []
    for label in labels:
        count = int(np.sum(clusters == label))
        covariance_blocks.append(
            residual_variance * np.eye(count)
            + random_intercept_variance * np.ones((count, count))
        )
    covariance = linalg.block_diag(*covariance_blocks)
    precision = linalg.solve(covariance, np.eye(len(values)), assume_a="pos")
    information = design.T @ precision @ design
    right_hand_side = design.T @ precision @ values
    reference_coefficients = linalg.solve(
        information, right_hand_side, assume_a="pos"
    )
    residuals = values - design @ reference_coefficients
    sign_covariance, logdet_covariance = np.linalg.slogdet(covariance)
    sign_information, logdet_information = np.linalg.slogdet(information)
    if sign_covariance <= 0.0 or sign_information <= 0.0:
        raise ValueError("reference REML covariance is not positive definite")
    reference_objective = 0.5 * (
        logdet_covariance
        + logdet_information
        + float(residuals @ precision @ residuals)
        + (len(values) - design.shape[1]) * math.log(2.0 * math.pi)
    )
    coefficient_difference = float(
        np.max(np.abs(reference_coefficients - coefficients))
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


def reference_bayesian_linear(
    design: np.ndarray,
    values: np.ndarray,
    prior: BayesianPrior,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
) -> dict[str, Any]:
    """Verify the conjugate posterior mean with an independent solver."""
    prior_mean = np.asarray(prior.mean, dtype=float)
    prior_precision = np.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    reference = linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + design.T @ values,
        assume_a="pos",
    )
    difference = float(np.max(np.abs(reference - coefficients)))
    return {
        "method": "scipy.linalg.solve:positive_definite",
        "max_absolute_difference": difference,
        "passed": difference <= policy.reference_tolerance,
    }


def reference_bayesian_ate(
    design: np.ndarray,
    values: np.ndarray,
    prior: BayesianPrior,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
) -> dict[str, Any]:
    """Independently verify the causal posterior mean and ATE coefficient."""
    validate_bayesian_prior_shape(prior, design.shape[1])
    prior_mean = np.asarray(prior.mean, dtype=float)
    prior_precision = np.asarray(prior.precision, dtype=float)
    posterior_precision = prior_precision + design.T @ design
    reference = linalg.solve(
        posterior_precision,
        prior_precision @ prior_mean + design.T @ values,
        assume_a="pos",
    )
    difference = float(np.max(np.abs(reference - coefficients)))
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
    design: np.ndarray,
    values: np.ndarray,
    coefficients: np.ndarray,
    policy: EstimationPolicy,
) -> dict[str, Any]:
    """Independently verify logistic coefficients with scipy optimization."""

    def objective(candidate: np.ndarray) -> tuple[float, np.ndarray]:
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
    difference = float(np.max(np.abs(result.x - coefficients)))
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
    values: np.ndarray, probabilities: np.ndarray
) -> float:
    """Return a numerically stable Bernoulli log likelihood."""
    logits = np.log(np.maximum(probabilities, 1e-15))
    complements = np.log(np.maximum(1.0 - probabilities, 1e-15))
    return float(np.sum(values * logits + (1.0 - values) * complements))


def sensitivity_report(
    matrix: np.ndarray,
    values: np.ndarray,
    policy: EstimationPolicy,
    design: np.ndarray,
    coefficients: np.ndarray,
    shift_reference: Any | None,
    estimand: EstimandSpec,
    clusters: np.ndarray | None,
    prior: BayesianPrior | None = None,
) -> dict[str, Any]:
    """Record declared sensitivity coverage without inventing causal claims."""
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
        validate_bayesian_prior_shape(prior, design.shape[1])
        baseline = np.asarray(coefficients, dtype=float)
        alternatives = []
        prior_mean = np.asarray(prior.mean, dtype=float)
        prior_precision = np.asarray(prior.precision, dtype=float)
        for label, precision_factor in (
            ("weakened_precision", 0.01),
            ("strengthened_precision", 100.0),
        ):
            alternative_precision = prior_precision * precision_factor
            alternative = np.linalg.solve(
                alternative_precision + design.T @ design,
                alternative_precision @ prior_mean + design.T @ values,
            )
            alternative_record = {
                "name": label,
                "max_absolute_coefficient_change": float(
                    np.max(np.abs(alternative - baseline))
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
        report["prior"] = {
            "status": "evaluated",
            "method": "precision_scale_sensitivity",
            "alternatives": alternatives,
        }
    if shift_reference is not None:
        reference = np.asarray(shift_reference, dtype=float)
        validate_matrix(reference, matrix.shape[1], "shift_reference")
        if not np.isfinite(reference).all():
            raise ValueError("shift_reference must contain only finite values")
        pooled = np.sqrt((matrix.var(axis=0) + reference.var(axis=0)) / 2.0)
        difference = np.abs(reference.mean(axis=0) - matrix.mean(axis=0)) / (
            np.maximum(pooled, 1e-12)
        )
        report["distribution_shift"] = {
            "status": "evaluated",
            "standardized_mean_difference": [
                float(item) for item in difference
            ],
            "max_standardized_mean_difference": float(np.max(difference)),
        }
    if policy.bootstrap_replicates and policy.estimator in {
        "ipw_ate",
        "aipw_ate",
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
        rng = np.random.default_rng(policy.seed)
        estimates: list[np.ndarray] = []
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
        bootstrap = np.asarray(estimates)
        alpha = (1.0 - policy.confidence_level) / 2.0
        report["sampling"]["bootstrap_intervals"] = [
            [float(item) for item in np.quantile(bootstrap, alpha, axis=0)],
            [
                float(item)
                for item in np.quantile(bootstrap, 1.0 - alpha, axis=0)
            ],
        ]
    return report


def hierarchical_bootstrap_sensitivity(
    matrix: np.ndarray,
    values: np.ndarray,
    estimand: EstimandSpec,
    policy: EstimationPolicy,
    coefficients: np.ndarray,
    clusters: np.ndarray,
) -> dict[str, Any]:
    """Estimate grouped-parameter sensitivity by resampling whole clusters.

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
        ValueError: If fewer than two reference-checked replicates succeed.
    """
    rng = np.random.default_rng(policy.seed)
    bootstrap_policy = dataclasses.replace(policy, bootstrap_replicates=0)
    estimates: list[np.ndarray] = []
    failures: list[dict[str, Any]] = []
    draws: list[dict[str, Any]] = []
    limitations = [
        "variance-component uncertainty is not propagated",
        "random slopes are unsupported",
        "nested effects are unsupported",
        "crossed effects are unsupported",
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
            result = estimate_parameters(
                sampled_matrix,
                sampled_values,
                estimand,
                bootstrap_policy,
                clusters=sampled_clusters.tolist(),
            )
            estimate = np.asarray(
                [parameter.estimate for parameter in result.parameters],
                dtype=float,
            )
            if estimate.shape != np.asarray(coefficients).shape:
                raise ValueError(
                    "bootstrap replicate returned an unexpected parameter shape"
                )
            if not np.isfinite(estimate).all():
                raise ValueError(
                    "bootstrap replicate returned non-finite estimates"
                )
            if result.reference.get("passed") is not True:
                raise ValueError(
                    "bootstrap replicate independent reference check failed"
                )
            estimates.append(estimate)
        except (
            FloatingPointError,
            np.linalg.LinAlgError,
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

    if len(estimates) < 2:
        raise ValueError(
            "hierarchical bootstrap needs at least two successful replicates; "
            f"received {len(estimates)} of {policy.bootstrap_replicates}"
        )

    bootstrap = np.asarray(estimates)
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
            [float(item) for item in np.quantile(bootstrap, alpha, axis=0)],
            [
                float(item)
                for item in np.quantile(bootstrap, 1.0 - alpha, axis=0)
            ],
        ],
    }


def draw_hierarchical_bootstrap_sample(
    matrix: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Draw whole clusters and assign fresh identities to selected copies.

    Args:
        matrix: Complete feature rows from the source estimate.
        values: Complete outcomes from the source estimate.
        clusters: Source cluster labels.
        rng: Seeded random generator owned by the sensitivity report.

    Returns:
        Resampled rows, outcomes, fresh cluster labels, and digest evidence.

    Raises:
        ValueError: If the source data contains fewer than two clusters.
    """
    try:
        labels = tuple(dict.fromkeys(clusters.tolist()))
    except TypeError as error:
        raise ValueError(
            "hierarchical cluster labels must be hashable for bootstrap"
        ) from error
    if len(labels) < 2:
        raise ValueError(
            "hierarchical bootstrap needs at least two source clusters"
        )
    source_indices = rng.integers(0, len(labels), size=len(labels))
    row_groups = [
        np.flatnonzero(clusters == labels[index]) for index in source_indices
    ]
    indices = np.concatenate(row_groups)
    sampled_clusters = np.repeat(
        np.arange(len(row_groups), dtype=object),
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
                np.arange(len(row_groups), dtype=np.int64)
            ),
        },
    )


def causal_bootstrap_sensitivity(
    matrix: np.ndarray,
    values: np.ndarray,
    estimand: EstimandSpec,
    policy: EstimationPolicy,
    coefficients: np.ndarray,
    clusters: np.ndarray | None,
) -> dict[str, Any]:
    """Estimate causal sampling sensitivity with nonparametric bootstrap.

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
            independently reference-checked replicates.
    """
    rng = np.random.default_rng(policy.seed)
    bootstrap_policy = dataclasses.replace(policy, bootstrap_replicates=0)
    estimates: list[np.ndarray] = []
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
            result = estimate_parameters(
                sampled_matrix,
                sampled_values,
                estimand,
                bootstrap_policy,
                clusters=bootstrap_clusters,
            )
            estimate = np.asarray(
                [parameter.estimate for parameter in result.parameters],
                dtype=float,
            )
            if estimate.shape != np.asarray(coefficients).shape:
                raise ValueError(
                    "bootstrap replicate returned an unexpected parameter shape"
                )
            if not np.isfinite(estimate).all():
                raise ValueError(
                    "bootstrap replicate returned non-finite estimates"
                )
            if result.reference.get("passed") is not True:
                raise ValueError(
                    "bootstrap replicate independent reference check failed"
                )
            estimates.append(estimate)
        except (
            FloatingPointError,
            np.linalg.LinAlgError,
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

    if len(estimates) < 2:
        raise ValueError(
            "causal bootstrap needs at least two successful replicates; "
            f"received {len(estimates)} of {policy.bootstrap_replicates}"
        )

    bootstrap = np.asarray(estimates)
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
            [float(item) for item in np.quantile(bootstrap, alpha, axis=0)],
            [
                float(item)
                for item in np.quantile(bootstrap, 1.0 - alpha, axis=0)
            ],
        ],
    }


def draw_causal_bootstrap_sample(
    matrix: np.ndarray,
    values: np.ndarray,
    clusters: np.ndarray | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, dict[str, Any]]:
    """Draw one causal bootstrap sample and its auditable unit evidence."""
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
    if len(labels) < 2:
        raise ValueError(
            "clustered bootstrap needs at least two source clusters"
        )
    source_indices = rng.integers(0, len(labels), size=len(labels))
    row_groups = [
        np.flatnonzero(clusters == labels[index]) for index in source_indices
    ]
    indices = np.concatenate(row_groups)
    sampled_clusters = np.repeat(
        np.arange(len(row_groups), dtype=object),
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


def bootstrap_sample_digest(indices: np.ndarray) -> str:
    """Return a digest of bootstrap draw indices without exposing raw rows."""
    canonical = np.asarray(indices, dtype="<i8").tobytes()
    return hashlib.sha256(canonical).hexdigest()


__all__ = [
    "BayesianPrior",
    "CausalAssumption",
    "CovarianceKind",
    "EstimandSpec",
    "EstimationPolicy",
    "EstimationResult",
    "EstimatorKind",
    "HierarchicalAssumption",
    "MissingPolicy",
    "ParameterEstimate",
    "estimate_parameters",
]
