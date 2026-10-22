"""Reproducible frequentist parameter estimation with uncertainty evidence.

This module answers a different question from model selection: what is the
estimated value of a declared population parameter, and how uncertain is that
estimate under an explicit likelihood and covariance policy?  It intentionally
starts with transparent linear and logistic likelihoods.  More specialised
estimators can implement the same result contract without changing the
artifact or review surface.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any
from typing import Literal

import numpy as np
from scipy import optimize
from scipy import special
from scipy import stats

EstimatorKind = Literal["linear", "logistic"]
CovarianceKind = Literal["iid", "hc3", "clustered"]
MissingPolicy = Literal["reject", "complete_case"]


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

    def __post_init__(self) -> None:
        """Validate the policy before data is touched."""
        if self.estimator not in ("linear", "logistic"):
            raise ValueError("estimator must be linear or logistic")
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

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible policy metadata."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class ParameterEstimate:
    """One coefficient estimate and its declared uncertainty."""

    name: str
    estimate: float
    standard_error: float
    interval: tuple[float, float]
    p_value: float

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
            assert temporary is not None  # for the type checker
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
        report = cls.from_dict(envelope["report"])
        if envelope.get("digest") != report.digest():
            raise ValueError("estimation report digest mismatch")
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
        return cls(
            EstimandSpec(**estimand_values),
            EstimationPolicy(**policy_record),
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
        )


def estimate_parameters(
    features: Any,
    outcome: Any,
    estimand: EstimandSpec,
    policy: EstimationPolicy,
    clusters: Sequence[Any] | None = None,
    shift_reference: Any | None = None,
) -> EstimationResult:
    """Estimate declared linear or logistic parameters with evidence.

    Args:
        features: Numeric matrix with one column per ``estimand.feature_names``.
        outcome: Numeric response, or binary labels for logistic estimation.
        estimand: Scientific definition and identification assumptions.
        policy: Estimator, covariance, missingness, and uncertainty policy.
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
    values = np.asarray(outcome, dtype=float).reshape(-1)
    validate_matrix(matrix, len(estimand.feature_names), "features")
    input_rows = matrix.shape[0]
    if matrix.shape[0] != len(values):
        raise ValueError("features and outcome must have the same row count")
    if clusters is not None and len(clusters) != len(values):
        raise ValueError("clusters must have the same row count as outcome")
    if policy.covariance == "clustered" and clusters is None:
        raise ValueError("clustered covariance needs cluster labels")
    matrix, values, kept, dropped = prepare_rows(
        matrix, values, policy.missing, clusters
    )
    if policy.estimator == "logistic" and not set(np.unique(values)).issubset(
        {0.0, 1.0}
    ):
        raise ValueError("logistic outcome must contain only 0 and 1")
    cluster_values = kept if clusters is not None else None
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
    else:
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

    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    parameters = tuple(
        ParameterEstimate(
            name,
            float(value),
            float(error),
            (float(value - pivot * error), float(value + pivot * error)),
            float(p_value),
        )
        for name, value, error, p_value in zip(
            names, coefficients, standard_errors, p_values, strict=True
        )
    )
    sensitivity = sensitivity_report(
        matrix,
        values,
        policy,
        design,
        coefficients,
        shift_reference,
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
    )


def validate_matrix(matrix: np.ndarray, columns: int, name: str) -> None:
    """Validate a numeric two-dimensional matrix."""
    if matrix.ndim != 2 or matrix.shape[1] != columns:
        raise ValueError(f"{name} must have shape (rows, {columns})")
    if matrix.shape[0] <= columns:
        raise ValueError("estimation needs more rows than parameters")


def prepare_rows(
    matrix: np.ndarray,
    values: np.ndarray,
    missing: MissingPolicy,
    clusters: Sequence[Any] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, int]:
    """Apply the declared missingness policy and retain cluster labels."""
    valid = np.isfinite(matrix).all(axis=1) & np.isfinite(values)
    dropped = int((~valid).sum())
    if dropped and missing == "reject":
        raise ValueError(
            "missing or non-finite values require an explicit policy"
        )
    if valid.sum() <= matrix.shape[1] + 1:
        raise ValueError("complete rows are insufficient for estimation")
    kept_clusters = None
    if clusters is not None:
        cluster_array = np.asarray(clusters, dtype=object)
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
    if shift_reference is not None:
        reference = np.asarray(shift_reference, dtype=float)
        validate_matrix(reference, matrix.shape[1], "shift_reference")
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
    if policy.bootstrap_replicates:
        rng = np.random.default_rng(policy.seed)
        estimates: list[np.ndarray] = []
        for _ in range(policy.bootstrap_replicates):
            sample = rng.integers(0, len(values), size=len(values))
            if policy.estimator == "linear":
                estimate, _ = fit_linear(
                    design[sample], values[sample], 1, 1e-8
                )
            else:
                estimate, _, _ = fit_logistic(
                    design[sample],
                    values[sample],
                    policy.max_iterations,
                    policy.tolerance,
                )
            estimates.append(estimate)
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


__all__ = [
    "CovarianceKind",
    "EstimandSpec",
    "EstimationPolicy",
    "EstimationResult",
    "EstimatorKind",
    "MissingPolicy",
    "ParameterEstimate",
    "estimate_parameters",
]
