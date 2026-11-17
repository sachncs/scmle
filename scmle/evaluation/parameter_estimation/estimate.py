"""The estimator dispatch and the public estimation entry point.

Calls written `parameter_estimation.<name>(...)` look the function up on
the package when they run, so a test that patches the package attribute
reaches them as it did when the package was one module.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
from typing import Any

import numpy

from scmle.contracts import errors
from scmle.evaluation import parameter_estimation
from scmle.evaluation.envelope import Counts
from scmle.evaluation.parameter_estimation.bayesian import bayes
from scmle.evaluation.parameter_estimation.causal import (
    fit_propensity_estimator,
)
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.continuous import (
    fit_continuous_estimator,
)
from scmle.evaluation.parameter_estimation.declaration import Estimand
from scmle.evaluation.parameter_estimation.declaration import ParameterEstimate
from scmle.evaluation.parameter_estimation.hierarchical import (
    fit_hierarchical_estimator,
)
from scmle.evaluation.parameter_estimation.inputs import admit
from scmle.evaluation.parameter_estimation.inputs import clustering
from scmle.evaluation.parameter_estimation.inputs import condition
from scmle.evaluation.parameter_estimation.inputs import prepare
from scmle.evaluation.parameter_estimation.inputs import raw_input_fingerprint
from scmle.evaluation.parameter_estimation.inputs import screen
from scmle.evaluation.parameter_estimation.inputs import validate_clusters
from scmle.evaluation.parameter_estimation.inputs import validate_matrix
from scmle.evaluation.parameter_estimation.linear import fit_logistic_estimator
from scmle.evaluation.parameter_estimation.linear import ols
from scmle.evaluation.parameter_estimation.report import Estimation
from scmle.evaluation.parameter_estimation.report import Fit
from scmle.evaluation.parameter_estimation.sensitivity import sensitivity_report


def compute(
    matrix: numpy.ndarray,
    design: numpy.ndarray,
    values: numpy.ndarray,
    estimand: Estimand,
    policy: Config,
    cluster_values: numpy.ndarray | None,
) -> Fit:
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
        return ols(design, values, policy, cluster_values, names)
    if policy.estimator == "logistic":
        return fit_logistic_estimator(
            design, values, policy, cluster_values, names
        )
    if policy.estimator in {"bayesian_linear", "bayesian_ate"}:
        return bayes(matrix, design, values, policy, estimand, names)
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
    matrix, values, cluster_values, design, input_rows, dropped, fingerprint = (
        ingest(features, outcome, estimand, policy, clusters)
    )
    fit = parameter_estimation.compute(  # type: ignore[attr-defined]
        matrix, design, values, estimand, policy, cluster_values
    )
    diagnostics, parameters = summarise(fit, policy, cluster_values)
    sensitivity = sensitivity_report(
        matrix,
        values,
        policy,
        design,
        fit.coefficients,
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
            tuple(float(value) for value in row) for row in fit.covariance
        ),
        rows=Counts(int(input_rows), len(values), int(dropped)),
        diagnostics=diagnostics,
        sensitivity=sensitivity,
        reference=fit.reference,
        digests={"frame": fingerprint},
    )


def ingest(
    features: Any,
    outcome: Any,
    estimand: Estimand,
    policy: Config,
    clusters: Sequence[Any] | None,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray | None,
    numpy.ndarray,
    int,
    int,
    str,
]:
    """Checks the raw inputs, fingerprints them and keeps the usable rows.

    Args:
        features: Numeric matrix with one column per estimand feature.
        outcome: Numeric response.
        estimand: Scientific definition and identification assumptions.
        policy: Estimator, covariance and missingness policy.
        clusters: Optional cluster labels, one per input row.

    Returns:
        The kept feature rows, the kept outcomes, the kept cluster labels
        (None without clusters), the design matrix with an intercept column,
        the number of input rows, the number of dropped rows, and the
        fingerprint of the raw inputs.

    Raises:
        ValueError: If the outcome is not one-dimensional; the matrix,
            cluster, policy and missing-row checks raise their own errors.
    """
    matrix = numpy.asarray(features, dtype=float)
    values = numpy.asarray(outcome, dtype=float)
    validate_matrix(matrix, len(estimand.feature_names), "features")
    if values.ndim != 1:
        raise errors.InputError("outcome must have shape (rows,)")
    input_clusters = validate_clusters(clusters, len(values))
    fingerprint = raw_input_fingerprint(matrix, values, input_clusters)
    input_rows = matrix.shape[0]
    admit(matrix, values, input_clusters, policy, estimand)
    matrix, values, kept, dropped = prepare(
        matrix, values, policy.missing, input_clusters
    )
    screen(matrix, values, clusters, policy)
    cluster_values = kept if input_clusters is not None else None
    design = numpy.column_stack((numpy.ones(matrix.shape[0]), matrix))
    return (
        matrix,
        values,
        cluster_values,
        design,
        input_rows,
        dropped,
        fingerprint,
    )


def summarise(
    fit: Fit, policy: Config, cluster_values: numpy.ndarray | None
) -> tuple[Mapping[str, Any], tuple[ParameterEstimate, ...]]:
    """Adds the covariance evidence to a fit and builds its coefficients.

    Args:
        fit: The fitted quantities of the chosen estimator.
        policy: Estimator policy; clustered covariance adds cluster evidence.
        cluster_values: Cluster labels of the kept rows, or None.

    Returns:
        The diagnostics with covariance conditioning (and cluster
        uncertainty for clustered covariance) and one estimate per
        parameter, with intervals scaled by the fit's pivot.

    Raises:
        ValueError: If clustered covariance has no cluster labels; the
            covariance and cluster checks raise their own errors.
    """
    covariance_conditioning = condition(fit.covariance, len(fit.names))
    diagnostics = {
        **fit.diagnostics,
        "covariance_conditioning": covariance_conditioning,
    }
    if policy.covariance == "clustered":
        if cluster_values is None:
            raise errors.InputError(
                "cluster labels are required for cluster diagnostics"
            )
        diagnostics["cluster_uncertainty"] = clustering(cluster_values)
    if fit.standard_errors is None:
        standard_errors = numpy.sqrt(
            numpy.maximum(numpy.diag(fit.covariance), 0.0)
        )
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
            fit.names,
            fit.coefficients,
            standard_errors,
            interval_errors,
            fit.p_values,
            strict=True,
        )
    )
    return diagnostics, parameters
