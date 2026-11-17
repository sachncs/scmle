"""Linear and logistic likelihoods and their sandwich covariance.

Calls written `parameter_estimation.<name>(...)` look the function up on
the package when they run, so a test that patches the package attribute
reaches them as it did when the package was one module.
"""

from typing import Any

import numpy
from scipy import optimize
from scipy import special
from scipy import stats

from scmle.contracts import errors
from scmle.evaluation import parameter_estimation
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import CovarianceKind
from scmle.evaluation.parameter_estimation.declaration import MIN_CLUSTERS
from scmle.evaluation.parameter_estimation.report import Fit


def ols(
    design: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    cluster_values: numpy.ndarray | None,
    names: tuple[str, ...],
) -> Fit:
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
    coefficients, diagnostics = regress(
        design, values, policy.max_iterations, policy.tolerance
    )
    fitted = design @ coefficients
    residuals = values - fitted
    covariance = parameter_estimation.covariance_matrix(  # type: ignore[attr-defined]
        design, residuals, policy.covariance, cluster_values, None
    )
    reference_coefficients = numpy.linalg.lstsq(design, values, rcond=None)[0]
    reference_difference = float(
        numpy.max(numpy.abs(reference_coefficients - coefficients))
    )
    reference = {
        "method": "numpy.linalg.lstsq",
        "max_absolute_difference": reference_difference,
        "passed": reference_difference <= policy.reference_tolerance,
    }
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
    return Fit(
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
) -> Fit:
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
    coefficients, probabilities, diagnostics = logit(
        design, values, policy.max_iterations, policy.tolerance
    )
    residuals = values - probabilities
    covariance = parameter_estimation.covariance_matrix(  # type: ignore[attr-defined]
        design, residuals, policy.covariance, cluster_values, probabilities
    )
    reference = corroborate(design, values, coefficients, policy)
    likelihood = "bernoulli"
    pivot = stats.norm.ppf(0.5 + policy.confidence_level / 2.0)
    p_values = 2.0 * stats.norm.sf(
        numpy.abs(
            coefficients
            / numpy.maximum(numpy.sqrt(numpy.diag(covariance)), 1e-15)
        )
    )
    diagnostics = {**diagnostics, "likelihood": likelihood}
    return Fit(
        names,
        coefficients,
        covariance,
        diagnostics,
        reference,
        likelihood,
        pivot,
        p_values,
    )


def regress(
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
        raise errors.InputError("linear design matrix is rank deficient")
    coefficients = numpy.linalg.solve(gram, design.T @ values)
    return coefficients, {"converged": True, "iterations": 1}


def logit(
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
            raise errors.InputError(
                "logistic design matrix is not identifiable"
            ) from error
        coefficients += step
        if numpy.linalg.norm(step) <= tolerance * (
            1.0 + numpy.linalg.norm(coefficients)
        ):
            probabilities = special.expit(design @ coefficients)
            log_likelihood = loglik(values, probabilities)
            return (
                coefficients,
                probabilities,
                {
                    "converged": True,
                    "iterations": iteration,
                    "log_likelihood": log_likelihood,
                },
            )
    raise errors.InputError("logistic estimation did not converge")


def loglik(values: numpy.ndarray, probabilities: numpy.ndarray) -> float:
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


def corroborate(
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
        value = -loglik(values, probabilities)
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
        raise errors.InputError("cluster labels are required")
    adjusted = residuals
    groups = []
    for label in dict.fromkeys(clusters.tolist()):
        group = clusters == label
        groups.append(design[group].T @ adjusted[group])
    if len(groups) < MIN_CLUSTERS:
        raise errors.InputError(
            "clustered covariance needs at least two clusters"
        )
    meat = sum(numpy.outer(score, score) for score in groups)
    correction = (len(groups) / (len(groups) - 1)) * (
        (len(design) - 1) / max(len(design) - design.shape[1], 1)
    )
    sandwich: numpy.ndarray = correction * bread @ meat @ bread
    return sandwich
