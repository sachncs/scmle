"""Independent reference calculations for the IPW and AIPW effects."""

from collections.abc import Mapping
from typing import Any

import numpy
from scipy import optimize
from scipy import special

from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.weighting import aipw
from scmle.evaluation.parameter_estimation.weighting import dispersion
from scmle.evaluation.parameter_estimation.weighting import hajek
from scmle.evaluation.parameter_estimation.weighting import outcomes


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
    result = bfgs(propensity_design, treatment, policy)
    reference_propensity = numpy.clip(
        special.expit(propensity_design @ result.x),
        policy.clip,
        1.0 - policy.clip,
    )
    (
        reference_ate,
        reference_variance,
        reference_evidence,
    ) = hajek(
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
        "passed": bool(result.success and difference <= policy.reference * 10),
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
    if policy.folds > 1:
        return reference_cross_fitted_aipw_ate(
            inputs, values, coefficients, policy, clusters
        )
    nuisance_design = inputs["nuisance_design"]
    treatment = inputs["treatment"]
    propensity_fit = bfgs(nuisance_design, treatment, policy)
    propensity = numpy.clip(
        special.expit(nuisance_design @ propensity_fit.x),
        policy.clip,
        1.0 - policy.clip,
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
    score = aipw(
        outcome_treated, outcome_control, treatment, values, propensity
    )
    reference_ate, reference_variance, variance_method, difference = verdict(
        score, coefficients, policy, clusters
    )
    return {
        "method": "scipy.optimize.minimize:BFGS_plus_lstsq",
        "converged": bool(propensity_fit.success),
        "reference_ate": reference_ate,
        "max_absolute_difference": difference,
        "variance": reference_variance,
        "variance_method": variance_method,
        "variance_status": (
            "declared_influence_calculation_not_independently_verified"
        ),
        "passed": bool(
            propensity_fit.success and difference <= policy.reference * 10
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
    fold_assignments = numpy.asarray(inputs["fold_assignments"], dtype=int)
    if len(fold_assignments) != len(values):
        raise ValueError("cross-fitting reference fold assignments are invalid")
    reference_scores = numpy.empty(len(values), dtype=float)
    fold_records = []
    for fold in range(policy.folds):
        validation = fold_assignments == fold
        reference_scores[validation], fold_record = refold(
            inputs, values, policy, fold_assignments != fold, fold
        )
        fold_records.append(fold_record)
    reference_ate, reference_variance, variance_method, difference = verdict(
        reference_scores, coefficients, policy, clusters
    )
    converged = all(record["converged"] for record in fold_records)
    return {
        "method": "scipy.optimize.minimize:BFGS_plus_lstsq_per_training_fold",
        "converged": converged,
        "reference_ate": reference_ate,
        "max_absolute_difference": difference,
        "variance": reference_variance,
        "variance_method": variance_method,
        "variance_status": (
            "declared_influence_calculation_not_independently_verified"
        ),
        "folds": fold_records,
        "passed": bool(converged and difference <= policy.reference * 10),
    }


def refold(
    inputs: Mapping[str, numpy.ndarray],
    values: numpy.ndarray,
    policy: Config,
    training: numpy.ndarray,
    fold: int,
) -> tuple[numpy.ndarray, dict[str, int | bool | str]]:
    """Refits one fold's nuisance models and scores its held-out rows.

    Args:
        inputs: Arrays from the primary fit: the nuisance design and the
            treatment.
        values: Outcome vector.
        policy: Supplies the reference tolerance and propensity clip.
        training: Mask of the fold's training rows.
        fold: Index of the fold.

    Returns:
        The reference AIPW scores of the held-out rows and the fold's
        optimizer record.

    Raises:
        ValueError: If an outcome design is rank deficient.
    """
    nuisance_design = inputs["nuisance_design"]
    treatment = inputs["treatment"]
    validation = ~training
    training_design = nuisance_design[training]
    training_treatment = treatment[training]
    validation_design = nuisance_design[validation]
    propensity_fit = bfgs(training_design, training_treatment, policy)
    propensity = numpy.clip(
        special.expit(validation_design @ propensity_fit.x),
        policy.clip,
        1.0 - policy.clip,
    )
    outcome_control, outcome_treated = outcomes(
        training_design,
        values[training],
        (training_treatment == 1.0, training_treatment == 0.0),
        validation_design,
        "aipw_ate reference outcome design is not identifiable",
    )
    scores = aipw(
        outcome_treated,
        outcome_control,
        treatment[validation],
        values[validation],
        propensity,
    )
    return scores, {
        "fold": fold,
        "converged": bool(propensity_fit.success),
        "message": str(propensity_fit.message),
    }


def bfgs(
    design: numpy.ndarray, treatment: numpy.ndarray, policy: Config
) -> optimize.OptimizeResult:
    """Fits the logistic propensity model independently with BFGS.

    Args:
        design: Propensity design matrix.
        treatment: 0/1 treatment of every row of `design`.
        policy: Supplies the reference tolerance.

    Returns:
        The optimizer result, started from zero coefficients.
    """

    def objective(candidate: numpy.ndarray) -> tuple[float, numpy.ndarray]:
        """Returns the negative Bernoulli log-likelihood and its gradient.

        Args:
            candidate: Propensity-model coefficients.

        Returns:
            The objective value and its gradient with respect to
            `candidate`.
        """
        logits = design @ candidate
        probabilities = special.expit(logits)
        value = -float(
            numpy.sum(treatment * logits - numpy.logaddexp(0.0, logits))
        )
        gradient = design.T @ (probabilities - treatment)
        return value, gradient

    return optimize.minimize(
        lambda candidate: objective(candidate)[0],
        numpy.zeros(design.shape[1]),
        jac=lambda candidate: objective(candidate)[1],
        method="BFGS",
        options={"gtol": policy.reference, "maxiter": 1000},
    )


def verdict(
    scores: numpy.ndarray,
    coefficients: numpy.ndarray,
    policy: Config,
    clusters: numpy.ndarray | None,
) -> tuple[float, float, str, float]:
    """Compares a reference AIPW estimate with the primary one.

    Args:
        scores: Reference AIPW score of every row.
        coefficients: ATE vector from the primary fit.
        policy: Supplies the covariance kind.
        clusters: Optional cluster labels for clustered uncertainty.

    Returns:
        The reference ATE, its variance, the variance method and the
        absolute difference from the primary ATE.  Errors from the variance
        calculation propagate.
    """
    reference_ate = float(numpy.mean(scores))
    reference_variance, reference_uncertainty = dispersion(
        scores - reference_ate,
        policy.covariance,
        clusters,
    )
    difference = float(abs(reference_ate - coefficients[0]))
    return (
        reference_ate,
        reference_variance,
        reference_uncertainty["method"],
        difference,
    )
