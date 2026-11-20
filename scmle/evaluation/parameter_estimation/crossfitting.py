"""The AIPW nuisance models, fitted on all rows or cross-fitted by fold."""

from collections.abc import Mapping
import dataclasses
from typing import Any

import numpy
from scipy import special

from scmle.contracts import errors
from scmle.evaluation.parameter_estimation.config import Config
from scmle.evaluation.parameter_estimation.declaration import MIN_FOLDS
from scmle.evaluation.parameter_estimation.declaration import MIN_ROWS_PER_ARM
from scmle.evaluation.parameter_estimation.linear import logit
from scmle.evaluation.parameter_estimation.weighting import outcomes


@dataclasses.dataclass(frozen=True, slots=True)
class Nuisance:
    """The AIPW nuisance predictions for every row and how they were fitted.

    Attributes:
        propensity: Clipped propensity score of every row.
        raw: Unclipped propensity score of every row.
        control: Predicted outcome under control.
        treated: Predicted outcome under treatment.
        folds: Fold index of every row (all zero without cross-fitting).
        diagnostics: Propensity iterations and log-likelihood and the outcome
            design ranks, or the per-fold records under cross-fitting.
        crossfitting: The cross-fitting record of the diagnostics.
    """

    propensity: numpy.ndarray
    raw: numpy.ndarray
    control: numpy.ndarray
    treated: numpy.ndarray
    folds: numpy.ndarray
    diagnostics: Mapping[str, Any]
    crossfitting: Mapping[str, Any]

    def records(self) -> dict[str, Mapping[str, Any]]:
        """Returns the propensity-model and outcome-model records.

        Without cross-fitting they carry the single fit's log-likelihood and
        ranks; with it they point to the per-fold records.
        """
        single = self.crossfitting["fold_count"] == 1
        return {
            "propensity_model": {
                "method": "logistic_mle",
                "converged": True,
                "iterations": self.diagnostics["propensity_iterations"],
                "log_likelihood": self.diagnostics.get(
                    "propensity_log_likelihood", "recorded_per_fold"
                ),
            },
            "outcome_models": {
                "method": "separate_ordinary_least_squares",
                "treated_rank": self.diagnostics["treated_rank"]
                if single
                else "recorded_per_fold",
                "control_rank": self.diagnostics["control_rank"]
                if single
                else "recorded_per_fold",
            },
        }


def nuisance(
    design: numpy.ndarray,
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
) -> Nuisance:
    """Fits the AIPW nuisance models, cross-fitted when the policy asks.

    Args:
        design: Nuisance design matrix with an intercept column.
        treatment: 0/1 treatment of every row.
        values: Outcome of every row.
        policy: Supplies the number of cross-fitting folds, the propensity
            clip, iteration limit and tolerance.

    Returns:
        The nuisance predictions for every row and their fit records.
    """
    if policy.folds != 1:
        return crossfit(design, treatment, values, policy)
    (
        propensity,
        raw_propensity,
        outcome_control,
        outcome_treated,
        nuisance_diagnostics,
    ) = train(
        design,
        treatment,
        values,
        design,
        policy,
    )
    return Nuisance(
        propensity,
        raw_propensity,
        outcome_control,
        outcome_treated,
        numpy.zeros(len(values), dtype=int),
        nuisance_diagnostics,
        {
            "status": "disabled",
            "fold_count": 1,
            "assignment_method": "not_applicable",
            "limitations": caveats(crossfit=False),
        },
    )


def crossfit(
    design: numpy.ndarray,
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
) -> Nuisance:
    """Fits the nuisance models on each training fold, scoring held-out rows.

    Args:
        design: Nuisance design matrix with an intercept column.
        treatment: 0/1 treatment of every row.
        values: Outcome of every row.
        policy: Supplies the number of folds, the propensity clip, iteration
            limit and tolerance.

    Returns:
        The held-out nuisance predictions for every row and the per-fold
        records.  Fold assignment and support errors propagate.
    """
    fold_count = policy.folds
    fold_assignments = interleave(len(values), fold_count)
    support = overlap(treatment, fold_assignments, fold_count)
    (
        propensity,
        raw_propensity,
        outcome_control,
        outcome_treated,
        fold_diagnostics,
    ) = holdout(design, treatment, values, policy, fold_assignments)
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
        "limitations": caveats(crossfit=True),
    }
    return Nuisance(
        propensity,
        raw_propensity,
        outcome_control,
        outcome_treated,
        fold_assignments,
        nuisance_diagnostics,
        cross_fitting,
    )


def holdout(
    design: numpy.ndarray,
    treatment: numpy.ndarray,
    values: numpy.ndarray,
    policy: Config,
    fold_assignments: numpy.ndarray,
) -> tuple[
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    numpy.ndarray,
    list[dict[str, Any]],
]:
    """Fits each fold's nuisance models on its training rows.

    Args:
        design: Nuisance design matrix with an intercept column.
        treatment: 0/1 treatment of every row.
        values: Outcome of every row.
        policy: Supplies the number of folds, the propensity clip, iteration
            limit and tolerance.
        fold_assignments: Fold index of every row.

    Returns:
        For every row, predicted from the folds that held it out: the
        clipped propensity, the unclipped propensity, the predicted outcome
        under control and under treatment; and one record per fold.
    """
    propensity = numpy.empty(len(values), dtype=float)
    raw_propensity = numpy.empty(len(values), dtype=float)
    outcome_control = numpy.empty(len(values), dtype=float)
    outcome_treated = numpy.empty(len(values), dtype=float)
    fold_diagnostics = []
    for fold in range(policy.folds):
        training = fold_assignments != fold
        validation = ~training
        (
            fold_propensity,
            fold_raw_propensity,
            fold_outcome_control,
            fold_outcome_treated,
            fold_nuisance_diagnostics,
        ) = train(
            design[training],
            treatment[training],
            values[training],
            design[validation],
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
    return (
        propensity,
        raw_propensity,
        outcome_control,
        outcome_treated,
        fold_diagnostics,
    )


def train(
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
        raise errors.InputError(
            "aipw_ate each training fold needs at least two rows per "
            "treatment arm"
        )
    propensity, evaluation_raw_propensity, propensity_diagnostics = (
        propensities(
            training_design, training_treatment, evaluation_design, policy
        )
    )
    outcome_control, outcome_treated = outcomes(
        training_design,
        training_values,
        (treated, control),
        evaluation_design,
        "aipw_ate outcome design is not identifiable",
    )
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
            "treated_rank": int(
                numpy.linalg.matrix_rank(training_design[treated])
            ),
            "control_rank": int(
                numpy.linalg.matrix_rank(training_design[control])
            ),
        },
    )


def interleave(rows: int, fold_count: int) -> numpy.ndarray:
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
        raise errors.InputError("cross-fitting needs at least two folds")
    if fold_count > rows:
        raise errors.InputError("cross_fitting_folds must not exceed rows")
    return numpy.arange(rows, dtype=int) % fold_count


def overlap(
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
            raise errors.InputError(
                "aipw_ate cross-fitting needs both treatment arms in every "
                "validation fold"
            )
        if (
            training_treated < MIN_ROWS_PER_ARM
            or training_control < MIN_ROWS_PER_ARM
        ):
            raise errors.InputError(
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


def caveats(crossfit: bool) -> list[str]:
    """Returns limitations of the AIPW estimator.

    Args:
        crossfit: Whether the nuisance models were cross-fitted.

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
    if not crossfit:
        limitations.append(
            "nuisance models are fitted on the same rows used for scoring"
        )
    return limitations


def propensities(
    training: numpy.ndarray,
    treatment: numpy.ndarray,
    evaluation: numpy.ndarray,
    policy: Config,
) -> tuple[numpy.ndarray, numpy.ndarray, Mapping[str, Any]]:
    """Fits the logistic propensity model and scores the evaluation rows.

    Args:
        training: Propensity design matrix of the training rows.
        treatment: 0/1 treatment of the training rows.
        evaluation: Propensity design matrix of the rows to score.
        policy: Supplies the propensity clip, iteration limit and tolerance.

    Returns:
        The clipped propensity and the unclipped propensity of the
        evaluation rows, and the logistic fit's diagnostics.  Errors from
        the logistic fit propagate.
    """
    coefficients, probabilities, diagnostics = logit(
        training,
        treatment,
        policy.iterations,
        policy.tolerance,
    )
    del probabilities
    raw = special.expit(evaluation @ coefficients)
    clipped = numpy.clip(raw, policy.clip, 1.0 - policy.clip)
    return clipped, raw, diagnostics
