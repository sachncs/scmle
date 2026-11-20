"""Model explainability: ablation, perturbation and fidelity."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy

from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.explainability import quality
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import containers
from scmle.pipeline.modality_lifecycle import state
from scmle.pipeline.modality_lifecycle import steps

if TYPE_CHECKING:
    from sklearn.linear_model import LogisticRegression


REPEATED_LIMITATIONS = (
    (
        "The suite perturbs the declared feature "
        "representation; it does not establish raw-modality "
        "saliency."
    ),
    (
        "The sample bound is a lifecycle resource limit, not a "
        "production-scale stress guarantee."
    ),
)
LIMITATIONS = (
    "Associational explanation only.",
    ("Input representation and linear approximation constrain interpretation."),
    (
        "Quality checks validate the feature-space linear "
        "baseline; they do not validate raw-modality causality."
    ),
)


@dataclasses.dataclass(frozen=True, slots=True)
class Stability:
    """Linear explanations of test rows under tiny feature perturbations.

    Attributes:
        stabilities: Ranking stability of every evaluated row.
        reports: Contribution report of every evaluated row.
        coefficients: Coefficients that explain the first row.
        perturbed: Contribution reports of the first row followed
            by those of its perturbed copies.
        stability: Ranking stability of the first row.
        consistency: Share of first-row zero-ablations whose
            decision change has the sign the linear model predicts.
    """

    stabilities: list[float]
    reports: list[dict[str, object]]
    coefficients: numpy.ndarray
    perturbed: list[dict[str, object]]
    stability: float
    consistency: float


def ablate(
    model: LogisticRegression,
    sample_row: numpy.ndarray,
    sample_coefficients: numpy.ndarray,
    sample_decision: numpy.ndarray,
    sample_class: int,
) -> float:
    """Measures whether zeroing a feature moves the decision as predicted.

    Args:
        model: Fitted linear classifier.
        sample_row: Feature values of the explained row.
        sample_coefficients: Coefficients of the explained class.
        sample_decision: The model's decision output for the row.
        sample_class: Index of the explained class.

    Returns:
        The share of non-negligible features whose zero-ablation changes the
        decision with the sign the linear coefficients predict; 0.0 when no
        feature contributes.
    """
    nonzero = numpy.flatnonzero(
        numpy.abs(sample_row * sample_coefficients)
        > constants.MIN_STABILITY_COEFFICIENT
    )
    if not len(nonzero):
        return 0.0
    perturbed = numpy.repeat(sample_row[None, :], len(nonzero), axis=0)
    perturbed[numpy.arange(len(nonzero)), nonzero] = 0.0
    perturbed_decisions = numpy.asarray(model.decision_function(perturbed))
    if perturbed_decisions.ndim == 1:
        observed_change = perturbed_decisions - sample_decision[0]
    else:
        observed_change = (
            perturbed_decisions[:, sample_class]
            - sample_decision[0, sample_class]
        )
    expected_change = -(sample_row[nonzero] * sample_coefficients[nonzero])
    return float(
        numpy.mean(numpy.sign(observed_change) == numpy.sign(expected_change))
    )


def sensitivities(
    batch: containers.Batch,
    sample_index: int,
    sample_coefficients: numpy.ndarray,
    perturbation_size: numpy.ndarray,
) -> list[dict[str, object]]:
    """Explains a row and copies of it with one feature nudged at a time.

    Args:
        batch: Feature batch holding the row.
        sample_index: Position of the row in the batch.
        sample_coefficients: Coefficients of the explained class.
        perturbation_size: Size of the nudge for each feature.

    Returns:
        The row's contribution report, then one report for each feature
        nudged down and up.
    """
    sample_row = batch.values[sample_index]
    top_k = min(10, len(batch.feature_names))
    reports = [
        quality.decompose(
            batch.sample_ids[sample_index],
            batch.feature_names,
            sample_row.tolist(),
            sample_coefficients.tolist(),
            top_k=top_k,
        )
    ]
    for feature_index in range(len(batch.feature_names)):
        for direction in (-1.0, 1.0):
            perturbed_row = sample_row.copy()
            perturbed_row[feature_index] += (
                direction * perturbation_size[feature_index]
            )
            reports.append(
                quality.decompose(
                    batch.sample_ids[sample_index],
                    batch.feature_names,
                    perturbed_row.tolist(),
                    sample_coefficients.tolist(),
                    top_k=top_k,
                )
            )
    return reports


def perturb(
    model: LogisticRegression,
    batch: containers.Batch,
    perturbation_size: numpy.ndarray,
    evaluated_sample_count: int,
) -> Stability:
    """Explains the first rows of a batch and tests how stable they are.

    Args:
        model: Fitted linear classifier.
        batch: Test feature batch.
        perturbation_size: Size of the nudge for each feature.
        evaluated_sample_count: Number of leading rows to evaluate.

    Returns:
        The reports and stabilities of every evaluated row, with the
        perturbation and zero-ablation evidence of the first row.

    Raises:
        RuntimeError: If no row was evaluated.
    """
    stabilities: list[float] = []
    explanation_reports: list[dict[str, object]] = []
    first: tuple[numpy.ndarray, list[dict[str, object]], float, float] | None
    first = None
    for sample_index in range(evaluated_sample_count):
        sample_row = batch.values[sample_index]
        sample_decision = numpy.asarray(
            model.decision_function(sample_row[None])
        )
        sample_class = 0
        if sample_decision.ndim > 1:
            sample_class = int(numpy.argmax(sample_decision[0]))
        sample_coefficients = model.coef_[sample_class]
        sample_reports = sensitivities(
            batch, sample_index, sample_coefficients, perturbation_size
        )
        explanation_reports.append(sample_reports[0])
        sample_stability = quality.topk(
            tuple(sample_reports),
            top_k=min(10, len(batch.feature_names)),
        )
        stabilities.append(sample_stability)
        if sample_index == 0:
            first = (
                sample_coefficients,
                sample_reports,
                sample_stability,
                ablate(
                    model,
                    sample_row,
                    sample_coefficients,
                    sample_decision,
                    sample_class,
                ),
            )
    if first is None:
        raise RuntimeError("repeated explanation evaluation produced no report")
    return Stability(
        stabilities=stabilities,
        reports=explanation_reports,
        coefficients=first[0],
        perturbed=first[1],
        stability=first[2],
        consistency=first[3],
    )


def replicate(model: LogisticRegression, batch: containers.Batch) -> float:
    """Scores how closely the linear reference reproduces the decision output.

    Args:
        model: Fitted linear classifier.
        batch: Test feature batch.

    Returns:
        The fidelity of `X @ coef_.T + intercept_` to `decision_function`.

    Raises:
        ValueError: If the linear reference does not match the model's
            decision output in shape.
    """
    decision_values = numpy.asarray(model.decision_function(batch.values))
    reference_values = batch.values @ model.coef_.T + model.intercept_
    if (
        reference_values.ndim == constants.MATRIX_RANK
        and reference_values.shape[1] == 1
    ):
        reference_values = reference_values[:, 0]
    if (
        decision_values.ndim == constants.MATRIX_RANK
        and decision_values.shape[1] == 1
    ):
        decision_values = decision_values[:, 0]
    if decision_values.shape != reference_values.shape:
        raise errors.InputError(
            "linear explanation reference shape does not match model "
            "decision output"
        )
    return quality.score(
        decision_values.astype(float).reshape(-1).tolist(),
        reference_values.astype(float).reshape(-1).tolist(),
    )


class Explainability(state.Stage):
    """Explain linear weights and test feature-space stability."""

    stage = modalities.Phase.MODEL_EXPLAINABILITY
    requires = (
        steps.Splitting,
        steps.Preparation,
        steps.Modelling,
        steps.Composition,
    )
    makes = state.Explanation

    @staticmethod
    def corroborate(
        board: state.Board,
        batch: containers.Batch,
        partitions: containers.Partitions,
        evaluated_sample_count: int,
        explanation_reports: list[dict[str, object]],
    ) -> dict[str, object]:
        """Compares the explanations with caller-authored ground truth.

        Args:
            board: The run's inputs.
            batch: Test feature batch.
            partitions: Train, validation and test partitions.
            evaluated_sample_count: Number of leading rows that were explained.
            explanation_reports: Contribution report of each of those rows.

        Returns:
            The ground-truth suite result with its limitations.

        Raises:
            ValueError: If independent ground truth is required and the suite
                is unavailable or failed.
        """
        references = []
        for sample_index in range(evaluated_sample_count):
            ground_truth = board.plan.adapter.corroborate(
                partitions.test[sample_index], batch.feature_names
            )
            if ground_truth is not None:
                references.append(
                    (
                        ground_truth,
                        explanation_reports[sample_index],
                        batch.sample_ids[sample_index],
                    )
                )
        ground_truth_evidence = {
            **quality.referee(evaluated_sample_count, references).to_dict(),
            "feature_space_only": True,
            "limitations": [
                (
                    "Feature-space stability and fidelity are not independent "
                    "semantic validation."
                ),
                "Raw-modality ground truth remains caller-owned evidence.",
            ],
        }
        if board.plan.independent and (
            ground_truth_evidence["status"] != "available"
            or ground_truth_evidence["passed"] is not True
        ):
            raise errors.InputError(
                "independent explanation ground-truth suite is required "
                "before packaging"
            )
        return ground_truth_evidence

    @staticmethod
    def prominence(
        model: LogisticRegression, feature_names: tuple[str, ...]
    ) -> list[list[object]]:
        """Ranks features by mean absolute coefficient across classes.

        Args:
            model: Fitted linear classifier.
            feature_names: Names of its features.

        Returns:
            The ten strongest features as [name, strength], strongest first and
            ties by name.
        """
        strength = numpy.mean(numpy.abs(model.coef_), axis=0)
        global_terms = sorted(
            (
                (name, float(strength[index]))
                for index, name in enumerate(feature_names)
            ),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        return [list(value) for value in global_terms]

    @staticmethod
    def repetition(
        stability: float, sample_count: int, evaluated_sample_count: int
    ) -> dict[str, object]:
        """Describes the repeated-perturbation suite and its bound.

        Args:
            stability: Minimum stability across the evaluated samples.
            sample_count: Number of test samples.
            evaluated_sample_count: Number of leading samples evaluated.

        Returns:
            The suite's score, counts, threshold, bound and limitations.
        """
        return {
            "score": stability,
            "sample_count": sample_count,
            "evaluated_sample_count": evaluated_sample_count,
            "minimum_across_samples": stability,
            "threshold": 0.9,
            "feature_space_only": True,
            "bounded_by": constants.MAX_REPEATED_EXPLANATION_SAMPLES,
            "limitations": list(REPEATED_LIMITATIONS),
        }

    def execute(self, board: state.Board) -> state.Explanation:
        """Records signed coefficient contributions with limitations.

        Args:
            board: The run's inputs, partitions, features and model.

        Returns:
            The model explanation.

        Raises:
            OrderError: If there are no test features to explain.
        """
        model = board.get(state.Fit).model
        batches = board.get(state.Features).batches
        batch = batches["test"]
        partitions = board.get(state.Partition).partitions
        if not len(batch.sample_ids):
            raise errors.OrderError(
                "model, partitions, and test features are required"
            )
        global_terms = self.prominence(model, batch.feature_names)
        fidelity = replicate(model, batch)
        train_values = batches["train"].values
        feature_scales = numpy.maximum(numpy.std(train_values, axis=0), 1.0)
        perturbation_size = feature_scales * 1e-6
        evaluated_sample_count = min(
            len(batch.sample_ids), constants.MAX_REPEATED_EXPLANATION_SAMPLES
        )
        repeated = perturb(
            model, batch, perturbation_size, evaluated_sample_count
        )
        ground_truth_evidence = self.corroborate(
            board, batch, partitions, evaluated_sample_count, repeated.reports
        )
        repeated_stability = min(repeated.stabilities)
        quality_evaluation = quality.evaluate(
            method="linear_reference_zero_ablation_and_perturbation",
            sample_count=len(batch.sample_ids),
            thresholds=quality.Thresholds(
                stability=0.9,
                minimum=0.999999,
            ),
            stability=repeated_stability,
            fidelity=fidelity,
        )
        return state.Explanation(
            {
                "method": "linear_coefficient_and_local_contribution",
                "model_family": "regularized_linear_classifier",
                "global_terms": global_terms,
                "local": {
                    "sample_id": batch.sample_ids[0],
                    "terms": repeated.perturbed[0]["terms"],
                },
                "zero_ablation_direction_consistency": (repeated.consistency),
                "perturbation_stability": {
                    "score": repeated.stability,
                    "report_count": len(repeated.perturbed),
                    "relative_step": 1e-6,
                    "feature_space_only": True,
                },
                "repeated_perturbation": self.repetition(
                    repeated_stability,
                    len(batch.sample_ids),
                    evaluated_sample_count,
                ),
                "independent_ground_truth": ground_truth_evidence,
                "modality_specific": board.plan.adapter.elucidate(
                    partitions.test[0],
                    model,
                    batch.feature_names,
                    repeated.coefficients,
                ),
                "quality": quality_evaluation.to_dict(),
                "limitations": list(LIMITATIONS),
            }
        )
