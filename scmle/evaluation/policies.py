"""Construction of explicit release policies from project requirements."""

from collections.abc import Mapping

from scmle.contracts import evaluation
from scmle.contracts import tasks

EVIDENCE = "diagnostics/metric_uncertainty.json"


def release(gates: list[evaluation.Hurdle]) -> evaluation.Gates:
    """The release gates: the metric hurdles and the uncertainty evidence.

    Args:
        gates: One hurdle per success criterion.

    Returns:
        The hurdles plus a blocking gate on the metric-uncertainty
        diagnostic, with the evidence every release needs.
    """
    return evaluation.Gates(
        metric_gates=tuple(gates),
        diagnostic_gates=(
            evaluation.Requirement(
                name="metric_uncertainty",
                action=evaluation.Remedy.BLOCK,
                required_evidence=(EVIDENCE,),
            ),
        ),
        required_evidence=(
            "dataset_manifest.json",
            "model/model_manifest.json",
            EVIDENCE,
        ),
    )


def derive(
    task: tasks.Task,
    primary_metric: str,
    primary_greater_is_better: bool,
    success_criteria: Mapping[str, float],
) -> evaluation.Criteria | None:
    """Builds a release policy when the project declares score thresholds.

    The project-level success criteria are the only source of numeric release
    gates. A project without them has no honest threshold to evaluate and
    therefore returns ``None`` instead of inventing one from the observed
    score. The metric-uncertainty diagnostic is a blocking evidence gate for
    every task because a point estimate without an uncertainty report is not
    sufficient release evidence.

    Args:
        task: The task type.
        primary_metric: Name of the primary metric.
        primary_greater_is_better: Whether larger values of the primary
            metric are better.
        success_criteria: Required score per metric name.

    Returns:
        A release policy with one metric gate per success criterion and a
        blocking gate on the metric-uncertainty diagnostic, or None when
        there are no success criteria.

    Raises:
        UnsupportedError: If a success criterion names a
            metric with no declared release direction.
    """
    if not success_criteria:
        return None
    primary_direction = (
        evaluation.Orientation.MAXIMIZE
        if primary_greater_is_better
        else evaluation.Orientation.MINIMIZE
    )
    declarations: dict[str, evaluation.Declaration] = {
        primary_metric: evaluation.Declaration(
            name=primary_metric,
            direction=primary_direction,
        )
    }
    gates: list[evaluation.Hurdle] = []
    for name, threshold in success_criteria.items():
        direction = evaluation.METRIC_DIRECTIONS.get(name)
        if direction is None:
            raise evaluation.UnsupportedError(
                f"metric '{name}' has no declared release direction"
            )
        declarations[name] = evaluation.Declaration(
            name=name,
            direction=direction,
        )
        gates.append(
            evaluation.Hurdle(
                metric=name, direction=direction, threshold=float(threshold)
            )
        )
    return evaluation.Criteria(
        task=task,
        primary_metric=primary_metric,
        metrics=tuple(declarations.values()),
        uncertainty=evaluation.Uncertainty(
            method=evaluation.UncertaintyMethod.NONE
        ),
        calibration=evaluation.Calibration(method=evaluation.Procedure.NONE),
        fairness=evaluation.Fairness(method=evaluation.Technique.NONE),
        small_groups=evaluation.Subgroups(warning_threshold=30),
        explanations=evaluation.Explanations(
            availability=evaluation.Explicability.NONE
        ),
        release=release(gates),
    )
