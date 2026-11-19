"""The planning stage and the plan it devises."""

import dataclasses
import logging
from typing import ClassVar

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.pipeline import engine
from scmle.pipeline import kinds
from scmle.pipeline import products
from scmle.pipeline import recommendations
from scmle.pipeline import roster
import scmle.pipeline.planner
from scmle.pipeline.stages import ingestion
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import profiling
import scmle.runtime.state

LOGGER = logging.getLogger(__name__)


def observe(board: engine.Board) -> None:
    """Persists the runtime facts immediately before capability planning.

    Args:
        board: The run's board, whose services hold the facts.

    Raises:
        ScmleError: If the services carry no resource observation.
    """
    observation = products.require(
        board.services.capability_observation, "capability_observation"
    )
    path = "evidence/capability_resources.json"
    board.journal.write_json(path, observation.to_dict())
    board.journal.emit(
        "capability.environment.observed",
        "runtime facts captured for the capability gate",
        (path,),
        observation.to_dict()["environment"],
    )


def devise(
    board: engine.Board,
) -> experiments.Plan:
    """Creates and validates the experiment plan from a profiled board.

    The planner remains responsible for adapter compatibility and search
    policy. The capability registry is the separate release boundary: every
    candidate in the resulting plan must have a supported capability
    declaration (an experimental one only if the candidate is listed in
    `models.experimental_models`) before the execution plan can reach
    ``TrainStage``. Candidates of the `custom` and `test` families without one
    are registered as extension-only instead.

    Args:
        board: The run's board, holding the profile, task and manifest.

    Returns:
        The validated experiment plan.

    Raises:
        capabilities.UnsupportedError: If a planned candidate has no
            accepted capability declaration, or the requested fairness
            mitigation is not implemented for the task.
    """
    approved = board.services.approved_plan
    planner = scmle.pipeline.planner.Planner(
        board.services.registry, board.services.catalog
    )
    partition = board.get(profiling.Partition)
    plan = planner.plan(
        scmle.pipeline.planner.Situation(
            board.services.spec,
            partition.profile,
            partition.task,
            approved.candidates if approved is not None else None,
            board.services.history,
            board.get(ingestion.Intake).kind.cost,
        )
    )
    task = partition.task
    manifest = board.get(ingestion.Intake).manifest
    for model_name in plan.candidates:
        adapter = board.services.registry.get(model_name)
        model_family = adapter.capability_family or adapter.family
        key = requisition(board, task, model_family)
        allowed_statuses = (
            {
                capabilities.Status.SUPPORTED,
                capabilities.Status.EXPERIMENTAL,
            }
            if model_name in board.services.spec.models.experimental_models
            else None
        )
        try:
            board.services.capability_registry.require(
                key,
                allowed_statuses=allowed_statuses,
                row_count=manifest.rows,
            )
        except capabilities.UnsupportedError as error:
            if model_family not in {"custom", "test"}:
                raise
            LOGGER.warning(
                "%s family %s accepted as extension-only: %s",
                key,
                model_family,
                error,
            )
            if board.services.capability_registry.records.get(key) is None:
                board.services.capability_registry.register(
                    roster.extension_capability(key)
                )
            board.services.capability_registry.require(
                key,
                allowed_statuses={capabilities.Status.EXTENSION_ONLY},
                row_count=manifest.rows,
            )
    return plan


def requisition(
    board: engine.Board,
    task: tasks.Task,
    model_family: str,
) -> capabilities.Coordinate:
    """Builds the complete capability request for a planned candidate.

    The key always requests library inference with no predictive uncertainty
    and no explainability output, evaluated as the tabular beta evaluates the
    task. Fairness is `mitigation` unless the specification requests none.

    Args:
        board: The run's board, supplying the manifest and specification.
        task: The resolved task.
        model_family: Family of the candidate model.

    Returns:
        The capability key requested for the candidate.

    Raises:
        capabilities.UnsupportedError: If the dataset modality is
            unknown, or group-threshold mitigation is requested for a task
            other than binary.
        ConfigError: If the manifest's modality differs from the data spec's.
    """
    manifest = board.get(ingestion.Intake).manifest
    if manifest.modality not in {m.value for m in capabilities.Modality}:
        raise capabilities.UnsupportedError(
            "unsupported dataset modality before training",
            {
                "status": "unsupported",
                "reason": "dataset manifest has an unknown modality",
                "modality": manifest.modality,
            },
        )
    modality = capabilities.Modality(manifest.modality)
    if manifest.modality != board.services.spec.data.modality:
        raise errors.ConfigError(
            "dataset manifest modality does not match the project data "
            f"spec: {manifest.modality!r} != "
            f"{board.services.spec.data.modality!r}"
        )
    fairness = (
        capabilities.Fairness.NONE
        if board.services.spec.fairness.mitigation == "none"
        else capabilities.Fairness.MITIGATION
    )
    key = capabilities.Coordinate(
        modality=modality,
        task=task,
        model_family=model_family,
        evaluation=kinds.TABULAR_BETA_EVALUATIONS[task],
        uncertainty=capabilities.Uncertainty.NONE,
        fairness=fairness,
        explainability=capabilities.Explainability.NONE,
        inference=capabilities.Inference.LIBRARY,
    )
    if (
        board.services.spec.fairness.mitigation == "group_thresholds"
        and task is not tasks.Task.BINARY
    ):
        board.services.capability_registry.refuse(
            key,
            "group-threshold mitigation is implemented for binary tasks "
            "only; reweighing mitigates multiclass and regression tasks",
        )
    return key


def advise(
    board: engine.Board,
) -> tuple[recommendations.Recommendation, ...]:
    """Derives the recommendations for the user.

    Args:
        board: A board whose profile is populated.

    Returns:
        The recommendations (kept on the board by `Blueprint.apply`).
    """
    return recommendations.recommend(
        board.get(profiling.Partition).profile,
        board.services.spec,
    )


@dataclasses.dataclass(frozen=True)
class Blueprint(products.Kept):
    """The execution plan, the advice for the user and the capabilities.

    Attributes:
        plan: The execution plan.
        recommendations: The advice derived from the profile.
        capability_registry: The capabilities, with the refusals planning
            recorded.
    """

    plan: experiments.Plan | None
    recommendations: tuple[recommendations.Recommendation, ...]
    capability_registry: capabilities.Registry

    def apply(self, board: object) -> None:
        """Sets the capabilities on the run's services and holds the plan.

        Args:
            board: The run's board.

        Raises:
            KindError: If `board` is not a run's board.
        """
        if not isinstance(board, engine.Board):
            raise errors.KindError("a blueprint applies to a run's board")
        board.services = dataclasses.replace(
            board.services, capability_registry=self.capability_registry
        )
        super().apply(board)


class Planning(interface.Stage):
    """Builds the experiment plan from the development-set profile."""

    requires: ClassVar[tuple[type, ...]] = (profiling.ProfileStage,)
    makes: ClassVar[type[products.Product] | None] = Blueprint
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.PLANNING
    )

    def describe(
        self,
        board: engine.Board,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            board: The run's board; not consulted.

        Returns:
            The text "build execution plan".
        """
        return "build execution plan"

    def execute(self, board: engine.Board) -> Blueprint:
        """Builds the execution plan and the recommendations.

        First records the runtime resources the capability gate plans
        against. Writes the execution plan, the capability registry, the
        decision log and the recommendations, and emits a
        `plan.recommendations` event. A candidate without an accepted
        capability declaration stops the run (`devise` raises
        `UnsupportedError`).

        Args:
            board: The run's board, holding the profile, task and manifest.

        Returns:
            The blueprint: the plan, the advice and the capabilities.
        """
        observe(board)
        plan = devise(board)
        registry = board.services.capability_registry
        board.journal.write_json("execution_plan.json", plan.to_dict())
        board.journal.write_json("capability_registry.json", registry.to_dict())
        board.journal.write_json(
            "diagnostics/decision_log.json",
            {"plan": [d.to_dict() for d in plan.decisions], "outcomes": []},
        )
        advice = advise(board)
        board.journal.write_json(
            "diagnostics/recommendations.json",
            [item.to_dict() for item in advice],
        )
        board.journal.emit(
            "plan.recommendations",
            f"{len(advice)} recommendation(s) for the user",
            ("diagnostics/recommendations.json",),
        )
        return Blueprint(plan, advice, registry)
