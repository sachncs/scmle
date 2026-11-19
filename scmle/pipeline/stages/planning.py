"""The planning stage and the plan it devises."""

import dataclasses
import logging
from typing import ClassVar

from scmle.contracts import capabilities
from scmle.contracts import errors
from scmle.contracts import experiments
from scmle.contracts import tasks
from scmle.pipeline import products
from scmle.pipeline import recommendations
from scmle.pipeline import roster
import scmle.pipeline.context
import scmle.pipeline.planner
from scmle.pipeline.stages import interface
from scmle.pipeline.stages import profiling
import scmle.runtime.state

LOGGER = logging.getLogger(__name__)


def observe(context: scmle.pipeline.context.Context) -> None:
    """Persists the runtime facts immediately before capability planning.

    Args:
        context: The run context, whose services hold the facts.

    Raises:
        ScmleError: If the services carry no resource observation.
    """
    observation = scmle.pipeline.context.require(
        context.services.capability_observation, "capability_observation"
    )
    path = "evidence/capability_resources.json"
    context.journal.write_json(path, observation.to_dict())
    context.journal.emit(
        "capability.environment.observed",
        "runtime facts captured for the capability gate",
        (path,),
        observation.to_dict()["environment"],
    )


def devise(
    context: scmle.pipeline.context.Context,
) -> experiments.Plan:
    """Creates and validates the experiment plan from a profiled context.

    The planner remains responsible for adapter compatibility and search
    policy. The capability registry is the separate release boundary: every
    candidate in the resulting plan must have a supported capability
    declaration (an experimental one only if the candidate is listed in
    `models.experimental_models`) before the execution plan can reach
    ``TrainStage``. Candidates of the `custom` and `test` families without one
    are registered as extension-only instead.

    Args:
        context: The run context, holding the profile, task and manifest.

    Returns:
        The validated experiment plan.

    Raises:
        capabilities.UnsupportedError: If a planned candidate has no
            accepted capability declaration, or the requested fairness
            mitigation is not implemented for the task.
    """
    approved = context.services.approved_plan
    planner = scmle.pipeline.planner.Planner(
        context.services.registry, context.services.catalog
    )
    plan = planner.plan(
        scmle.pipeline.planner.Situation(
            context.services.spec,
            scmle.pipeline.context.require(context.profile, "profile"),
            scmle.pipeline.context.require(context.task, "task"),
            approved.candidates if approved is not None else None,
            context.services.history,
        )
    )
    task = scmle.pipeline.context.require(context.task, "task")
    manifest = scmle.pipeline.context.require(context.manifest, "manifest")
    for model_name in plan.candidates:
        adapter = context.services.registry.get(model_name)
        model_family = adapter.capability_family or adapter.family
        key = requisition(context, task, model_family)
        allowed_statuses = (
            {
                capabilities.Status.SUPPORTED,
                capabilities.Status.EXPERIMENTAL,
            }
            if model_name in context.services.spec.models.experimental_models
            else None
        )
        try:
            context.services.capability_registry.require(
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
            if context.services.capability_registry.records.get(key) is None:
                context.services.capability_registry.register(
                    roster.extension_capability(key)
                )
            context.services.capability_registry.require(
                key,
                allowed_statuses={capabilities.Status.EXTENSION_ONLY},
                row_count=manifest.rows,
            )
    return plan


def requisition(
    context: scmle.pipeline.context.Context,
    task: tasks.Task,
    model_family: str,
) -> capabilities.Coordinate:
    """Builds the complete capability request for a planned candidate.

    The key always requests library inference with no predictive uncertainty
    and no explainability output, evaluated as the tabular beta evaluates the
    task. Fairness is `mitigation` unless the specification requests none.

    Args:
        context: The run context, supplying the manifest and specification.
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
    manifest = scmle.pipeline.context.require(context.manifest, "manifest")
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
    if manifest.modality != context.services.spec.data.modality:
        raise errors.ConfigError(
            "dataset manifest modality does not match the project data "
            f"spec: {manifest.modality!r} != "
            f"{context.services.spec.data.modality!r}"
        )
    fairness = (
        capabilities.Fairness.NONE
        if context.services.spec.fairness.mitigation == "none"
        else capabilities.Fairness.MITIGATION
    )
    key = capabilities.Coordinate(
        modality=modality,
        task=task,
        model_family=model_family,
        evaluation=roster.TABULAR_BETA_EVALUATIONS[task],
        uncertainty=capabilities.Uncertainty.NONE,
        fairness=fairness,
        explainability=capabilities.Explainability.NONE,
        inference=capabilities.Inference.LIBRARY,
    )
    if (
        context.services.spec.fairness.mitigation == "group_thresholds"
        and task is not tasks.Task.BINARY
    ):
        context.services.capability_registry.refuse(
            key,
            "group-threshold mitigation is implemented for binary tasks "
            "only; reweighing mitigates multiclass and regression tasks",
        )
    return key


def advise(
    context: scmle.pipeline.context.Context,
) -> tuple[recommendations.Recommendation, ...]:
    """Derives the recommendations for the user.

    Args:
        context: A context whose profile is populated.

    Returns:
        The recommendations (kept on the context by `Blueprint.apply`).
    """
    return recommendations.recommend(
        scmle.pipeline.context.require(context.profile, "profile"),
        context.services.spec,
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


class Planning(interface.Stage):
    """Builds the experiment plan from the development-set profile."""

    requires: ClassVar[tuple[type, ...]] = (profiling.ProfileStage,)
    makes: ClassVar[type[products.Product] | None] = Blueprint
    state: ClassVar[scmle.runtime.state.Progress] = (
        scmle.runtime.state.Progress.PLANNING
    )

    def describe(
        self,
        context: scmle.pipeline.context.Context,  # noqa: ARG002 -- Stage.describe
    ) -> str:
        """Names the stage.

        Args:
            context: The run context; not consulted.

        Returns:
            The text "build execution plan".
        """
        return "build execution plan"

    def execute(self, context: scmle.pipeline.context.Context) -> Blueprint:
        """Builds the blueprint, which the engine applies to the context.

        Args:
            context: The run context, holding the profile, task and manifest.

        Returns:
            The execution plan and the recommendations.
        """
        return self.draft(context)

    def draft(self, context: scmle.pipeline.context.Context) -> Blueprint:
        """Builds the execution plan and the recommendations.

        First records the runtime resources the capability gate plans
        against. Writes the execution plan, the capability registry, the
        decision log and the recommendations, and emits a
        `plan.recommendations` event. A candidate without an accepted
        capability declaration stops the run (`devise` raises
        `UnsupportedError`).

        Args:
            context: The run context, holding the profile, task and manifest.

        Returns:
            The blueprint: the plan, the advice and the capabilities.
        """
        observe(context)
        plan = devise(context)
        registry = context.services.capability_registry
        context.journal.write_json("execution_plan.json", plan.to_dict())
        context.journal.write_json(
            "capability_registry.json", registry.to_dict()
        )
        context.journal.write_json(
            "diagnostics/decision_log.json",
            {"plan": [d.to_dict() for d in plan.decisions], "outcomes": []},
        )
        advice = advise(context)
        context.journal.write_json(
            "diagnostics/recommendations.json",
            [item.to_dict() for item in advice],
        )
        context.journal.emit(
            "plan.recommendations",
            f"{len(advice)} recommendation(s) for the user",
            ("diagnostics/recommendations.json",),
        )
        return Blueprint(plan, advice, registry)
