"""The runner that executes the lifecycle stages in order."""

from __future__ import annotations

import time
from typing import cast
from typing import ClassVar
from typing import TYPE_CHECKING

from scmle.contracts import data
from scmle.contracts import errors
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.governance import attestation
from scmle.pipeline.modality_lifecycle import checkpoints
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import explanation
from scmle.pipeline.modality_lifecycle import plans
from scmle.pipeline.modality_lifecycle import reporting
from scmle.pipeline.modality_lifecycle import state
from scmle.pipeline.modality_lifecycle import steps

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from pathlib import Path


def clocked(context: state.Context, stage: state.Stage) -> None:
    """Executes one stage and records its duration when it is measured.

    Args:
        context: Shared lifecycle state, updated in place.
        stage: The stage to execute.
    """
    started = time.perf_counter()
    stage.execute(context)
    operation = stage.performance_operation
    if operation is not None:
        if operation is performance.Operation.INGEST:
            sample_count = len(context.adapter.samples)
        else:
            sample_count = len(context.feature_batches["train"].sample_ids)
        context.clock(
            operation,
            time.perf_counter() - started,
            sample_count,
        )


class Lifecycle:
    """Executes the fifteen shared lifecycle stages in deterministic order.

    Attributes:
        stages: The stage objects, in required lifecycle order.
    """

    stages: ClassVar[tuple[state.Stage, ...]] = (
        steps.Ingestion(),
        steps.Validation(),
        steps.Profiling(),
        steps.Curating(),
        steps.Splitting(),
        steps.Preparation(),
        steps.Modelling(),
        steps.Evaluation(),
        steps.Composition(),
        explanation.ModelExplainabilityStage(),
        steps.Packaging(),
        steps.Inference(),
        steps.Monitoring(),
        steps.Documentation(),
        steps.Recovery(),
    )

    def run_stages(
        self,
        plan: plans.Plan,
        context: state.Context,
        resumed: tuple[int, tuple[str, ...]],
        progress: tuple[list[str], list[str]],
    ) -> None:
        """Executes every stage, replaying the ones a checkpoint completed.

        Args:
            plan: The run's fixed inputs.
            context: Shared lifecycle state, updated in place.
            resumed: Number of leading stages that a checkpoint completed,
                and their evidence digests.
            progress: Values of the completed stages and their digests; both
                lists grow as stages finish and are persisted at every
                boundary.

        Raises:
            ValueError: If a replayed stage's digest differs from the
                checkpoint's.
        """
        resume_count, resume_digests = resumed
        completed_stages, stage_digests = progress
        for stage_index, stage in enumerate(self.stages):
            plan.poll()
            context.resume_replay = stage_index < resume_count
            clocked(context, stage)
            stage_digest = attestation.checksum(
                context.stage_records[stage.stage]
            )
            if stage_index < resume_count:
                if stage_digest != resume_digests[stage_index]:
                    raise errors.InputError(
                        "lifecycle checkpoint replay diverged at stage "
                        f"{stage.stage.value}"
                    )
            else:
                completed_stages.append(stage.stage.value)
                stage_digests.append(stage_digest)
                plan.persist(completed_stages, stage_digests, "running")
            context.resume_replay = False
            plan.poll()

    def run(
        self,
        adapter: curators.Preparer,
        package_path: str | Path,
        signing_key: str | bytes,
        seed: int = 0,
        observed_adapter: curators.Preparer | None = None,
        report_path: str | Path | None = None,
        performance_budget: performance.Budget | None = None,
        performance_baseline: performance.Baseline | None = None,
        manifest: data.Passport | None = None,
        require_independent_ground_truth: bool = False,
        cancellation_check: Callable[[], bool] | None = None,
        checkpoint_path: str | Path | None = None,
        resume: bool = False,
    ) -> reporting.Record:
        """Runs every stage and optionally persists its evidence report.

        Args:
            adapter: Adapter for the reference dataset.
            package_path: Directory the model package is written to; it must not
                exist unless a checkpoint replay is resuming onto it.
            signing_key: Key that signs the package, report and checkpoints.
            seed: Random seed for the split and the model.
            observed_adapter: Adapter over the data to monitor against the
                reference; defaults to `adapter`. Its modality must match.
            report_path: File the signed report is written to, if any.
            performance_budget: Limits the measured operations must meet, if
                any.
            performance_baseline: Earlier run the measurements are compared
                with, if any.
            manifest: Modality manifest, if any. It is checked against the
                adapter before any stage executes, which keeps source, task,
                privacy, and lineage claims bound to the exact modality content.
            require_independent_ground_truth: Whether to fail unless the
                independent explanation ground-truth suite is available and
                passes.
            cancellation_check: Callable polled at stage boundaries. When it
                returns True, artifacts created by this run are removed and
                `LifecycleCancelledError` is raised.
            checkpoint_path: File where signed stage checkpoints are kept, if
                any; requires `report_path` and must differ from it.
            resume: Whether to resume from an existing checkpoint at
                `checkpoint_path`, replaying its completed stages and verifying
                their digests.

        Returns:
            The signed lifecycle report. If a completed checkpoint is resumed,
            the previously written report after verification.

        Raises:
            TypeError: If `require_independent_ground_truth` or `resume` is
                not a bool, or `cancellation_check` is not callable.
            ValueError: If the option combination is invalid, the observed
                modality or manifest does not match the adapter, a checkpoint or
                report does not match the resumed run, a stage fails, or the
                performance budget or baseline is violated.
            RuntimeError: If the stage order is invalid or packaging produced no
                package.
            FileExistsError: If the checkpoint exists and `resume` is False.
            FileNotFoundError: If `resume` is True and the checkpoint does not
                exist.
        """
        plan = plans.prepare(
            adapter,
            package_path,
            signing_key,
            seed,
            observed_adapter,
            report_path,
            manifest,
            require_independent_ground_truth,
            cancellation_check,
            checkpoint_path,
            resume,
        )
        return self.complete(
            plan, resume, performance_budget, performance_baseline
        )

    def complete(
        self,
        plan: plans.Plan,
        resume: bool,
        performance_budget: performance.Budget | None,
        performance_baseline: performance.Baseline | None,
    ) -> reporting.Record:
        """Executes the stages of a planned run and signs its report.

        Args:
            plan: The run's checked, fixed inputs.
            resume: Whether to resume from the plan's checkpoint.
            performance_budget: Limits the measured operations must meet, if
                any.
            performance_baseline: Earlier run the measurements are compared
                with, if any.

        Returns:
            The signed lifecycle report. If a completed checkpoint is resumed,
            the previously written report after verification.

        Raises:
            RuntimeError: If the stage order is invalid.
        """
        plan.poll()
        plan.conform()
        plan.audit()
        stage_names = tuple(stage.stage.value for stage in self.stages)
        required_stage_names = tuple(
            stage.value for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        )
        if stage_names != required_stage_names:
            raise RuntimeError("modality lifecycle stage order is invalid")
        resume_count = 0
        resume_digests: tuple[str, ...] = ()
        checkpoint: checkpoints.Checkpoint | None = None
        if plan.checkpoint_target is not None:
            plan.expect(resume=resume)
            if resume:
                checkpoint = plan.reload(stage_names)
                if checkpoint.status == "completed":
                    return plan.verify()
                resume_count = len(checkpoint.completed_stages)
                resume_digests = checkpoint.stage_digests
        completed_stages = list(
            checkpoint.completed_stages if checkpoint is not None else ()
        )
        stage_digests = list(
            checkpoint.stage_digests if checkpoint is not None else ()
        )
        plan.persist(completed_stages, stage_digests, "running")
        context = state.Context(
            adapter=plan.adapter,
            package_path=plan.package_target,
            signing_key=plan.signing_key,
            seed=plan.seed,
            observed_adapter=plan.observed,
            manifest=plan.manifest,
            require_independent_ground_truth=plan.require_independent_ground_truth,
        )
        self.run_stages(
            plan,
            context,
            (resume_count, resume_digests),
            (completed_stages, stage_digests),
        )
        plan.poll()
        report = plans.publish(
            plan, context, performance_budget, performance_baseline
        )
        if plan.checkpoint_target is not None:
            plan.persist(completed_stages, stage_digests, "completed")
        return report

    def declaration(self, report: reporting.Record) -> modalities.Declaration:
        """Converts a report into an explicit experimental declaration.

        Args:
            report: Completed lifecycle report.

        Returns:
            A modality declaration with the report's stage evidence and
            limitations; its status is extension-only for extension modalities
            and experimental otherwise.
        """
        return modalities.Declaration(
            modality=modalities.restore(report.modality),
            status=(
                modalities.Status.EXTENSION_ONLY
                if report.modality.startswith("extension:")
                else modalities.Status.EXPERIMENTAL
            ),
            stage_evidence=report.stage_evidence,
            required_extras=("core",),
            failure_modes=(
                "malformed or unsupported modality input",
                "representation or codec dependency unavailable",
                "resource limits exceeded",
                "package signature or inference parity failure",
            ),
            limitations=tuple(
                str(value)
                for value in cast(
                    "Sequence[object]", report.documentation["limitations"]
                )
            ),
        )
