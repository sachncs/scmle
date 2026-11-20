"""The runner that executes the lifecycle stages on the pipeline engine."""

from __future__ import annotations

from typing import cast
from typing import ClassVar
from typing import TYPE_CHECKING

from scmle.contracts import data
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.pipeline import engine
from scmle.pipeline.modality_lifecycle import checkpoints
from scmle.pipeline.modality_lifecycle import constants
from scmle.pipeline.modality_lifecycle import curators
from scmle.pipeline.modality_lifecycle import explanation
from scmle.pipeline.modality_lifecycle import plans
from scmle.pipeline.modality_lifecycle import reporting
from scmle.pipeline.modality_lifecycle import state
from scmle.pipeline.modality_lifecycle import steps
import scmle.runtime.state

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence
    from pathlib import Path


def measure(
    pipeline: engine.Pipeline[state.Board], board: state.Board
) -> tuple[performance.Observation, ...]:
    """The measured operations of a finished run, in operation order.

    The engine timed every stage; a stage with a `operation`
    reports its time, and inference reports its load and predict times.

    Args:
        pipeline: The engine that ran the stages.
        board: The run's board after the last stage.

    Returns:
        One observation per measured operation.
    """
    found = {
        observation.operation: observation
        for observation in board.get(state.Reproduction).observations
    }
    for stage in pipeline.stages:
        operation = cast("state.Stage", stage).operation
        if operation is not None:
            found[operation] = state.observe(
                operation,
                pipeline.timings[pipeline.dependency.title(stage)],
                cast("state.Stage", stage).samples(board),
            )
    return tuple(
        found[operation]
        for operation in performance.Operation
        if operation in found
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
        explanation.Explainability(),
        steps.Packaging(),
        steps.Inference(),
        steps.Monitoring(),
        steps.Documentation(),
        steps.Recovery(),
    )

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
                `AbortedError` is raised.
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
        pipeline = engine.Pipeline[state.Board](self.stages)
        stage_names = tuple(
            cast("state.Stage", stage).stage.value for stage in pipeline.stages
        )
        required_stage_names = tuple(
            stage.value for stage in modalities.REQUIRED_LIFECYCLE_STAGES
        )
        if stage_names != required_stage_names:
            raise RuntimeError("modality lifecycle stage order is invalid")
        checkpoint: checkpoints.Checkpoint | None = None
        if plan.target is not None:
            plan.expect(resume=resume)
            if resume:
                checkpoint = plan.reload(stage_names)
                if checkpoint.status == "completed":
                    return plan.verify()
        replay = checkpoints.Replay(plan, checkpoint)
        replay.start()
        pipeline.records = replay
        board = state.Board(
            plan, () if checkpoint is None else tuple(checkpoint.completed)
        )
        machine = scmle.runtime.state.Machine(
            state.TABLE, scmle.runtime.state.Progress.CREATED
        )
        pipeline.advance(board, machine)
        plan.poll()
        report = plans.publish(
            plan,
            board,
            measure(pipeline, board),
            performance_budget,
            performance_baseline,
        )
        replay.finish()
        machine.transition(scmle.runtime.state.Progress.COMPLETED)
        return report

    def declaration(self, report: reporting.Record) -> modalities.Declaration:
        """Converts a report into an explicit experimental declaration.

        Args:
            report: Completed lifecycle report.

        Returns:
            A modality declaration with the report's stage evidence and
            limitations; its status is extension-only for extension modalities
            and experimental otherwise, when the limitations also give the
            reason it is not yet at parity with the tabular path.
        """
        bespoke = report.modality.startswith("extension:")
        limitations = tuple(
            str(value)
            for value in cast(
                "Sequence[object]", report.documentation["limitations"]
            )
        )
        return modalities.Declaration(
            modality=modalities.restore(report.modality),
            status=(
                modalities.Status.EXTENSION_ONLY
                if bespoke
                else modalities.Status.EXPERIMENTAL
            ),
            stage_evidence=report.evidence,
            required_extras=("core",),
            failure_modes=(
                "malformed or unsupported modality input",
                "representation or codec dependency unavailable",
                "resource limits exceeded",
                "package signature or inference parity failure",
            ),
            limitations=(
                limitations if bespoke else (*limitations, constants.SHORTFALL)
            ),
        )
