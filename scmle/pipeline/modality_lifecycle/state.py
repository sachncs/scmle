"""The shared lifecycle state and the stage interface."""

from __future__ import annotations

import abc
import dataclasses
import platform
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING

from scmle.contracts import data
from scmle.contracts import events
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.runtime import limits

if TYPE_CHECKING:
    from pathlib import Path

    from sklearn.linear_model import LogisticRegression

    from scmle.pipeline.modality_lifecycle import containers
    from scmle.pipeline.modality_lifecycle import curators


@dataclasses.dataclass
class Context:
    """Mutable products shared by the fifteen lifecycle stages.

    Attributes:
        adapter: Modality adapter the lifecycle runs on.
        package_path: Directory the model package is written to.
        signing_key: Key that signs the package and report.
        seed: Random seed of the run.
        observed_adapter: Adapter over the data used for monitoring.
        manifest: Modality manifest, if the caller supplied one.
        require_independent_ground_truth: Whether explanation quality needs
            independently authored ground truth.
        resume_replay: Whether the run replays stages already checkpointed.
        stage_records: Result of each completed stage.
        curation_report: Curation report produced by the curation stage.
        partitions: Train, validation and test partitions, once split.
        feature_batches: Feature batch per partition name.
        model: Fitted classifier, once trained.
        evaluation: Held-out evaluation evidence.
        performance_observations: Measured operations, one per kind.
        data_explanation: Data explainability evidence.
        model_explanation: Model explainability evidence.
        package: Identity of the written package, once packaged.
        inference: Inference smoke-test evidence.
        monitoring: Drift-monitoring evidence.
        documentation: Documentation evidence.
        recovery: Recovery and resource evidence.
    """

    adapter: curators.Preparer
    package_path: Path
    signing_key: str | bytes
    seed: int
    observed_adapter: curators.Preparer
    manifest: data.Passport | None = None
    require_independent_ground_truth: bool = False
    resume_replay: bool = False
    stage_records: dict[modalities.Phase, dict[str, object]] = (
        dataclasses.field(default_factory=dict)
    )
    curation_report: Any = None
    partitions: containers.Partitions | None = None
    feature_batches: dict[str, containers.Batch] = dataclasses.field(
        default_factory=dict
    )
    model: LogisticRegression | None = None
    evaluation: dict[str, object] = dataclasses.field(default_factory=dict)
    performance_observations: dict[
        performance.Operation,
        performance.Observation,
    ] = dataclasses.field(default_factory=dict)
    data_explanation: dict[str, object] = dataclasses.field(
        default_factory=dict
    )
    model_explanation: dict[str, object] = dataclasses.field(
        default_factory=dict
    )
    package: containers.Package | None = None
    inference: dict[str, object] = dataclasses.field(default_factory=dict)
    monitoring: dict[str, object] = dataclasses.field(default_factory=dict)
    documentation: dict[str, object] = dataclasses.field(default_factory=dict)
    recovery: dict[str, object] = dataclasses.field(default_factory=dict)

    def record(
        self,
        stage: modalities.Phase,
        payload: dict[str, object],
    ) -> None:
        """Records one stage result for the final evidence index.

        Args:
            stage: Stage that produced the result.
            payload: JSON-native result of the stage.
        """
        self.stage_records[stage] = payload

    def clock(
        self,
        operation: performance.Operation,
        elapsed_seconds: float,
        sample_count: int,
    ) -> None:
        """Stores one measured operation with machine and timestamp context.

        The observation also carries the process's peak memory, the platform
        and Python version, and a UTC timestamp. It replaces any earlier
        observation of the same operation.

        Args:
            operation: Operation that was measured.
            elapsed_seconds: Wall-clock duration of the operation.
            sample_count: Number of samples the operation processed.
        """
        self.performance_observations[operation] = performance.Observation(
            operation=operation,
            elapsed=elapsed_seconds,
            memory=limits.peak_memory_gb(),
            samples=sample_count,
            machine=platform.platform(),
            observed=events.stamp(),
            software=(
                f"{platform.python_implementation()} "
                f"{platform.python_version()}"
            ),
        )


class Stage(abc.ABC):
    """One polymorphic stage in a modality lifecycle.

    Attributes:
        stage: Lifecycle stage this class implements.
        performance_operation: Operation whose duration is measured for this
            stage, or None if it is not timed.
    """

    stage: ClassVar[modalities.Phase]
    performance_operation: ClassVar[performance.Operation | None] = None

    @abc.abstractmethod
    def execute(self, context: Context) -> None:
        """Executes the stage and records its evidence.

        Args:
            context: Shared lifecycle state, updated in place.
        """
