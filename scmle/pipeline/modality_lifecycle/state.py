"""The lifecycle board, the stage products and the stage interface.

The fifteen lifecycle stages are nodes of the one pipeline engine
(`scmle.pipeline.engine`): the engine orders them by what they need and by
the run state their phase belongs to (`STATES`), checks cancellation before
each, caps the math-library threads, times each one and hands each finished
stage to the run's archive, the signed checkpoint (`checkpoints.Replay`).
Each stage reads the products of the stages it needs from the `Board` and
returns its own product, a frozen `Proof`; the engine applies it to the
board. The board holds the run's fixed inputs and the products made so far,
and no stage assigns an attribute on it.

Every product carries `evidence`, the JSON-native stage record the report
holds and the checkpoint digests, unchanged from before the engine. Products
are derived (`products.Derived`): a resumed run replays every stage against
the caller's adapter and checks each replayed digest against the checkpoint,
so no fitted model or adapter state is trusted from disk.
"""

from __future__ import annotations

import abc
import dataclasses
import platform
from typing import Any
from typing import ClassVar
from typing import TYPE_CHECKING
from typing import TypeVar

from scmle.contracts import errors
from scmle.contracts import events
from scmle.contracts import modalities
from scmle.contracts import performance
from scmle.contracts import tasks
from scmle.pipeline import products
from scmle.runtime import limits
import scmle.runtime.state

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sklearn.linear_model import LogisticRegression

    from scmle.pipeline import engine
    from scmle.pipeline.modality_lifecycle import containers
    from scmle.pipeline.modality_lifecycle import plans

ProofT = TypeVar("ProofT", bound="Proof")

STATES = {
    modalities.Phase.INGESTION: scmle.runtime.state.Progress.VALIDATING,
    modalities.Phase.VALIDATION: scmle.runtime.state.Progress.VALIDATING,
    modalities.Phase.PROFILING: scmle.runtime.state.Progress.PROFILING,
    modalities.Phase.CURATION: scmle.runtime.state.Progress.PROFILING,
    modalities.Phase.SPLITTING: scmle.runtime.state.Progress.PROFILING,
    modalities.Phase.FEATURE_PREPARATION: (
        scmle.runtime.state.Progress.TRAINING
    ),
    modalities.Phase.MODELLING: scmle.runtime.state.Progress.TRAINING,
    modalities.Phase.EVALUATION: scmle.runtime.state.Progress.TESTING,
    modalities.Phase.DATA_EXPLAINABILITY: scmle.runtime.state.Progress.TESTING,
    modalities.Phase.MODEL_EXPLAINABILITY: (
        scmle.runtime.state.Progress.TESTING
    ),
    modalities.Phase.PACKAGING: scmle.runtime.state.Progress.PACKAGING,
    modalities.Phase.INFERENCE: scmle.runtime.state.Progress.PACKAGING,
    modalities.Phase.MONITORING: scmle.runtime.state.Progress.PACKAGING,
    modalities.Phase.DOCUMENTATION: scmle.runtime.state.Progress.PACKAGING,
    modalities.Phase.RECOVERY_AND_RESOURCE_TESTS: (
        scmle.runtime.state.Progress.PACKAGING
    ),
}
"""The run state each lifecycle phase belongs to, as the tabular stages: the
engine runs earlier states first, so with each stage's `requires` the order
is the required lifecycle order."""

TABLE = scmle.runtime.state.chain(
    (
        scmle.runtime.state.Progress.CREATED,
        *dict.fromkeys(STATES.values()),
        scmle.runtime.state.Progress.COMPLETED,
    ),
    scmle.runtime.state.ABORT_STATES,
)
"""The lifecycle's transition table on the shared `Machine`: its run states
forward in order (it has no planning or selecting state), or an abort."""


@dataclasses.dataclass(frozen=True)
class Proof(products.Derived):
    """What one lifecycle stage made: its evidence, and what later stages read.

    Attributes:
        phase: The lifecycle stage the product belongs to.
        evidence: The JSON-native evidence the stage recorded.
    """

    phase: ClassVar[modalities.Phase]
    evidence: dict[str, object]

    def summary(self) -> object:
        """The stage's evidence, which the checkpoint digests."""
        return self.evidence

    def apply(self, board: object) -> None:
        """Adds the product to the lifecycle board.

        Args:
            board: The lifecycle `Board`.

        Raises:
            KindError: If `board` is not a lifecycle board.
        """
        if not isinstance(board, Board):
            raise errors.KindError("a lifecycle product applies to a Board")
        board.made[self.phase] = self


@dataclasses.dataclass(frozen=True)
class Intake(Proof):
    """The source identity and sample count."""

    phase = modalities.Phase.INGESTION


@dataclasses.dataclass(frozen=True)
class Census(Proof):
    """The label and class counts."""

    phase = modalities.Phase.VALIDATION


@dataclasses.dataclass(frozen=True)
class Profile(Proof):
    """The modality's curation report.

    Attributes:
        report: The curation report the adapter produced.
    """

    phase = modalities.Phase.PROFILING
    report: Any


@dataclasses.dataclass(frozen=True)
class Clearance(Proof):
    """The finding that no blocking curation finding is unresolved."""

    phase = modalities.Phase.CURATION


@dataclasses.dataclass(frozen=True)
class Partition(Proof):
    """The train, validation and test partitions.

    Attributes:
        partitions: The partitions.
    """

    phase = modalities.Phase.SPLITTING
    partitions: containers.Partitions


@dataclasses.dataclass(frozen=True)
class Features(Proof):
    """The feature batch of each partition.

    Attributes:
        batches: The batch of each partition, by partition name.
    """

    phase = modalities.Phase.FEATURE_PREPARATION
    batches: Mapping[str, containers.Batch]


@dataclasses.dataclass(frozen=True)
class Fit(Proof):
    """The classifier fitted on the training partition.

    Attributes:
        model: The fitted classifier.
    """

    phase = modalities.Phase.MODELLING
    model: LogisticRegression


@dataclasses.dataclass(frozen=True)
class Score(Proof):
    """The held-out metrics and predictions."""

    phase = modalities.Phase.EVALUATION


@dataclasses.dataclass(frozen=True)
class Lineage(Proof):
    """The data lineage and feature summaries."""

    phase = modalities.Phase.DATA_EXPLAINABILITY


@dataclasses.dataclass(frozen=True)
class Explanation(Proof):
    """The model explanation and its quality checks."""

    phase = modalities.Phase.MODEL_EXPLAINABILITY


@dataclasses.dataclass(frozen=True)
class Bundle(Proof):
    """The written, signed model package.

    Attributes:
        package: The package's identity.
    """

    phase = modalities.Phase.PACKAGING
    package: containers.Package


@dataclasses.dataclass(frozen=True)
class Reproduction(Proof):
    """The packaged model reproducing the held-out predictions.

    Attributes:
        observations: The load and inference measurements.
    """

    phase = modalities.Phase.INFERENCE
    observations: tuple[performance.Observation, ...]


@dataclasses.dataclass(frozen=True)
class Drift(Proof):
    """The reference-to-observed drift evidence."""

    phase = modalities.Phase.MONITORING


@dataclasses.dataclass(frozen=True)
class Card(Proof):
    """The dataset and model card."""

    phase = modalities.Phase.DOCUMENTATION


@dataclasses.dataclass(frozen=True)
class Resilience(Proof):
    """The tampered-package rejection and the resource bounds."""

    phase = modalities.Phase.RECOVERY_AND_RESOURCE_TESTS


def governor() -> limits.Governor:
    """The thread cap of a lifecycle run: the default resource limits.

    Returns:
        A governor over the default resources; with no CPU limit its cap
        leaves the math-library threads as they are.
    """
    return limits.Governor(tasks.Resources())


@dataclasses.dataclass(frozen=True)
class Board:
    """The fixed inputs of a lifecycle run and the products made so far.

    Attributes:
        plan: The run's fixed inputs.
        resumed: Values of the stages the resumed checkpoint completed; a
            stage among them replays onto what the interrupted run wrote.
        regulator: The thread cap the engine applies around each stage.
        made: The product of each finished stage, by lifecycle stage; only
            `Proof.apply` adds one.
    """

    plan: plans.Plan
    resumed: tuple[str, ...] = ()
    regulator: limits.Governor = dataclasses.field(default_factory=governor)
    made: dict[modalities.Phase, Proof] = dataclasses.field(
        default_factory=dict
    )

    def check_cancelled(self) -> None:
        """Polls the run's cancellation check (`plans.Plan.poll`)."""
        self.plan.poll()

    def get(self, kind: type[ProofT]) -> ProofT:
        """The product of a finished stage.

        Args:
            kind: The product type.

        Returns:
            The product.

        Raises:
            OrderError: If the stage that makes it has not run.
        """
        found = self.made.get(kind.phase)
        if not isinstance(found, kind):
            raise errors.OrderError(f"{kind.__name__} is not made yet")
        return found

    def replaying(self, phase: modalities.Phase) -> bool:
        """Whether a stage replays one a checkpoint already completed.

        Args:
            phase: The stage.

        Returns:
            True if the run resumes a checkpoint that completed the stage.
        """
        return phase.value in self.resumed


def observe(
    operation: performance.Operation, lap: engine.Lap, samples: int
) -> performance.Observation:
    """One measured operation with its machine and time.

    Args:
        operation: What was measured.
        lap: Its seconds and the process's peak memory after it.
        samples: How many samples it processed.

    Returns:
        The observation, stamped now in UTC.
    """
    return performance.Observation(
        operation=operation,
        elapsed=lap.seconds,
        memory=lap.memory,
        samples=samples,
        machine=platform.platform(),
        observed=events.stamp(),
        software=(
            f"{platform.python_implementation()} {platform.python_version()}"
        ),
    )


class Stage(abc.ABC):
    """One lifecycle stage: a node of the pipeline engine.

    Attributes:
        stage: Lifecycle stage this class implements.
        state: The run state the stage belongs to (`STATES`); with
            `requires` it orders the stages in the required lifecycle order.
        requires: The stages whose products this one reads.
        makes: The product the stage returns, a `Proof` (typed as the
            engine node's product, which a class variable cannot narrow).
        performance_operation: Operation the engine's timing of this stage
            reports, or None if it is not reported.
    """

    stage: ClassVar[modalities.Phase]
    state: ClassVar[scmle.runtime.state.Progress]
    requires: ClassVar[tuple[type, ...]] = ()
    makes: ClassVar[type[products.Product] | None]
    performance_operation: ClassVar[performance.Operation | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Gives a stage the run state of its phase.

        Args:
            **kwargs: Passed on to `object.__init_subclass__`.
        """
        super().__init_subclass__(**kwargs)
        cls.state = STATES[cls.stage]

    def describe(self, board: Board | None) -> str:  # noqa: ARG002 -- engine.Node
        """Returns the reason recorded on entering the stage.

        Args:
            board: Unused: the reason names only the stage.

        Returns:
            The reason text.
        """
        return f"modality {self.stage.value}"

    def samples(self, board: Board) -> int:
        """How many samples the reported operation processed.

        Args:
            board: The run's board after the stage.

        Returns:
            The training rows.
        """
        return len(board.get(Features).batches["train"].sample_ids)

    @abc.abstractmethod
    def execute(self, board: Board) -> Proof:
        """Executes the stage.

        Args:
            board: The run's inputs and the products made so far.

        Returns:
            The stage's product.
        """
