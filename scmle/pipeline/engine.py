"""The pipeline engine: runs the ordered stages under the run state.

The order comes from `scmle.runtime.engine` (`order`, with the `CLASSES`
dependency by default: a need is a stage class, met by an instance of it or
of a subclass; earlier lifecycle states first).
`Pipeline` runs that order: before each stage it checks cancellation, it
moves the state machine only when the stage's state differs from the
current one, it limits the math-library threads, times the stage, consults
the reviewers once the last stage of a state has run, and maps a failure to
the run's terminal state. With an archive it writes each stage's product and
record once the stage and its review are done, and the completion marker
after the completed state; on a resumed run a stage whose product the
archive restores is not run again, its product is applied instead
(`Archive.restore`). A stage that returns its product has it applied to the
board (`Product.apply`), the one path by which a product enters the board
whether the stage ran or was restored; a board that keeps products holds
each one by type (`Product.hold`), and later stages read it from there.
`Pipeline.advance(until=...)` runs only the stages a given stage needs,
which is the side-effect-free plan preview. The tabular run and the
modality lifecycle are two boards and two archives on this one engine: the
run directory's stage records (`archive.Directory`) and the modality run's
signed checkpoint (`modality_lifecycle.checkpoints.Replay`).
"""

from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Sequence
import contextlib
import dataclasses
import enum
import time
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import Protocol
from typing import TypeVar

from scmle.contracts import errors
from scmle.contracts import records
from scmle.pipeline import products
from scmle.runtime import engine
from scmle.runtime import limits
from scmle.runtime import state
from scmle.runtime import tracing

ContextT = TypeVar("ContextT", contravariant=True)  # noqa: PLC0105 -- no underscores (naming rule)
MadeT = TypeVar("MadeT")


class Regulator(Protocol):
    """Limits the math-library threads while a stage runs."""

    def cap(self) -> contextlib.AbstractContextManager[object]:
        """Returns a context manager that applies the limit while open."""
        ...


class Board(Protocol):
    """What the engine needs of a run context: cancellation and a thread cap.

    The tabular run's context and the modality lifecycle's board
    (`modality_lifecycle.state.Board`) are both boards.
    """

    @property
    def regulator(self) -> Regulator:
        """Limits the math-library threads while a stage runs."""
        ...

    def check_cancelled(self) -> None:
        """Raises if the run was cancelled."""
        ...


BoardT = TypeVar("BoardT", bound=Board)


class Archive(Protocol[MadeT]):
    """Where a run keeps each finished stage and from which it resumes.

    It is generic over what a stage makes. `archive.Directory` keeps the
    tabular run's stage records and the modality lifecycle's signed
    checkpoint is `modality_lifecycle.checkpoints.Replay`, both over
    products; the workflow executor's object store is
    `workflow.executor.records.Bucket`, over a step's outputs.
    """

    def record(
        self,
        index: int,
        name: str,
        progress: enum.StrEnum,
        seconds: float,
        made: MadeT | None = None,
        needs: dict[str, str | None] | None = None,
    ) -> records.Layout:
        """Keeps one finished stage.

        Args:
            index: The stage's position in the run, from zero.
            name: The stage's name.
            progress: The state the stage ran in, or a step ended in.
            seconds: Seconds the stage took.
            made: What the stage made, or None.
            needs: The product digests of the stages it needs, by name.

        Returns:
            The record kept; its `product` entry is the product's digest.
        """
        ...

    def restore(
        self,
        index: int,
        name: str,
        makes: type[MadeT] | None,
        needs: dict[str, str | None],
    ) -> tuple[MadeT, str] | None:
        """Loads a finished stage's product, if the stage need not run again.

        Args:
            index: The stage's position in the run, from zero.
            name: The stage's name.
            makes: The stage's product type, or None.
            needs: The product digests this attempt has for its needs.

        Returns:
            The product and its digest, or None when the stage must run.
        """
        ...

    def finish(self) -> None:
        """Marks the run complete, after everything else."""
        ...


class Node(Protocol[ContextT]):
    """The structural contract of a stage the engine can order and run.

    The engine depends on this protocol, not on the stage hierarchy, so a
    replacement stage needs no base class. It is generic over the context
    the stage reads and writes.
    """

    state: ClassVar[state.Progress]
    requires: ClassVar[tuple[type, ...]]
    makes: ClassVar[type[products.Product] | None]

    def describe(self, context: ContextT) -> str:
        """Returns the reason recorded on entering the stage's state.

        Args:
            context: The run context.

        Returns:
            The reason text attached to the state transition.
        """
        ...

    def execute(self, context: ContextT) -> products.Product | None:
        """Performs the stage, reading the context.

        Args:
            context: The run context.

        Returns:
            The stage's product, which the engine applies to the context,
            or None for a stage that sets the context itself.
        """


@dataclasses.dataclass(frozen=True, slots=True)
class Lap:
    """What one stage cost.

    Attributes:
        seconds: Wall-clock seconds the stage's `execute` took.
        memory: Peak resident memory of the process in GB after the stage.
    """

    seconds: float
    memory: float


Review = Callable[[state.Progress, BoardT], None]
"""Consults the approval gates of a state once its last stage has run."""


class Pipeline(Generic[BoardT]):
    """Runs an ordered stage sequence under one state machine.

    Attributes:
        stages: The stages in execution order.
        records: Where each finished stage's product and record and the
            completion marker are written, and from which a resumed run
            restores products, or None for a run that keeps none.
        dependency: How the stages name and meet their needs.
        timings: The cost of each stage that ran in the latest run, keyed by
            the stage's name.
        digests: The product digest of each stage of the latest run, by
            name; None when a product could not be digested.
        restored: The names of the stages whose product was restored instead
            of running them, in order.
    """

    def __init__(
        self,
        stages: Sequence[Node[BoardT]],
        records: Archive[products.Product] | None = None,
        dependency: engine.Dependency[Any] = engine.CLASSES,
    ) -> None:
        """Orders the stages.

        Args:
            stages: The stages; see `order` for the errors.
            records: The run's stage records, or None.
            dependency: How the stages name and meet their needs.
        """
        self.stages = engine.order(stages, dependency)
        self.records = records
        self.dependency = dependency
        self.timings: dict[str, Lap] = {}
        self.digests: dict[str, str | None] = {}
        self.restored: list[str] = []

    def select(self, until: Hashable) -> tuple[Node[BoardT], ...]:
        """Returns the stages meeting `until` and every stage they need.

        Args:
            until: The need the last stages to run meet (a stage class).

        Returns:
            The stages to run, in execution order.

        Raises:
            ConfigError: If no stage meets `until`.
        """
        needed = [s for s in self.stages if self.dependency.meets(s, until)]
        if not needed:
            raise errors.ConfigError(
                f"no {self.dependency.title(until)} in the workflow"
            )
        for stage in reversed(self.stages):
            if stage not in needed and any(
                self.dependency.meets(stage, need)
                for dependent in needed
                for need in self.dependency.needs(dependent)
            ):
                needed.append(stage)
        return tuple(stage for stage in self.stages if stage in needed)

    def ancestry(self, stage: Node[BoardT]) -> dict[str, str | None]:
        """The product digests of every stage `stage` needs, directly or not.

        Args:
            stage: A stage that is about to run or be restored.

        Returns:
            The digests this run made, by stage name.
        """
        title = self.dependency.title
        needed = [
            other
            for other in self.select(type(stage))
            if other is not stage and title(other) in self.digests
        ]
        return {title(other): self.digests[title(other)] for other in needed}

    def advance(
        self,
        context: BoardT,
        machine: state.Machine[state.Progress] | None = None,
        review: Review[BoardT] | None = None,
        until: Hashable | None = None,
    ) -> None:
        """Runs the stages, or those `until` needs, without ending the run.

        With records, a stage whose product they restore is not executed:
        its product is applied to the context, its state still entered and
        its review still consulted, and its record is kept as it is.

        Args:
            context: The run context every stage reads and writes.
            machine: The run's state machine, or None for a preview.
            review: Approval gates, consulted after the last stage of each
                state, or None.
            until: The need the last stages to run meet, or None for all.
        """
        stages = self.stages if until is None else self.select(until)
        for index, stage in enumerate(stages):
            context.check_cancelled()
            name = self.dependency.title(stage)
            restored = self.restore(stage)
            if machine is not None and stage.state is not machine.state:
                reason = stage.describe(context)
                if restored is not None:
                    reason = f"{reason} (restored from its record)"
                machine.transition(stage.state, reason)
            lap = Lap(0.0, 0.0)
            made: products.Product | None = None
            if restored is not None:
                restored.apply(context)
                self.restored.append(name)
            else:
                started = time.perf_counter()
                with (
                    tracing.TRACING.span(
                        f"stage {name}", {"scmle.stage": name}
                    ),
                    context.regulator.cap(),
                ):
                    made = stage.execute(context)
                if made is not None:
                    made.apply(context)
                lap = Lap(
                    time.perf_counter() - started, limits.peak_memory_gb()
                )
                self.timings[name] = lap
            following = stages[index + 1] if index + 1 < len(stages) else None
            if review is not None and (
                following is None or following.state is not stage.state
            ):
                review(stage.state, context)
            if made is None and restored is None and stage.makes is not None:
                # A stage that sets the context itself: its product is read
                # back and held, so the stages after it read it as a product.
                made = stage.makes.capture(context)
                made.hold(context)
            if self.records is not None and restored is None:
                needs = self.ancestry(stage)
                record = self.records.record(
                    self.stages.index(stage),
                    name,
                    stage.state,
                    lap.seconds,
                    made,
                    needs,
                )
                self.digests[name] = record["product"]

    def restore(self, stage: Node[BoardT]) -> products.Product | None:
        """The product of a finished stage a resumed run need not run again.

        Args:
            stage: The stage about to run.

        Returns:
            The product the records restore, its digest noted in `digests`,
            or None when the stage must run.
        """
        if self.records is None:
            return None
        name = self.dependency.title(stage)
        found = self.records.restore(
            self.stages.index(stage), name, stage.makes, self.ancestry(stage)
        )
        if found is None:
            return None
        made, self.digests[name] = found
        return made

    def run(
        self,
        context: BoardT,
        machine: state.Machine[state.Progress],
        review: Review[BoardT],
        details: Callable[[], dict[str, str | list[str]]],
        artifacts: tuple[str, ...] = (),
    ) -> None:
        """Runs every stage and ends the run in its terminal state.

        On a cancellation, block or failure the error's `details` gain the
        entries of `details()` (a cancellation keeps a `run_dir` it already
        has; a keyboard interrupt gets only `run_dir`).

        Args:
            context: The run context every stage reads and writes.
            machine: The run's state machine.
            review: Approval gates, consulted after the last stage of each
                state.
            details: Builds the error details at the time of the failure.
            artifacts: Paths recorded with the completed state.

        Raises:
            BlockedError: If a stage or an approval gate blocks the run.
            CancelledError: If the run is cancelled, or interrupted from the
                keyboard.
            ScmleError: If a stage fails; any other exception is wrapped in it.
        """
        try:
            self.advance(context, machine, review)
            machine.transition(
                state.Progress.COMPLETED, "run complete", artifacts
            )
            if self.records is not None:
                self.records.finish()
        except errors.CancelledError as error:
            found = details()
            error.details.setdefault("run_dir", found["run_dir"])
            error.details.update(
                {k: v for k, v in found.items() if k != "run_dir"}
            )
            machine.transition(state.Progress.CANCELLED, str(error))
            raise
        except KeyboardInterrupt as error:
            machine.transition(state.Progress.CANCELLED, "interrupted")
            raise errors.CancelledError(
                "interrupted", {"run_dir": details()["run_dir"]}
            ) from error
        except errors.BlockedError as error:
            error.details.update(details())
            machine.transition(state.Progress.BLOCKED, str(error))
            raise
        except errors.ScmleError as error:
            # This also takes an `InputError` (and `KindError`,
            # `OrderError`): before ERR-20 those were bare ValueError,
            # TypeError or RuntimeError raises that fell to the catch-all
            # below, failing as `internal:` and escaping wrapped. Kept on
            # purpose: an InputError is by contract a user or data problem,
            # not a defect, so the run fails as `input:` and the error
            # escapes as itself; a bare ValueError still fails as internal.
            error.details.update(details())
            machine.transition(
                state.Progress.FAILED, f"{error.category}: {error}"
            )
            raise
        except Exception as error:
            machine.transition(state.Progress.FAILED, f"internal: {error!r}")
            raise errors.ScmleError(str(error), details()) from error
