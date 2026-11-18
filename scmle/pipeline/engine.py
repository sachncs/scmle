"""The pipeline engine: orders the stages and runs them under the run state.

Each stage names the stages it needs (`requires`). `order` checks that every
dependency is present, rejects duplicates and cycles, and returns a
deterministic topological order (earlier lifecycle states first, then the
order the stages were given).
`Pipeline` runs that order: before each stage it checks cancellation, it
moves the state machine only when the stage's state differs from the
current one, it limits the math-library threads, times the stage, consults
the reviewers once the last stage of a state has run, and maps a failure to
the run's terminal state. With an archive it writes each stage's record once
the stage and its review are done, and the completion marker after the
completed state. `Pipeline.advance(until=...)` runs only the stages a given
stage needs, which is the side-effect-free plan preview.
"""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import time
from typing import ClassVar
from typing import Protocol
from typing import TypeVar

from scmle.contracts import errors
from scmle.pipeline import archive
import scmle.pipeline.context
from scmle.runtime import limits
from scmle.runtime import state


class Node(Protocol):
    """The structural contract of a stage the engine can order and run.

    The engine depends on this protocol, not on the stage hierarchy, so a
    replacement stage needs no base class.
    """

    state: ClassVar[state.Progress]
    requires: ClassVar[tuple[type, ...]]

    def describe(self, context: scmle.pipeline.context.Context) -> str:
        """Returns the reason recorded on entering the stage's state.

        Args:
            context: The run context.

        Returns:
            The reason text attached to the state transition.
        """
        ...

    def execute(self, context: scmle.pipeline.context.Context) -> None:
        """Performs the stage, reading and writing the context.

        Args:
            context: The run context.
        """


StageT = TypeVar("StageT", bound=Node)
LIFECYCLE = tuple(state.Progress)
"""The lifecycle states in order; `order` runs earlier states first."""


def order(stages: Sequence[StageT]) -> tuple[StageT, ...]:
    """Orders stages so that every stage runs after the stages it requires.

    A requirement is met by an instance of the required class or of a
    subclass, so a replacement stage satisfies its predecessor's dependents.
    Among the stages whose requirements have run, the one earliest in the
    lifecycle runs first, and ties keep the order the stages were given, so
    the state machine only ever moves forward.

    Args:
        stages: Stage instances, each with a `requires` tuple of classes.

    Returns:
        The stages in execution order.

    Raises:
        ConfigError: If a required stage is missing, a stage appears twice,
            or the dependencies contain a cycle.
    """
    kinds = [type(stage) for stage in stages]
    if len(set(kinds)) != len(kinds):
        raise errors.ConfigError("a stage appears more than once")
    for stage in stages:
        missing = [
            kind.__name__
            for kind in stage.requires
            if not any(isinstance(other, kind) for other in stages)
        ]
        if missing:
            raise errors.ConfigError(
                f"{type(stage).__name__} requires {missing}, which are not in "
                "the workflow"
            )
    remaining = list(stages)
    ordered: list[StageT] = []
    while remaining:
        candidates = [
            stage
            for stage in remaining
            if all(
                any(isinstance(done, kind) for done in ordered)
                for kind in stage.requires
            )
        ]
        if not candidates:
            cycle = [type(stage).__name__ for stage in remaining]
            raise errors.ConfigError(
                f"stage dependencies contain a cycle among {cycle}"
            )
        ready = min(candidates, key=lambda stage: LIFECYCLE.index(stage.state))
        ordered.append(ready)
        remaining.remove(ready)
    return tuple(ordered)


@dataclasses.dataclass(frozen=True, slots=True)
class Lap:
    """What one stage cost.

    Attributes:
        seconds: Wall-clock seconds the stage's `execute` took.
        memory: Peak resident memory of the process in GB after the stage.
    """

    seconds: float
    memory: float


Review = Callable[[state.Progress, scmle.pipeline.context.Context], None]
"""Consults the approval gates of a state once its last stage has run."""


class Pipeline:
    """Runs an ordered stage sequence under one state machine.

    Attributes:
        stages: The stages in execution order.
        records: Where each finished stage's record and the completion
            marker are written, or None for a run that keeps none.
        timings: The cost of each stage of the latest run, keyed by the
            stage's class name.
    """

    def __init__(
        self, stages: Sequence[Node], records: archive.Directory | None = None
    ) -> None:
        """Orders the stages.

        Args:
            stages: The stages; see `order` for the errors.
            records: The run's stage records, or None.
        """
        self.stages = order(stages)
        self.records = records
        self.timings: dict[str, Lap] = {}

    def select(self, until: type) -> tuple[Node, ...]:
        """Returns the stage `until` and every stage it needs, in order.

        Args:
            until: The class of the last stage to run.

        Returns:
            The stages to run, in execution order.

        Raises:
            ConfigError: If no stage is an instance of `until`.
        """
        needed = [stage for stage in self.stages if isinstance(stage, until)]
        if not needed:
            raise errors.ConfigError(f"no {until.__name__} in the workflow")
        for stage in reversed(self.stages):
            if stage not in needed and any(
                isinstance(stage, kind)
                for dependent in needed
                for kind in dependent.requires
            ):
                needed.append(stage)
        return tuple(stage for stage in self.stages if stage in needed)

    def advance(
        self,
        context: scmle.pipeline.context.Context,
        machine: state.Tracker | None = None,
        review: Review | None = None,
        until: type | None = None,
    ) -> None:
        """Runs the stages, or those `until` needs, without ending the run.

        Args:
            context: The run context every stage reads and writes.
            machine: The run's state machine, or None for a preview.
            review: Approval gates, consulted after the last stage of each
                state, or None.
            until: The class of the last stage to run, or None for all.
        """
        stages = self.stages if until is None else self.select(until)
        for index, stage in enumerate(stages):
            context.check_cancelled()
            if machine is not None and stage.state is not machine.state:
                machine.transition(stage.state, stage.describe(context))
            started = time.perf_counter()
            with context.regulator.cap():
                stage.execute(context)
            lap = Lap(time.perf_counter() - started, limits.peak_memory_gb())
            self.timings[type(stage).__name__] = lap
            following = stages[index + 1] if index + 1 < len(stages) else None
            if review is not None and (
                following is None or following.state is not stage.state
            ):
                review(stage.state, context)
            if self.records is not None:
                self.records.record(
                    self.stages.index(stage),
                    type(stage).__name__,
                    stage.state,
                    lap.seconds,
                )

    def run(
        self,
        context: scmle.pipeline.context.Context,
        machine: state.Tracker,
        review: Review,
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
