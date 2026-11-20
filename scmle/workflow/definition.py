"""The workflow definition: one type for every place it runs.

A `Workflow` is a set of named `Step`s with explicit dependencies. A step has
a kind (`ingest`, `train`, `evaluate`, `monitor`, `batch_predict`, or
`approval`), a timeout, a retry policy with backoff, an optional cost budget
and an optional concurrency group. The workflow has an optional schedule, a
cap on concurrent runs and a cost budget for the whole run.

**Automatic versus approved.** A step marked ``effect="approved"`` changes
something that people depend on (for example a batch prediction that is
published, or a model that is promoted). It may only run after an
``approval`` step that it depends on, directly or through other steps. The
definition refuses to exist otherwise, so no run can reach an approved
action without a person's decision.
"""

from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
import hashlib
import json
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records
from scmle.runtime import engine
from scmle.runtime import retry

NAME_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
MAX_NAME_LENGTH = 64
AUTOMATIC = "automatic"
APPROVED = "approved"


class Kind(enum.StrEnum):
    """What a step does.

    Attributes:
        INGEST: Reads data in.
        TRAIN: Trains a model.
        EVALUATE: Evaluates a model.
        MONITOR: Monitors a model.
        BATCH_PREDICT: Scores a batch of records.
        APPROVAL: Waits for a person's decision.
    """

    INGEST = "ingest"
    TRAIN = "train"
    EVALUATE = "evaluate"
    MONITOR = "monitor"
    BATCH_PREDICT = "batch_predict"
    APPROVAL = "approval"


class Backfill(enum.StrEnum):
    """What to do about schedule slots that were missed.

    Attributes:
        RUN_ALL: Run every missed slot, up to a limit.
        LATEST_ONLY: Run only the newest due slot.
        SKIP: Run a due slot only if it is recent; miss the rest.
    """

    RUN_ALL = "run_all"
    LATEST_ONLY = "latest_only"
    SKIP = "skip"


def vet(name: str, what: str) -> str:
    """Returns `name` if it is 1-64 letters, digits, `-` or `_`.

    Args:
        name: The candidate name.
        what: What the name is, for the error message.

    Returns:
        `name`, unchanged.

    Raises:
        ConfigError: If the name is empty, longer than 64 characters or holds
            another character.
    """
    if not name or len(name) > MAX_NAME_LENGTH or set(name) - NAME_CHARACTERS:
        raise errors.ConfigError(
            f"{what} must be 1-64 characters of letters, digits, - and _"
        )
    return name


@dataclasses.dataclass(frozen=True, slots=True)
class Step:
    """One unit of work.

    Attributes:
        name: Unique name within the workflow.
        kind: What it does.
        needs: Names of steps that must have succeeded first.
        timeout: Limit for one attempt.
        retries: Attempts and backoff.
        budget: Ceiling for this step's cost, if any.
        concurrency: Steps of one group never run at the same time
            beyond the group's limit (see `Workflow.limits`).
        effect: `automatic`, or `approved` (needs an approval upstream).
        parameters: JSON-ready parameters for the handler.
        approvers: For an approval step, the identities that may decide.
        expiry: For an approval step, how long it waits.
        KEYS: Version-1 keys to the written ones (`needs`, `timeout` and
            `expiry` in seconds, `budget`, `concurrency`); `downgrade`
            restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.Migration(
        renames={
            "depends_on": "needs",
            "timeout_seconds": "timeout",
            "cost_budget": "budget",
            "concurrency_group": "concurrency",
            "expires_seconds": "expiry",
        }
    )

    name: str
    kind: Kind
    needs: tuple[str, ...] = ()
    timeout: float = 3600.0
    retries: retry.Retries = dataclasses.field(
        default_factory=lambda: retry.Retries(3, 5.0, 2.0, 300.0)
    )
    budget: float | None = None
    concurrency: str | None = None
    effect: str = AUTOMATIC
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    approvers: tuple[str, ...] = ()
    expiry: float = 86400.0

    def __post_init__(self) -> None:
        """Validates the step.

        Raises:
            ConfigError: If a value is out of range or an approval step has
                no approvers.
        """
        vet(self.name, "step name")
        if self.timeout <= 0 or self.expiry <= 0:
            raise errors.ConfigError("timeouts must be positive")
        if self.budget is not None and self.budget <= 0:
            raise errors.ConfigError("cost_budget must be positive")
        if self.effect not in (AUTOMATIC, APPROVED):
            raise errors.ConfigError("effect must be automatic or approved")
        if self.name in self.needs:
            raise errors.ConfigError(f"step {self.name} depends on itself")
        if self.kind is Kind.APPROVAL and not self.approvers:
            raise errors.ConfigError("an approval step needs approvers")
        if self.kind is Kind.APPROVAL and self.effect == APPROVED:
            raise errors.ConfigError("an approval step is itself automatic")

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready form.

        Returns:
            A dict of all fields with the written keys, the kind as its
            value and the retry policy as a dict.
        """
        return self.KEYS.upgrade(self.layout(self.retries.to_dict()))

    def anchor(self) -> records.Layout:
        """Builds the step in its version-1 layout, the one digests name.

        Returns:
            The step as version 1 wrote it, with the retry policy's field
            names.
        """
        return self.layout(self.retries.anchor())

    def layout(self, retries: Mapping[str, Any]) -> records.Layout:
        """Builds the version-1 layout around a written retry policy.

        Args:
            retries: The retry policy as it is to be written.

        Returns:
            The step's fields with their version-1 keys.
        """
        return {
            "name": self.name,
            "kind": self.kind.value,
            "depends_on": list(self.needs),
            "timeout_seconds": self.timeout,
            "retries": dict(retries),
            "cost_budget": self.budget,
            "concurrency_group": self.concurrency,
            "effect": self.effect,
            "parameters": dict(self.parameters),
            "approvers": list(self.approvers),
            "expires_seconds": self.expiry,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Schedule:
    """When runs start.

    Attributes:
        interval_seconds: Time between slots.
        start_epoch: Epoch seconds of the first slot.
        catch_up: What to do about missed slots.
        max_catch_up: Most slots `run_all` executes in one tick.
    """

    interval_seconds: float
    start_epoch: float = 0.0
    catch_up: Backfill = Backfill.LATEST_ONLY
    max_catch_up: int = 10

    def __post_init__(self) -> None:
        """Validates the schedule.

        Raises:
            ConfigError: If the interval or `max_catch_up` is not positive.
        """
        if self.interval_seconds <= 0 or self.max_catch_up < 1:
            raise errors.ConfigError(
                "interval and max_catch_up must be positive"
            )


@dataclasses.dataclass(frozen=True, slots=True)
class Workflow:
    """The definition.

    Attributes:
        name: Workflow name.
        version: Version label; part of every run's identity.
        steps: The steps.
        schedule: When runs start, if scheduled.
        concurrency: Runs of this workflow allowed at once.
        budget: Ceiling for one run's total cost, if any.
        limits: Concurrency group name to the steps allowed at once (a
            group not listed allows one).
        parallel: Steps the local executor runs at once.
        KEYS: Version-1 keys to the written ones (`concurrency`, `budget`,
            `limits`, `parallel`, and each step's own); `downgrade`
            restores version 1.
    """

    KEYS: ClassVar[records.Migration] = records.nest(
        {
            "max_concurrent_runs": "concurrency",
            "cost_budget": "budget",
            "group_limits": "limits",
            "max_parallel": "parallel",
        },
        parts={"steps": Step.KEYS},
    )

    name: str
    version: str
    steps: tuple[Step, ...]
    schedule: Schedule | None = None
    concurrency: int = 1
    budget: float | None = None
    limits: Mapping[str, int] = dataclasses.field(default_factory=dict)
    parallel: int = 4

    def __post_init__(self) -> None:
        """Validates the whole definition.

        Raises:
            ConfigError: On duplicate names, unknown or cyclic dependencies,
                an approved step with no approval upstream, or bad limits.
        """
        vet(self.name, "workflow name")
        if not self.version or not self.steps:
            raise errors.ConfigError("a workflow needs a version and steps")
        if self.concurrency < 1 or self.parallel < 1:
            raise errors.ConfigError("concurrency limits must be positive")
        if self.budget is not None and self.budget <= 0:
            raise errors.ConfigError("cost_budget must be positive")
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise errors.ConfigError("step names must be unique")
        for step in self.steps:
            missing = [d for d in step.needs if d not in names]
            if missing:
                raise errors.ConfigError(
                    f"step {step.name} depends on unknown steps {missing}"
                )
        self.order()
        for step in self.steps:
            if step.effect == APPROVED and not self.gated(step.name):
                raise errors.ConfigError(
                    f"step {step.name} is an approved action but no approval "
                    "step precedes it"
                )

    def step(self, name: str) -> Step:
        """Looks up a step by name.

        Args:
            name: The step name.

        Returns:
            The step called `name`.

        Raises:
            ConfigError: If there is no such step.
        """
        for candidate in self.steps:
            if candidate.name == name:
                return candidate
        raise errors.ConfigError(f"no step {name!r}")

    def order(self) -> tuple[str, ...]:
        """Computes a deterministic topological order (ties broken by name).

        The engine's waves (`scmle.runtime.engine.layers`), flattened.

        Returns:
            The step names, each after all of its dependencies.
        """
        return tuple(name for layer in self.layers() for name in layer)

    def layers(self) -> tuple[tuple[str, ...], ...]:
        """Groups steps by depth: each layer depends only on earlier ones.

        Returns:
            The layers in order, each a sorted tuple of step names. The
            engine raises `ConfigError` if the dependencies contain a cycle.
        """
        return tuple(
            tuple(step.name for step in layer)
            for layer in engine.layers(self.steps, engine.NAMES)
        )

    def ancestors(self, name: str) -> frozenset[str]:
        """Finds every step `name` depends on, directly or not.

        Args:
            name: The step name.

        Returns:
            The names of all upstream steps.
        """
        found: set[str] = set()
        pending = list(self.step(name).needs)
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(self.step(current).needs)
        return frozenset(found)

    def gated(self, name: str) -> bool:
        """Reports whether an approval step precedes `name`.

        Args:
            name: The step name.

        Returns:
            True if any upstream step is an approval step.
        """
        return any(
            self.step(a).kind is Kind.APPROVAL for a in self.ancestors(name)
        )

    def quota(self, group: str) -> int:
        """Looks up how many steps of a concurrency group may run at once.

        Args:
            group: The group name.

        Returns:
            The configured limit, or 1 for a group that is not listed.
        """
        return int(self.limits.get(group, 1))

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready form.

        Returns:
            A dict of all fields with the written keys, the steps and
            schedule as dicts.
        """
        written = self.KEYS.upgrade(self.anchor())
        written["steps"] = [step.to_dict() for step in self.steps]
        return written

    def anchor(self) -> records.Layout:
        """Builds the definition in its version-1 layout, named by `digest`.

        Returns:
            The definition as version 1 wrote it, every step included.
        """
        return self.layout([step.anchor() for step in self.steps])

    def layout(self, steps: Sequence[Mapping[str, Any]]) -> records.Layout:
        """Builds the version-1 layout around the given steps.

        Args:
            steps: The steps as they are to be written.

        Returns:
            The definition's fields with their version-1 keys.
        """
        return {
            "name": self.name,
            "version": self.version,
            "steps": [dict(step) for step in steps],
            "schedule": None
            if self.schedule is None
            else {
                "interval_seconds": self.schedule.interval_seconds,
                "start_epoch": self.schedule.start_epoch,
                "catch_up": self.schedule.catch_up.value,
                "max_catch_up": self.schedule.max_catch_up,
            },
            "max_concurrent_runs": self.concurrency,
            "cost_budget": self.budget,
            "group_limits": dict(self.limits),
            "max_parallel": self.parallel,
        }

    def digest(self) -> str:
        """Digests the definition, anchored at version 1.

        Renaming a written key leaves the digest of a recorded run
        unchanged.

        Returns:
            The hex SHA-256 of the key-sorted JSON of `anchor()`.
        """
        text = json.dumps(self.anchor(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


def scaffold(
    name: str = "scmle-pipeline",
    version: str = "1",
    schedule: Schedule | None = None,
    approvers: Sequence[str] = ("ml-owner",),
    cost_budget: float | None = None,
) -> Workflow:
    """Builds the usual pipeline: ingest through approval to publication.

    Batch prediction publishes results to consumers, so it is an approved action
    behind the approval step; monitoring is automatic.

    Args:
        name: Workflow name.
        version: Version label.
        schedule: When runs start, if scheduled.
        approvers: Identities that may decide the approval step.
        cost_budget: Ceiling for one run's total cost, if any.

    Returns:
        The workflow.
    """
    return Workflow(
        name=name,
        version=version,
        schedule=schedule,
        budget=cost_budget,
        steps=(
            Step("ingest", Kind.INGEST),
            Step("train", Kind.TRAIN, ("ingest",), concurrency="train"),
            Step("evaluate", Kind.EVALUATE, ("train",)),
            Step("monitor", Kind.MONITOR, ("evaluate",)),
            Step(
                "approve",
                Kind.APPROVAL,
                ("evaluate",),
                approvers=tuple(approvers),
            ),
            Step(
                "batch_predict",
                Kind.BATCH_PREDICT,
                ("approve", "monitor"),
                effect=APPROVED,
            ),
        ),
    )
