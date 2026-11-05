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

from scmle.contracts import errors
from scmle.core import retry

NAME_CHARACTERS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
)
AUTOMATIC = "automatic"
APPROVED = "approved"


class StepKind(enum.StrEnum):
    """What a step does."""

    INGEST = "ingest"
    TRAIN = "train"
    EVALUATE = "evaluate"
    MONITOR = "monitor"
    BATCH_PREDICT = "batch_predict"
    APPROVAL = "approval"


class CatchUp(enum.StrEnum):
    """What to do about schedule slots that were missed."""

    RUN_ALL = "run_all"
    LATEST_ONLY = "latest_only"
    SKIP = "skip"


def check_name(name: str, what: str) -> str:
    """Returns `name` if it is 1-64 letters, digits, `-` or `_`."""
    if not name or len(name) > 64 or set(name) - NAME_CHARACTERS:
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
        depends_on: Names of steps that must have succeeded first.
        timeout_seconds: Limit for one attempt.
        retries: Attempts and backoff.
        cost_budget: Ceiling for this step's cost, if any.
        concurrency_group: Steps of one group never run at the same time
            beyond the group's limit (see `Workflow.group_limits`).
        effect: `automatic`, or `approved` (needs an approval upstream).
        parameters: JSON-ready parameters for the handler.
        approvers: For an approval step, the identities that may decide.
        expires_seconds: For an approval step, how long it waits.
    """

    name: str
    kind: StepKind
    depends_on: tuple[str, ...] = ()
    timeout_seconds: float = 3600.0
    retries: retry.RetryPolicy = dataclasses.field(
        default_factory=lambda: retry.RetryPolicy(3, 5.0, 2.0, 300.0)
    )
    cost_budget: float | None = None
    concurrency_group: str | None = None
    effect: str = AUTOMATIC
    parameters: Mapping[str, Any] = dataclasses.field(default_factory=dict)
    approvers: tuple[str, ...] = ()
    expires_seconds: float = 86400.0

    def __post_init__(self) -> None:
        """Validates the step.

        Raises:
            ConfigError: If a value is out of range or an approval step has
                no approvers.
        """
        check_name(self.name, "step name")
        if self.timeout_seconds <= 0 or self.expires_seconds <= 0:
            raise errors.ConfigError("timeouts must be positive")
        if self.cost_budget is not None and self.cost_budget <= 0:
            raise errors.ConfigError("cost_budget must be positive")
        if self.effect not in (AUTOMATIC, APPROVED):
            raise errors.ConfigError("effect must be automatic or approved")
        if self.name in self.depends_on:
            raise errors.ConfigError(f"step {self.name} depends on itself")
        if self.kind is StepKind.APPROVAL and not self.approvers:
            raise errors.ConfigError("an approval step needs approvers")
        if self.kind is StepKind.APPROVAL and self.effect == APPROVED:
            raise errors.ConfigError("an approval step is itself automatic")

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "name": self.name,
            "kind": self.kind.value,
            "depends_on": list(self.depends_on),
            "timeout_seconds": self.timeout_seconds,
            "retries": self.retries.to_dict(),
            "cost_budget": self.cost_budget,
            "concurrency_group": self.concurrency_group,
            "effect": self.effect,
            "parameters": dict(self.parameters),
            "approvers": list(self.approvers),
            "expires_seconds": self.expires_seconds,
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
    catch_up: CatchUp = CatchUp.LATEST_ONLY
    max_catch_up: int = 10

    def __post_init__(self) -> None:
        """Validates the schedule."""
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
        max_concurrent_runs: Runs of this workflow allowed at once.
        cost_budget: Ceiling for one run's total cost, if any.
        group_limits: Concurrency group name to the steps allowed at once
            (a group not listed allows one).
        max_parallel: Steps the local executor runs at once.
    """

    name: str
    version: str
    steps: tuple[Step, ...]
    schedule: Schedule | None = None
    max_concurrent_runs: int = 1
    cost_budget: float | None = None
    group_limits: Mapping[str, int] = dataclasses.field(default_factory=dict)
    max_parallel: int = 4

    def __post_init__(self) -> None:
        """Validates the whole definition.

        Raises:
            ConfigError: On duplicate names, unknown or cyclic dependencies,
                an approved step with no approval upstream, or bad limits.
        """
        check_name(self.name, "workflow name")
        if not self.version or not self.steps:
            raise errors.ConfigError("a workflow needs a version and steps")
        if self.max_concurrent_runs < 1 or self.max_parallel < 1:
            raise errors.ConfigError("concurrency limits must be positive")
        if self.cost_budget is not None and self.cost_budget <= 0:
            raise errors.ConfigError("cost_budget must be positive")
        names = [step.name for step in self.steps]
        if len(set(names)) != len(names):
            raise errors.ConfigError("step names must be unique")
        for step in self.steps:
            missing = [d for d in step.depends_on if d not in names]
            if missing:
                raise errors.ConfigError(
                    f"step {step.name} depends on unknown steps {missing}"
                )
        self.order()
        for step in self.steps:
            if step.effect == APPROVED and not self.has_approval_upstream(
                step.name
            ):
                raise errors.ConfigError(
                    f"step {step.name} is an approved action but no approval "
                    "step precedes it"
                )

    def step(self, name: str) -> Step:
        """The step called `name`."""
        for candidate in self.steps:
            if candidate.name == name:
                return candidate
        raise errors.ConfigError(f"no step {name!r}")

    def order(self) -> tuple[str, ...]:
        """A deterministic topological order (ties broken by name).

        Raises:
            ConfigError: If the dependencies contain a cycle.
        """
        remaining = {s.name: set(s.depends_on) for s in self.steps}
        ordered: list[str] = []
        while remaining:
            ready = sorted(n for n, deps in remaining.items() if not deps)
            if not ready:
                raise errors.ConfigError(
                    f"dependency cycle among {sorted(remaining)}"
                )
            ordered.extend(ready)
            for name in ready:
                del remaining[name]
            for deps in remaining.values():
                deps.difference_update(ready)
        return tuple(ordered)

    def layers(self) -> tuple[tuple[str, ...], ...]:
        """Steps grouped by depth: each layer depends only on earlier ones."""
        depth: dict[str, int] = {}
        for name in self.order():
            parents = self.step(name).depends_on
            depth[name] = 1 + max((depth[p] for p in parents), default=-1)
        count = 1 + max(depth.values())
        return tuple(
            tuple(sorted(n for n, d in depth.items() if d == level))
            for level in range(count)
        )

    def ancestors(self, name: str) -> frozenset[str]:
        """Every step `name` depends on, directly or not."""
        found: set[str] = set()
        pending = list(self.step(name).depends_on)
        while pending:
            current = pending.pop()
            if current not in found:
                found.add(current)
                pending.extend(self.step(current).depends_on)
        return frozenset(found)

    def has_approval_upstream(self, name: str) -> bool:
        """Whether an approval step precedes `name`."""
        return any(
            self.step(a).kind is StepKind.APPROVAL for a in self.ancestors(name)
        )

    def group_limit(self, group: str) -> int:
        """Steps of a concurrency group allowed at once."""
        return int(self.group_limits.get(group, 1))

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return {
            "name": self.name,
            "version": self.version,
            "steps": [step.to_dict() for step in self.steps],
            "schedule": None
            if self.schedule is None
            else {
                "interval_seconds": self.schedule.interval_seconds,
                "start_epoch": self.schedule.start_epoch,
                "catch_up": self.schedule.catch_up.value,
                "max_catch_up": self.schedule.max_catch_up,
            },
            "max_concurrent_runs": self.max_concurrent_runs,
            "cost_budget": self.cost_budget,
            "group_limits": dict(self.group_limits),
            "max_parallel": self.max_parallel,
        }

    def digest(self) -> str:
        """SHA-256 of the canonical definition."""
        text = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(text.encode()).hexdigest()


def standard_workflow(
    name: str = "scmle-pipeline",
    version: str = "1",
    schedule: Schedule | None = None,
    approvers: Sequence[str] = ("ml-owner",),
    cost_budget: float | None = None,
) -> Workflow:
    """The usual pipeline: ingest, train, evaluate, monitor, approve, publish.

    Batch prediction publishes results to consumers, so it is an approved
    action behind the approval step; monitoring is automatic.
    """
    return Workflow(
        name=name,
        version=version,
        schedule=schedule,
        cost_budget=cost_budget,
        steps=(
            Step("ingest", StepKind.INGEST),
            Step(
                "train", StepKind.TRAIN, ("ingest",), concurrency_group="train"
            ),
            Step("evaluate", StepKind.EVALUATE, ("train",)),
            Step("monitor", StepKind.MONITOR, ("evaluate",)),
            Step(
                "approve",
                StepKind.APPROVAL,
                ("evaluate",),
                approvers=tuple(approvers),
            ),
            Step(
                "batch_predict",
                StepKind.BATCH_PREDICT,
                ("approve", "monitor"),
                effect=APPROVED,
            ),
        ),
    )
