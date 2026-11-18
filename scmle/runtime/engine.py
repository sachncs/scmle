"""The engine's ordering: which stage or step runs after which.

Each stage or workflow step names what it needs. How a need is met and how
ties are broken is a `Dependency`: `CLASSES` for the tabular stages (a need
is a stage class, met by an instance of it or of a subclass; earlier
lifecycle states first, then the order the stages were given) and `NAMES`
for workflow steps (a need is a step name; ties by name). `ready` lists the
items whose needs are done, which is what a parallel dispatcher starts next.
`order` checks that every dependency is present, rejects duplicates and
cycles, and returns a deterministic topological order; `layers` groups the
same items into waves that may run side by side.

This module imports only the run states and the error types, so the
workflow executor, which runs on it, stays free of the pipeline's imports.
"""

from collections.abc import Callable
from collections.abc import Hashable
from collections.abc import Iterable
from collections.abc import Sequence
import dataclasses
from typing import Any
from typing import Generic
from typing import NoReturn
from typing import TypeVar

from scmle.contracts import errors
from scmle.runtime import state

ItemT = TypeVar("ItemT")

LIFECYCLE = tuple(state.Progress)
"""The lifecycle states in order; `order` runs earlier states first."""


@dataclasses.dataclass(frozen=True, slots=True)
class Dependency(Generic[ItemT]):
    """How items name their needs, how a need is met and how ties break.

    Attributes:
        needs: The needs of an item.
        meets: Whether a done item meets a need.
        rank: The tie-break key: among ready items, the smallest runs first;
            equal keys keep the order the items were given.
        title: The name of an item or of a need, for records and messages.
    """

    needs: Callable[[ItemT], Iterable[Hashable]]
    meets: Callable[[ItemT, Hashable], bool]
    rank: Callable[[ItemT], tuple[int, str]]
    title: Callable[[Any], str]


CLASSES: Dependency[Any] = Dependency(
    needs=lambda stage: stage.requires,
    meets=lambda done, need: isinstance(need, type) and isinstance(done, need),
    rank=lambda stage: (LIFECYCLE.index(stage.state), ""),
    title=lambda item: (
        item.__name__ if isinstance(item, type) else type(item).__name__
    ),
)
"""Tabular stages: a need is a stage class, met by an instance of it or of a
subclass (so a replacement stage satisfies its predecessor's dependents);
earlier lifecycle states run first, so the state machine only moves forward.
"""

NAMES: Dependency[Any] = Dependency(
    needs=lambda step: step.depends_on,
    meets=lambda done, need: done.name == need,
    rank=lambda step: (0, step.name),
    title=lambda item: item if isinstance(item, str) else item.name,
)
"""Workflow steps (`name`, `depends_on`): a need is a step name; ties are
broken by name."""


def ready(
    done: Iterable[ItemT],
    remaining: Iterable[ItemT],
    dependency: Dependency[ItemT] = CLASSES,
) -> tuple[ItemT, ...]:
    """Lists the remaining items whose every need a done item meets.

    Args:
        done: The items that finished.
        remaining: The items not started.
        dependency: How needs are named and met.

    Returns:
        The items that may start now, in the order given.
    """
    finished = tuple(done)
    return tuple(
        item
        for item in remaining
        if all(
            any(dependency.meets(other, need) for other in finished)
            for need in dependency.needs(item)
        )
    )


def vet(stages: Sequence[ItemT], dependency: Dependency[ItemT]) -> None:
    """Checks that every need is present and no item appears twice.

    Args:
        stages: The stages or steps.
        dependency: How needs are named and met.

    Raises:
        ConfigError: If a needed item is missing or an item appears twice.
    """
    titles = [dependency.title(stage) for stage in stages]
    if len(set(titles)) != len(titles):
        raise errors.ConfigError("a stage appears more than once")
    for stage in stages:
        missing = [
            dependency.title(need)
            for need in dependency.needs(stage)
            if not any(dependency.meets(other, need) for other in stages)
        ]
        if missing:
            raise errors.ConfigError(
                f"{dependency.title(stage)} requires {missing}, which are not "
                "in the workflow"
            )


def stuck(
    remaining: Sequence[ItemT], dependency: Dependency[ItemT]
) -> NoReturn:
    """Refuses items none of which can start: their needs form a cycle.

    Args:
        remaining: The items left when nothing is ready.
        dependency: How items are named.

    Raises:
        ConfigError: Always, naming the items.
    """
    cycle = [dependency.title(stage) for stage in remaining]
    raise errors.ConfigError(
        f"stage dependencies contain a cycle among {cycle}"
    )


def order(
    stages: Sequence[ItemT], dependency: Dependency[ItemT] = CLASSES
) -> tuple[ItemT, ...]:
    """Orders items so that every item runs after the items it needs.

    Among the items whose needs have run, the one with the smallest
    `dependency.rank` runs first, and ties keep the order the items were
    given.

    Args:
        stages: The stages or steps.
        dependency: How needs are named and met and how ties break.

    Returns:
        The items in execution order. `vet` and `stuck` raise `ConfigError`
        when a needed item is missing, an item appears twice, or the
        dependencies contain a cycle.
    """
    vet(stages, dependency)
    remaining = list(stages)
    ordered: list[ItemT] = []
    while remaining:
        candidates = ready(ordered, remaining, dependency)
        if not candidates:
            stuck(remaining, dependency)
        first = min(candidates, key=dependency.rank)
        ordered.append(first)
        remaining.remove(first)
    return tuple(ordered)


def layers(
    stages: Sequence[ItemT], dependency: Dependency[ItemT] = CLASSES
) -> tuple[tuple[ItemT, ...], ...]:
    """Groups items into waves; each wave needs only earlier waves.

    A wave holds every item whose needs the earlier waves meet, smallest
    `dependency.rank` first, so its items may run side by side. Flattened,
    the waves are a topological order.

    Args:
        stages: The stages or steps.
        dependency: How needs are named and met and how ties break.

    Returns:
        The waves in order. `vet` and `stuck` raise `ConfigError` when a
        needed item is missing, an item appears twice, or the dependencies
        contain a cycle.
    """
    vet(stages, dependency)
    remaining = list(stages)
    done: list[ItemT] = []
    waves: list[tuple[ItemT, ...]] = []
    while remaining:
        wave = ready(done, remaining, dependency)
        if not wave:
            stuck(remaining, dependency)
        waves.append(tuple(sorted(wave, key=dependency.rank)))
        done.extend(wave)
        remaining = [item for item in remaining if item not in wave]
    return tuple(waves)
