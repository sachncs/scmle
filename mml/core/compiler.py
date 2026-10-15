"""Workflow compiler: turns declared stage dependencies into a run order.

Each stage names the stages it needs (`Stage.requires`). The compiler checks
that every dependency is present, rejects cycles, and returns a deterministic
topological order (ties keep the order the stages were given), so the
workflow is a directed acyclic graph rather than a hard-coded sequence.
"""

from collections.abc import Sequence
from typing import Any

from mml.contracts import errors


def compile_workflow(stages: Sequence[Any]) -> tuple[Any, ...]:
    """Orders stages so every stage runs after the stages it requires.

    Args:
        stages: Stage instances, each with a `requires` tuple of stage
            classes.

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
            k.__name__
            for k in stage.requires
            if not any(isinstance(other, k) for other in stages)
        ]
        if missing:
            raise errors.ConfigError(
                f"{type(stage).__name__} requires {missing}, which are not in "
                "the workflow"
            )
    remaining = list(stages)
    done: list[Any] = []
    ordered: list[Any] = []
    while remaining:
        ready = next(
            (
                s
                for s in remaining
                if all(any(isinstance(d, k) for d in done) for k in s.requires)
            ),
            None,
        )
        if ready is None:
            cycle = [type(s).__name__ for s in remaining]
            raise errors.ConfigError(
                f"stage dependencies contain a cycle among {cycle}"
            )
        ordered.append(ready)
        done.append(ready)
        remaining.remove(ready)
    return tuple(ordered)
