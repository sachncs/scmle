"""Policy checkpoints: human (or automated) approval gates in a run.

A checkpoint is consulted after a stage completes. Rejecting at the
`PLANNING` gate stops before any training; rejecting at the `SELECTING` gate
stops after the winner is chosen but *before the locked test set is opened*.
Every verdict is an audit event.
"""

import abc
from collections.abc import Callable
import dataclasses
from typing import Any

from mml.core import context as context_module
from mml.core import state

PLAN_GATE = state.RunState.PLANNING
SELECTION_GATE = state.RunState.SELECTING


@dataclasses.dataclass(frozen=True)
class GateReview:
    """What a reviewer is shown at a gate.

    Attributes:
        gate: The stage that just completed.
        summary: Human-readable description of what is awaiting approval.
        details: Structured facts (plan, selected trial, ...).
    """

    gate: state.RunState
    summary: str
    details: dict[str, Any]


@dataclasses.dataclass(frozen=True)
class Approval:
    """A reviewer's verdict.

    Attributes:
        approved: Whether the run may continue.
        reviewer: Who decided (person or system).
        reason: Why.
    """

    approved: bool
    reviewer: str = "system"
    reason: str = ""


class PolicyCheckpoint(abc.ABC):
    """An approval gate.

    Attributes:
        gates: Stages after which the checkpoint is consulted.
    """

    gates: frozenset[state.RunState] = frozenset({PLAN_GATE, SELECTION_GATE})

    @abc.abstractmethod
    def review(self, review: GateReview) -> Approval:
        """Decides whether the run may continue past the gate."""


class AutoApprove(PolicyCheckpoint):
    """Approves everything (fully automatic operation)."""

    def review(self, review: GateReview) -> Approval:
        """Always approves."""
        return Approval(True, "auto", "automatic approval")


class CallbackCheckpoint(PolicyCheckpoint):
    """Delegates the decision to a function (a UI, a ticket, a script)."""

    def __init__(
        self,
        decide: Callable[[GateReview], Approval | bool],
        gates: frozenset[state.RunState] | None = None,
    ) -> None:
        """Initializes the checkpoint.

        Args:
            decide: Function returning an `Approval` or a plain bool.
            gates: Gates to consult it at; defaults to plan and selection.
        """
        self.decide = decide
        if gates is not None:
            self.gates = gates

    def review(self, review: GateReview) -> Approval:
        """Calls the function and normalizes its answer."""
        verdict = self.decide(review)
        if isinstance(verdict, Approval):
            return verdict
        return Approval(bool(verdict), "callback")


def describe(
    gate: state.RunState, context: context_module.RunContext
) -> GateReview:
    """Builds the review shown at `gate`."""
    if gate is PLAN_GATE:
        plan = context_module.require(context.plan, "plan")
        return GateReview(gate, plan.summary(), plan.to_dict())
    selected = context_module.require(context.selected, "selected")
    return GateReview(
        gate,
        f"selected {selected.model_name}/{selected.strategy} "
        f"(cv {selected.score_mean:.4f}); the test set is still locked",
        selected.to_dict(),
    )
