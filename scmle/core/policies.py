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

from scmle.core import state
import scmle.core.context


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


class Reviewer(abc.ABC):
    """An approval gate.

    Attributes:
        gates: Stages after which the checkpoint is consulted.
    """

    gates: frozenset[state.RunState] = frozenset(
        {state.RunState.PLANNING, state.RunState.SELECTING}
    )

    @abc.abstractmethod
    def review(self, review: GateReview) -> Approval:
        """Decides whether the run may continue past the gate.

        Args:
            review: What the reviewer is shown at the gate.

        Returns:
            The reviewer's verdict.
        """


class Callback(Reviewer):
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
        """Calls the function and normalizes its answer.

        Args:
            review: What the reviewer is shown at the gate, passed to the
                callback.

        Returns:
            The callback's `Approval` as is, or an approval by reviewer
            "callback" whose outcome is the truth value of its answer.
        """
        verdict = self.decide(review)
        if isinstance(verdict, Approval):
            return verdict
        return Approval(bool(verdict), "callback")


def describe(
    gate: state.RunState, context: scmle.core.context.Context
) -> GateReview:
    """Builds the review shown at `gate`.

    Args:
        gate: The stage that just completed.
        context: The run context holding that stage's product.

    Returns:
        A review summarizing the execution plan at the gate after planning,
        or the selected model and strategy (with its cross-validation
        score) at any other gate. A missing stage product (no plan or no
        selected trial in the context) fails with the context's
        unavailable-product error.
    """
    if gate is state.RunState.PLANNING:
        plan = scmle.core.context.require(context.plan, "plan")
        return GateReview(gate, plan.summary(), plan.to_dict())
    selected = scmle.core.context.require(context.selected, "selected")
    return GateReview(
        gate,
        f"selected {selected.model_name}/{selected.strategy} "
        f"(cv {selected.score_mean:.4f}); the test set is still locked",
        selected.to_dict(),
    )
