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

from scmle.pipeline import engine
from scmle.pipeline import products
from scmle.pipeline.stages import planning
from scmle.pipeline.stages import selecting
from scmle.runtime import state


@dataclasses.dataclass(frozen=True)
class Review:
    """What a reviewer is shown at a gate.

    Attributes:
        gate: The stage that just completed.
        summary: Human-readable description of what is awaiting approval.
        details: Structured facts (plan, selected trial, ...).
    """

    gate: state.Progress
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

    gates: frozenset[state.Progress] = frozenset(
        {state.Progress.PLANNING, state.Progress.SELECTING}
    )

    @abc.abstractmethod
    def review(self, review: Review) -> Approval:
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
        decide: Callable[[Review], Approval | bool],
        gates: frozenset[state.Progress] | None = None,
    ) -> None:
        """Initializes the checkpoint.

        Args:
            decide: Function returning an `Approval` or a plain bool.
            gates: Gates to consult it at; defaults to plan and selection.
        """
        self.decide = decide
        if gates is not None:
            self.gates = gates

    def review(self, review: Review) -> Approval:
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


def describe(gate: state.Progress, board: engine.Board) -> Review:
    """Builds the review shown at `gate`.

    Args:
        gate: The stage that just completed.
        board: The run's board holding that stage's product.

    Returns:
        A review summarizing the execution plan at the gate after planning,
        or the selected model and strategy (with its cross-validation
        score) at any other gate.

    Raises:
        OrderError: If the board holds no blueprint (after planning) or no
            choice (at any other gate).
        ScmleError: If the blueprint has no plan or the choice no trial.
    """
    if gate is state.Progress.PLANNING:
        plan = products.require(board.get(planning.Blueprint).plan, "plan")
        return Review(gate, plan.summary(), plan.to_dict())
    selected = products.require(
        board.get(selecting.Choice).selected, "selected"
    )
    return Review(
        gate,
        f"selected {selected.model_name}/{selected.strategy} "
        f"(cv {selected.score_mean:.4f}); the test set is still locked",
        selected.to_dict(),
    )
