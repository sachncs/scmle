"""The retraining policy: when retraining is recommended or triggered."""

from collections.abc import Sequence
import dataclasses
from typing import Any

from scmle.contracts import errors
from scmle.operations import signals

RECOMMEND = "recommend"
TRIGGER = "trigger"
MODES = (RECOMMEND, TRIGGER)
RETRAIN_QUESTION = "retrain_recommendation"
DEFAULT_IF_UNANSWERED = "no_retrain"
DATA_FAULTS = frozenset(
    {"invalid_request_rate", "feature_missing_increase", "unseen_category_rate"}
)


@dataclasses.dataclass(frozen=True, slots=True)
class RetrainPolicy:
    """When retraining is recommended or triggered.

    A *primary* reason is needed: a windowed loss above the reference plus
    tolerance with at least ``min_outcomes`` outcomes, an L1 alarm of the loop,
    or a model older than ``max_age_days``. Input drift supports a primary
    reason and is never enough alone. Retraining is held back while an
    infrastructure fault or a data-quality fault is open, because both would
    teach the next model the fault, and during ``cooldown_batches`` after the
    last proposal.

    Attributes:
        mode: ``recommend`` (default) or ``trigger``.
        min_outcomes: Fewest arrived outcomes behind a performance reason.
        max_age_days: Age that alone justifies a proposal, or None.
        cooldown_batches: Batches between proposals.
    """

    __module__ = "scmle.operations.continuous"

    mode: str = RECOMMEND
    min_outcomes: int = 100
    max_age_days: float | None = None
    cooldown_batches: int = 20

    def __post_init__(self) -> None:
        """Validates the choice and the bounds.

        Raises:
            ValueError: If the mode is unknown, the minimum outcomes is below
                one, the cooldown is negative, or the maximum age is not
                positive.
        """
        if self.mode not in MODES:
            raise errors.InputError(f"mode must be one of {MODES}")
        if self.min_outcomes < 1 or self.cooldown_batches < 0:
            raise errors.InputError(
                "min_outcomes >= 1 and cooldown_batches >= 0"
            )
        if self.max_age_days is not None and self.max_age_days <= 0:
            raise errors.InputError("max_age_days must be positive")


@dataclasses.dataclass(frozen=True, slots=True)
class RetrainDecision:
    """What the policy decided for one review.

    Attributes:
        action: ``none``, ``recommend`` or ``trigger``.
        reasons: Primary and supporting reasons.
        held_back: Why a proposal was not made though a reason existed.
        escalation_id: The pending escalation carrying the question.
        challenger: Name of the challenger a trigger started.
        alarm_gates_retrain: Copied from the loop policy, for the record.
    """

    __module__ = "scmle.operations.continuous"

    action: str
    reasons: tuple[str, ...]
    held_back: str | None
    escalation_id: str | None
    challenger: str | None
    alarm_gates_retrain: bool

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready record.

        Returns:
            The decision's fields as a dict.
        """
        return dataclasses.asdict(self)


def decide(
    policy: RetrainPolicy,
    rows: Sequence[signals.Signal],
    alarm_since_last: bool,
    age_days: float | None,
    batches_since_proposal: int | None,
) -> tuple[list[str], list[str], str | None]:
    """The pure part of the policy.

    Args:
        policy: The retraining policy.
        rows: The current window's signals.
        alarm_since_last: Whether the loop's alarm fired since the last
            proposal.
        age_days: Age of the model in days, or None.
        batches_since_proposal: Batches since the last proposal, or None if
            there was none.

    Returns:
        Primary reasons, supporting reasons and the reason for holding back
        (None if nothing holds a proposal back).
    """
    by_name = {row.name: row for row in rows}
    primary: list[str] = []
    support: list[str] = []
    loss = by_name.get("windowed_loss")
    if (
        loss is not None
        and loss.breached
        and loss.samples >= policy.min_outcomes
    ):
        primary.append(
            f"windowed loss {loss.value:.3f} is above {loss.threshold:.3f} "
            f"over {loss.samples} outcomes"
        )
    if alarm_since_last:
        primary.append("the loop's L1 alarm fired")
    if (
        policy.max_age_days is not None
        and age_days is not None
        and age_days > policy.max_age_days
    ):
        primary.append(f"the model is {age_days:.0f} days old")
    psi = by_name.get("feature_psi_max")
    if psi is not None and psi.breached:
        support.append("input drift is above its threshold")
    held = None
    infra = [
        r.name
        for r in rows
        if r.breached and r.category.value == "infrastructure"
    ]
    data = [r.name for r in rows if r.breached and r.name in DATA_FAULTS]
    if infra:
        held = "an infrastructure fault is open: " + ", ".join(infra)
    elif data:
        held = "a data-quality fault is open: " + ", ".join(data)
    elif (
        batches_since_proposal is not None
        and batches_since_proposal < policy.cooldown_batches
    ):
        held = (
            f"the last proposal was {batches_since_proposal} batches ago "
            f"(cooldown {policy.cooldown_batches})"
        )
    return primary, support, held
