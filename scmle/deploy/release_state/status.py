"""Release states, their transitions and the records a release returns."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
from typing import Any
from typing import NoReturn

from scmle.contracts import errors
from scmle.deploy import control
from scmle.deploy import observation

STATE_FILE = "release_state.json"
STATE_FORMAT = 1
PRODUCTION_ENVIRONMENT = "production"
FULL_ROLLOUT = 100  # percent of traffic when a revision serves everything


class State(enum.StrEnum):
    """Where a registered version stands in its release."""

    CANDIDATE = "candidate"
    CHALLENGER = "challenger"
    APPROVED = "approved"
    PRODUCTION = "production"
    RETIRED = "retired"


ALLOWED: dict[State, frozenset[State]] = {
    State.CANDIDATE: frozenset(
        {State.CHALLENGER, State.APPROVED, State.RETIRED}
    ),
    State.CHALLENGER: frozenset({State.APPROVED, State.RETIRED}),
    State.APPROVED: frozenset({State.PRODUCTION, State.RETIRED}),
    State.PRODUCTION: frozenset({State.RETIRED}),
    State.RETIRED: frozenset({State.PRODUCTION}),
}
"""Transitions. Into ``production`` only `promote` and `rollback` move."""
MANUAL = frozenset(
    {
        State.CHALLENGER,
        State.RETIRED,
    }
)
"""States `transition` may set by hand (``approved`` needs `approve`)."""


class DisagreementError(errors.ScmleError):
    """The registry's claim and the platform's report differ."""

    category = "runtime"


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One disagreement between the registry, the ledger and the runtime."""

    __module__ = "scmle.deploy.release_state"  # pinned signatures name it

    kind: str
    detail: str

    def to_dict(self) -> dict[str, str]:
        """Returns a JSON-ready record."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class Reconciliation:
    """The result of comparing records with a platform observation."""

    model: str
    environment: str
    findings: tuple[Finding, ...]

    @property
    def consistent(self) -> bool:
        """Whether registry, ledger and runtime agree."""
        return not self.findings

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record."""
        return {
            "model": self.model,
            "environment": self.environment,
            "consistent": self.consistent,
            "findings": [item.to_dict() for item in self.findings],
        }


def halt(percent: int, reasons: Sequence[str]) -> NoReturn:
    """Stops a rollout because the health check objected.

    Args:
        percent: The traffic share at which the check objected.
        reasons: What the health check reported.

    Raises:
        UnconfirmedError: Always.
    """
    raise control.UnconfirmedError(
        f"health stopped the rollout at {percent}%: " + "; ".join(reasons)
    )


def predecessor(
    previous: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    """Returns the previous revision a staged rollout splits traffic with.

    Args:
        previous: What the registry records as running, or None.

    Returns:
        `previous`, unchanged.

    Raises:
        ConfigError: If there is no previous revision.
    """
    if previous is None:
        raise errors.ConfigError("a staged rollout needs a previous revision")
    return previous


@dataclasses.dataclass(frozen=True, slots=True)
class Outcome:
    """What a promotion or rollback did.

    Attributes:
        done: Whether the platform confirmed and the registry changed.
        reasons: Why not, when it was not done.
        restored: Whether the previous revision was confirmed back in place
            after a failed attempt (None when there was nothing to restore).
        version: The version now serving, if the call succeeded.
        revision: The revision now holding all traffic, if the call
            succeeded.
    """

    __module__ = "scmle.deploy.release_state"  # pinned signatures name it

    done: bool
    reasons: tuple[str, ...]
    restored: bool | None = None
    version: int | None = None
    revision: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-ready record."""
        return {
            "done": self.done,
            "reasons": list(self.reasons),
            "restored": self.restored,
            "version": self.version,
            "revision": self.revision,
        }


Health = Callable[[observation.Observation, int], list[str]]
"""``health(observation, percent)`` returns reasons to stop a rollout."""
