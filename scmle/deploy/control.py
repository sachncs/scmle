"""The control plane a release talks to, and how it waits for confirmation.

A `ControlPlane` is the thin interface the registry's promotion, staged
traffic allocation and rollback go through. An implementation knows one
platform (or is a test double); it can read what runs
(`observe`), start a new revision of an image digest (`deploy`) and send
traffic to revisions (`set_traffic`). It never decides anything and never
touches the registry.

The registry changes only after `confirm` sees the platform report what was
asked for: the revision exists, runs the expected image digest, is ready and
holds the requested share of traffic. A platform that does not confirm in
time is treated as not having done it.
"""

import abc
from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import time

from scmle.contracts import errors
from scmle.deploy import observation


class RuntimeUnconfirmed(errors.ScmleError):
    """The platform did not report what a request asked for in time."""

    category = "runtime"


class Unsupported(errors.ScmleError):
    """The platform cannot do what was asked (for example split traffic)."""

    category = "runtime"


@dataclasses.dataclass(frozen=True, slots=True)
class ConfirmPolicy:
    """How long to wait for the platform to settle.

    Attributes:
        attempts: Reads of the platform before giving up.
        delay: Seconds between reads.
        sleep: The wait function (replaced in tests).
    """

    attempts: int = 30
    delay: float = 10.0
    sleep: Callable[[float], None] = time.sleep

    def __post_init__(self) -> None:
        """Validates the policy.

        Raises:
            ConfigError: If `attempts` is below 1 or `delay` is negative.
        """
        if self.attempts < 1:
            raise errors.ConfigError("attempts must be at least 1")
        if self.delay < 0:
            raise errors.ConfigError("delay must not be negative")


class ControlPlane(abc.ABC):
    """What a release needs from a platform; nothing provider-specific else.

    Attributes:
        provider: Name of the platform.
    """

    provider: str = ""

    @abc.abstractmethod
    def observe(self) -> observation.Observation:
        """Reads what is running and how traffic is split.

        Returns:
            The platform's current state.
        """

    @abc.abstractmethod
    def deploy(self, image_digest: str) -> str:
        """Starts a revision of the image and returns its identifier.

        A platform that can hold a revision without traffic does so; one that
        cannot (a rolling update) sends it all the traffic, which is why the
        caller confirms and can restore the previous revision.

        Args:
            image_digest: Digest of the image to run, as `sha256:<hex>`.

        Returns:
            The identifier of the new revision.
        """

    @abc.abstractmethod
    def set_traffic(self, allocation: Mapping[str, int]) -> None:
        """Requests a split (revision to percent, summing to 100).

        Args:
            allocation: Percent of traffic per revision identifier.

        Raises:
            Unsupported: If the platform cannot make that split.
        """


def confirm(
    plane: ControlPlane,
    check: Callable[[observation.Observation], list[str]],
    policy: ConfirmPolicy | None = None,
) -> observation.Observation:
    """Reads the platform until ``check`` finds nothing wrong.

    An observation that cannot be read counts as a failed attempt.

    Args:
        plane: The platform.
        check: Returns the reasons an observation is not yet what was asked; an
            empty list means it is.
        policy: Attempts and delay; the defaults of `ConfirmPolicy` if None.

    Returns:
        The confirming observation.

    Raises:
        RuntimeUnconfirmed: If the last read still differs, with its reasons.
    """
    policy = policy or ConfirmPolicy()
    reasons: list[str] = []
    for attempt in range(policy.attempts):
        if attempt:
            policy.sleep(policy.delay)
        try:
            seen = plane.observe()
        except errors.ArtifactError as error:
            reasons = [f"the platform's state cannot be read yet: {error}"]
            continue
        reasons = check(seen)
        if not reasons:
            return seen
    raise RuntimeUnconfirmed(
        "the platform did not confirm: " + "; ".join(reasons),
        {"reasons": reasons},
    )


def expecting(
    revision: str, image_digest: str, percent: int
) -> Callable[[observation.Observation], list[str]]:
    """Builds a `confirm` check for one revision.

    Args:
        revision: Identifier of the revision expected.
        image_digest: Image digest the revision must run.
        percent: Share of traffic the revision must hold.

    Returns:
        A function that takes an observation and returns the reasons it does not
        show ``revision`` running ``image_digest`` at ``percent`` of the traffic
        with the service ready; empty if it does.
    """

    def check(seen: observation.Observation) -> list[str]:
        """Compares one observation with the expectation.

        Args:
            seen: The observation to check.

        Returns:
            The reasons the observation differs from what was expected; empty if
            it matches.
        """
        reasons = []
        digest = seen.digest_of(revision)
        if digest is None:
            reasons.append(f"revision {revision} is not reported")
        elif digest != image_digest:
            reasons.append(f"revision {revision} runs {digest}")
        if seen.share_of(revision) != percent:
            reasons.append(
                f"revision {revision} holds {seen.share_of(revision)}% "
                f"of traffic, not {percent}%"
            )
        if not seen.ready:
            reasons.append("the platform does not report the service ready")
        return reasons

    return check
