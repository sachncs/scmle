"""Controlled retries: a bounded number of attempts with exponential backoff.

The policy is data (attempts, first delay, factor, ceiling) and the delays
are a pure function of it, with no random jitter, so a retry schedule is
reproducible and testable. Sleeping and the decision of what counts as
transient are injected.
"""

from collections.abc import Callable
import dataclasses
import math
import time
from typing import TypeVar

from scmle.contracts import errors

T = TypeVar("T")


class TransientError(errors.ScmleError):
    """A failure that may succeed if tried again (throttling, timeout)."""

    category = "transient"


class RetryExhaustedError(errors.ScmleError):
    """Every attempt failed.

    Attributes:
        attempts: How many attempts were made.
    """

    category = "retry_exhausted"

    def __init__(self, attempts: int, message: str) -> None:
        """Initializes the error.

        Args:
            attempts: Attempts made.
            message: Description of the last failure.
        """
        super().__init__(message, {"attempts": attempts})
        self.attempts = attempts


@dataclasses.dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How often and how patiently an operation is retried.

    Attributes:
        max_attempts: Total attempts including the first (at least 1).
        base_delay_seconds: Wait after the first failure.
        factor: Growth of the wait per failure (at least 1).
        max_delay_seconds: Ceiling of one wait.
    """

    max_attempts: int = 3
    base_delay_seconds: float = 0.5
    factor: float = 2.0
    max_delay_seconds: float = 30.0

    def __post_init__(self) -> None:
        """Validates the policy.

        Raises:
            ConfigError: If a value is out of range.
        """
        if self.max_attempts < 1:
            raise errors.ConfigError("max_attempts must be at least 1")
        for name in ("base_delay_seconds", "factor", "max_delay_seconds"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise errors.ConfigError(f"{name} must be finite and >= 0")
        if self.factor < 1:
            raise errors.ConfigError("factor must be at least 1")

    def delay_after(self, failures: int) -> float:
        """The wait after `failures` failed attempts (1 for the first)."""
        wait = self.base_delay_seconds * self.factor ** (failures - 1)
        return min(wait, self.max_delay_seconds)

    def delays(self) -> list[float]:
        """The waits between all attempts, in order."""
        return [self.delay_after(n) for n in range(1, self.max_attempts)]

    def to_dict(self) -> dict[str, float | int]:
        """JSON-ready form."""
        return dataclasses.asdict(self)


def call_with_retries(
    operation: Callable[[], T],
    policy: RetryPolicy,
    *,
    is_transient: Callable[[Exception], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Runs `operation`, retrying transient failures under `policy`.

    Args:
        operation: Zero-argument callable.
        policy: Attempts and backoff.
        is_transient: Whether an exception may be retried; anything else
            propagates at once. By default only `TransientError`.
        sleep: Waits between attempts.

    Returns:
        The operation's result.

    Raises:
        RetryExhaustedError: If every attempt failed with a transient error.
    """
    last: Exception | None = None
    transient = is_transient or (
        lambda error: isinstance(error, TransientError)
    )
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if not transient(error):
                raise
            last = error
            if attempt < policy.max_attempts:
                sleep(policy.delay_after(attempt))
    raise RetryExhaustedError(
        policy.max_attempts,
        f"{policy.max_attempts} attempts failed; last: {last!r}",
    ) from last
