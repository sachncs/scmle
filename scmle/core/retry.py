"""Controlled retries: a bounded number of attempts with exponential backoff.

The policy is data (attempts, first delay, factor, ceiling) and the delays
are a pure function of it, with no random jitter, so a retry schedule is
reproducible and testable. Sleeping and the decision of what counts as
transient are injected.
"""

from collections.abc import Callable
import dataclasses
import errno
import math
import time
from typing import TypeVar
import urllib.error

from scmle.contracts import errors

T = TypeVar("T")
# Error numbers of failures that pass: the resource is busy, the call was
# interrupted, or the network dropped or timed out. Missing files, denied
# permissions and a full disk are not here: retrying cannot fix them.
PASSING = frozenset(
    code
    for code in (
        getattr(errno, name, None)
        for name in (
            "EAGAIN",
            "EWOULDBLOCK",
            "EBUSY",
            "EINTR",
            "ETIMEDOUT",
            "ECONNRESET",
            "ECONNABORTED",
            "EPIPE",
            "ENETDOWN",
            "ENETUNREACH",
            "ENETRESET",
            "EHOSTUNREACH",
        )
    )
    if code is not None
)
# Windows sharing and lock violations: another process holds the file.
SHARING = frozenset({32, 33})
# HTTP statuses a client may retry: throttling and server-side failures.
RETRYABLE_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class TransientError(errors.ScmleError):
    """A failure that may succeed if tried again (throttling, timeout)."""

    category = "transient"


class RetryError(errors.ScmleError):
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
class Retries:
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

    def delay(self, failures: int) -> float:
        """Returns the wait after `failures` failed attempts (1 for the first).

        Args:
            failures: Number of failed attempts so far; at least 1.

        Returns:
            The wait in seconds: the base delay grown by the factor for each
            earlier failure, capped at the maximum delay.
        """
        wait = self.base_delay_seconds * self.factor ** (failures - 1)
        return min(wait, self.max_delay_seconds)

    def delays(self) -> list[float]:
        """Returns the waits between all attempts, in order.

        Returns:
            The waits in seconds between consecutive attempts, in order; one
            fewer than the maximum number of attempts.
        """
        return [self.delay(n) for n in range(1, self.max_attempts)]

    def to_dict(self) -> dict[str, float | int]:
        """Returns the JSON-ready form.

        Returns:
            The policy's fields as a dict.
        """
        return dataclasses.asdict(self)


def transient(error: BaseException) -> bool:
    """Whether a failure may succeed if the same call is made again.

    The default classification for `call` (ERR-6: before it,
    only `TransientError`, which nothing raised, was retried).

    Args:
        error: The failure.

    Returns:
        True for `TransientError`, timeouts, dropped connections, an
        `OSError` whose errno passes (`PASSING`) or that is a Windows sharing
        violation, and an HTTP error with a retryable status; False for
        anything else, including missing files and denied permissions.
    """
    if isinstance(error, (TransientError, TimeoutError, ConnectionError)):
        return True
    if isinstance(error, urllib.error.HTTPError):
        return error.code in RETRYABLE_STATUS
    if isinstance(error, urllib.error.URLError):
        return isinstance(error.reason, (TimeoutError, ConnectionError))
    if isinstance(error, OSError):
        return (
            error.errno in PASSING
            or getattr(error, "winerror", None) in SHARING
        )
    return False


def call(
    operation: Callable[[], T],
    policy: Retries,
    *,
    is_transient: Callable[[Exception], bool] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Runs `operation`, retrying transient failures under `policy`.

    Args:
        operation: Zero-argument callable.
        policy: Attempts and backoff.
        is_transient: Whether an exception may be retried; anything else
            propagates at once. By default `transient`.
        sleep: Called with each wait in seconds between attempts, not after
            the last one.

    Returns:
        The operation's result.

    Raises:
        RetryError: If every attempt failed with a transient error.
    """
    last: Exception | None = None
    judge = is_transient or transient
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return operation()
        except Exception as error:
            if not judge(error):
                raise
            last = error
            if attempt < policy.max_attempts:
                sleep(policy.delay(attempt))
    raise RetryError(
        policy.max_attempts,
        f"{policy.max_attempts} attempts failed; last: {last!r}",
    ) from last
