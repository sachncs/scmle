"""Connector failures and the retry budget every operation runs under."""

from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import math
import time
from typing import Any
from typing import TypeVar

from scmle.contracts import errors

T = TypeVar("T")


class ConnectorError(errors.DataError):
    """A connector failure with a stable code and structured details.

    Attributes:
        code: Stable machine-readable failure code.
        uri: Canonical source URI associated with the failure.
        details: Structured context; always includes ``code`` and
            ``uri`` plus any caller-supplied entries.
    """

    def __init__(
        self,
        code: str,
        message: str,
        source_uri: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes a structured connector failure.

        Args:
            code: Stable machine-readable failure code.
            message: Human-readable failure description.
            source_uri: Canonical source URI associated with the failure.
            details: Additional structured context for callers and logs.
        """
        self.code = code
        self.uri = source_uri
        self.details = {
            "code": code,
            "source_uri": source_uri,
            **dict(details or {}),
        }
        super().__init__(message, self.details)

    def serialize(self) -> dict[str, Any]:
        """Serializes the failure.

        Returns:
            A JSON-compatible mapping with the error ``category``,
            ``code``, ``message``, ``uri`` and a copy of
            ``details``.
        """
        return {
            "category": self.category,
            "code": self.code,
            "message": str(self),
            "source_uri": self.uri,
            "details": dict(self.details),
        }


class SchemeError(ConnectorError):
    """A connector was given a URI scheme it does not implement.

    Attributes:
        scheme: The rejected scheme (empty for a bare path).
        schemes: The schemes the connector accepts.
    """

    def __init__(
        self, source_uri: str, scheme: str, supported_schemes: Sequence[str]
    ) -> None:
        """Initializes an unsupported-scheme failure.

        Args:
            source_uri: The URI that was rejected.
            scheme: Its scheme (empty for a bare path).
            supported_schemes: Schemes the connector accepts.
        """
        supported = tuple(supported_schemes)
        super().__init__(
            "unsupported_uri_scheme",
            f"unsupported URI scheme '{scheme or '<path>'}'; "
            f"supported schemes: {supported}",
            source_uri,
            {"scheme": scheme, "supported_schemes": list(supported)},
        )
        self.scheme = scheme
        self.schemes = supported


class RetryError(ConnectorError):
    """All attempts for a connector operation failed.

    Attributes:
        operation: Name of the failed operation.
        attempts: Number of attempts made.
        error: The exception raised by the final attempt.
    """

    def __init__(
        self,
        source_uri: str,
        operation: str,
        attempts: int,
        last_error: Exception,
    ) -> None:
        """Initializes a retry exhaustion failure.

        Args:
            source_uri: Canonical source URI of the operation.
            operation: Name of the failed operation.
            attempts: Number of attempts made.
            last_error: The exception raised by the final attempt.
        """
        super().__init__(
            "retry_exhausted",
            f"{operation} failed after {attempts} attempt(s): {last_error}",
            source_uri,
            {
                "operation": operation,
                "attempts": attempts,
                "last_error_type": type(last_error).__name__,
                "last_error": str(last_error),
            },
        )
        self.operation = operation
        self.attempts = attempts
        self.error = last_error


class CheckpointError(ConnectorError):
    """A checkpoint cannot safely be used with a connector."""

    def __init__(
        self,
        source_uri: str,
        code: str,
        message: str,
        details: Mapping[str, Any],
    ) -> None:
        """Initializes a checkpoint failure.

        Args:
            source_uri: Canonical source URI of the connector.
            code: Stable machine-readable failure code.
            message: Human-readable failure description.
            details: Structured context for callers and logs.
        """
        super().__init__(code, message, source_uri, details)


@dataclasses.dataclass(frozen=True)
class Retries:
    """Retry limits applied independently to each connector operation.

    Attributes:
        limit: Number of retries after the initial attempt.
        delay: Delay between attempts. Zero is recommended for tests.
    """

    limit: int = 2
    delay: float = 0.0

    def __post_init__(self) -> None:
        """Validates retry settings before a connector uses them.

        Raises:
            ValueError: If `limit` is negative or `delay` is
                negative or not finite.
        """
        if self.limit < 0:
            raise errors.InputError("max_retries must be non-negative")
        if not math.isfinite(self.delay) or self.delay < 0:
            raise errors.InputError(
                "delay_seconds must be finite and non-negative"
            )

    @property
    def attempts(self) -> int:
        """The total number of attempts: the initial attempt plus retries."""
        return self.limit + 1


def persevere(
    operation: Callable[[], T],
    operation_name: str,
    source_uri: str,
    retry_budget: Retries,
) -> T:
    """Executes an operation until it succeeds or its retry budget is spent.

    Any exception other than a `ConnectorError` triggers a retry, after a
    pause of `delay_seconds` when that is nonzero. A `ConnectorError` raised
    by the operation is re-raised immediately without retrying.

    Args:
        operation: Zero-argument operation to execute.
        operation_name: Stable operation name for failure evidence.
        source_uri: Canonical source URI for failure evidence.
        retry_budget: Retry policy for this operation.

    Returns:
        The operation result.

    Raises:
        RetryError: If every attempt fails.
        ConnectorError: If the operation raises one; it is not retried.
        AssertionError: If the retry loop ends without returning or
            raising, which cannot happen because the budget allows at least
            one attempt.
    """
    for attempt in range(1, retry_budget.attempts + 1):
        try:
            return operation()
        except ConnectorError:
            raise
        except Exception as error:
            if attempt == retry_budget.attempts:
                raise RetryError(
                    source_uri, operation_name, attempt, error
                ) from error
            if retry_budget.delay:
                time.sleep(retry_budget.delay)
    raise AssertionError("retry loop must return or raise")
