"""The structured error a streaming operation raises."""

from collections.abc import Mapping
from typing import Any

from scmle.contracts import errors


class StreamingError(errors.DataError):
    """A bounded streaming operation cannot safely advance its state.

    Attributes:
        code: Short machine-readable failure code.
        identity: Identity of the source involved.
        details: Evidence mapping holding `code`, `identity` and
            any extra detail supplied.
    """

    def __init__(
        self,
        code: str,
        message: str,
        source_identity: str,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes a structured, transport-neutral streaming failure.

        Args:
            code: Short machine-readable failure code.
            message: Human-readable description of the failure.
            source_identity: Identity of the source involved.
            details: Extra evidence merged into `details` after the code and
                source identity.
        """
        self.code = code
        self.identity = source_identity
        self.details = {
            "code": code,
            "source_identity": source_identity,
            **dict(details or {}),
        }
        super().__init__(message, self.details)
