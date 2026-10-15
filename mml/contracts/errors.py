"""Error taxonomy.

Every framework failure derives from `MagicMLError` and carries a stable
`category` plus an optional structured `details` mapping so callers can react
programmatically instead of parsing messages.
"""

from typing import Any


class MagicMLError(Exception):
    """Base class for all framework errors.

    Attributes:
        category: Stable machine-readable error category.
        details: Structured context for the failure.
    """

    category = "internal"

    def __init__(
        self, message: str = "", details: dict[str, Any] | None = None
    ) -> None:
        """Initializes the error.

        Args:
            message: Human-readable description.
            details: Structured context (for example the run directory).
        """
        super().__init__(message)
        self.details: dict[str, Any] = details or {}


class ConfigError(MagicMLError):
    """The project configuration is invalid or inconsistent."""

    category = "config"


class DataError(MagicMLError):
    """The dataset cannot be read or is malformed."""

    category = "data"


class SchemaError(DataError):
    """Input data does not match the schema a model was trained on."""

    category = "schema"


class BlockedError(MagicMLError):
    """The framework cannot safely continue automatically."""

    category = "blocked"


class LeakageError(MagicMLError):
    """An operation would violate a data-leakage invariant."""

    category = "leakage"


class ArtifactError(MagicMLError):
    """A saved artifact is missing, corrupt, or incompatible."""

    category = "artifact"


class CancelledError(MagicMLError):
    """The run was cancelled before completion."""

    category = "cancelled"
