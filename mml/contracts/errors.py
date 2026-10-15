"""Error taxonomy. Every framework failure derives from MagicMLError."""

from __future__ import annotations


class MagicMLError(Exception):
    category = "internal"

    def __init__(self, message: str = "", details: dict | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class ConfigError(MagicMLError):
    category = "config"


class DataError(MagicMLError):
    category = "data"


class SchemaError(DataError):
    category = "schema"


class BlockedError(MagicMLError):
    """The framework cannot safely continue automatically (run enters BLOCKED)."""

    category = "blocked"


class LeakageError(MagicMLError):
    category = "leakage"


class BudgetExhausted(MagicMLError):
    category = "budget"


class ArtifactError(MagicMLError):
    category = "artifact"
