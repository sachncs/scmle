"""Error taxonomy. Every framework failure derives from MLForgeError."""

from __future__ import annotations


class MLForgeError(Exception):
    category = "internal"


class ConfigError(MLForgeError):
    category = "config"


class DataError(MLForgeError):
    category = "data"


class SchemaError(DataError):
    category = "schema"


class BlockedError(MLForgeError):
    """The framework cannot safely continue automatically (run enters BLOCKED)."""

    category = "blocked"


class LeakageError(MLForgeError):
    category = "leakage"


class BudgetExhausted(MLForgeError):
    category = "budget"


class ArtifactError(MLForgeError):
    category = "artifact"
