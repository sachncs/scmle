"""Error taxonomy.

Every framework failure derives from `ScmleError` and carries a stable
`category` plus an optional structured `details` mapping so callers can react
programmatically instead of parsing messages.
"""

from collections.abc import Mapping
import dataclasses
import traceback
from typing import Any

# Bounds on what a stored failure keeps: the last characters of the
# traceback (the innermost frames, where the failure is) and the nearest
# causes. Policy: enough to diagnose, small enough for a run record.
TRACE = 8000
CAUSES = 5


class ScmleError(Exception):
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


class ConfigError(ScmleError):
    """The project configuration is invalid or inconsistent."""

    category = "config"


class DataError(ScmleError):
    """The dataset cannot be read or is malformed."""

    category = "data"


class InputError(DataError, ValueError):
    """The input is unusable for this step: a user or data problem, not a bug.

    It is also a `ValueError`, so code that catches `ValueError` still works
    while a module moves to it; a gate catches `InputError` alone and lets a
    plain `ValueError`, which marks a defect, propagate (ERR-20).
    """

    category = "input"


class KindError(InputError, TypeError):
    """An input of the wrong kind: a user problem, not a bug.

    It is also a `TypeError`, so code that catches `TypeError` still works,
    and an `InputError`, so a gate treats it like any other bad input
    (ERR-20).
    """


class OrderError(InputError, RuntimeError):
    """A step was called before the step it needs: a caller's mistake.

    Predicting before fitting is one example; it is not a bug. It is also
    a `RuntimeError`, so code that catches `RuntimeError` still works
    (ERR-20). A plain `RuntimeError` is kept for real defects.
    """


class SchemaError(DataError):
    """Input data does not match the schema a model was trained on."""

    category = "schema"


class BlockedError(ScmleError):
    """The framework cannot safely continue automatically."""

    category = "blocked"


class LeakageError(ScmleError):
    """An operation would violate a data-leakage invariant."""

    category = "leakage"


class ArtifactError(ScmleError):
    """A saved artifact is missing, corrupt, or incompatible."""

    category = "artifact"


class CancelledError(ScmleError):
    """The run was cancelled before completion."""

    category = "cancelled"


class ReliabilityError(ScmleError):
    """A fault-injection execution cannot produce trustworthy evidence."""

    category = "reliability"


@dataclasses.dataclass(frozen=True)
class Failure:
    """A stored failure: what was raised, where, and what caused it.

    Records that keep only `"Type: message"` lose the traceback and the
    cause, so a failure found in a run record cannot be diagnosed (ERR-19).
    Every place that stores a failure stores this instead.

    Attributes:
        kind: The exception's qualified type name.
        message: Its message.
        trace: The formatted traceback, innermost frame last.
        causes: `"Type: message"` of each explicit cause or implicit
            context, nearest first.
    """

    kind: str
    message: str
    trace: str = ""
    causes: tuple[str, ...] = ()

    @classmethod
    def of(cls, error: BaseException) -> "Failure":
        """Captures a raised exception.

        Args:
            error: The exception, ideally inside its `except` block so its
                traceback is attached.

        Returns:
            The failure, with the traceback and the chain of causes. An
            error that carries a failure from another process in
            `details["failure"]` (as `scmle.runtime.limits` and the
            deep-learning worker raise) gets that failure as its first cause
            and the remote traceback appended, so the child's frames are not
            lost.
        """
        causes: list[str] = []
        trace = "".join(traceback.format_exception(error))
        remote = getattr(error, "details", {}).get("failure")
        if isinstance(remote, Mapping):
            causes.append(f"{remote.get('kind')}: {remote.get('message')}")
            trace += "\n-- in the child process --\n" + str(
                remote.get("trace", "")
            )
        link = error.__cause__ or error.__context__
        while link is not None and len(causes) < CAUSES:
            causes.append(f"{type(link).__qualname__}: {link}")
            link = link.__cause__ or link.__context__
        return cls(
            type(error).__qualname__,
            str(error),
            trace[-TRACE:],
            tuple(causes),
        )

    @property
    def text(self) -> str:
        """The one-line form, `"Type: message"`, for messages and logs."""
        return f"{self.kind}: {self.message}"

    def dump(self) -> Mapping[str, object]:
        """Returns a JSON-ready record of every field."""
        return {
            "kind": self.kind,
            "message": self.message,
            "trace": self.trace,
            "causes": list(self.causes),
        }
