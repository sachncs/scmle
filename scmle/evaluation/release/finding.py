"""One release finding, its kinds and the counts over a set of findings."""

from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import enum
import math
from numbers import Real
from typing import Any
from typing import ClassVar

from scmle.contracts import errors
from scmle.contracts import records


class Kind(enum.StrEnum):
    """Category of one release finding."""

    DIAGNOSTIC = "diagnostic"
    EVIDENCE = "evidence"
    EXPLANATION = "explanation"
    FAIRNESS = "fairness"
    INPUT = "input"
    METRIC = "metric"
    RELIABILITY = "reliability"
    SECURITY = "security"


class Readiness(enum.StrEnum):
    """Machine-readable release readiness status."""

    READY = "ready"
    BLOCKED = "blocked"


def finite_real(value: Any) -> float | None:
    """Reads a finite real number.

    Args:
        value: Any value.

    Returns:
        `value` as a float when it is a real number that is neither a bool
        nor infinite nor NaN, otherwise None.
    """
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def quantify(field_name: str, value: Any) -> float | str | None:
    """Validates an observed or expected value of a finding.

    Args:
        field_name: Name of the field, used in error messages.
        value: The candidate value.

    Returns:
        A real number as a float, text or None unchanged.

    Raises:
        TypeError: If `value` is a bool, or is neither a number, text nor
            None.
        ValueError: If `value` is a number that is not finite.
    """
    if isinstance(value, bool):
        raise errors.KindError(f"{field_name} must not be bool")
    if isinstance(value, Real):
        numeric = float(value)
        if not math.isfinite(numeric):
            raise errors.InputError(f"{field_name} must be finite")
        return numeric
    if value is not None and not isinstance(value, str):
        raise errors.KindError(f"{field_name} must be numeric, text, or None")
    return value


@dataclasses.dataclass(frozen=True, slots=True)
class Finding:
    """One immutable pass or fail finding from release evaluation.

    Attributes:
        kind: Gate or input category that produced the finding.
        name: Stable metric, diagnostic, fairness metric, or evidence name.
        passed: Whether the supplied result satisfied this finding.
        blocking: Whether a failed finding blocks release. A passed finding
            may still have ``blocking=True`` because the declared gate is a
            blocking gate.
        reason: Human-readable, deterministic explanation of the outcome.
        observed: Value observed from the supplied report, when applicable.
        expected: Threshold or expected state, when applicable.
        evidence: Evidence-index references required for this finding.
    """

    kind: Kind
    name: str
    passed: bool
    blocking: bool
    reason: str
    observed: float | str | None = None
    expected: float | str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validates the immutable finding shape.

        Numeric `observed` and `expected` values are converted to floats.

        Raises:
            TypeError: If `kind` is not a `Kind`, `passed` or
                `blocking` is not a bool, `evidence` is not a tuple,
                `observed` or `expected` is a bool, or is neither a number,
                text nor None.
            ValueError: If `name` or `reason` is not a string or is blank,
                an evidence item is not a string, is blank or is repeated,
                or a numeric `observed` or `expected` is not finite.
        """
        if not isinstance(self.kind, Kind):
            raise errors.KindError("kind must be Kind")
        name: object = self.name  # may be a non-string at runtime
        if not isinstance(name, str) or not name.strip():
            raise errors.InputError("name must not be empty")
        if not isinstance(self.passed, bool):
            raise errors.KindError("passed must be bool")
        if not isinstance(self.blocking, bool):
            raise errors.KindError("blocking must be bool")
        reason: object = self.reason  # may be a non-string at runtime
        if not isinstance(reason, str) or not reason.strip():
            raise errors.InputError("reason must not be empty")
        if not isinstance(self.evidence, tuple):
            raise errors.KindError("evidence must be a tuple of strings")
        entries: Iterable[object] = self.evidence  # may hold non-strings
        if any(
            not isinstance(item, str) or not item.strip() for item in entries
        ):
            raise errors.InputError("evidence must contain non-empty strings")
        if len(self.evidence) != len(set(self.evidence)):
            raise errors.InputError("evidence must not contain duplicates")
        for field_name in ("observed", "expected"):
            object.__setattr__(
                self,
                field_name,
                quantify(field_name, getattr(self, field_name)),
            )

    def to_dict(self) -> dict[str, Any]:
        """Returns a deterministic JSON-compatible finding.

        Returns:
            A mapping of the finding's fields. `observed` and `expected`
            appear only when they are not None.
        """
        record: dict[str, Any] = {
            "kind": self.kind.value,
            "name": self.name,
            "passed": self.passed,
            "blocking": self.blocking,
            "reason": self.reason,
            "evidence": list(self.evidence),
        }
        if self.observed is not None:
            record["observed"] = self.observed
        if self.expected is not None:
            record["expected"] = self.expected
        return record


@dataclasses.dataclass(frozen=True, slots=True)
class Tally:
    """Validated aggregate counts for a set of release findings.

    Attributes:
        total: Number of findings.
        passed: Number of passed findings.
        failed: Number of failed findings.
        blocking: Number of findings that are blocking gates.
        blocking_failures: Number of failed blocking findings.
        migrations: The written layout's version steps, oldest first;
            version 2 writes `blocking_failures` as `blockers`.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(renames={"blocking_failures": "blockers"}),
    )

    total: int
    passed: int
    failed: int
    blocking: int
    blocking_failures: int

    def __post_init__(self) -> None:
        """Validates count relationships before exposing the aggregate.

        Raises:
            TypeError: If a count is not an integer (bools are rejected).
            ValueError: If a count is negative, `total` differs from
                `passed` plus `failed`, `blocking` exceeds `total`, or
                `blocking_failures` exceeds `failed` or `blocking`.
        """
        values = {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "blocking": self.blocking,
            "blocking_failures": self.blocking_failures,
        }
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, int):
                raise errors.KindError(f"{name} must be a non-negative integer")
            if value < 0:
                raise errors.InputError(
                    f"{name} must be a non-negative integer"
                )
        if self.total != self.passed + self.failed:
            raise errors.InputError("total must equal passed plus failed")
        if self.blocking > self.total:
            raise errors.InputError("blocking must not exceed total")
        if self.blocking_failures > self.failed:
            raise errors.InputError("blocking_failures must not exceed failed")
        if self.blocking_failures > self.blocking:
            raise errors.InputError(
                "blocking_failures must not exceed blocking"
            )

    def to_dict(self) -> dict[str, int]:
        """Returns deterministic JSON-compatible aggregate counts.

        Returns:
            A mapping from each count's name to its value, in the current
            layout.
        """
        return records.upgrade(
            self.migrations,
            {
                "total": self.total,
                "passed": self.passed,
                "failed": self.failed,
                "blocking": self.blocking,
                "blocking_failures": self.blocking_failures,
            },
            records.ORIGIN,
        )


def tallies(values: Mapping[str, Any]) -> None:
    """Checks that every named count is a non-negative integer.

    Args:
        values: The counts, keyed by field name.

    Raises:
        TypeError: If a count is a bool or not an integer.
        ValueError: If a count is negative.
    """
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise errors.KindError(f"{name} must be a non-negative integer")
        if value < 0:
            raise errors.InputError(f"{name} must be a non-negative integer")


def exhaustive(counts_by_kind: Any) -> None:
    """Checks that the counts cover every finding kind and only those.

    Args:
        counts_by_kind: The candidate mapping from kind value to counts.

    Raises:
        TypeError: If it is not a mapping of strings to `Tally`.
        ValueError: If its keys are not exactly the finding kinds.
    """
    if not isinstance(counts_by_kind, Mapping):
        raise errors.KindError("counts_by_kind must be a mapping")
    expected_kinds = {kind.value for kind in Kind}
    if set(counts_by_kind) != expected_kinds:
        raise errors.InputError(
            "counts_by_kind must contain every finding kind"
        )
    if not all(
        isinstance(kind, str) and isinstance(counts, Tally)
        for kind, counts in counts_by_kind.items()
    ):
        raise errors.KindError("counts_by_kind must map strings to Tally")
