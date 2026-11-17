"""Graph nodes, edges, event times, validation reports and their errors."""

import dataclasses
from datetime import datetime
from datetime import UTC
import math
from typing import Any

from scmle.contracts import errors


class GraphError(ValueError):
    """Base error for invalid graph data or graph input."""


class ParseError(GraphError):
    """Raised when a graph input file cannot be parsed."""


class TemporalError(GraphError):
    """Raised when temporal graph evidence is incomplete or invalid."""


class ValidationError(GraphError):
    """Raised when a graph fails structural validation.

    Attributes:
        report: The complete validation report that caused the error.
    """

    def __init__(self, report: "Validation") -> None:
        """Builds an error containing the complete validation report.

        Args:
            report: The validation report; its messages, joined by ``; ``,
                form the error message.
        """
        self.report = report
        super().__init__("; ".join(issue.message for issue in report.issues))


@dataclasses.dataclass(frozen=True, slots=True)
class Node:
    """Immutable node with an optional numeric feature vector and label.

    Attributes:
        node_id: Non-empty node identifier.
        features: Finite numeric feature vector, stored as floats; may be
            empty.
        label: Optional node label.
    """

    node_id: str
    features: tuple[float, ...] = ()
    label: str | int | float | bool | None = None

    def __post_init__(self) -> None:
        """Validates the node identifier and numeric feature vector.

        Raises:
            ValueError: If node_id is not a non-empty string or a feature is
                not finite.
            TypeError: If a feature is not an int or float.
        """
        node_id: object = self.node_id
        if not isinstance(node_id, str) or not node_id.strip():
            raise errors.InputError("node_id must be a non-empty string")
        if any(not isinstance(value, (int, float)) for value in self.features):
            raise errors.KindError("node features must be numeric")
        if any(not math.isfinite(float(value)) for value in self.features):
            raise errors.InputError("node features must be finite")
        object.__setattr__(
            self,
            "features",
            tuple(float(value) for value in self.features),
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Edge:
    """Immutable weighted edge between two node identifiers.

    Attributes:
        source: Non-empty identifier of the source node.
        target: Non-empty identifier of the target node.
        weight: Finite edge weight, stored as a float.
        relation: Non-empty relation name.
        event_time: Optional timezone-aware time, stored in UTC.
    """

    source: str
    target: str
    weight: float = 1.0
    relation: str = "default"
    event_time: datetime | None = None

    def __post_init__(self) -> None:
        """Validates endpoint, relation, and weight values.

        Raises:
            ValueError: If an endpoint or the relation is not a non-empty
                string, the weight is not finite, or the event time has no
                timezone.
            TypeError: If the weight is not numeric or the event time is not
                a datetime.
        """
        source: object = self.source
        target: object = self.target
        if not isinstance(source, str) or not source.strip():
            raise errors.InputError("edge source must be a non-empty string")
        if not isinstance(target, str) or not target.strip():
            raise errors.InputError("edge target must be a non-empty string")
        if not isinstance(self.weight, (int, float)):
            raise errors.KindError("edge weight must be numeric")
        if not math.isfinite(float(self.weight)):
            raise errors.InputError("edge weight must be finite")
        relation: object = self.relation
        if not isinstance(relation, str) or not relation.strip():
            raise errors.InputError("edge relation must be a non-empty string")
        if self.event_time is not None:
            if not isinstance(self.event_time, datetime):
                raise errors.KindError("event_time must be a datetime")
            if (
                self.event_time.tzinfo is None
                or self.event_time.utcoffset() is None
            ):
                raise errors.InputError("event_time must be timezone-aware")
            object.__setattr__(
                self,
                "event_time",
                self.event_time.astimezone(UTC),
            )
        object.__setattr__(self, "weight", float(self.weight))


@dataclasses.dataclass(frozen=True, slots=True)
class Chronology:
    """Time-ordered graph edges separated at an explicit UTC cutoff.

    Attributes:
        cutoff: Timezone-aware cutoff, converted to UTC.
        train_edges: Edges with event time at or before the cutoff.
        future_edges: Edges with event time after the cutoff.
    """

    cutoff: datetime
    train_edges: tuple[Edge, ...]
    future_edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        """Validates the cutoff and timestamp completeness of both partitions.

        Raises:
            ValueError: If the cutoff is not timezone-aware or an edge in
                either partition has no event time.
        """
        if self.cutoff.tzinfo is None or self.cutoff.utcoffset() is None:
            raise errors.InputError("cutoff must be timezone-aware")
        object.__setattr__(
            self,
            "cutoff",
            self.cutoff.astimezone(UTC),
        )
        if any(edge.event_time is None for edge in self.train_edges):
            raise errors.InputError("train_edges must contain event times")
        if any(edge.event_time is None for edge in self.future_edges):
            raise errors.InputError("future_edges must contain event times")

    @property
    def lookahead(self) -> bool:
        """Whether a training edge occurs after the cutoff."""
        cutoff = self.cutoff.astimezone(UTC)
        return any(
            edge.event_time is not None and edge.event_time > cutoff
            for edge in self.train_edges
        )


def normalize(value: datetime | str) -> datetime:
    """Parses and normalizes a timezone-aware graph event timestamp to UTC.

    Args:
        value: A datetime, or ISO 8601 text; a trailing ``Z`` means UTC.

    Returns:
        The same instant as a UTC datetime.

    Raises:
        TemporalError: If the text is not ISO 8601, the value is
            neither datetime nor text, or the time has no timezone.
    """
    if isinstance(value, str):
        normalized = value.strip()
        if normalized.endswith(("Z", "z")):
            normalized = normalized[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError as error:
            raise TemporalError(
                f"invalid graph event time: {value!r}"
            ) from error
    elif isinstance(value, datetime):
        parsed = value
    else:
        raise TemporalError("graph event time must be datetime or text")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise TemporalError("graph event time must be timezone-aware")
    return parsed.astimezone(UTC)


@dataclasses.dataclass(frozen=True, slots=True)
class Issue:
    """One deterministic graph validation finding.

    Attributes:
        code: Machine-readable issue code.
        message: Human-readable description.
        affected: Identifiers of the nodes or edge parts concerned.
    """

    code: str
    message: str
    affected: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible issue mapping.

        Returns:
            A dict with `code`, `message` and `affected` (as a list).
        """
        return {
            "code": self.code,
            "message": self.message,
            "affected": list(self.affected),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class Validation:
    """Complete graph validation result.

    Attributes:
        issues: Every issue found; empty when the graph is valid.
    """

    issues: tuple[Issue, ...]

    @property
    def is_valid(self) -> bool:
        """Whether no validation issues were found."""
        return not self.issues

    @property
    def error_codes(self) -> tuple[str, ...]:
        """The unique issue codes, sorted."""
        return tuple(sorted({issue.code for issue in self.issues}))

    def to_dict(self) -> dict[str, Any]:
        """Returns a JSON-compatible report mapping.

        Returns:
            A dict with `is_valid`, `error_codes` and `issues` (each issue as
            a dict).
        """
        return {
            "is_valid": self.is_valid,
            "error_codes": list(self.error_codes),
            "issues": [issue.to_dict() for issue in self.issues],
        }
