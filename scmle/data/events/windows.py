"""Windowed features: a statistic of one field over a span before a cutoff.

A window `purchase.amount:total:7d` sums the `amount` of an entity's
`purchase` events whose time lies in `(cutoff - 7d, cutoff]`; the kind `*`
takes events of every kind, and a window without a field
(`login:count:1d`) counts events. Each statistic is a `Statistic` subclass;
defining a new subclass with its own `name` makes it available to every
window and to the command line, with no registry to edit.
"""

import abc
from collections.abc import Sequence
import dataclasses
import math
from typing import ClassVar

from scmle.data.events import event

# The kind that matches events of every kind.
ANY = "*"
NANOSECONDS = 1_000_000_000
# Parts of a specification: source, statistic and span.
PARTS = 3


def numbers(values: Sequence[object]) -> list[float]:
    """Keeps the numeric values (int or float, not bool) of a slice.

    Args:
        values: Field values, in time order.

    Returns:
        The numeric values as floats, NaN left out.
    """
    return [
        float(v)
        for v in values
        if isinstance(v, int | float)
        and not isinstance(v, bool)
        and not math.isnan(v)
    ]


class Statistic(abc.ABC):
    """A summary of the values a window holds.

    Attributes:
        name: The statistic's name in a window specification.
    """

    name: ClassVar[str]

    @abc.abstractmethod
    def measure(
        self, times: Sequence[int], values: Sequence[object], cutoff: int
    ) -> object:
        """Summarizes the window.

        Args:
            times: Event times in the window, ascending, in nanoseconds.
            values: The field's value for each of those events.
            cutoff: The prediction time the window ends at.

        Returns:
            The feature value.
        """

    @classmethod
    def named(cls, name: str) -> "Statistic":
        """Finds a statistic by name among every subclass defined.

        Args:
            name: The statistic's name.

        Returns:
            A new instance of that statistic.

        Raises:
            EventError: If no statistic has that name.
        """
        found = {kind.name: kind for kind in every(cls)}
        if name not in found:
            raise event.EventError(
                f"unknown statistic {name!r}; known: {sorted(found)}"
            )
        return found[name]()


def every(root: type[Statistic]) -> list[type[Statistic]]:
    """Lists the concrete statistics below a class.

    Args:
        root: The class to search below.

    Returns:
        Every subclass, at any depth, that defines a `name`.
    """
    found = []
    for kind in root.__subclasses__():
        if "name" in vars(kind):
            found.append(kind)
        found.extend(every(kind))
    return found


class Count(Statistic):
    """Number of events in the window (carrying the field, if one is named)."""

    name = "count"

    def measure(
        self,
        times: Sequence[int],
        values: Sequence[object],  # noqa: ARG002 -- the count ignores values
        cutoff: int,  # noqa: ARG002 -- the count ignores the cutoff
    ) -> object:
        """Counts the events.

        Args:
            times: Event times in the window.
            values: Unused.
            cutoff: Unused.

        Returns:
            The number of events.
        """
        return len(times)


class Total(Statistic):
    """Exactly rounded sum of the numeric values; 0 for an empty window."""

    name = "total"

    def measure(
        self,
        times: Sequence[int],  # noqa: ARG002 -- the sum ignores times
        values: Sequence[object],
        cutoff: int,  # noqa: ARG002 -- the sum ignores the cutoff
    ) -> object:
        """Sums the numeric values with `math.fsum`.

        Args:
            times: Unused.
            values: Field values in the window.
            cutoff: Unused.

        Returns:
            The sum.
        """
        return math.fsum(numbers(values))


class Mean(Statistic):
    """Mean of the numeric values; NaN for a window without any."""

    name = "mean"

    def measure(
        self,
        times: Sequence[int],  # noqa: ARG002 -- the mean ignores times
        values: Sequence[object],
        cutoff: int,  # noqa: ARG002 -- the mean ignores the cutoff
    ) -> object:
        """Averages the numeric values.

        Args:
            times: Unused.
            values: Field values in the window.
            cutoff: Unused.

        Returns:
            The exactly rounded sum over the count, or NaN.
        """
        found = numbers(values)
        return math.fsum(found) / len(found) if found else math.nan


class Least(Statistic):
    """Smallest numeric value; NaN for a window without any."""

    name = "min"

    def measure(
        self,
        times: Sequence[int],  # noqa: ARG002 -- the minimum ignores times
        values: Sequence[object],
        cutoff: int,  # noqa: ARG002 -- the minimum ignores the cutoff
    ) -> object:
        """Takes the minimum.

        Args:
            times: Unused.
            values: Field values in the window.
            cutoff: Unused.

        Returns:
            The minimum, or NaN.
        """
        return min(numbers(values), default=math.nan)


class Most(Statistic):
    """Largest numeric value; NaN for a window without any."""

    name = "max"

    def measure(
        self,
        times: Sequence[int],  # noqa: ARG002 -- the maximum ignores times
        values: Sequence[object],
        cutoff: int,  # noqa: ARG002 -- the maximum ignores the cutoff
    ) -> object:
        """Takes the maximum.

        Args:
            times: Unused.
            values: Field values in the window.
            cutoff: Unused.

        Returns:
            The maximum, or NaN.
        """
        return max(numbers(values), default=math.nan)


class Latest(Statistic):
    """The value of the newest event; None for an empty window.

    Events at the same time are ordered by arrival, so the last to arrive
    is the latest.
    """

    name = "last"

    def measure(
        self,
        times: Sequence[int],  # noqa: ARG002 -- order is given by position
        values: Sequence[object],
        cutoff: int,  # noqa: ARG002 -- the last value ignores the cutoff
    ) -> object:
        """Takes the newest value.

        Args:
            times: Unused.
            values: Field values in the window.
            cutoff: Unused.

        Returns:
            The newest value, or None.
        """
        return values[-1] if values else None


class Distinct(Statistic):
    """Number of different values in the window."""

    name = "distinct"

    def measure(
        self,
        times: Sequence[int],  # noqa: ARG002 -- distinctness ignores times
        values: Sequence[object],
        cutoff: int,  # noqa: ARG002 -- distinctness ignores the cutoff
    ) -> object:
        """Counts the different values.

        Args:
            times: Unused.
            values: Field values in the window.
            cutoff: Unused.

        Returns:
            The number of different values.
        """
        return len(set(values))


class Recency(Statistic):
    """Seconds from the newest event to the cutoff; NaN for an empty window."""

    name = "recency"

    def measure(
        self,
        times: Sequence[int],
        values: Sequence[object],  # noqa: ARG002 -- recency ignores values
        cutoff: int,
    ) -> object:
        """Measures the age of the newest event.

        Args:
            times: Event times in the window.
            values: Unused.
            cutoff: The prediction time.

        Returns:
            The age in seconds, or NaN.
        """
        return (cutoff - times[-1]) / NANOSECONDS if times else math.nan


@dataclasses.dataclass(frozen=True, slots=True)
class Window:
    """A statistic of one field of one kind over a span before a cutoff.

    Attributes:
        kind: The event kind the window reads, or `ANY`.
        field: The payload field it reads; None reads event presence.
        statistic: What it computes over the values in the span.
        span: The span's length in nanoseconds; the window holds events
            with `cutoff - span < time <= cutoff`.
        label: The span as written, used in the feature name.
    """

    kind: str
    field: str | None
    statistic: Statistic
    span: int
    label: str

    @property
    def name(self) -> str:
        """The feature's name: the window's specification."""
        source = (
            self.kind if self.field is None else f"{self.kind}.{self.field}"
        )
        return f"{source}:{self.statistic.name}:{self.label}"

    @classmethod
    def parse(cls, text: str) -> "Window":
        """Reads a specification `KIND[.FIELD]:STATISTIC:SPAN`.

        The kind ends at the first dot, so a field may itself be dotted (a
        flattened payload object). `*` is every kind.

        Args:
            text: The specification, such as `purchase.amount:total:7d`.

        Returns:
            The window.

        Raises:
            EventError: If the text has not three parts, its kind or a
                field after a dot is empty, it names an unknown
                statistic, or its span is not a positive duration.
        """
        parts = text.split(":")
        if len(parts) != PARTS or not all(p.strip() for p in parts):
            raise event.EventError(
                f"a window is KIND[.FIELD]:STATISTIC:SPAN, got {text!r}"
            )
        source, name, label = (p.strip() for p in parts)
        kind, dot, field = source.partition(".")
        if not kind or (dot and not field):
            raise event.EventError(
                f"a window is KIND[.FIELD]:STATISTIC:SPAN, got {text!r}"
            )
        length = event.span(label)
        if length == 0:
            raise event.EventError(
                f"a window's span must be positive: {text!r}"
            )
        statistic = Statistic.named(name)
        if not dot and not isinstance(statistic, Count | Recency):
            raise event.EventError(
                f"{name!r} needs a field: write KIND.FIELD:{name}:{label}"
            )
        return cls(kind, field if dot else None, statistic, length, label)
