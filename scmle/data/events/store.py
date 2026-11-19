"""Point-in-time entity snapshots and windowed features, kept incrementally.

A `Store` takes events one at a time, in any order, and keeps each entity's
values per (kind, field) in time order. A query names an entity and a
cutoff; it sees exactly the events with `time <= cutoff`, so a feature for
prediction at the cutoff never reads the future. Adding an event costs a
binary search plus the shift of the events newer than it, which the
lateness bound keeps small; answering a query costs a binary search plus
the events in its windows, never a pass over the log.

Out-of-order events are admitted while they are no older than the
watermark, the newest time seen less the declared lateness bound. An older
event is late: the `policy` drops it (counted), rejects it
(`LatenessError`), or accepts it and counts the revision. Under `drop` and
`reject` a query whose cutoff lies before the watermark is final: no event
the store keeps can change it. Under `accept` no query is ever final, since
an event of any age may still arrive and revise it.
"""

import bisect
from collections import Counter
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import enum
import itertools

import pandas

from scmle.data.events import event
from scmle.data.events import windows
from scmle.data.events.windows import ANY
from scmle.data.readers.incremental.timeline import Lateness

# Highest arrival rank: a search key that sorts after every event at a time.
LAST = 2**63


class Arrival(enum.StrEnum):
    """How an event arrived, relative to the events before it.

    Attributes:
        TIMELY: At or after the newest time seen.
        DISORDERED: Out of order, but within the lateness bound.
        LATE: Older than the watermark, and kept by the `accept` policy.
        DROPPED: Older than the watermark, and discarded by the `drop`
            policy.
    """

    TIMELY = "timely"
    DISORDERED = "disordered"
    LATE = "late"
    DROPPED = "dropped"


class Series:
    """One field of one kind of one entity, in (time, arrival) order.

    Attributes:
        keys: The (time, arrival rank) of each value, ascending.
        values: The field's value for each key.
    """

    def __init__(self) -> None:
        """Starts an empty series."""
        self.keys: list[tuple[int, int]] = []
        self.values: list[object] = []

    def insert(self, time: int, rank: int, value: object) -> None:
        """Places one value at its time; an in-order value is appended.

        Args:
            time: The event's time in nanoseconds.
            rank: The event's arrival rank, which orders equal times.
            value: The field's value.
        """
        key = (time, rank)
        if not self.keys or key > self.keys[-1]:
            self.keys.append(key)
            self.values.append(value)
            return
        position = bisect.bisect_right(self.keys, key)
        self.keys.insert(position, key)
        self.values.insert(position, value)

    def slice(self, start: int, stop: int) -> tuple[list[int], list[object]]:
        """The times and values with `start < time <= stop`.

        Args:
            start: Exclusive lower time bound.
            stop: Inclusive upper time bound.

        Returns:
            The times and the values, in order.
        """
        low = bisect.bisect_right(self.keys, (start, LAST))
        high = bisect.bisect_right(self.keys, (stop, LAST))
        return [k[0] for k in self.keys[low:high]], self.values[low:high]

    def latest(self, cutoff: int) -> object:
        """The newest value at or before a cutoff.

        Args:
            cutoff: Inclusive upper time bound.

        Returns:
            The value, or None when the series starts after the cutoff.
        """
        position = bisect.bisect_right(self.keys, (cutoff, LAST))
        return self.values[position - 1] if position else None


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    """An entity as known at a cutoff.

    Attributes:
        entity: The entity.
        cutoff: The prediction time, in nanoseconds.
        state: The newest value of each `kind.field` at the cutoff.
        features: Each window's value, by window name.
        final: Whether no later event the store keeps can change the
            snapshot: the cutoff lies before the watermark and the policy
            is not `accept`, which keeps events of any age.
    """

    entity: str
    cutoff: int
    state: Mapping[str, object]
    features: Mapping[str, object]
    final: bool


class Store:
    """An append-only event log kept as per-entity, per-field series.

    Attributes:
        windows: The windowed features every query computes.
        lateness: The lateness bound in nanoseconds.
        policy: What happens to an event older than the watermark.
        entities: Each entity's series by (kind, field); field None holds
            event presence.
        frontier: The newest event time seen, or None before any event.
        tally: How many events arrived each way.
        ranks: The source of arrival ranks.
        wide: Whether some window reads every kind, so `*` series are
            kept.
    """

    def __init__(
        self,
        windows: Sequence[windows.Window] = (),
        lateness: int = 0,
        policy: Lateness = Lateness.DROP,
    ) -> None:
        """Starts an empty store.

        Args:
            windows: The windowed features every query computes.
            lateness: The lateness bound in nanoseconds; 0 admits no
                out-of-order event older than the newest time.
            policy: What happens to an event older than the watermark.

        Raises:
            EventError: If the bound is negative or two windows share a name.
        """
        if lateness < 0:
            raise event.EventError("the lateness bound must not be negative")
        names = Counter(w.name for w in windows)
        if repeated := sorted(n for n, c in names.items() if c > 1):
            raise event.EventError(f"windows named twice: {repeated}")
        self.windows = tuple(windows)
        self.lateness = lateness
        self.policy = Lateness(policy)
        self.entities: dict[str, dict[tuple[str, str | None], Series]] = {}
        self.frontier: int | None = None
        self.tally: Counter[Arrival] = Counter()
        self.ranks = itertools.count()
        self.wide = any(w.kind == ANY for w in self.windows)

    @property
    def watermark(self) -> int | None:
        """The oldest time an event may have and still be on time."""
        if self.frontier is None:
            return None
        return self.frontier - self.lateness

    def classify(self, time: int) -> Arrival:
        """Tells how an event at a time would arrive now.

        Args:
            time: The event's time in nanoseconds.

        Returns:
            TIMELY, DISORDERED, or LATE (before the policy is applied).
        """
        if self.frontier is None or time >= self.frontier:
            return Arrival.TIMELY
        watermark = self.frontier - self.lateness
        return Arrival.DISORDERED if time >= watermark else Arrival.LATE

    def add(self, item: event.Event) -> Arrival:
        """Adds one event.

        Args:
            item: The event.

        Returns:
            How it arrived; DROPPED when the policy discarded it.

        Raises:
            LatenessError: If it is late and the policy is `reject`.
        """
        arrival = self.classify(item.time)
        if arrival is Arrival.LATE and self.policy is Lateness.REJECT:
            raise event.LatenessError(
                f"event of {item.entity!r} at {pandas.Timestamp(item.time)} "
                f"is older than the watermark "
                f"{pandas.Timestamp(self.watermark or 0)}",
                {"entity": item.entity, "time": item.time},
            )
        if arrival is Arrival.LATE and self.policy is Lateness.DROP:
            arrival = Arrival.DROPPED
        self.tally[arrival] += 1
        if arrival is Arrival.DROPPED:
            return arrival
        rank = next(self.ranks)
        series = self.entities.setdefault(item.entity, {})
        kinds = (item.kind, ANY) if self.wide else (item.kind,)
        for kind in kinds:
            series.setdefault((kind, None), Series()).insert(
                item.time, rank, True
            )
            for field, value in item.payload.items():
                series.setdefault((kind, field), Series()).insert(
                    item.time, rank, value
                )
        if self.frontier is None or item.time > self.frontier:
            self.frontier = item.time
        return arrival

    def extend(self, items: Iterable[event.Event]) -> Counter[Arrival]:
        """Adds events in arrival order.

        Args:
            items: The events.

        Returns:
            How many arrived each way in this call.
        """
        found: Counter[Arrival] = Counter()
        for item in items:
            found[self.add(item)] += 1
        return found

    def final(self, cutoff: int) -> bool:
        """Whether no later event the store keeps can change a cutoff.

        Args:
            cutoff: The prediction time in nanoseconds.

        Returns:
            True when the cutoff lies before the watermark and the policy
            drops or rejects older events; always False under `accept`,
            which keeps an event of any age.
        """
        watermark = self.watermark
        return (
            self.policy is not Lateness.ACCEPT
            and watermark is not None
            and cutoff < watermark
        )

    def snapshot(self, entity: str, cutoff: int) -> Snapshot:
        """The entity as known at a cutoff, with every window's value.

        Args:
            entity: The entity.
            cutoff: The prediction time in nanoseconds; events at exactly
                the cutoff are included.

        Returns:
            The snapshot; an unknown entity has an empty state and empty
            windows.
        """
        series = self.entities.get(entity, {})
        state = {}
        for (kind, field), history in sorted(
            series.items(), key=lambda pair: (pair[0][0], pair[0][1] or "")
        ):
            value = history.latest(cutoff)
            if field is not None and kind != ANY and value is not None:
                state[f"{kind}.{field}"] = value
        features = {}
        empty = Series()
        for window in self.windows:
            times, values = series.get(
                (window.kind, window.field), empty
            ).slice(cutoff - window.span, cutoff)
            features[window.name] = window.statistic.measure(
                times, values, cutoff
            )
        return Snapshot(entity, cutoff, state, features, self.final(cutoff))

    def frame(
        self,
        queries: pandas.DataFrame,
        entity: str = "entity",
        cutoff: str = "cutoff",
    ) -> pandas.DataFrame:
        """Answers a table of (entity, cutoff) queries.

        Args:
            queries: One row per prediction to make.
            entity: The column holding the entity.
            cutoff: The column holding the cutoff.

        Returns:
            The queries' rows, in order, with one column per window, one
            per state field any snapshot has, and `final`.

        Raises:
            EventError: If a column is missing, a cutoff is not a
                timestamp, or an answer column shares a query column's name.
        """
        for column in (entity, cutoff):
            if column not in queries.columns:
                raise event.EventError(f"the queries need a {column!r} column")
        rows = []
        for who, when in zip(
            queries[entity].tolist(), queries[cutoff].tolist(), strict=True
        ):
            found = self.snapshot(event.identity(who), event.instant(when))
            rows.append(
                {
                    **found.features,
                    **found.state,
                    "final": found.final,
                }
            )
        answers = pandas.DataFrame(rows, index=queries.index)
        names = [w.name for w in self.windows]
        state = sorted(set(answers.columns) - {*names, "final"})
        ordered = answers.reindex(columns=[*names, *state, "final"])
        if clash := sorted(set(ordered.columns) & set(queries.columns)):
            raise event.EventError(
                f"answer columns would overwrite query columns: {clash}",
                {"columns": clash},
            )
        return pandas.concat([queries, ordered], axis=1)
