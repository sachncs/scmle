"""Event contract and the append-only event log."""

from collections.abc import Callable
from collections.abc import Sequence
import dataclasses
import datetime
import json
import logging
import pathlib
from typing import Any

LOGGER = logging.getLogger(__name__)


def utc_timestamp() -> str:
    """Returns the current UTC time as an ISO-8601 string."""
    return datetime.datetime.now(datetime.UTC).isoformat()


@dataclasses.dataclass(frozen=True, slots=True)
class Event:
    """A single auditable occurrence during a run.

    Attributes:
        run_id: Identifier of the run that produced the event.
        kind: Dotted event type, e.g. `state.TRAINING` or `trial.ok`.
        reason: Why the event happened.
        config_hash: Hash of the configuration in force.
        artifacts: Run-relative paths of artifacts the event refers to.
        data: Free-form structured payload.
        timestamp: UTC creation time.
    """

    run_id: str
    kind: str
    reason: str = ""
    config_hash: str = ""
    artifacts: tuple[str, ...] = ()
    data: dict[str, Any] = dataclasses.field(default_factory=dict)
    timestamp: str = dataclasses.field(default_factory=utc_timestamp)

    def to_json(self) -> str:
        """Serializes the event as one canonical JSON line."""
        return json.dumps(dataclasses.asdict(self), default=str, sort_keys=True)


class EventLog:
    """Append-only event sink: in memory always, JSONL on disk if configured."""

    def __init__(self, path: pathlib.Path | None = None) -> None:
        """Initializes the log.

        Args:
            path: Optional JSONL file that receives every emitted event.
        """
        self.path = path
        self.events: list[Event] = []
        self.listeners: list[Callable[[Event], None]] = []

    def subscribe(self, listener: Callable[[Event], None]) -> None:
        """Registers a callback invoked for every emitted event.

        Args:
            listener: Function receiving each event after it is recorded.
        """
        self.listeners.append(listener)

    def emit(self, event: Event) -> None:
        """Records an event.

        Args:
            event: The event to append.
        """
        self.events.append(event)
        LOGGER.info("[%s] %s: %s", event.run_id, event.kind, event.reason)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(event.to_json() + "\n")
        for listener in self.listeners:
            listener(event)

    def kinds(self) -> Sequence[str]:
        """Returns the kinds of all recorded events, in order."""
        return [event.kind for event in self.events]
