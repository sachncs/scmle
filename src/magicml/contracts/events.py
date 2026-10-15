"""Event contract and JSONL event log."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Event:
    run_id: str
    kind: str
    reason: str = ""
    config_hash: str = ""
    artifacts: tuple[str, ...] = ()
    data: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str, sort_keys=True)


class EventLog:
    """Append-only event sink; in-memory always, JSONL on disk when a path is given."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path
        self.events: list[Event] = []

    def emit(self, event: Event) -> None:
        self.events.append(event)
        if self.path is not None:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(event.to_json() + "\n")
