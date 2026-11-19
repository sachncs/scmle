"""Event streams: an append-only log turned into features at any cutoff.

Read events (entity, time, type, payload) from files (`File`), a text
stream (`Lines`) or a file another process appends to (`Tail`); add them
to a `Store` in any order; ask it for a `Snapshot` of an entity at a cutoff
or for a table of features at many. Windows (`Window.parse(
"purchase.amount:total:7d")`) see only events at or before the cutoff.
Out-of-order events within the declared lateness bound are placed where
they belong; older ones are dropped, rejected or accepted as revisions by
the policy, and `lateness.bound` estimates the bound from history.
`python -m scmle.data.events` is the command line.
"""

from scmle.data.events.event import Columns
from scmle.data.events.event import Event
from scmle.data.events.event import EventError
from scmle.data.events.event import LatenessError
from scmle.data.events.sources import File
from scmle.data.events.sources import Lines
from scmle.data.events.sources import Source
from scmle.data.events.sources import Tail
from scmle.data.events.store import Arrival
from scmle.data.events.store import Snapshot
from scmle.data.events.store import Store
from scmle.data.events.windows import Statistic
from scmle.data.events.windows import Window
from scmle.data.readers.incremental.timeline import Lateness

__all__ = [
    "Arrival",
    "Columns",
    "Event",
    "EventError",
    "File",
    "Lateness",
    "LatenessError",
    "Lines",
    "Snapshot",
    "Source",
    "Statistic",
    "Store",
    "Tail",
    "Window",
]
