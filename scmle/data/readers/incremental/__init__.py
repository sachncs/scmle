"""Transport-neutral incremental event ingestion contracts.

This package owns the semantics that every event source must provide: stable
event identity, ordered checkpoints, watermarks, late-data policy, duplicate
classification, and an atomic durable commit. It deliberately does not
import a broker or warehouse SDK. A transport adapter supplies available
events through ``Source``.

The package keeps every name the single module incremental.py had, so callers
are unchanged; `connectors` stays reachable here because tests name
`incremental.connectors`. It splits by concern: `timeline` (events,
checkpoints and batches), `evidence` (receipts, acknowledgements and state),
`sources` (the source contract and its adapters), `store` (the durable
`Ledger`) and `consumer` (the `Ingestor`).
"""

from scmle.data.readers import connectors
from scmle.data.readers.incremental.consumer import Ingestor
from scmle.data.readers.incremental.evidence import Ack
from scmle.data.readers.incremental.evidence import Consumption
from scmle.data.readers.incremental.evidence import EventAcknowledgement
from scmle.data.readers.incremental.evidence import Settlement
from scmle.data.readers.incremental.sources import SequenceEventSource
from scmle.data.readers.incremental.sources import Source
from scmle.data.readers.incremental.sources import Stream
from scmle.data.readers.incremental.store import Ledger
from scmle.data.readers.incremental.timeline import Batch
from scmle.data.readers.incremental.timeline import Checkpoint
from scmle.data.readers.incremental.timeline import Event
from scmle.data.readers.incremental.timeline import IncrementalError
from scmle.data.readers.incremental.timeline import Lateness
from scmle.data.readers.incremental.timeline import parse_timestamp

__all__ = [
    "Ack",
    "Batch",
    "Checkpoint",
    "Consumption",
    "Event",
    "EventAcknowledgement",
    "IncrementalError",
    "Ingestor",
    "Lateness",
    "Ledger",
    "SequenceEventSource",
    "Settlement",
    "Source",
    "Stream",
    "connectors",
    "parse_timestamp",
]
