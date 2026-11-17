"""Reading tables larger than memory, in batches.

A big file is never loaded whole. One pass over its batches picks the locked
test rows by hashing each row's position (so the choice needs no memory and
is the same on every run), and keeps a bounded random sample of the other
rows for development. Each row's sampling priority is also a hash of its
position, so the sample does not depend on the batch size.

The package keeps every name the single module streaming.py had, so callers
are unchanged. It splits by concern: `failure` (the error), `progress`
(checkpoints, watermarks and batches), `feed` (the per-format readers),
`transaction` (acknowledgements, receipts and durable state), `archive`
(reading and checking that state), `journal` (the commits), `consumer` (the
acknowledging consumer) and `sampling` (the one-pass split and sample).
"""

from scmle.data.readers.streaming.consumer import Consumer
from scmle.data.readers.streaming.failure import StreamingError
from scmle.data.readers.streaming.feed import BATCH_ROWS
from scmle.data.readers.streaming.feed import BYTES_PER_CELL
from scmle.data.readers.streaming.feed import Chunked
from scmle.data.readers.streaming.feed import Csv
from scmle.data.readers.streaming.feed import CSV_EXPANSION
from scmle.data.readers.streaming.feed import Feed
from scmle.data.readers.streaming.feed import locate
from scmle.data.readers.streaming.feed import Ndjson
from scmle.data.readers.streaming.feed import SOURCES
from scmle.data.readers.streaming.journal import Journal
from scmle.data.readers.streaming.progress import advance
from scmle.data.readers.streaming.progress import assay
from scmle.data.readers.streaming.progress import Batch
from scmle.data.readers.streaming.progress import Checkpoint
from scmle.data.readers.streaming.progress import SHA256_HEX_LENGTH
from scmle.data.readers.streaming.progress import validate_timestamp
from scmle.data.readers.streaming.sampling import allot
from scmle.data.readers.streaming.sampling import budgets
from scmle.data.readers.streaming.sampling import DEVELOPMENT
from scmle.data.readers.streaming.sampling import LOCKED
from scmle.data.readers.streaming.sampling import MIN_SAMPLE_ROWS
from scmle.data.readers.streaming.sampling import MIX
from scmle.data.readers.streaming.sampling import MIX_A
from scmle.data.readers.streaming.sampling import MIX_B
from scmle.data.readers.streaming.sampling import retain
from scmle.data.readers.streaming.sampling import row_hash
from scmle.data.readers.streaming.sampling import SAMPLE_SHARE
from scmle.data.readers.streaming.sampling import SPLIT_COLUMN
from scmle.data.readers.streaming.sampling import sweep
from scmle.data.readers.streaming.sampling import Totals
from scmle.data.readers.streaming.transaction import Acknowledgement
from scmle.data.readers.streaming.transaction import Commitment
from scmle.data.readers.streaming.transaction import Receipt
from scmle.data.readers.streaming.transaction import Settlement
from scmle.data.readers.streaming.transaction import Sink
from scmle.data.readers.streaming.transaction import (
    STREAM_ACKNOWLEDGEMENT_SCHEMA,
)
from scmle.data.readers.streaming.transaction import STREAM_STATE_SCHEMA
from scmle.data.readers.streaming.transaction import STREAM_TRANSACTION_SCHEMA
from scmle.data.readers.streaming.transaction import Voucher

__all__ = [
    "BATCH_ROWS",
    "BYTES_PER_CELL",
    "CSV_EXPANSION",
    "DEVELOPMENT",
    "LOCKED",
    "MIN_SAMPLE_ROWS",
    "MIX",
    "MIX_A",
    "MIX_B",
    "SAMPLE_SHARE",
    "SHA256_HEX_LENGTH",
    "SOURCES",
    "SPLIT_COLUMN",
    "STREAM_ACKNOWLEDGEMENT_SCHEMA",
    "STREAM_STATE_SCHEMA",
    "STREAM_TRANSACTION_SCHEMA",
    "Acknowledgement",
    "Batch",
    "Checkpoint",
    "Chunked",
    "Commitment",
    "Consumer",
    "Csv",
    "Feed",
    "Journal",
    "Ndjson",
    "Receipt",
    "Settlement",
    "Sink",
    "StreamingError",
    "Totals",
    "Voucher",
    "advance",
    "allot",
    "assay",
    "budgets",
    "locate",
    "retain",
    "row_hash",
    "sweep",
    "validate_timestamp",
]
