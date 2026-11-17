"""Deterministic source connectors for files, objects, and DB-API queries.

Connectors expose the smallest useful ingestion contract: identify a logical
source, inspect its schema, and read bounded row batches from a checkpoint.
Network clients remain caller-owned: transport functions are injected and
their boundaries are recorded by the connector rather than hidden behind a
cloud SDK import.

The package keeps every name the single module connectors.py had, so
callers are unchanged. It splits by concern: `failure` (errors and the retry
budget), `locator` (canonical URIs, identities and bookmarks), `layout`
(schemas and pages), `adapter` (the shared contract) and one module per
connector: `filesystem`, `database`, `lakehouse`, `bucket` and `stream`.
"""

from scmle.data.readers.connectors.adapter import Adapter
from scmle.data.readers.connectors.bucket import Bucket
from scmle.data.readers.connectors.bucket import CSV_COMPRESSION
from scmle.data.readers.connectors.bucket import ObjectDecoder
from scmle.data.readers.connectors.bucket import Payload
from scmle.data.readers.connectors.bucket import unpack
from scmle.data.readers.connectors.database import DbApiConnector
from scmle.data.readers.connectors.database import feed
from scmle.data.readers.connectors.database import LITERALS_AND_COMMENTS
from scmle.data.readers.connectors.database import READ_STARTS
from scmle.data.readers.connectors.database import readonly
from scmle.data.readers.connectors.database import WORD
from scmle.data.readers.connectors.database import WRITE_KEYWORDS
from scmle.data.readers.connectors.failure import CheckpointError
from scmle.data.readers.connectors.failure import ConnectorError
from scmle.data.readers.connectors.failure import persevere
from scmle.data.readers.connectors.failure import Retries
from scmle.data.readers.connectors.failure import RetryError
from scmle.data.readers.connectors.failure import SchemeError
from scmle.data.readers.connectors.failure import T
from scmle.data.readers.connectors.filesystem import checksum
from scmle.data.readers.connectors.filesystem import Filesystem
from scmle.data.readers.connectors.lakehouse import Lakehouse
from scmle.data.readers.connectors.layout import Columns
from scmle.data.readers.connectors.layout import Field
from scmle.data.readers.connectors.layout import Page
from scmle.data.readers.connectors.layout import Row
from scmle.data.readers.connectors.locator import Bookmark
from scmle.data.readers.connectors.locator import bookmark
from scmle.data.readers.connectors.locator import canonicalize
from scmle.data.readers.connectors.locator import SHA256_HEX_LENGTH
from scmle.data.readers.connectors.locator import source_identity
from scmle.data.readers.connectors.locator import unwrap
from scmle.data.readers.connectors.stream import JSON_TYPE_NAMES
from scmle.data.readers.connectors.stream import RangeReader
from scmle.data.readers.connectors.stream import SizeReader
from scmle.data.readers.connectors.stream import StreamingObjectStoreConnector

__all__ = [
    "CSV_COMPRESSION",
    "JSON_TYPE_NAMES",
    "LITERALS_AND_COMMENTS",
    "READ_STARTS",
    "SHA256_HEX_LENGTH",
    "WORD",
    "WRITE_KEYWORDS",
    "Adapter",
    "Bookmark",
    "Bucket",
    "CheckpointError",
    "Columns",
    "ConnectorError",
    "DbApiConnector",
    "Field",
    "Filesystem",
    "Lakehouse",
    "ObjectDecoder",
    "Page",
    "Payload",
    "RangeReader",
    "Retries",
    "RetryError",
    "Row",
    "SchemeError",
    "SizeReader",
    "StreamingObjectStoreConnector",
    "T",
    "bookmark",
    "canonicalize",
    "checksum",
    "feed",
    "persevere",
    "readonly",
    "source_identity",
    "unpack",
    "unwrap",
]
