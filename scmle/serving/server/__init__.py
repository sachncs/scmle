"""Serving adapters: how a bundle is exposed to callers.

`Server` is the contract; `Batcher` scores files and
`HttpServing` exposes a JSON API using only the standard library, so serving
needs no extra dependencies. Online and batch scoring share one path,
`Server.infer`, so they cannot drift apart: the same
validation, the same preprocessing, the same model call, the same response.

The package splits the adapters: `contract` (`Server`, the limits and the
logger), `batch` (`Batcher`), `web` (`HttpServing`, its `Observer` and
`Listener`) and `handler` (`Responder`, which answers each request).
"""

# Callers patch `server.durability.write_atomically` to watch batch writes.
from scmle import durability  # noqa: F401
from scmle.serving.server.batch import Batcher
from scmle.serving.server.contract import BATCH_CHUNK_ROWS
from scmle.serving.server.contract import Loader
from scmle.serving.server.contract import LOGGER
from scmle.serving.server.contract import MAX_CONCURRENT_REQUESTS
from scmle.serving.server.contract import MAX_REQUEST_BYTES
from scmle.serving.server.contract import MAX_REQUEST_RECORDS
from scmle.serving.server.contract import SCORING_TIMEOUT_SECONDS
from scmle.serving.server.contract import Server
from scmle.serving.server.handler import Responder
from scmle.serving.server.web import HttpServing
from scmle.serving.server.web import Listener
from scmle.serving.server.web import Observer

__all__ = [
    "BATCH_CHUNK_ROWS",
    "LOGGER",
    "MAX_CONCURRENT_REQUESTS",
    "MAX_REQUEST_BYTES",
    "MAX_REQUEST_RECORDS",
    "SCORING_TIMEOUT_SECONDS",
    "Batcher",
    "HttpServing",
    "Listener",
    "Loader",
    "Observer",
    "Responder",
    "Server",
]
