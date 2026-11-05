"""Mock services for the local environment: an object store and a metrics sink.

They exist so the local demo exercises the same moves a real deployment makes
(upload an artifact and read it back by digest, ship metrics somewhere else)
with no cloud account. They are not S3 or a metrics backend: no
authentication, no multipart, no retention. Everything is in the standard
library and bound to a port chosen by the caller.

    python -m scmle.deploy.mocks object-store --root /data --port 9000
    python -m scmle.deploy.mocks metrics-sink --file metrics.jsonl --port 9100
"""

import argparse
from http import server
import json
import pathlib
import threading
import time
from typing import Any
import urllib.error
import urllib.request

from scmle import durability
from scmle.contracts import errors
from scmle.contracts import observability
from scmle.deploy import provenance

MAX_OBJECT_BYTES = 256 * 1024 * 1024
MAX_METRICS_BYTES = 1024 * 1024
DIGEST_HEADER = "X-Content-SHA256"


def safe_key(raw: str) -> pathlib.PurePosixPath:
    """Validates an object path of the form `bucket/key...`.

    Raises:
        ValueError: If the path escapes the store or lacks a bucket and key.
    """
    path = pathlib.PurePosixPath(raw.split("?", 1)[0].lstrip("/"))
    if len(path.parts) < 2 or any(
        part in {"", ".", ".."} or "\\" in part for part in path.parts
    ):
        raise ValueError("path must be /<bucket>/<key> without '..'")
    return path


class MockServer:
    """A background HTTP server with a lifecycle."""

    def __init__(self, host: str = "127.0.0.1", port: int = 0) -> None:
        """Remembers where to bind; nothing listens until `start`."""
        self.host = host
        self.port = port
        self.httpd: server.ThreadingHTTPServer | None = None
        self.thread: threading.Thread | None = None

    def handler(self) -> type[server.BaseHTTPRequestHandler]:
        """The request handler class; subclasses provide it."""
        raise NotImplementedError

    @property
    def address(self) -> str:
        """Base URL once started."""
        if self.httpd is None:
            raise errors.ScmleError("server is not running")
        return f"http://{self.host}:{self.httpd.server_address[1]}"

    def start(self) -> None:
        """Binds and serves in a daemon thread."""
        if self.httpd is not None:
            raise errors.ScmleError("server is already running")
        self.httpd = server.ThreadingHTTPServer(
            (self.host, self.port), self.handler()
        )
        self.thread = threading.Thread(
            target=self.httpd.serve_forever, daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        """Stops the server and joins its thread."""
        if self.httpd is not None:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.thread is not None:
            self.thread.join(timeout=5)
        self.httpd = self.thread = None


class ObjectStore(MockServer):
    """PUT, GET, HEAD and list over `/<bucket>/<key>`, with SHA-256 etags."""

    def __init__(
        self, root: str | pathlib.Path, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        """Stores objects under `root`."""
        super().__init__(host, port)
        self.root = pathlib.Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def handler(self) -> type[server.BaseHTTPRequestHandler]:
        """Builds the request handler bound to this store."""
        store = self

        class Handler(server.BaseHTTPRequestHandler):
            def reply(
                self, status: int, body: bytes, headers: dict[str, str]
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Length", str(len(body)))
                for name, value in headers.items():
                    self.send_header(name, value)
                self.end_headers()
                if self.command != "HEAD":
                    self.wfile.write(body)

            def problem(self, status: int, message: str) -> None:
                body = json.dumps({"error": message}).encode()
                self.reply(status, body, {"Content-Type": "application/json"})

            def target(self) -> pathlib.Path | None:
                try:
                    relative = safe_key(self.path)
                except ValueError as error:
                    self.problem(400, str(error))
                    return None
                return store.root.joinpath(*relative.parts)

            def do_PUT(self) -> None:
                path = self.target()
                if path is None:
                    return
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > MAX_OBJECT_BYTES:
                    self.problem(413, "object size not allowed")
                    return
                body = self.rfile.read(length)
                digest = provenance.sha256_digest(body)
                claimed = self.headers.get(DIGEST_HEADER)
                if claimed is not None and claimed != digest:
                    self.problem(400, "content digest mismatch")
                    return
                path.parent.mkdir(parents=True, exist_ok=True)
                durability.write_atomically(path, body)
                self.reply(200, b"", {"ETag": digest})

            def do_GET(self) -> None:
                if self.path == "/healthz":
                    self.reply(200, b"ok", {"Content-Type": "text/plain"})
                    return
                if "?list" in self.path:
                    prefix = self.path.split("?", 1)[0].strip("/")
                    base = store.root / prefix
                    entries = [
                        {
                            "key": p.relative_to(store.root).as_posix(),
                            "etag": provenance.file_digest(p),
                            "size": p.stat().st_size,
                        }
                        for p in sorted(base.rglob("*"))
                        if p.is_file()
                    ]
                    body = json.dumps({"objects": entries}).encode()
                    self.reply(200, body, {"Content-Type": "application/json"})
                    return
                path = self.target()
                if path is None:
                    return
                if not path.is_file():
                    self.problem(404, "no such object")
                    return
                body = path.read_bytes()
                self.reply(
                    200,
                    body,
                    {
                        "ETag": provenance.sha256_digest(body),
                        "Content-Type": "application/octet-stream",
                    },
                )

            def do_HEAD(self) -> None:
                self.do_GET()

            def log_message(self, format: str, *args: Any) -> None:
                """Silences per-request logging."""

        return Handler


class MetricsSink(MockServer):
    """Accepts metric samples on POST /metrics and replays them on GET."""

    def __init__(
        self, file: str | pathlib.Path, host: str = "127.0.0.1", port: int = 0
    ) -> None:
        """Appends validated samples to `file` (JSON lines)."""
        super().__init__(host, port)
        self.file = pathlib.Path(file)
        self.file.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def samples(self) -> list[dict[str, Any]]:
        """Every stored sample."""
        with self.lock:
            if not self.file.exists():
                return []
            return [
                json.loads(line)
                for line in self.file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

    def handler(self) -> type[server.BaseHTTPRequestHandler]:
        """Builds the request handler bound to this sink."""
        sink = self

        class Handler(server.BaseHTTPRequestHandler):
            def reply(self, status: int, payload: dict[str, Any]) -> None:
                body = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self) -> None:
                if self.path == "/healthz":
                    self.reply(200, {"status": "ok"})
                elif self.path == "/metrics":
                    stored = sink.samples()
                    self.reply(200, {"count": len(stored), "samples": stored})
                else:
                    self.reply(404, {"error": "not found"})

            def do_POST(self) -> None:
                if self.path != "/metrics":
                    self.reply(404, {"error": "not found"})
                    return
                length = int(self.headers.get("Content-Length", "-1"))
                if length < 0 or length > MAX_METRICS_BYTES:
                    self.reply(413, {"error": "payload size not allowed"})
                    return
                try:
                    payload = json.loads(self.rfile.read(length))
                    batch = payload if isinstance(payload, list) else [payload]
                    samples = [
                        observability.MetricSample.from_dict(item).to_dict()
                        for item in batch
                    ]
                except (ValueError, KeyError, TypeError) as error:
                    self.reply(400, {"error": f"invalid sample: {error}"})
                    return
                with sink.lock, self.server_file() as handle:
                    for sample in samples:
                        handle.write(json.dumps(sample) + "\n")
                self.reply(200, {"accepted": len(samples)})

            def server_file(self) -> Any:
                return sink.file.open("a", encoding="utf-8", newline="\n")

            def log_message(self, format: str, *args: Any) -> None:
                """Silences per-request logging."""

        return Handler


def put_object(base: str, path: str, data: bytes, timeout: float = 30.0) -> str:
    """Uploads `data` to `base/path`; returns the etag the store computed.

    Raises:
        ArtifactError: If the store's digest differs from the local one.
    """
    local = provenance.sha256_digest(data)
    request = urllib.request.Request(
        f"{base.rstrip('/')}/{path.lstrip('/')}",
        data,
        {DIGEST_HEADER: local},
        method="PUT",
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:  # nosec B310
        etag = str(reply.headers.get("ETag"))
    if etag != local:
        raise errors.ArtifactError("object store stored different bytes")
    return etag


def get_object(
    base: str, path: str, expected: str | None = None, timeout: float = 30.0
) -> bytes:
    """Downloads `base/path`, refusing bytes whose digest is not `expected`.

    Raises:
        ArtifactError: If the digest differs from `expected` or the store's
            etag.
    """
    url = f"{base.rstrip('/')}/{path.lstrip('/')}"
    with urllib.request.urlopen(url, timeout=timeout) as reply:  # nosec B310
        body: bytes = reply.read()
        etag = reply.headers.get("ETag")
    actual = provenance.sha256_digest(body)
    if actual != etag or (expected is not None and actual != expected):
        raise errors.ArtifactError(f"{path} does not match its digest")
    return body


def post_metrics(
    base: str, samples: list[observability.MetricSample], timeout: float = 30.0
) -> int:
    """Sends samples to a sink; returns how many it accepted."""
    request = urllib.request.Request(
        f"{base.rstrip('/')}/metrics",
        json.dumps([sample.to_dict() for sample in samples]).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:  # nosec B310
        return int(json.loads(reply.read())["accepted"])


def main(arguments: list[str] | None = None) -> int:
    """Runs one mock service until interrupted."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    actions = parser.add_subparsers(dest="service", required=True)
    store = actions.add_parser("object-store")
    store.add_argument("--root", required=True)
    sink = actions.add_parser("metrics-sink")
    sink.add_argument("--file", required=True)
    for action in (store, sink):
        action.add_argument("--host", default="127.0.0.1")
        action.add_argument("--port", type=int, required=True)
    options = parser.parse_args(arguments)
    service: MockServer = (
        ObjectStore(options.root, options.host, options.port)
        if options.service == "object-store"
        else MetricsSink(options.file, options.host, options.port)
    )
    service.start()
    print(f"{options.service} listening on {service.address}", flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        service.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
