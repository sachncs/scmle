"""A dedicated process that runs every PyTorch computation.

PyTorch ships its own OpenMP runtime. Loaded into a process that has already
run scikit-learn, XGBoost or LightGBM threads, it can crash or deadlock. So
the framework process never imports torch: training and prediction happen
here, and only plain numpy arrays and dictionaries cross the pipe.
"""

import atexit
import collections
import importlib
from multiprocessing import connection
import os
import subprocess  # nosec B404
import threading
from typing import Any

from scmle.contracts import errors
from scmle.core import ipc

CACHE_SIZE = 8
NEED_STATE = "need_state"


class WorkerError(errors.ScmleError):
    """The worker could not complete a request."""

    category = "model"


def resolve(path: str) -> Any:
    """Imports an engine class from its dotted path.

    Args:
        path: Dotted `module.ClassName` path.

    Returns:
        The class object found at `path`.
    """
    module, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module), name)


def rebuild(path: str, state: dict[str, Any]) -> Any:
    """Recreates a fitted engine from its numpy state.

    Args:
        path: Dotted `module.ClassName` path of the engine class.
        state: State mapping produced by the engine's `__getstate__`.

    Returns:
        The engine instance, restored through `__setstate__` without calling
        its constructor.
    """
    engine = resolve(path).__new__(resolve(path))
    engine.__setstate__(state)
    return engine


def handle(
    cache: collections.OrderedDict[str, Any],
    operation: str,
    payload: tuple[Any, ...],
) -> Any:
    """Executes one request inside the worker.

    Args:
        cache: Fitted engines already held, by state id (least recent first).
        operation: `fit`, `call`, `probe` or `embed`.
        payload: Operation arguments.

    Returns:
        The reply value: whether CUDA is available for `probe`, the
        embedding matrix and encoded count for `embed`, the fitted engine's
        state for `fit`, and otherwise the result of the called method.
        Returns the `need_state` marker when the engine is not cached and no
        state was sent.
    """
    if operation == "probe":
        return bool(importlib.import_module("torch").cuda.is_available())
    if operation == "embed":
        name, texts = payload
        return importlib.import_module(
            "scmle.models.deeplearning.embeddings"
        ).embed(name, texts)
    if operation == "fit":
        path, params, features, target, weights = payload
        engine = resolve(path)(**params)
        engine.fit(features, target, weights)
        return engine.__getstate__()
    path, state_id, state, method, features = payload
    if state_id not in cache:
        if state is None:
            return NEED_STATE
        cache[state_id] = rebuild(path, state)
        while len(cache) > CACHE_SIZE:
            cache.popitem(last=False)
    cache.move_to_end(state_id)
    return getattr(cache[state_id], method)(features)


def serve(channel: connection.Connection) -> None:
    """Worker main loop: answer requests until the parent goes away.

    Args:
        channel: Connection to the parent process.
    """
    cache: collections.OrderedDict[str, Any] = collections.OrderedDict()
    while True:
        try:
            operation, payload = channel.recv()
        except EOFError:
            return
        try:
            channel.send(("ok", handle(cache, operation, payload)))
        except Exception as error:  # Isolation: report, keep serving.
            channel.send(("error", f"{type(error).__name__}: {error}"))


def main() -> None:
    """Entry point of the worker process.

    The parent passes the channel address as the first argument and the key
    on standard input, so only the parent can talk to this process.
    """
    serve(ipc.connect_to_parent())


class Client:
    """Owns the worker process and serializes requests to it."""

    def __init__(self) -> None:
        """Creates a client; the process starts on first use."""
        self.lock = threading.Lock()
        self.process: subprocess.Popen[bytes] | None = None
        self.channel: connection.Connection | None = None

    def ensure_running(self) -> connection.Connection:
        """Starts (or restarts) the worker process.

        Returns:
            The connection to the live worker process.
        """
        if (
            self.process is not None
            and self.process.poll() is None
            and self.channel is not None
        ):
            return self.channel
        self.shutdown()
        self.process, parent = ipc.start_helper("scmle.models.deeplearning")
        self.channel = parent
        return parent

    def request(self, operation: str, payload: tuple[Any, ...]) -> Any:
        """Sends one request and returns the reply.

        Args:
            operation: `fit` or `call`.
            payload: Operation arguments.

        Returns:
            The value the worker replied with.

        Raises:
            WorkerError: If the worker reports an error or dies.
        """
        with self.lock:
            channel = self.ensure_running()
            try:
                channel.send((operation, payload))
                status, value = channel.recv()
            except (EOFError, OSError) as error:
                self.shutdown()
                raise WorkerError("the PyTorch worker process died") from error
        if status == "error":
            raise WorkerError(value)
        return value

    def shutdown(self) -> None:
        """Stops the worker process if it is running."""
        if self.channel is not None:
            self.channel.close()
            self.channel = None
        if self.process is not None:
            if self.process.poll() is None:
                self.process.kill()
            self.process.wait(timeout=5)
            self.process = None


CLIENTS: dict[int, Client] = {}
CLIENTS_LOCK = threading.Lock()


def client() -> Client:
    """Returns this process's worker client (one worker per process).

    Safe to call from several threads: the first creates the client, and
    every other caller gets that same one.

    Returns:
        The worker client registered for the current process id.
    """
    pid = os.getpid()
    found = CLIENTS.get(pid)
    if found is not None:
        return found
    with CLIENTS_LOCK:
        found = CLIENTS.get(pid)
        if found is None:
            found = CLIENTS[pid] = Client()
            atexit.register(found.shutdown)
        return found
