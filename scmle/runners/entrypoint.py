"""What a job process runs: ``python -m scmle.runners.entrypoint``.

Reads ``SCMLE_RUN_ID``, ``SCMLE_ATTEMPT``, ``SCMLE_STORE_URI`` and
``SCMLE_SPEC_DIGEST`` from the environment (set by the runner that started
the process or container), opens the store (confined to the namespace of
``SCMLE_TENANT`` when it is set), loads ``runs/<id>/spec.json``,
refuses a spec whose digest differs from the one the runner started, and runs
the workload named by ``SCMLE_WORKLOAD`` (default
`scmle.runners.project:project`) under the training role.

The exit status is a `base.Exit`: 0 when the workload returned, 75 for a
transient error (a retry may pass), 143 when SIGTERM interrupted it (a
preemption or a stop: the workload is interrupted at once and again at its
next checkpoint, should it swallow the first), and 1 for anything else, with
the traceback on stderr.
"""

from collections.abc import Callable
from collections.abc import Iterator
from collections.abc import Mapping
import contextlib
import importlib
import json
import os
import signal
import sys
import threading
from types import FrameType
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

from scmle.contracts import errors
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import local
from scmle.runners import roles
from scmle.runtime import retry
from scmle.storage import objectstore
from scmle.storage import tenancy

DEFAULT_WORKLOAD = "scmle.runners.project:project"


def mount(
    uri: str, environ: Mapping[str, str] | None = None
) -> objectstore.Depot:
    """Opens a `file://` store, confined to the tenant `SCMLE_TENANT` names.

    The confinement follows `environ`, the job's environment: a runner
    started with `SCMLE_TENANT` registers its specs in the tenant's
    namespace and its jobs inherit the variable, so both open the store
    the same way (`tenancy.confine`).

    Args:
        uri: Store URI.
        environ: The environment naming the tenant; None reads the
            process's own.

    Returns:
        A local object store at the path of the URI; with `SCMLE_TENANT`
        set, that store confined to the tenant's namespace (`tenancy`).

    Raises:
        ConfigError: If the scheme is not supported.
        TenantError: If `SCMLE_TENANT` is not a valid tenant name.
    """
    parts = urlsplit(uri)
    if parts.scheme == "file":
        # url2pathname turns /C:/dir into C:\\dir on Windows
        store = objectstore.Disk(url2pathname(parts.path))
        return tenancy.confine(
            store, os.environ if environ is None else environ
        )
    raise errors.ConfigError(
        f"unsupported store URI scheme {parts.scheme!r}; only file:// is "
        "built in, so open any other store in your own workload"
    )


def resolve(name: str) -> Callable[[local.Context], None]:
    """Imports ``module:function``.

    An unimportable module raises ImportError and a missing function raises
    AttributeError.

    Args:
        name: The workload as `module:function`.

    Returns:
        The function.

    Raises:
        ConfigError: If the name has no `:`.
    """
    module, separator, function = name.partition(":")
    if not separator:
        raise errors.ConfigError("SCMLE_WORKLOAD must be module:function")
    found = getattr(importlib.import_module(module), function)
    workload: Callable[[local.Context], None] = found
    return workload


def respec(store: objectstore.Depot, run_id: str) -> base.Job:
    """Rebuilds the stored spec.

    Args:
        store: Durable storage.
        run_id: The run.

    Returns:
        The job spec stored at `runs/<run_id>/spec.json`, written in the
        current layout or in version 1 (see `base.Job.read`).
    """
    data = store.get(f"runs/{run_id}/spec.json")[0]
    return base.Job.read(json.loads(data))


def verify(spec: base.Job, expected: str | None) -> None:
    """Checks that the stored spec is the one that was started.

    Args:
        spec: The spec read from the store.
        expected: The digest the launcher recorded; empty or None skips the
            check.

    Raises:
        ConfigError: If `expected` is set and the spec's digest differs.
    """
    if expected and spec.digest() != expected:
        raise errors.ConfigError("the stored spec is not the one started")


class Alarm:
    """Turns SIGTERM into `local.PreemptedError` for the running workload.

    The first signal raises in the workload at once; `check`, called after
    every checkpoint, raises again in case the workload caught the first.

    Attributes:
        raised: Whether SIGTERM arrived.
    """

    def __init__(self) -> None:
        """Initializes an alarm that has not gone off."""
        self.raised = False

    def ring(self, number: int, frame: FrameType | None) -> None:  # noqa: ARG002 -- signal handler signature
        """Handles SIGTERM.

        Args:
            number: The signal.
            frame: The interrupted frame. Unused.

        Raises:
            PreemptedError: Always, so the workload stops now.
        """
        self.raised = True
        raise local.PreemptedError(f"signal {number}")

    def check(self, step: int) -> None:  # noqa: ARG002 -- hook signature
        """Raises if SIGTERM arrived (the checkpoint hook).

        Args:
            step: The step just saved. Unused.

        Raises:
            PreemptedError: If SIGTERM arrived.
        """
        if self.raised:
            raise local.PreemptedError("signal after checkpoint")

    @contextlib.contextmanager
    def armed(self) -> Iterator[None]:
        """Handles SIGTERM inside the block, restoring the old handler after.

        Outside the main thread (a test calling `main` from a worker) no
        handler can be installed and the block runs without one.

        Yields:
            Nothing.
        """
        if threading.current_thread() is not threading.main_thread():
            yield
            return
        previous = signal.signal(signal.SIGTERM, self.ring)
        try:
            yield
        finally:
            signal.signal(signal.SIGTERM, previous)


def main(environ: Mapping[str, str], store: Any = None) -> int:
    """Runs the job described by `environ`.

    Any exception from loading the spec or running the workload is printed to
    stderr with its traceback instead of being raised.

    Args:
        environ: The environment (`SCMLE_RUN_ID`, `SCMLE_ATTEMPT`,
            `SCMLE_STORE_URI`, `SCMLE_SPEC_DIGEST`, `SCMLE_WORKLOAD`,
            `SCMLE_TENANT`).
        store: The training store; opened from `SCMLE_STORE_URI` (confined
            to the tenant of `environ`) if None.

    Returns:
        A `base.Exit`: `SUCCEEDED` if the workload returned, `INTERRUPTED`
        after SIGTERM or a `PreemptedError`, `TRANSIENT` for a transient
        error, otherwise `FAILED`. That includes a stored spec whose digest
        differs from `SCMLE_SPEC_DIGEST`.
    """
    alarm = Alarm()
    try:
        with alarm.armed():
            run_id = environ["SCMLE_RUN_ID"]
            attempt = int(environ.get("SCMLE_ATTEMPT", "1"))
            training = store or mount(environ["SCMLE_STORE_URI"], environ)
            spec = respec(training, run_id)
            verify(spec, environ.get("SCMLE_SPEC_DIGEST"))
            context = local.Context(
                roles.Store(training, roles.TRAINING_ROLE),
                run_id,
                attempt,
                spec,
                checkpoint.Store(training, run_id),
                alarm.check,
            )
            resolve(environ.get("SCMLE_WORKLOAD", DEFAULT_WORKLOAD))(context)
    except local.PreemptedError as error:
        # handled: reported as the interrupted exit status
        sys.stderr.write(f"interrupted: {error}\n")
        return base.Exit.INTERRUPTED
    except retry.TransientError as error:
        # handled: the traceback goes to stderr, a retryable exit status
        sys.stderr.write(errors.Failure.of(error).trace)
        return base.Exit.TRANSIENT
    except Exception as error:  # noqa: BLE001 -- process boundary
        # handled: the full traceback is written to stderr, exit code 1
        sys.stderr.write(errors.Failure.of(error).trace)
        return base.Exit.FAILED
    return base.Exit.SUCCEEDED


if __name__ == "__main__":
    sys.exit(main(os.environ))
