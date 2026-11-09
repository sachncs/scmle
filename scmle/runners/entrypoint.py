"""What a container job runs: ``python -m scmle.runners.entrypoint``.

Reads ``SCMLE_RUN_ID``, ``SCMLE_ATTEMPT``, ``SCMLE_STORE_URI`` and
``SCMLE_SPEC_DIGEST`` from the environment (set by the runner
adapter), opens the store, loads ``runs/<id>/spec.json``, refuses a spec
whose digest differs from the one the runner started, and runs the workload
named by ``SCMLE_WORKLOAD`` (default `scmle.runners.project:run_project`)
under the training role. Exit status 0 means the workload returned; any
other exception exits 1 with its type and message on stderr.
"""

from collections.abc import Callable
from collections.abc import Mapping
import importlib
import json
import sys
from typing import Any
from urllib.parse import urlsplit
from urllib.request import url2pathname

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.runners import base
from scmle.runners import checkpoint
from scmle.runners import local
from scmle.runners import roles

DEFAULT_WORKLOAD = "scmle.runners.project:run_project"


def open_store(uri: str) -> objectstore.ObjectStore:
    """Opens a `file://` store.

    Args:
        uri: Store URI.

    Returns:
        A local object store at the path of the URI.

    Raises:
        ConfigError: If the scheme is not supported.
    """
    parts = urlsplit(uri)
    if parts.scheme == "file":
        # url2pathname turns /C:/dir into C:\\dir on Windows
        return objectstore.LocalObjectStore(url2pathname(parts.path))
    raise errors.ConfigError(
        f"unsupported store URI scheme {parts.scheme!r}; only file:// is "
        "built in, so open any other store in your own workload"
    )


def load_workload(name: str) -> Callable[[local.JobContext], None]:
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
    module_name, separator, function = name.partition(":")
    if not separator:
        raise errors.ConfigError("SCMLE_WORKLOAD must be module:function")
    found = getattr(importlib.import_module(module_name), function)
    workload: Callable[[local.JobContext], None] = found
    return workload


def spec_from_store(store: objectstore.ObjectStore, run_id: str) -> base.Job:
    """Rebuilds the stored spec.

    Args:
        store: Durable storage.
        run_id: The run.

    Returns:
        The job spec stored at `runs/<run_id>/spec.json`.
    """
    data, ignored = store.get(f"runs/{run_id}/spec.json")
    record = json.loads(data)
    budget = record["budget"]
    return base.Job(
        run_id=record["run_id"],
        name=record["name"],
        project=record["project"],
        identity=record["identity"],
        resources=base.Resources(**record["resources"]),
        timeout_seconds=record["timeout_seconds"],
        retries=retry.Retries(**record["retries"]),
        budget=None
        if budget is None
        else base.CostBudget(
            budget["max_cost"], base.CostRate(**budget["rate"])
        ),
        environment=record["environment"],
    )


def main(environ: Mapping[str, str], store: Any = None) -> int:
    """Runs the job described by `environ`.

    Any exception from loading the spec or running the workload is printed to
    stderr as its type and message instead of being raised.

    Args:
        environ: The environment (`SCMLE_RUN_ID`, `SCMLE_ATTEMPT`,
            `SCMLE_STORE_URI`, `SCMLE_SPEC_DIGEST`, `SCMLE_WORKLOAD`).
        store: The training store; opened from `SCMLE_STORE_URI` if None.

    Returns:
        0 if the workload returned, otherwise 1.

    Raises:
        ConfigError: Raised when the stored spec's digest differs from
            `SCMLE_SPEC_DIGEST`, but caught inside the function and reported on
            stderr with exit status 1; it never reaches the caller.
    """
    try:
        run_id = environ["SCMLE_RUN_ID"]
        attempt = int(environ.get("SCMLE_ATTEMPT", "1"))
        training_store = store or open_store(environ["SCMLE_STORE_URI"])
        spec = spec_from_store(training_store, run_id)
        expected = environ.get("SCMLE_SPEC_DIGEST")
        if expected and spec.digest() != expected:
            raise errors.ConfigError("the stored spec is not the one started")
        context = local.JobContext(
            roles.RoleStore(training_store, roles.TRAINING_ROLE),
            run_id,
            attempt,
            spec,
            checkpoint.CheckpointStore(training_store, run_id),
        )
        load_workload(environ.get("SCMLE_WORKLOAD", DEFAULT_WORKLOAD))(context)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    import os

    sys.exit(main(os.environ))
