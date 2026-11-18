"""Run ids, record keys, and a run's records in the object store.

A run's records live under `workflows/<workflow>/runs/<run id>/`: `run.json`
(the run record, its checkpoint), `steps/<step>.json` and
`approvals/<step>.json`. `Bucket` is that prefix as the engine's archive
(`scmle.pipeline.engine.Archive`, over a step's outputs): `record` writes a
step's record and `restore` hands back a succeeded step's outputs. The keys
and the record fields are those the executor wrote before the archive, so
stored runs load and resume unchanged.
"""

from collections.abc import Callable
from collections.abc import Mapping
import enum
import hashlib
import json
import threading
from typing import Any

from scmle.contracts import errors
from scmle.contracts import records
from scmle.storage import objectstore
from scmle.workflow import definition
from scmle.workflow.executor import context


def parse(key: str, data: bytes) -> records.Layout:
    """Decodes a stored run, step, approval or lease record.

    Args:
        key: The object key, named in the error.
        data: The stored bytes.

    Returns:
        The record.

    Raises:
        ArtifactError: If the bytes are not a JSON object; a raw
            `JSONDecodeError` used to escape with no key (ERR-17).
    """
    try:
        record = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise errors.ArtifactError(
            f"workflow record {key} is corrupt: {error}", {"key": key}
        ) from error
    if not isinstance(record, dict):
        raise errors.ArtifactError(
            f"workflow record {key} is not a JSON object", {"key": key}
        )
    return record


def distill(value: Any) -> str:
    """Digests a value as canonical JSON.

    Args:
        value: A JSON-serializable value; other types are converted with `str`.

    Returns:
        The hex SHA-256 of the key-sorted JSON.
    """
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def run_id_for(workflow: definition.Workflow, trigger_key: str) -> str:
    """Derives the durable run id of a trigger.

    Args:
        workflow: The workflow.
        trigger_key: Identifies the trigger (a schedule slot, a request).

    Returns:
        The workflow name and a 16-character digest of its name, version and the
        trigger key, truncated to 64 characters.
    """
    short = distill([workflow.name, workflow.version, trigger_key])[:16]
    return f"{workflow.name}-{short}"[:64]


class Bucket:
    """A run's records in the object store: the workflow's engine archive.

    Attributes:
        base: The key prefix of the run, ending in `/`.
        read: Reads a record by its full key (None when absent).
        write: Writes a record by its full key; the executor's write, fenced
            by the run's lease (`Leases.write`).
    """

    def __init__(
        self,
        base: str,
        read: Callable[[str], records.Layout | None],
        write: Callable[[str, Mapping[str, Any]], None],
    ) -> None:
        """Opens a run's records.

        Args:
            base: The key prefix of the run.
            read: Reads a record by its full key.
            write: Writes a record by its full key.
        """
        self.base = base
        self.read = read
        self.write = write

    def fetch(self, path: str) -> records.Layout | None:
        """Reads one record of the run.

        Args:
            path: The key under the run, such as `run.json`.

        Returns:
            The record, or None if there is none.
        """
        return self.read(self.base + path)

    def keep(self, path: str, body: Mapping[str, Any]) -> None:
        """Writes one record of the run, such as its checkpoint `run.json`.

        Args:
            path: The key under the run.
            body: The record.
        """
        self.write(self.base + path, body)

    def record(
        self,
        index: int,  # noqa: ARG002 -- engine.Archive
        name: str,
        progress: enum.StrEnum,
        seconds: float,  # noqa: ARG002 -- engine.Archive
        made: Mapping[str, Any] | None = None,
        needs: dict[str, str | None] | None = None,  # noqa: ARG002 -- engine.Archive
        **fields: Any,
    ) -> records.Layout:
        """Writes a step's record, `steps/<step>.json`.

        Args:
            index: Unused: a step is keyed by its name.
            name: The step.
            progress: The step's state (`status.Phase`).
            seconds: Unused: a step record keeps its attempts, not a time.
            made: The step's outputs, stored as `outputs`, or None.
            needs: Unused: the idempotency key hashes the upstream outputs.
            **fields: Further fields of the record.

        Returns:
            The record written, with `product`, the digest of the outputs
            (None without them), which is not stored.
        """
        body = {"step": name, "state": progress.value, **fields}
        if made is not None:
            body["outputs"] = made
        self.keep(f"steps/{name}.json", body)
        return {**body, "product": None if made is None else distill(made)}

    def restore(
        self,
        index: int,  # noqa: ARG002 -- engine.Archive
        name: str,
        makes: type[Mapping[str, Any]] | None,  # noqa: ARG002 -- engine.Archive
        needs: dict[str, str | None],  # noqa: ARG002 -- engine.Archive
    ) -> tuple[Mapping[str, Any], str] | None:
        """The outputs of a step that already succeeded in this run.

        Args:
            index: Unused: a step is keyed by its name.
            name: The step.
            makes: Unused: outputs are JSON.
            needs: Unused: whether a succeeded step runs again is decided by
                its idempotency key when it is started.

        Returns:
            The outputs and their digest, or None unless the step's record
            is a success.
        """
        saved = self.fetch(f"steps/{name}.json")
        if saved is None or saved["state"] != "succeeded":
            return None
        return saved["outputs"], distill(saved["outputs"])

    def finish(self) -> None:
        """Writes nothing: the run record written last is the run's end."""


class Records:
    """The executor's state, and the keys and records of its runs.

    Attributes:
        workflow: The definition.
        store: Durable state.
        handlers: Step name to handler.
        approver_keys: Approver identity to the key their decisions are signed
            with.
        clock: Returns epoch seconds.
        timer: Returns monotonic seconds, for how long a step has run.
        sleep: Waits between retries.
        lease_seconds: Validity of a run lease.
        owner: Name written into the lease.
        guard: Serializes writes of step records.
        fences: The fencing token of each run this executor leases.
    """

    workflow: definition.Workflow
    store: objectstore.Depot
    handlers: dict[str, context.Handler]
    approver_keys: dict[str, str]
    clock: Callable[[], float]
    timer: Callable[[], float]
    sleep: Callable[[float], None]
    lease_seconds: float
    owner: str
    guard: threading.Lock
    fences: dict[str, int]

    def base(self, run_id: str) -> str:
        """Builds the key prefix of a run.

        Args:
            run_id: The run.

        Returns:
            The prefix under which the run's records are stored.
        """
        return f"workflows/{self.workflow.name}/runs/{run_id}/"

    def read(self, key: str) -> records.Layout | None:
        """Reads a stored JSON record.

        Args:
            key: Object key.

        Returns:
            The record, or None if there is no such object.

        Raises:
            ArtifactError: If the object is not a JSON object (see `parse`).
        """
        try:
            data = self.store.get(key)[0]
        except objectstore.AbsentError:
            # handled: missing object returned as None, documented
            return None
        return parse(key, data)

    def write(self, key: str, record: Mapping[str, Any]) -> None:
        """Stores a JSON record, unfenced (`Leases.write` fences it).

        Args:
            key: Object key.
            record: The record to store.
        """
        body = json.dumps(record, indent=2, sort_keys=True).encode()
        self.store.put(key, body)

    def bucket(self, run_id: str) -> Bucket:
        """Opens a run's records.

        Args:
            run_id: The run.

        Returns:
            The run's archive, reading and writing through this executor.
        """
        return Bucket(self.base(run_id), self.read, self.write)

    def run_record(self, run_id: str) -> records.Layout | None:
        """Reads the stored run record.

        Args:
            run_id: The run.

        Returns:
            The run record, or None if the run is unknown.
        """
        return self.bucket(run_id).fetch("run.json")

    def step_record(self, run_id: str, name: str) -> records.Layout | None:
        """Reads the stored record of one step.

        Args:
            run_id: The run.
            name: The step name.

        Returns:
            The step record, or None if the step has not been recorded.
        """
        return self.bucket(run_id).fetch(f"steps/{name}.json")
