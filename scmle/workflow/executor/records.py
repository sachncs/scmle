"""Run ids, record keys and reading the stored records of a run."""

from collections.abc import Callable
import hashlib
import json
import threading
from typing import Any

from scmle.contracts import errors
from scmle.data.store import objectstore
from scmle.workflow import definition
from scmle.workflow.executor import context


def parse(key: str, data: bytes) -> dict[str, Any]:
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

    def read(self, key: str) -> dict[str, Any] | None:
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

    def run_record(self, run_id: str) -> dict[str, Any] | None:
        """Reads the stored run record.

        Args:
            run_id: The run.

        Returns:
            The run record, or None if the run is unknown.
        """
        return self.read(self.base(run_id) + "run.json")

    def step_record(self, run_id: str, name: str) -> dict[str, Any] | None:
        """Reads the stored record of one step.

        Args:
            run_id: The run.
            name: The step name.

        Returns:
            The step record, or None if the step has not been recorded.
        """
        return self.read(f"{self.base(run_id)}steps/{name}.json")
