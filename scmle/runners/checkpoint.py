"""Resumable job state in durable storage.

A workload saves `(step, state)` as it goes. Each checkpoint is a JSON object
that carries the digest of its own content, written atomically under
``runs/<run id>/checkpoints/<step>.json``. On restart the workload asks for
`latest()`: the highest step whose digest verifies. A damaged newest
checkpoint (a write cut short, a changed object) is skipped and reported, and
the one before it is used, so an interrupted job loses at most the work since
its last good checkpoint.
"""

import dataclasses
import hashlib
import json
from typing import Any

from scmle.contracts import errors
from scmle.data.store import objectstore

STEP_WIDTH = 10


class CheckpointError(errors.ScmleError):
    """A checkpoint could not be saved."""

    category = "checkpoint"


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoint:
    """One verified checkpoint.

    Attributes:
        step: Step number.
        state: The JSON state saved at that step.
        sha256: Digest of the checkpoint content.
    """

    step: int
    state: Any
    sha256: str


def content_digest(step: int, state: Any) -> str:
    """Digest of the canonical JSON of a step and its state."""
    body = json.dumps({"step": step, "state": state}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


class CheckpointStore:
    """The checkpoints of one run."""

    def __init__(self, store: objectstore.ObjectStore, run_id: str) -> None:
        """Initializes the store view.

        Args:
            store: Durable storage.
            run_id: The run whose checkpoints these are.
        """
        self.store = store
        self.prefix = f"runs/{run_id}/checkpoints/"
        self.skipped: list[str] = []

    def key(self, step: int) -> str:
        """The object key for a step."""
        return f"{self.prefix}{step:0{STEP_WIDTH}d}.json"

    def save(self, step: int, state: Any) -> Checkpoint:
        """Saves a checkpoint.

        Args:
            step: Non-negative step number.
            state: JSON-serializable state.

        Returns:
            The saved checkpoint.

        Raises:
            CheckpointError: If the step is invalid or the state cannot be
                serialized.
        """
        if step < 0:
            raise CheckpointError("step must not be negative")
        try:
            digest = content_digest(step, state)
            body = json.dumps(
                {"step": step, "state": state, "sha256": digest},
                sort_keys=True,
            ).encode()
        except (TypeError, ValueError) as error:
            raise CheckpointError("state is not JSON-serializable") from error
        self.store.put(self.key(step), body)
        return Checkpoint(step, state, digest)

    def steps(self) -> list[int]:
        """Steps that have an object, ascending."""
        found = []
        for meta in self.store.list_objects(self.prefix):
            stem = meta.key.removeprefix(self.prefix).removesuffix(".json")
            if stem.isdigit():
                found.append(int(stem))
        return sorted(found)

    def load(self, step: int) -> Checkpoint | None:
        """A verified checkpoint, or None if absent or damaged."""
        try:
            data, ignored = self.store.get(self.key(step))
            document = json.loads(data)
            if document["step"] != step or document["sha256"] != content_digest(
                step, document["state"]
            ):
                raise ValueError("digest mismatch")
            return Checkpoint(step, document["state"], document["sha256"])
        except (objectstore.StoreError, ValueError, KeyError, TypeError):
            return None

    def latest(self) -> Checkpoint | None:
        """The newest checkpoint that verifies (damaged ones are skipped)."""
        self.skipped = []
        for step in reversed(self.steps()):
            found = self.load(step)
            if found is not None:
                return found
            self.skipped.append(self.key(step))
        return None
