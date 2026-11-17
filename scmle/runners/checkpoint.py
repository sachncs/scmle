"""Resumable job state in durable storage.

A workload saves `(step, state)` as it goes. Each checkpoint is a JSON object
that carries the digest of its own content, written atomically under
``runs/<run id>/checkpoints/<step>.json``. On restart the workload asks for
`latest()`: the highest step whose digest verifies. A damaged newest
checkpoint (a write cut short, a changed object) is skipped and reported, and
the one before it is used, so an interrupted job loses at most the work since
its last good checkpoint.

`load` returns a typed `Reading`: absent, corrupt or loaded, never one value
for two of them (ERR-10). A store outage raises; it is not "no checkpoint".
Every caller handles a corrupt reading explicitly: `latest` passes over it and
keeps it in `skipped`, `record` writes the skipped ones to durable storage
for the attempt that resumed past them, and the run manifest lists each with
its damage.
"""

import dataclasses
import enum
import hashlib
import json
import logging
from typing import Any

from scmle.contracts import errors
from scmle.storage import objectstore

LOGGER = logging.getLogger(__name__)


STEP_WIDTH = 10


class CheckpointError(errors.ScmleError):
    """A checkpoint could not be saved."""

    category = "checkpoint"


LEDGER = "damaged.json"


class Condition(enum.Enum):
    """What a checkpoint read found."""

    ABSENT = "absent"
    CORRUPT = "corrupt"
    LOADED = "loaded"


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


@dataclasses.dataclass(frozen=True, slots=True)
class Reading:
    """The result of reading one checkpoint.

    Attributes:
        key: Object key that was read.
        condition: Absent, corrupt or loaded.
        checkpoint: The verified checkpoint when loaded, otherwise None.
        reason: Why a corrupt checkpoint was rejected; empty otherwise.
    """

    key: str
    condition: Condition
    checkpoint: Checkpoint | None = None
    reason: str = ""


def content_digest(step: int, state: Any) -> str:
    """Digests the canonical JSON of a step and its state.

    Args:
        step: Step number.
        state: JSON-serializable state.

    Returns:
        The hex SHA-256 digest.
    """
    body = json.dumps({"step": step, "state": state}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


class CheckpointStore:
    """The checkpoints of one run.

    Attributes:
        store: Durable storage.
        prefix: Key prefix of this run's checkpoints.
        ledger: Key of the record of damaged checkpoints a resume passed.
        skipped: Corrupt readings passed over by the last `latest`.
    """

    def __init__(self, store: objectstore.Depot, run_id: str) -> None:
        """Initializes the store view.

        Args:
            store: Durable storage.
            run_id: The run whose checkpoints these are.
        """
        self.store = store
        self.prefix = f"runs/{run_id}/checkpoints/"
        self.ledger = self.prefix + LEDGER
        self.skipped: list[Reading] = []

    def key(self, step: int) -> str:
        """Builds the object key for a step.

        Args:
            step: Step number.

        Returns:
            The key, with the step zero-padded to `STEP_WIDTH` digits.
        """
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
        """Lists the steps that have an object.

        Returns:
            The step numbers in ascending order; damaged checkpoints are
            included.
        """
        found = []
        for meta in self.store.list_objects(self.prefix):
            stem = meta.key.removeprefix(self.prefix).removesuffix(".json")
            if stem.isdigit():
                found.append(int(stem))
        return sorted(found)

    def load(self, step: int) -> Reading:
        """Reads and verifies a checkpoint.

        Args:
            step: Step number.

        Returns:
            The reading: `LOADED` with the checkpoint, `ABSENT` if there is no
            object, or `CORRUPT` with the reason if it is unreadable, not
            JSON, a different step, or a digest that does not match; a
            corrupt one is logged as a warning.

        Raises:
            StoreError: If the store fails for another reason (for example
                the backend is down): an outage is not "no checkpoint", which
                would restart the work from the beginning (ERR-10).
        """
        key = self.key(step)
        try:
            data = self.store.get(key)[0]
        except objectstore.AbsentError:
            # handled: an absent checkpoint is an ABSENT reading
            return Reading(key, Condition.ABSENT)
        except objectstore.MismatchError as error:
            # handled: a corrupt object is a DAMAGED reading; damage logs it
            return damage(key, f"corrupt object: {error}")
        try:
            document = json.loads(data)
            state = document["state"]
            digest = document["sha256"]
            stored = document["step"]
        except (ValueError, KeyError, TypeError) as error:
            # handled: a malformed record is a DAMAGED reading; damage logs it
            return damage(key, f"malformed: {type(error).__name__}: {error}")
        if stored != step:
            return damage(key, f"does not verify: holds step {stored!r}")
        if digest != content_digest(step, state):
            return damage(key, "does not verify: digest mismatch")
        return Reading(key, Condition.LOADED, Checkpoint(step, state, digest))

    def latest(self) -> Checkpoint | None:
        """Finds the newest checkpoint that verifies (corrupt ones are skipped).

        Returns:
            The checkpoint, or None if there is none that verifies. The
            corrupt readings passed over are kept in `skipped`; a step whose
            object vanished after listing is passed over silently.
        """
        self.skipped = []
        for step in reversed(self.steps()):
            reading = self.load(step)
            if reading.condition is Condition.LOADED:
                return reading.checkpoint
            if reading.condition is Condition.CORRUPT:
                self.skipped.append(reading)
        return None

    def record(self, attempt: int, found: Checkpoint | None) -> str | None:
        """Writes the corrupt checkpoints the last `latest` passed over.

        The record lands under the run's prefix, so the run manifest lists
        it as evidence: a restart from an older step, or from zero, past
        damaged checkpoints is on file, not only in a log.

        Args:
            attempt: The attempt that resumed.
            found: The checkpoint it resumed from, or None for a fresh start.

        Returns:
            The record's key, or None if nothing was skipped.
        """
        if not self.skipped:
            return None
        body = json.dumps(
            {
                "attempt": attempt,
                "resumed_from": None if found is None else found.step,
                "damaged": [
                    {"key": r.key, "reason": r.reason} for r in self.skipped
                ],
            },
            indent=2,
            sort_keys=True,
        ).encode()
        self.store.put(self.ledger, body)
        return self.ledger


def damage(key: str, reason: str) -> Reading:
    """Logs a corrupt checkpoint and builds its reading.

    Args:
        key: Object key of the checkpoint.
        reason: Why it was rejected.

    Returns:
        A `CORRUPT` reading.
    """
    LOGGER.warning("checkpoint %s is %s; passed over", key, reason)
    return Reading(key, Condition.CORRUPT, reason=reason)
