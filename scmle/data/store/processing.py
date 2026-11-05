"""Deterministic incremental processing with retries, dedup and recovery.

`IncrementalProcessor` turns the objects under an input prefix into output
objects, once each, in sorted key order:

* **Deterministic.** Objects are taken in key order, rows keep their order,
  the first of a repeated key wins, and outputs are written as canonical CSV
  under a key derived from the input key. The same inputs give the same
  output bytes.
* **Deduplicated.** A row whose key (the contract's `key_columns`) was
  already processed, in this object or an earlier one, is dropped and
  counted. An object whose bytes equal an already processed object's is
  skipped as a duplicate upload.
* **Checkpointed.** After each object the checkpoint (processed objects with
  their hashes, seen row keys) is written atomically to the store with its
  own digest. A checkpoint that fails its digest stops the run; it is never
  silently reset.
* **Retried.** Transient errors from the store or the transform are retried
  under a `RetryPolicy`; anything else stops the run at that object with the
  checkpoint unchanged.
* **Recoverable.** The output is written before the checkpoint. A crash
  between the two leaves the object unrecorded, so the next run processes it
  again and rewrites the same bytes under the same key. Delivery is
  therefore at-least-once for the work and idempotent for the output; it is
  not exactly-once, and a transform with side effects other than its output
  must be idempotent itself.
* **One writer.** A lock object created with a conditional put keeps two
  runs from sharing a checkpoint; a crashed run leaves the lock, which a
  person removes with `break_lock`.

The seen-key set grows with the data and lives in one JSON object, which is
adequate for millions of keys and not beyond.
"""

from collections.abc import Callable
import dataclasses
import hashlib
import json
import time
from typing import Any

import pandas as pd

from scmle.contracts import errors
from scmle.core import retry
from scmle.data.store import objectstore
from scmle.data.store import quality
from scmle.data.store import snapshot as snapshot_module

FORMAT = 1
SEPARATOR = "\x1f"


class CheckpointCorruptError(errors.DataError):
    """The checkpoint does not match its digest or format."""

    category = "checkpoint_corrupt"


class ChangedInputError(errors.DataError):
    """An already processed input object now has different bytes."""

    category = "input_changed"


class ConcurrentRunError(errors.ScmleError):
    """Another run holds the lock."""

    category = "concurrent_run"


@dataclasses.dataclass(slots=True)
class Checkpoint:
    """What has been processed.

    Attributes:
        processed: Input key to the SHA-256 it had when processed.
        seen: Sorted row keys already emitted.
    """

    processed: dict[str, str] = dataclasses.field(default_factory=dict)
    seen: set[str] = dataclasses.field(default_factory=set)

    def to_bytes(self) -> bytes:
        """Canonical JSON with a digest of its own content."""
        body = {
            "format": FORMAT,
            "processed": dict(sorted(self.processed.items())),
            "seen": sorted(self.seen),
        }
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()
        ).hexdigest()
        return json.dumps({**body, "digest": digest}, sort_keys=True).encode()

    @classmethod
    def from_bytes(cls, data: bytes) -> "Checkpoint":
        """Parses and verifies a stored checkpoint.

        Raises:
            CheckpointCorruptError: If it is malformed or its digest is wrong.
        """
        try:
            document = json.loads(data)
            digest = document.pop("digest")
            expected = hashlib.sha256(
                json.dumps(document, sort_keys=True).encode()
            ).hexdigest()
            if digest != expected or document["format"] != FORMAT:
                raise ValueError("digest or format mismatch")
            return cls(dict(document["processed"]), set(document["seen"]))
        except (ValueError, KeyError, TypeError) as error:
            raise CheckpointCorruptError(
                f"checkpoint is corrupt: {error}"
            ) from error


@dataclasses.dataclass(frozen=True, slots=True)
class RunResult:
    """What one run did.

    Attributes:
        processed: Input keys processed in this run.
        skipped: Input keys already done before this run.
        duplicate_objects: Input keys skipped as repeated uploads.
        duplicate_rows: Rows dropped as repeated keys.
        rows_out: Rows written.
        outputs: Output keys written in this run.
        failed: The input key that stopped the run, if any.
        error: Why it stopped, if it did.
        retries: Retry waits taken, in seconds.
    """

    processed: tuple[str, ...]
    skipped: tuple[str, ...]
    duplicate_objects: tuple[str, ...]
    duplicate_rows: int
    rows_out: int
    outputs: tuple[str, ...]
    failed: str | None
    error: str | None
    retries: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready form."""
        return dataclasses.asdict(self)


def row_keys(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    """One string key per row, from the key columns."""
    return frame[list(columns)].astype(str).agg(SEPARATOR.join, axis=1)


class IncrementalProcessor:
    """Processes new input objects once each, resuming after a crash."""

    def __init__(
        self,
        store: objectstore.ObjectStore,
        contract: snapshot_module.DataContract,
        transform: Callable[[pd.DataFrame], pd.DataFrame],
        *,
        input_prefix: str = "input/",
        output_prefix: str = "output/",
        state_key: str = "state/checkpoint.json",
        lock_key: str = "state/lock",
        gate: quality.QualityGate | None = None,
        stage: quality.Stage = quality.Stage.TRAIN,
        policy: retry.RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
        reader: snapshot_module.SnapshotReader | None = None,
        before_commit: Callable[[str], None] | None = None,
    ) -> None:
        """Initializes the processor.

        Args:
            store: Holds inputs, outputs, the checkpoint and the lock.
            contract: Declares the schema and the row key columns.
            transform: Maps a clean input frame to an output frame.
            input_prefix: Where inputs are listed.
            output_prefix: Where outputs are written.
            state_key: The checkpoint object.
            lock_key: The lock object.
            gate: Quality gate applied to each input before transforming.
            stage: The gate stage.
            policy: Retry policy for transient errors.
            sleep: Waits between retries.
            reader: Snapshot reader to record reads with.
            before_commit: Called after an output is written and before the
                checkpoint is; lets a test inject a crash there.

        Raises:
            ConfigError: If the contract has no key columns.
        """
        if not contract.key_columns:
            raise errors.ConfigError("deduplication needs key_columns")
        self.store = store
        self.contract = contract
        self.transform = transform
        self.input_prefix = input_prefix
        self.output_prefix = output_prefix
        self.state_key = state_key
        self.lock_key = lock_key
        self.gate = gate
        self.stage = stage
        self.policy = policy or retry.RetryPolicy()
        self.sleep = sleep
        self.reader = reader or snapshot_module.SnapshotReader(store)
        self.before_commit = before_commit
        self.waits: list[float] = []

    def load_checkpoint(self) -> Checkpoint:
        """The stored checkpoint, or an empty one on the first run."""
        try:
            data, ignored = self.store.get(self.state_key)
        except objectstore.NotFoundError:
            return Checkpoint()
        return Checkpoint.from_bytes(data)

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        """Writes the checkpoint atomically."""
        self.store.put(self.state_key, checkpoint.to_bytes())

    def break_lock(self) -> None:
        """Removes a lock left by a crashed run (a human decision)."""
        self.store.delete(self.lock_key)

    def retrying(self, operation: Callable[[], Any]) -> Any:
        """Runs an operation, retrying transient errors and noting waits."""

        def sleeper(seconds: float) -> None:
            self.waits.append(seconds)
            self.sleep(seconds)

        return retry.call_with_retries(operation, self.policy, sleep=sleeper)

    def run(self, max_objects: int | None = None) -> RunResult:
        """Processes the new input objects.

        Args:
            max_objects: Stop after this many objects (to bound a run).

        Returns:
            What happened. A failure on one object is in the result, with
            the earlier objects recorded.

        Raises:
            ConcurrentRunError: If another run holds the lock.
            CheckpointCorruptError: If the checkpoint is damaged.
            ChangedInputError: If a processed input now differs.
        """
        try:
            self.store.put(self.lock_key, b"locked", if_absent=True)
        except objectstore.AlreadyExistsError as error:
            raise ConcurrentRunError("another run holds the lock") from error
        try:
            return self.run_locked(max_objects)
        finally:
            self.store.delete(self.lock_key)

    def run_locked(self, max_objects: int | None) -> RunResult:
        """The run, with the lock held."""
        checkpoint = self.load_checkpoint()
        self.waits.clear()
        done: list[str] = []
        skipped: list[str] = []
        duplicate_objects: list[str] = []
        outputs: list[str] = []
        duplicate_rows = 0
        rows_out = 0
        failed: str | None = None
        problem: str | None = None
        listing = self.retrying(
            lambda: self.store.list_objects(self.input_prefix)
        )
        for meta in listing:
            known = checkpoint.processed.get(meta.key)
            if known is not None:
                if known != meta.sha256:
                    raise ChangedInputError(
                        f"{meta.key} changed after it was processed"
                    )
                skipped.append(meta.key)
                continue
            if max_objects is not None and len(done) >= max_objects:
                break
            if meta.sha256 in checkpoint.processed.values():
                checkpoint.processed[meta.key] = meta.sha256
                self.save_checkpoint(checkpoint)
                duplicate_objects.append(meta.key)
                continue
            try:
                dropped, written, output_key = self.process(meta, checkpoint)
            except (errors.ScmleError, retry.RetryExhaustedError) as error:
                failed, problem = meta.key, f"{type(error).__name__}: {error}"
                break
            done.append(meta.key)
            duplicate_rows += dropped
            rows_out += written
            outputs.append(output_key)
        return RunResult(
            tuple(done),
            tuple(skipped),
            tuple(duplicate_objects),
            duplicate_rows,
            rows_out,
            tuple(outputs),
            failed,
            problem,
            tuple(self.waits),
        )

    def process(
        self, meta: objectstore.ObjectMeta, checkpoint: Checkpoint
    ) -> tuple[int, int, str]:
        """Processes one object; returns dropped rows, rows out, output key."""
        frame, record = self.retrying(
            lambda: self.reader.read(meta.key, self.contract)
        )
        if self.gate is not None:
            frame = self.gate.run(
                frame, record, self.contract, self.stage
            ).frame
        keys = row_keys(frame, self.contract.key_columns)
        fresh = ~keys.isin(checkpoint.seen) & ~keys.duplicated(keep="first")
        dropped = int((~fresh).sum())
        result = self.retrying(lambda: self.transform(frame[fresh.to_numpy()]))
        name = meta.key.removeprefix(self.input_prefix).rsplit(".", 1)[0]
        output_key = f"{self.output_prefix}{name}.csv"
        payload = result.to_csv(index=False, lineterminator="\n").encode()
        self.retrying(lambda: self.store.put(output_key, payload))
        if self.before_commit is not None:
            self.before_commit(meta.key)
        checkpoint.processed[meta.key] = meta.sha256
        checkpoint.seen.update(keys[fresh].tolist())
        self.retrying(lambda: self.save_checkpoint(checkpoint))
        return dropped, len(result), output_key
