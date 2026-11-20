"""Deterministic incremental processing with retries, dedup and recovery.

`Processor` turns the objects under an input prefix into output
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
  under a `Retries`; anything else stops the run at that object with the
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
import logging
import time
from typing import Any
from typing import ClassVar

import pandas

from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.store import quality
from scmle.data.store import snapshot
from scmle.runtime import retry
from scmle.storage import objectstore

LOGGER = logging.getLogger(__name__)


FORMAT = 1
SEPARATOR = "\x1f"


def matches(digest: str, expected: str, document: dict[str, Any]) -> None:
    """Checks a checkpoint's digest and format.

    Args:
        digest: The digest stored in the checkpoint.
        expected: The digest recomputed from the document.
        document: The decoded checkpoint without its digest; its format is
            read only when the digests agree.

    Raises:
        ValueError: If the digest differs or the format is not `FORMAT`.
    """
    if digest != expected or document["format"] != FORMAT:
        raise errors.InputError("digest or format mismatch")


class CorruptError(errors.DataError):
    """The checkpoint does not match its digest or format."""

    category = "checkpoint_corrupt"


class ChangedError(errors.DataError):
    """An already processed input object now has different bytes."""

    category = "input_changed"


class LockError(errors.ScmleError):
    """Another run holds the lock."""

    category = "concurrent_run"


@dataclasses.dataclass(slots=True)
class Checkpoint:
    """What has been processed.

    Attributes:
        processed: Input key to the SHA-256 it had when processed.
        seen: Row keys already emitted (stored sorted).
    """

    processed: dict[str, str] = dataclasses.field(default_factory=dict)
    seen: set[str] = dataclasses.field(default_factory=set)

    def encode(self) -> bytes:
        """Serializes the checkpoint with a digest of its own content.

        Returns:
            UTF-8 JSON with sorted keys: `format`, `processed`, `seen` (a
            sorted list) and `digest`, the SHA-256 of the same JSON without
            the `digest` key.
        """
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
    def decode(cls, data: bytes) -> "Checkpoint":
        """Parses and verifies a stored checkpoint.

        Args:
            data: Bytes produced by `encode`.

        Returns:
            The checkpoint they describe.

        Raises:
            ValueError: Raised internally on a digest or format mismatch and
                converted to the error below.
            CorruptError: If the bytes are not valid JSON, lack a
                field, or have a wrong digest or format.
        """
        try:
            document = json.loads(data)
            digest = document.pop("digest")
            expected = hashlib.sha256(
                json.dumps(document, sort_keys=True).encode()
            ).hexdigest()
            matches(digest, expected, document)
            return cls(dict(document["processed"]), set(document["seen"]))
        except (ValueError, KeyError, TypeError, AttributeError) as error:
            raise CorruptError(f"checkpoint is corrupt: {error}") from error


@dataclasses.dataclass(frozen=True, slots=True)
class Processing:
    """What one run did.

    Attributes:
        processed: Input keys processed in this run.
        skipped: Input keys already done before this run.
        duplicates: Input keys skipped as repeated uploads.
        duplicate_rows: Rows dropped as repeated keys.
        rows_out: Rows written.
        outputs: Output keys written in this run.
        failed: The input key that stopped the run, if any.
        error: Why it stopped, if it did.
        retries: Retry waits taken, in seconds.
    """

    processed: tuple[str, ...]
    skipped: tuple[str, ...]
    duplicates: tuple[str, ...]
    duplicate_rows: int
    rows_out: int
    outputs: tuple[str, ...]
    failed: str | None
    error: str | None
    retries: tuple[float, ...]

    KEYS: ClassVar[records.Migration] = records.nest(
        {"duplicate_objects": "duplicates"},
        groups={"rows": {"duplicate_rows": "dropped", "rows_out": "used"}},
    )
    """Version-1 keys to the written ones; `downgrade` restores version 1."""

    def to_dict(self) -> dict[str, Any]:
        """Converts the result to plain data.

        Returns:
            A dict of the fields in single-word keys, with the row counts
            under `rows.{dropped, used}`; the tuples stay tuples, which
            JSON writes as arrays.
        """
        return self.KEYS.upgrade(dataclasses.asdict(self))


class Processor:
    """Processes new input objects once each, resuming after a crash.

    Attributes:
        store: Holds inputs, outputs, the checkpoint and the lock.
        contract: Declares the schema and the row key columns.
        transform: Maps a clean input frame to an output frame.
        prefix: Where inputs are listed.
        destination: Where outputs are written.
        state: The checkpoint object.
        guard: The lock object.
        gate: Quality gate applied to each input, if any.
        stage: The gate stage.
        policy: Retry policy for transient errors.
        sleep: Waits between retries.
        reader: Snapshot reader that records each read.
        hook: Hook called between writing an output and writing the
            checkpoint, if any.
        waits: The retry waits taken in the current or latest run, in
            seconds.
    """

    def __init__(
        self,
        store: objectstore.Depot,
        contract: snapshot.Contract,
        transform: Callable[[pandas.DataFrame], pandas.DataFrame],
        *,
        input_prefix: str = "input/",
        output_prefix: str = "output/",
        state_key: str = "state/checkpoint.json",
        lock_key: str = "state/lock",
        gate: quality.Gate | None = None,
        stage: quality.Stage = quality.Stage.TRAIN,
        policy: retry.Retries | None = None,
        sleep: Callable[[float], None] = time.sleep,
        reader: snapshot.Reader | None = None,
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
        self.prefix = input_prefix
        self.destination = output_prefix
        self.state = state_key
        self.guard = lock_key
        self.gate = gate
        self.stage = stage
        self.policy = policy or retry.Retries()
        self.sleep = sleep
        self.reader = reader or snapshot.Reader(store)
        self.hook = before_commit
        self.waits: list[float] = []

    def restore(self) -> Checkpoint:
        """Loads the stored checkpoint.

        A damaged checkpoint is not replaced: `CorruptError`
        from `Checkpoint.decode` propagates, as do errors from the
        store.

        Returns:
            The stored checkpoint, or an empty one if none exists yet.
        """
        try:
            data = self.retrying(lambda: self.store.get(self.state))[0]
        except objectstore.AbsentError:
            # handled: absent checkpoint returns documented empty checkpoint
            return Checkpoint()
        return Checkpoint.decode(data)

    def stash(self, checkpoint: Checkpoint) -> None:
        """Writes the checkpoint atomically.

        Args:
            checkpoint: The state to store under `state`.
        """
        self.store.put(self.state, checkpoint.encode())

    def lock(self) -> None:
        """Removes a lock left by a crashed run (a human decision)."""
        self.store.delete(self.guard)

    def retrying(self, operation: Callable[[], Any]) -> Any:
        """Runs an operation, retrying transient errors and noting waits.

        Each wait between attempts is appended to `waits`. Errors that are
        not transient propagate at once, and `RetryError`
        propagates when every attempt failed with a transient error.

        Args:
            operation: Zero-argument callable to run.

        Returns:
            Whatever the operation returned on the attempt that succeeded.
        """

        def sleeper(seconds: float) -> None:
            """Records a retry wait, then sleeps for it.

            Args:
                seconds: Length of the wait.
            """
            self.waits.append(seconds)
            self.sleep(seconds)

        return retry.call(operation, self.policy, sleep=sleeper)

    def run(self, max_objects: int | None = None) -> Processing:
        """Processes the new input objects.

        Args:
            max_objects: Stop after this many objects (to bound a run).

        Returns:
            What happened. A failure on one object is in the result, with
            the earlier objects recorded.

        Raises:
            LockError: If another run holds the lock.

        The lock is released when the run ends, whatever the outcome. Every
        store call is retried on a transient error except taking the lock:
        retrying a conditional put could turn a timeout whose first try
        landed into `ExistsError`, so that call is tried once.
        `CorruptError` propagates if the checkpoint is damaged,
        and `ChangedError` if an input that was already processed now
        has different bytes; both leave the stored checkpoint as it was.
        An `errors.InputError` from processing an object propagates too,
        as the `ValueError` it was before ERR-20 did.
        """
        try:
            self.store.put(self.guard, b"locked", if_absent=True)
        except objectstore.ExistsError as error:
            raise LockError("another run holds the lock") from error
        try:
            return self.sweep(max_objects)
        finally:
            self.retrying(lambda: self.store.delete(self.guard))

    def sweep(self, max_objects: int | None) -> Processing:
        """Runs with the lock held.

        Lists the inputs in key order. An input already in the checkpoint
        with the same hash is skipped; a new input whose bytes equal a
        processed one is recorded as a duplicate upload; any other is
        processed. The run stops at the first input that fails with a
        framework or retry error, and records it in the result.

        Args:
            max_objects: Stop after processing this many new objects, or
                None for no limit. Skipped and duplicate objects do not
                count.

        Returns:
            What happened in this run.

        Raises:
            ChangedError: If a processed input now has different bytes.
        """
        self.waits.clear()
        checkpoint = self.restore()
        done: list[str] = []
        skipped: list[str] = []
        duplicate_objects: list[str] = []
        outputs: list[str] = []
        duplicate_rows = 0
        rows_out = 0
        failed: str | None = None
        problem: str | None = None
        listing = self.retrying(lambda: self.store.catalog(self.prefix))
        for meta in listing:
            known = checkpoint.processed.get(meta.key)
            if known is not None:
                if known != meta.sha256:
                    raise ChangedError(
                        f"{meta.key} changed after it was processed"
                    )
                skipped.append(meta.key)
                continue
            if max_objects is not None and len(done) >= max_objects:
                break
            if meta.sha256 in checkpoint.processed.values():
                checkpoint.processed[meta.key] = meta.sha256
                try:
                    self.retrying(lambda: self.stash(checkpoint))
                except retry.RetryError as error:
                    LOGGER.exception("checkpoint save failed at %s", meta.key)
                    failed, problem = meta.key, errors.Failure.of(error).text
                    break
                duplicate_objects.append(meta.key)
                continue
            try:
                dropped, written, output_key = self.process(meta, checkpoint)
            except errors.InputError:
                # ERR-20: an InputError was a bare ValueError (or TypeError,
                # RuntimeError) that propagated out of the run; it still does.
                raise
            except (errors.ScmleError, retry.RetryError) as error:
                LOGGER.exception("processing failed at %s", meta.key)
                failed, problem = meta.key, errors.Failure.of(error).text
                break
            done.append(meta.key)
            duplicate_rows += dropped
            rows_out += written
            outputs.append(output_key)
        return Processing(
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
        self, meta: objectstore.Metadata, checkpoint: Checkpoint
    ) -> tuple[int, int, str]:
        """Processes one object and commits it to the checkpoint.

        Reads and validates the object, applies the gate if there is one,
        drops rows whose key was seen before or repeats earlier in the
        object, transforms the rest, writes the output, then adds the
        object and its row keys to `checkpoint` and saves it. Errors from
        any step propagate and leave the stored checkpoint unchanged.

        Args:
            meta: The input object.
            checkpoint: The state to update in place.

        Returns:
            The number of rows dropped as repeated keys, the number of rows
            in the output, and the key the output was written under.
        """
        frame, record = self.retrying(
            lambda: self.reader.read(meta.key, self.contract)
        )
        gate = self.gate
        if gate is not None:
            frame = self.retrying(
                lambda: gate.run(frame, record, self.contract, self.stage).frame
            )
        keys = (
            frame[list(self.contract.key_columns)]
            .astype(str)
            .agg(SEPARATOR.join, axis=1)
        )
        fresh = ~keys.isin(checkpoint.seen) & ~keys.duplicated(keep="first")
        dropped = int((~fresh).sum())
        result = self.retrying(lambda: self.transform(frame[fresh.to_numpy()]))
        name = meta.key.removeprefix(self.prefix).rsplit(".", 1)[0]
        output_key = f"{self.destination}{name}.csv"
        payload = result.to_csv(index=False, lineterminator="\n").encode()
        self.retrying(lambda: self.store.put(output_key, payload))
        if self.hook is not None:
            self.hook(meta.key)
        checkpoint.processed[meta.key] = meta.sha256
        checkpoint.seen.update(keys[fresh].tolist())
        self.retrying(lambda: self.stash(checkpoint))
        return dropped, len(result), output_key
