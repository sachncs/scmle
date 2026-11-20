"""The atomic durable commit boundary for incremental events."""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
import dataclasses
import json
import os
import pathlib
import tempfile
from typing import Any

from scmle import durability
from scmle.contracts import errors
from scmle.data.readers import sidecar
from scmle.data.readers.incremental import evidence
from scmle.data.readers.incremental import sources
from scmle.data.readers.incremental import timeline


def publish(target: pathlib.Path, record: Mapping[str, Any]) -> None:
    """Replaces a JSON file through a durable temporary file.

    If writing fails, the temporary file is removed and the error
    propagates.

    Args:
        target: The file to replace; its directory is created if missing.
        record: The JSON-compatible content, written with sorted keys and a
            trailing newline.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", dir=target.parent
    )
    try:
        with os.fdopen(
            descriptor, "w", encoding="utf-8", newline="\n"
        ) as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        pathlib.Path(temporary_name).replace(target)
        durability.flush(target.parent)
    except BaseException:
        pathlib.Path(temporary_name).unlink(missing_ok=True)
        raise


def partition(
    records: Iterable[timeline.Event], seen: Iterable[str]
) -> tuple[list[timeline.Event], set[str]]:
    """Separates unseen events from duplicates, in sequence then identity order.

    Args:
        records: Events to commit.
        seen: Identities already committed.

    Returns:
        The unseen events, each identity once, and the identities skipped as
        already committed. A repeat within `records` of an identity accepted
        here is dropped, not reported as a duplicate, so the two results
        stay disjoint.
    """
    committed = set(seen)
    known = set(committed)
    duplicates: set[str] = set()
    accepted: list[timeline.Event] = []
    for event in sorted(
        records, key=lambda value: (value.sequence, value.identifier)
    ):
        if event.identifier in committed:
            duplicates.add(event.identifier)
        elif event.identifier not in known:
            known.add(event.identifier)
            accepted.append(event)
    return accepted, duplicates


@dataclasses.dataclass(frozen=True, slots=True)
class Ledger(sidecar.Sidecar):
    """Atomic durable commit boundary for incremental event processing.

    Attributes:
        path: JSON file holding the committed state.
        identity: Identity of the one source the ledger serves.
    """

    identity: str

    def load(self) -> evidence.Settlement:
        """Loads committed state or initialize an empty source state.

        An unreadable or malformed state file raises the parsing error
        (for example JSONDecodeError or ValueError) unchanged.

        Returns:
            The committed state, or an empty state for this source when the
            state file does not exist.

        Raises:
            IncrementalError: With code "state_source_mismatch" if the stored
                state belongs to another source.
        """
        if not self.path.exists():
            return evidence.Settlement(timeline.Checkpoint(self.identity))
        record = json.loads(self.path.read_text(encoding="utf-8"))
        state = evidence.Settlement.from_dict(record)
        if state.checkpoint.source != self.identity:
            raise timeline.IncrementalError(
                "state_source_mismatch",
                "durable state belongs to another source",
                self.identity,
                {"state_source_identity": state.checkpoint.source},
            )
        return state

    def read(
        self, source: sources.Source, limit: int = 1_000
    ) -> timeline.Batch:
        """Reads an uncommitted batch using the durable deduplication state.

        Args:
            source: The event source to read from.
            limit: Maximum number of new events to return.

        Returns:
            The batch selected by `source.read` from the durable
            checkpoint, skipping already committed event identities.

        Raises:
            IncrementalError: With code "source_mismatch" if the source's
                identity differs from the ledger's.
        """
        if source.identity != self.identity:
            raise timeline.IncrementalError(
                "source_mismatch",
                "source does not match ingestion ledger",
                self.identity,
                {"source_identity": source.identity},
            )
        state = self.load()
        return source.read(state.checkpoint, state.seen, limit)

    def write(self, state: evidence.Settlement) -> None:
        """Publishes state with a durable temporary-file replacement.

        Callers that combine a read and write must hold ``lock``. This method
        remains public so transports can persist a prepared state explicitly.
        If writing fails, the temporary file is removed and the error
        propagates.

        Args:
            state: The state to persist.

        Raises:
            IncrementalError: With code "state_source_mismatch" if the state's
                checkpoint belongs to another source.
        """
        if state.checkpoint.source != self.identity:
            raise timeline.IncrementalError(
                "state_source_mismatch",
                "state does not match ingestion ledger",
                self.identity,
                {"state_source_identity": state.checkpoint.source},
            )
        publish(self.path, state.to_dict())

    def earmark(self, acknowledgement: evidence.Acknowledgement) -> None:
        """Stage an acknowledgement with durable atomic replacement.

        If writing fails, the temporary file is removed and the error
        propagates.

        Args:
            acknowledgement: The acknowledgement to stage for delivery.

        Raises:
            IncrementalError: With code "acknowledgement_source_mismatch" if
                the acknowledgement belongs to another source.
        """
        if acknowledgement.source != self.identity:
            raise timeline.IncrementalError(
                "acknowledgement_source_mismatch",
                "acknowledgement does not match ingestion ledger",
                self.identity,
                {"ack_source_identity": acknowledgement.source},
            )
        publish(self.pending, acknowledgement.to_dict())

    def reclaim(self) -> evidence.Acknowledgement | None:
        """Loads the acknowledgement that still needs transport delivery.

        Returns:
            The staged acknowledgement, or None if there is none.

        Raises:
            IncrementalError: If the staged file is unreadable or malformed
                (code "invalid_pending_acknowledgement"), or it belongs to
                another source (code "acknowledgement_source_mismatch").
        """
        if not self.pending.exists():
            return None
        try:
            record = json.loads(self.pending.read_text(encoding="utf-8"))
            acknowledgement = evidence.Acknowledgement.from_dict(record)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise timeline.IncrementalError(
                "invalid_pending_acknowledgement",
                "pending acknowledgement is not valid JSON evidence",
                self.identity,
                {"path": str(self.pending)},
            ) from error
        if acknowledgement.source != self.identity:
            raise timeline.IncrementalError(
                "acknowledgement_source_mismatch",
                "pending acknowledgement belongs to another source",
                self.identity,
                {"ack_source_identity": acknowledgement.source},
            )
        return acknowledgement

    def retract(self) -> None:
        """Removes a delivered acknowledgement and flushes the directory entry.

        Nothing is removed if no acknowledgement is staged.
        """
        self.pending.unlink(missing_ok=True)
        durability.flush(self.pending.parent)

    def commit(self, batch: timeline.Batch) -> evidence.Settlement:
        """Atomically publishes a processed batch's IDs and checkpoint.

        Args:
            batch: The processed batch.

        Returns:
            The state now committed. When every event of the batch is already
            committed and its checkpoint is not ahead of the current one, the
            current state is returned and nothing is written.

        Raises:
            IncrementalError: If the batch belongs to another source (code
                "batch_source_mismatch"), or its checkpoint sequence is behind
                the committed one (code "checkpoint_regression"), or the
                stored state belongs to another source.
        """
        if batch.identity != self.identity:
            raise timeline.IncrementalError(
                "batch_source_mismatch",
                "batch does not match ingestion ledger",
                self.identity,
                {"batch_source_identity": batch.identity},
            )
        with self.lock():
            current = self.load()
            candidate_ids = set(batch.identities)
            current_ids = set(current.seen)
            if (
                candidate_ids <= current_ids
                and batch.checkpoint.sequence <= current.checkpoint.sequence
            ):
                return current
            if batch.checkpoint.sequence < current.checkpoint.sequence:
                raise timeline.IncrementalError(
                    "checkpoint_regression",
                    "batch checkpoint moves backwards",
                    self.identity,
                    {
                        "current": current.checkpoint.to_dict(),
                        "candidate": batch.checkpoint.to_dict(),
                    },
                )
            state = evidence.Settlement(
                batch.checkpoint,
                tuple(sorted(set(current.seen) | set(batch.identities))),
            )
            self.write(state)
            return state

    def finalize(
        self,
        events: Iterable[timeline.Event],
        writer: Callable[[tuple[timeline.Event, ...]], None],
        acknowledgement: evidence.Acknowledgement | None = None,
    ) -> evidence.Ack:
        """Writes unseen events and commit their IDs under one ledger lock.

        The writer receives only events not already present in the durable
        ledger. If it raises, the ledger is unchanged and the caller may
        retry. A process crash after the writer succeeds but before the
        ledger replacement can cause a replay; external writers must accept
        ``event_id`` as an idempotency key. This is an at-least-once boundary,
        not a claim of distributed exactly-once delivery.

        Args:
            events: Events to commit; processed in sequence then identity
                order.
            writer: Callable receiving the tuple of unseen events. It is not
                called when every event is already committed.
            acknowledgement: Acknowledgement to stage, or None. It is staged,
                rebound to the resulting checkpoint, only if it lists at
                least one event identity.

        Returns:
            A receipt listing the sorted identities accepted and the sorted
            identities skipped as duplicates (including repeats within
            `events`), and the resulting checkpoint: advanced to the highest
            sequence and latest event time when anything was accepted,
            otherwise the unchanged current checkpoint.

        Raises:
            TypeError: If `events` holds something other than Event
                values or `writer` is not callable.
            IncrementalError: If the acknowledgement belongs to another source
                (code "acknowledgement_source_mismatch") or the stored state
                belongs to another source.
        """
        records = tuple(events)
        if not all(isinstance(event, timeline.Event) for event in records):
            raise errors.KindError("events must contain Event values")
        if not callable(writer):
            raise errors.KindError("writer must be callable")
        if (
            acknowledgement is not None
            and acknowledgement.source != self.identity
        ):
            raise timeline.IncrementalError(
                "acknowledgement_source_mismatch",
                "acknowledgement does not match ingestion ledger",
                self.identity,
                {"ack_source_identity": acknowledgement.source},
            )
        with self.lock():
            current = self.load()
            accepted, duplicates = partition(records, current.seen)
            accepted_ids = tuple(event.identifier for event in accepted)
            checkpoint = current.checkpoint
            if accepted:
                writer(tuple(accepted))
                checkpoint = timeline.forward(checkpoint, accepted)
                seen = {*current.seen, *accepted_ids}
                self.write(evidence.Settlement(checkpoint, tuple(sorted(seen))))
            if acknowledgement is not None and acknowledgement.events:
                self.earmark(
                    dataclasses.replace(acknowledgement, checkpoint=checkpoint)
                )
            return evidence.Ack(
                self.identity,
                tuple(sorted(accepted_ids)),
                tuple(sorted(duplicates)),
                checkpoint,
            )
