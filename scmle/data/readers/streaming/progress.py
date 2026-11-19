"""Durable row positions, watermarks and the batches that advance them."""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import datetime
import hashlib
from typing import Any
from typing import ClassVar
from typing import Self

import pandas

from scmle.contracts import errors
from scmle.contracts import records
from scmle.data.readers.streaming import failure

SHA256_HEX_LENGTH = 64


def assay(value: str, field_name: str) -> None:
    """Validates a SHA-256 digest used as durable source evidence.

    Args:
        value: The candidate digest.
        field_name: Name used in the error message.

    Raises:
        ValueError: If `value` is not 64 lower-case hexadecimal characters.
    """
    if len(value) != SHA256_HEX_LENGTH or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise errors.InputError(f"{field_name} must be a SHA-256 hex digest")


def validate(value: str, field_name: str) -> datetime.datetime:
    """Parses a timezone-aware watermark timestamp.

    Args:
        value: ISO-8601 timestamp; a trailing "Z" means UTC.
        field_name: Name used in the error message.

    Returns:
        The timestamp converted to UTC.

    Raises:
        ValueError: If `value` is not ISO-8601 or carries no timezone.
    """
    try:
        parsed = datetime.datetime.fromisoformat(value)
    except ValueError as error:
        raise errors.InputError(f"{field_name} must be ISO-8601") from error
    if parsed.tzinfo is None:
        raise errors.InputError(f"{field_name} must include a timezone")
    return parsed.astimezone(datetime.UTC)


def nest(layout: records.Layout) -> records.Layout:
    """Moves a renamed version-1 checkpoint's snapshot digest under digests.

    Args:
        layout: The checkpoint after its other keys were renamed.

    Returns:
        The checkpoint with ``snapshot_sha256`` as ``digests.snapshot``.
    """
    if "snapshot_sha256" not in layout:
        return layout
    rest = {k: v for k, v in layout.items() if k != "snapshot_sha256"}
    return {**rest, "digests": {"snapshot": layout["snapshot_sha256"]}}


def flatten(layout: records.Layout) -> records.Layout:
    """Undoes `nest`.

    Args:
        layout: A version-2 checkpoint.

    Returns:
        The checkpoint with its snapshot digest as ``snapshot_sha256``.
    """
    if "digests" not in layout:
        return layout
    rest = {k: v for k, v in layout.items() if k != "digests"}
    return {**rest, "snapshot_sha256": layout["digests"]["snapshot"]}


def shift(
    layout: records.Layout,
    key: str,
    step: Callable[[records.Layout], records.Layout],
) -> records.Layout:
    """Applies a step to one nested part of a stored record.

    Args:
        layout: The stored record.
        key: The key of the nested part.
        step: Moves the stored part.

    Returns:
        The record with the part moved, or unchanged when the part is not a
        mapping (loading then rejects it).
    """
    part = layout.get(key)
    if not isinstance(part, Mapping):
        return layout
    return {**layout, key: step(dict(part))}


@dataclasses.dataclass(frozen=True, slots=True)
class Checkpoint:
    """A durable row position bound to a source and exact raw snapshot.

    A checkpoint carries no version of its own: the record that holds it is
    versioned and applies `migrations` to it.

    Attributes:
        source: Identity of the source the position belongs to.
        digests: Digests keyed by the hashed thing; ``snapshot`` is the
            SHA-256 hex digest of the file the position was taken on.
        offset: Number of rows already consumed, so the next row to read
            is the one at this zero-based position.
        watermark: Greatest timestamp seen so far as an ISO-8601 UTC
            string; empty when none has been seen.
        column: Column the watermark is taken from, or None.
        migrations: The checkpoint's layout steps, applied by its holder.
    """

    migrations: ClassVar[tuple[records.Migration, ...]] = (
        records.Migration(
            renames={"source_identity": "source", "watermark_column": "column"},
            change=nest,
            revert=flatten,
        ),
    )

    source: str
    digests: Mapping[str, str]
    offset: int = 0
    watermark: str = ""
    column: str | None = None

    def __post_init__(self) -> None:
        """Validates source, snapshot, offset, and optional watermark.

        Raises:
            ValueError: If the source identity is blank, the digests are not
                exactly a SHA-256 hex ``snapshot`` digest, the offset is
                negative, the watermark column is an empty string, or the
                watermark is not a timezone-aware ISO-8601 timestamp.
            TypeError: If the offset is not an integer.
        """
        if not self.source.strip():
            raise errors.InputError("source must not be empty")
        if set(self.digests) != {"snapshot"}:
            raise errors.InputError("digests must hold the snapshot digest")
        assay(self.digests["snapshot"], "snapshot digest")
        object.__setattr__(self, "digests", dict(self.digests))
        if isinstance(self.offset, bool) or not isinstance(self.offset, int):
            raise errors.KindError("offset must be an integer")
        if self.offset < 0:
            raise errors.InputError("offset must be non-negative")
        if self.column is not None and not self.column:
            raise errors.InputError("column must not be empty")
        if self.watermark:
            validate(self.watermark, "watermark")

    def to_dict(self) -> dict[str, Any]:
        """Returns JSON-compatible checkpoint evidence.

        Returns:
            A dictionary with the five checkpoint fields.
        """
        return {
            "source": self.source,
            "digests": dict(self.digests),
            "offset": self.offset,
            "watermark": self.watermark,
            "column": self.column,
        }

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> Self:
        """Rebuilds a checkpoint from durable JSON.

        Args:
            record: Mapping in the form produced by `to_dict`.

        Returns:
            The checkpoint described by `record`.
        """
        return cls(
            source=str(record["source"]),
            digests={"snapshot": str(record["digests"]["snapshot"])},
            offset=int(record["offset"]),
            watermark=str(record.get("watermark", "")),
            column=(
                str(record["column"])
                if record.get("column") is not None
                else None
            ),
        )


def advance(
    frame: pandas.DataFrame,
    column: str | None,
    current: str,
    source_identity: str,
) -> str:
    """Returns the greatest timezone-aware timestamp in a batch.

    A malformed `current` value is rejected with ValueError.

    Args:
        frame: The batch of rows.
        column: Column holding timestamps, or None to disable tracking.
        current: The watermark so far (ISO-8601 UTC string, possibly
            empty).
        source_identity: Identity of the source, used in error evidence.

    Returns:
        The later of `current` and the greatest timestamp in `column`, as
        an ISO-8601 UTC string. `current` is returned unchanged when
        `column` is None or every value in the column is null.

    Raises:
        StreamingError: If `column` is not in the frame or its values
            cannot be parsed as timestamps.
    """
    if column is None:
        return current
    if column not in frame.columns:
        raise failure.StreamingError(
            "watermark_column_missing",
            f"watermark column '{column}' is not present in the batch",
            source_identity,
            {"column": column},
        )
    try:
        values = pandas.to_datetime(frame[column], utc=True, errors="raise")
    except (TypeError, ValueError) as error:
        raise failure.StreamingError(
            "invalid_watermark",
            f"watermark column '{column}' contains invalid timestamps",
            source_identity,
            {"column": column},
        ) from error
    observed = values.dropna()
    if observed.empty:
        return current
    candidate = observed.max().to_pydatetime().astimezone(datetime.UTC)
    if not current:
        return candidate.isoformat()
    return max(validate(current, "watermark"), candidate).isoformat()


@dataclasses.dataclass(frozen=True, slots=True)
class Batch:
    """A bounded frame plus the checkpoint it may advance.

    Attributes:
        identity: Identity of the source the rows came from.
        snapshot: Digest of the file snapshot the rows came from.
        offset: Zero-based position of the frame's first row in the
            source.
        frame: The rows; never empty.
        checkpoint: The checkpoint to persist once the batch is
            committed.
        more: Whether rows remain after this batch.
        key: Idempotency key derived from the source, snapshot and
            row range; set on construction.
    """

    identity: str
    snapshot: str
    offset: int
    frame: pandas.DataFrame = dataclasses.field(repr=False)
    checkpoint: Checkpoint
    more: bool = False
    key: str = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        """Validates the batch and derives its deterministic idempotency key.

        Raises:
            ValueError: If the checkpoint's source or snapshot differs from the
                batch's, the start offset is negative, the frame is empty or
                not a DataFrame, or the checkpoint does not advance past the
                start offset.
        """
        if self.checkpoint.source != self.identity:
            raise errors.InputError("batch and checkpoint sources differ")
        if self.checkpoint.digests["snapshot"] != self.snapshot:
            raise errors.InputError("batch and checkpoint snapshots differ")
        if self.offset < 0:
            raise errors.InputError("start_offset must be non-negative")
        frame: object = self.frame
        if not isinstance(frame, pandas.DataFrame) or frame.empty:
            raise errors.InputError("frame must contain at least one row")
        if self.checkpoint.offset <= self.offset:
            raise errors.InputError(
                "checkpoint must advance past the batch start"
            )
        # The key is stable for one source row range: it hashes the source,
        # the snapshot and both offsets.
        payload = (
            f"{self.identity}:{self.snapshot}:"
            f"{self.offset}:{self.checkpoint.offset}"
        ).encode()
        object.__setattr__(
            self, "key", f"batch_{hashlib.sha256(payload).hexdigest()}"
        )
