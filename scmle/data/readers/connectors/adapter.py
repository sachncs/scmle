"""The polymorphic contract every source connector implements."""

import abc
import pathlib

from scmle.contracts import errors
from scmle.data.readers.connectors import failure
from scmle.data.readers.connectors import layout
from scmle.data.readers.connectors import locator


class Adapter(abc.ABC):
    """Public polymorphic contract for bounded source retrieval.

    Attributes:
        source_uri: Canonical URI of the source.
        source_identity: Stable identity of the source.
        retry_budget: Retry policy applied to each operation.
    """

    def __init__(
        self,
        source_uri: str | pathlib.Path,
        retry_budget: failure.Retries | None = None,
    ) -> None:
        """Initializes a connector identity and retry policy.

        Args:
            source_uri: URI or local path of the source; it is canonicalized.
            retry_budget: Retry policy; defaults to `Retries()`.
        """
        self.source_uri = locator.canonicalize(source_uri)
        self.source_identity = locator.source_identity(self.source_uri)
        self.retry_budget = retry_budget or failure.Retries()

    @abc.abstractmethod
    def schema(self) -> layout.Columns:
        """Describes the source schema without consuming a row checkpoint.

        Returns:
            The ordered schema of the source.
        """

    @abc.abstractmethod
    def read_rows(
        self, checkpoint: locator.Bookmark | None = None, limit: int = 1_000
    ) -> layout.Page:
        """Reads at most ``limit`` rows from a checkpoint.

        Args:
            checkpoint: Position to resume from; None starts at the
                beginning.
            limit: Maximum number of rows to return; must be positive.

        Returns:
            The batch and the checkpoint from which to continue.
        """

    def checkpoint(self, offset: int = 0) -> locator.Bookmark:
        """Creates a source-bound checkpoint for explicit offset reads.

        Args:
            offset: Position to start from, in the connector's unit.

        Returns:
            A checkpoint bound to this connector's source at `offset`.
        """
        return locator.Bookmark(
            offset=offset,
            cursor=locator.bookmark(self.source_identity, offset),
            source_identity=self.source_identity,
        )

    def rebind(self, checkpoint: locator.Bookmark | None) -> locator.Bookmark:
        """Binds and validates a checkpoint before reading rows.

        Args:
            checkpoint: Checkpoint to validate; None means the start.

        Returns:
            The checkpoint bound to this connector's source identity.

        Raises:
            CheckpointError: ``checkpoint_mismatch`` if the checkpoint is
                bound to a different source.
        """
        value = checkpoint or locator.Bookmark()
        try:
            return value.bind(self.source_identity)
        except ValueError as error:
            raise failure.CheckpointError(
                self.source_uri,
                "checkpoint_mismatch",
                str(error),
                {
                    "checkpoint_source_identity": value.source_identity,
                    "connector_source_identity": self.source_identity,
                    "offset": value.offset,
                },
            ) from error

    def validate_limit(self, limit: int) -> None:
        """Rejects unbounded or invalid batch requests.

        Args:
            limit: Requested maximum batch size.

        Raises:
            ValueError: If `limit` is not positive.
        """
        if limit <= 0:
            raise errors.InputError("limit must be positive")
