"""A separate process that holds the locked test rows until release.

In-process protection can only be a convention: any code running in the
orchestrator process can read any attribute. The vault is an operating-system
boundary. After the split the test rows exist only in the vault process; the
orchestrator keeps their ids. The vault hands the rows over once, to the
holder of the release token, over an authenticated channel whose key only the
parent and the vault hold, so trial processes cannot reach it.
"""

from multiprocessing import connection
import secrets
from typing import Any

import pandas

from scmle.contracts import errors
from scmle.core import ipc
from scmle.data.splitting import splitters

OPEN = "open"
HEADER_LENGTH = 2  # the operation and the presented token; extras follow
BIND = "bind"


def serve(
    channel: connection.Connection,
    token: str,
    rows: tuple[pandas.DataFrame, pandas.Series],
) -> None:
    """Runs the vault loop: bind once, release the rows once.

    Every message is a tuple of an operation, the token and an optional
    argument. A message with an unknown operation or a wrong token is
    answered `("denied", "invalid token")`.

    A `bind` message fixes a digest. From then on `open` releases the rows
    only to a caller that presents the same digest, so the release cannot be
    spent on a selection other than the one that was bound.

    Args:
        channel: Connection to the caller; the loop ends when it closes.
        token: Secret that every message must present, compared in constant
            time.
        rows: The features and target held in custody and released once.
    """
    released = False
    binding: str | None = None
    while True:
        try:
            message = channel.recv()
        except EOFError:
            return
        operation, presented = message[0], message[1]
        extra = message[2] if len(message) > HEADER_LENGTH else None
        if operation not in (OPEN, BIND) or not secrets.compare_digest(
            presented, token
        ):
            channel.send(("denied", "invalid token"))
        elif operation == BIND:
            if binding is not None or released:
                channel.send(("denied", "already bound or released"))
            else:
                binding = str(extra)
                channel.send(("ok", None))
        elif released:
            channel.send(("denied", "already released"))
        elif binding is not None and not secrets.compare_digest(
            str(extra).encode(), binding.encode()
        ):
            channel.send(("denied", "binding mismatch"))
        else:
            released = True
            channel.send(("ok", rows))


def main() -> None:
    """Runs the vault process after connecting to its parent.

    The parent passes the channel address as the first argument and the key
    on standard input, then sends one message with the token and the rows.
    """
    channel = ipc.connect_to_parent()
    token, rows = channel.recv()
    serve(channel, token, rows)


class SealedTestSet(splitters.LockedTestSet):
    """The locked test set, held by a vault process.

    Attributes:
        ids: Row ids of the test rows (the rows themselves are not here).
        token: Release token, a random hex string.
        opened: Whether the rows were released.
        process: The vault process.
        channel: The parent end of the connection to the vault.
    """

    def __init__(
        self, features: pandas.DataFrame, target: pandas.Series, ids: Any
    ) -> None:
        """Starts the vault and hands it the rows.

        Args:
            features: Test features.
            target: Test labels.
            ids: Row ids of the test rows.
        """
        self.ids = tuple(ids)
        self.token = secrets.token_hex(16)
        self.opened = False
        self.process, parent = ipc.start_helper("scmle.governance")
        self.channel = parent
        parent.send((self.token, (features, target)))

    def issue_token(self) -> str:
        """Returns the release token; call only once selection is frozen."""
        return self.token

    def bind(self, binding: str) -> None:
        """Tells the vault what the single release is for.

        Args:
            binding: Digest the vault will require at `open`.

        Raises:
            LeakageError: If the rows were already released, the vault
                refuses (already bound or released) or the vault is gone.
        """
        if self.opened:
            raise errors.LeakageError("test set is already bound or used")
        try:
            self.channel.send((BIND, self.token, binding))
            status, value = self.channel.recv()
        except (EOFError, OSError) as error:
            raise errors.LeakageError("the test-set vault is gone") from error
        if status != "ok":
            raise errors.LeakageError(f"test set is locked: {value}")

    def open(
        self, token: str, binding: str | None = None
    ) -> tuple[pandas.DataFrame, pandas.Series]:
        """Asks the vault for the rows, once.

        Args:
            token: The token from `issue_token`.
            binding: The digest given to `bind`, if one was set.

        On success the vault process is ended after the rows are received.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: If the rows were already released, the vault
                refuses (wrong token, binding mismatch) or the vault is
                gone.
        """
        if self.opened:
            raise errors.LeakageError(
                "test set was already used for final evaluation"
            )
        try:
            self.channel.send((OPEN, token, binding))
            status, value = self.channel.recv()
        except (EOFError, OSError) as error:
            raise errors.LeakageError("the test-set vault is gone") from error
        if status != "ok":
            raise errors.LeakageError(f"test set is locked: {value}")
        self.opened = True
        self.close()
        rows: tuple[pandas.DataFrame, pandas.Series] = value
        return rows

    def close(self) -> None:
        """Ends the vault process."""
        self.channel.close()
        if self.process.poll() is None:
            self.process.wait(timeout=10)
