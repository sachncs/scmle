"""A separate process that holds the locked test rows until release.

In-process protection can only be a convention: any code running in the
orchestrator process can read any attribute. The vault is an operating-system
boundary. After the split the test rows exist only in the vault process; the
orchestrator keeps their ids. The vault hands the rows over once, to the
holder of the release token, over a pipe whose file descriptor is not
inheritable, so trial processes cannot reach it.
"""

import multiprocessing
from multiprocessing import connection
import secrets
import subprocess  # nosec B404
import sys
from typing import Any

import pandas as pd

from mml.contracts import errors
from mml.data.splitting import splitters

OPEN = "open"


def serve(
    channel: connection.Connection,
    token: str,
    rows: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Vault main loop: release the rows once, on the right token."""
    released = False
    while True:
        try:
            operation, presented = channel.recv()
        except EOFError:
            return
        if operation != OPEN or not secrets.compare_digest(presented, token):
            channel.send(("denied", "invalid token"))
        elif released:
            channel.send(("denied", "already released"))
        else:
            released = True
            channel.send(("ok", rows))


def main() -> None:
    """Entry point of the vault process.

    The parent passes the pipe as an inherited file descriptor (first
    argument) and sends one message with the token and the rows.
    """
    channel = connection.Connection(int(sys.argv[1]))
    token, rows = channel.recv()
    serve(channel, token, rows)


class SealedTestSet(splitters.LockedTestSet):
    """The locked test set, held by a vault process.

    Attributes:
        ids: Row ids of the test rows (the rows themselves are not here).
        token: Release token (excluded from repr).
        opened: Whether the rows were released.
    """

    def __init__(
        self, features: pd.DataFrame, target: pd.Series, ids: Any
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
        parent, child = multiprocessing.Pipe(duplex=True)
        self.process = subprocess.Popen(  # nosec B603
            [
                sys.executable,
                "-m",
                "mml.governance",
                str(child.fileno()),
            ],
            pass_fds=(child.fileno(),),
        )
        child.close()
        self.channel = parent
        parent.send((self.token, (features, target)))

    def issue_token(self) -> str:
        """Returns the release token; call only once selection is frozen."""
        return self.token

    def open(self, token: str) -> tuple[pd.DataFrame, pd.Series]:
        """Asks the vault for the rows, once.

        Args:
            token: The token from `issue_token`.

        Returns:
            Test features and labels.

        Raises:
            LeakageError: On a wrong token, a repeated release, or a vault
                that is gone.
        """
        if self.opened:
            raise errors.LeakageError(
                "test set was already used for final evaluation"
            )
        try:
            self.channel.send((OPEN, token))
            status, value = self.channel.recv()
        except (EOFError, OSError) as error:
            raise errors.LeakageError("the test-set vault is gone") from error
        if status != "ok":
            raise errors.LeakageError(f"test set is locked: {value}")
        self.opened = True
        self.close()
        return value

    def close(self) -> None:
        """Ends the vault process."""
        self.channel.close()
        if self.process.poll() is None:
            self.process.wait(timeout=10)
