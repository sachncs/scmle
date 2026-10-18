"""Portable channel to a helper process (Linux, macOS and Windows).

The parent opens an authenticated listener (a Unix socket in a private
directory, or a named pipe on Windows) and starts the helper with the
address as its only argument; the secret key travels over the helper's
standard input, never the command line. Only a process that holds the key
can connect, so unrelated local processes cannot reach the channel.
"""

from multiprocessing import connection
import secrets
import subprocess  # nosec B404
import sys
import threading
import time

KEY_BYTES = 32
CONNECT_SECONDS = 120.0
POLL_SECONDS = 0.05


class HelperStartError(OSError):
    """Raised when a helper process exits or never connects."""


def start_helper(
    module: str,
) -> tuple[subprocess.Popen[bytes], connection.Connection]:
    """Starts `python -m module ADDRESS` and connects to it.

    Args:
        module: Module whose `__main__` calls `connect_to_parent`.

    Returns:
        The helper process and the parent end of the channel.

    Raises:
        HelperStartError: If the helper dies or does not connect in time.
    """
    key = secrets.token_bytes(KEY_BYTES)
    listener = connection.Listener(authkey=key)
    address = str(listener.address)
    process = subprocess.Popen(  # nosec B603
        [sys.executable, "-m", module, address], stdin=subprocess.PIPE
    )
    assert process.stdin is not None  # nosec B101
    process.stdin.write(key.hex().encode("ascii"))
    process.stdin.close()
    accepted: list[connection.Connection] = []
    failures: list[BaseException] = []

    def accept() -> None:
        try:
            accepted.append(listener.accept())
        except BaseException as error:  # Reported to the caller.
            failures.append(error)

    thread = threading.Thread(target=accept, daemon=True)
    thread.start()
    deadline = time.monotonic() + CONNECT_SECONDS
    try:
        while thread.is_alive():
            thread.join(POLL_SECONDS)
            if process.poll() is not None and thread.is_alive():
                raise HelperStartError(
                    f"{module} exited with code {process.returncode} "
                    "before connecting"
                )
            if time.monotonic() > deadline:
                raise HelperStartError(f"{module} did not connect in time")
    except HelperStartError:
        process.kill()
        process.wait()
        raise
    finally:
        if accepted or failures:
            listener.close()
    if failures:
        process.kill()
        process.wait()
        raise HelperStartError(f"{module} failed to connect: {failures[0]}")
    return process, accepted[0]


def connect_to_parent() -> connection.Connection:
    """Helper side: connects using the address argument and key on stdin."""
    key = bytes.fromhex(sys.stdin.buffer.read().decode("ascii"))
    return connection.Client(sys.argv[1], authkey=key)
