"""Runs a heavy command only when the machine has room for it.

Usage:
    uv run python scripts/admit.py [options] -- COMMAND [ARG ...]

The command starts when all of these hold:
* the one-minute load average is below `--load`;
* free memory is above `--memory` gigabytes;
* fewer than `--slots` other admitted commands are running.

While it waits it polls every `--poll` seconds and never starts the command
early. Math-library thread counts are set to 1 unless `--threads` says
otherwise, so one job cannot occupy every core. The slot is released when the
command ends, including on interrupt. A slot left by a dead process is
reclaimed.

Exit status is the command's own, 75 if `--timeout` passes while waiting, and
2 for a usage error.
"""

import argparse
import os
import pathlib
import subprocess  # nosec B404
import sys
import tempfile
import time

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
WAITED_TOO_LONG = 75
CLAIM_GRACE = 5.0
BYTES_PER_GIGABYTE = 1024**3


def parse(arguments: list[str]) -> argparse.Namespace:
    """Parses the command line.

    Args:
        arguments: Arguments after the program name.

    Returns:
        The parsed options; `command` holds what follows `--`.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--load", type=float, default=8.0)
    parser.add_argument("--memory", type=float, default=4.0)
    parser.add_argument("--slots", type=int, default=2)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--poll", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    options = parser.parse_args(arguments)
    if options.command[:1] == ["--"]:
        options.command = options.command[1:]
    if not options.command:
        parser.error("no command given after --")
    return options


def measure() -> float:
    """Measures memory available to new work.

    Returns:
        Free plus reclaimable memory in gigabytes (available memory on
        Windows and Linux), or infinity when the platform offers no cheap way
        to read it.
    """
    if sys.platform == "darwin":
        report = subprocess.run(  # nosec B603 B607
            ["vm_stat"], capture_output=True, text=True, check=True
        ).stdout
        lines = report.splitlines()
        size = int(lines[0].split("page size of")[1].split()[0])
        pages = 0
        for line in lines[1:]:
            name, _, count = line.partition(":")
            if name in ("Pages free", "Pages inactive", "Pages speculative"):
                pages += int(count.strip().rstrip("."))
        return pages * size / BYTES_PER_GIGABYTE
    if sys.platform == "win32":
        import psutil  # Windows-only dependency

        return float(psutil.virtual_memory().available) / BYTES_PER_GIGABYTE
    meminfo = pathlib.Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024 / BYTES_PER_GIGABYTE
    return float("inf")


def living(pid: int) -> bool:
    """Whether a process with this id exists.

    Args:
        pid: Process id.

    Returns:
        True if the process exists.
    """
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def claim(root: pathlib.Path, slots: int) -> pathlib.Path | None:
    """Takes a free slot, reclaiming slots whose owner has died.

    Args:
        root: Directory that holds the slot directories.
        slots: Number of slots available.

    Returns:
        The claimed slot directory, or None when all slots are taken.
    """
    root.mkdir(parents=True, exist_ok=True)
    for number in range(slots):
        slot = root / f"slot{number}"
        try:
            slot.mkdir()
        except FileExistsError:
            owner = slot / "pid"
            try:
                if living(int(owner.read_text(encoding="utf-8"))):
                    continue
            except (OSError, ValueError):
                try:
                    young = time.time() - slot.stat().st_mtime < CLAIM_GRACE
                except OSError:
                    continue
                if young:
                    continue
            owner.unlink(missing_ok=True)
            try:
                slot.rmdir()
                slot.mkdir()
            except OSError:
                continue
        try:
            (slot / "pid").write_text(str(os.getpid()), encoding="utf-8")
        except OSError:
            continue
        return slot
    return None


def release(slot: pathlib.Path) -> None:
    """Gives a slot back.

    Args:
        slot: The slot directory returned by `claim`.
    """
    (slot / "pid").unlink(missing_ok=True)
    slot.rmdir()


def wait(
    options: argparse.Namespace, root: pathlib.Path
) -> pathlib.Path | None:
    """Waits until the machine has room, then claims a slot.

    Args:
        options: Parsed options.
        root: Directory that holds the slot directories.

    Returns:
        The claimed slot, or None if the timeout passed first.
    """
    deadline = time.monotonic() + options.timeout
    while time.monotonic() < deadline:
        roomy = os.getloadavg()[0] < options.load and measure() > options.memory
        if roomy:
            slot = claim(root, options.slots)
            if slot is not None:
                return slot
        time.sleep(options.poll)
    return None


def main() -> int:
    """Waits for room, runs the command, and returns its exit status.

    Returns:
        The command's exit status, or a code from the module docstring.
    """
    options = parse(sys.argv[1:])
    root = pathlib.Path(tempfile.gettempdir()) / "scmle-admit"
    slot = wait(options, root)
    if slot is None:
        print("admit: timed out waiting for room", file=sys.stderr)
        return WAITED_TOO_LONG
    environment = dict(os.environ)
    for name in THREAD_VARIABLES:
        environment.setdefault(name, str(options.threads))
    try:
        return subprocess.run(  # nosec B603
            options.command, env=environment, check=False
        ).returncode
    finally:
        release(slot)


if __name__ == "__main__":
    sys.exit(main())
