"""Child process for the resilience tests.

Usage:
    failpoint_child.py kill ROOT LABEL PHASE INDEX gate|plain
    failpoint_child.py hold ROOT MARKER RELEASE

`kill` runs the processor over the store at ROOT and kills itself with
SIGKILL at the named boundary. `hold` runs it with a transform that creates
MARKER and then waits until RELEASE exists, so the lock stays held.
"""

import pathlib
import sys
import time

import failpoints

from scmle.data.store import objectstore
from scmle.data.store import processing

HOLD_SECONDS = 120


def kill_run(root, label, phase, index, gated):
    inner = objectstore.LocalObjectStore(root)
    trigger = failpoints.Trigger(label, phase, index, failpoints.kill_process)
    store = failpoints.FailpointStore(inner, trigger)
    failpoints.processor(store, gated).run()


def hold_run(root, marker, release):
    store = objectstore.LocalObjectStore(root)

    def waiting(frame):
        pathlib.Path(marker).write_text("held")
        deadline = time.monotonic() + HOLD_SECONDS
        while not pathlib.Path(release).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("release never came")
            time.sleep(0.02)
        return failpoints.upper(frame)

    processor = processing.IncrementalProcessor(
        store, failpoints.CONTRACT, waiting
    )
    processor.run()


def main(arguments):
    mode = arguments[0]
    if mode == "kill":
        root, label, phase, index, kind = arguments[1:6]
        kill_run(root, label, phase, int(index), kind == "gate")
    else:
        hold_run(*arguments[1:4])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
