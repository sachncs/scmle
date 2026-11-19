"""Workloads a job process imports by name (`module:function`)."""

import signal
import time

from scmle.contracts import errors
from scmle.runtime import retry


def count(context):
    """Counts to `project.steps`, checkpointing each step; resumes."""
    found = context.resume()
    step = found.step if found else 0
    total = int(context.spec.project.get("steps", 3))
    pause = float(context.spec.project.get("pause", 0.0))
    while step < total:
        step += 1
        time.sleep(pause)
        context.step(step, {"n": step})
    context.finish(
        {
            "steps": step,
            "attempt": context.attempt,
            "resumed": found.step if found else 0,
        }
    )


def flaky(context):
    """Fails transiently on the first attempt, then succeeds."""
    if context.attempt == 1:
        raise retry.TransientError("throttled")
    context.finish({"attempt": context.attempt})


def broken(context):
    """Fails for good."""
    raise errors.InputError("bad column")


def stubborn(context):
    """Never ends on its own; with `project.deaf`, ignores SIGTERM too."""
    if context.spec.project.get("deaf"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    context.step(1, {"n": 1})
    while True:
        time.sleep(0.05)
