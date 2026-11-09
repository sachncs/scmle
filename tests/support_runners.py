"""Fake time, a deterministic workload and an in-process container."""

from scmle.runners import entrypoint


class FakeClock:
    """Time that moves only when told to (or when `sleep` is called)."""

    def __init__(self):
        self.seconds = 0.0
        self.sleeps = []

    def now(self):
        return self.seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.seconds += seconds

    def advance(self, seconds):
        self.seconds += seconds


def toy_workload(clock, steps=6, seconds_per_step=60.0, log=None):
    """Converges w toward 3.0 by halving the gap each step, checkpointing."""

    def workload(context):
        found = context.resume()
        step, weight = (found.step, found.state["w"]) if found else (0, 0.0)
        if log is not None:
            log.append((context.attempt, step))
        while step < steps:
            step += 1
            weight += 0.5 * (3.0 - weight)
            clock.advance(seconds_per_step)
            context.step(step, {"w": weight})
        context.finish({"metrics": {"w": weight}, "steps": step})

    return workload


class Killed(BaseException):
    """A container that is taken away mid-run (not catchable as Exception)."""


CURRENT = [None]


def current(context):
    CURRENT[0](context)


def run_container(store, environ, workload):
    """Runs the entrypoint with `workload`; returns its exit code or Killed."""
    CURRENT[0] = workload
    env = {**environ, "SCMLE_WORKLOAD": "tests.support_runners:current"}
    try:
        return entrypoint.main(env, store)
    except Killed:
        return Killed
