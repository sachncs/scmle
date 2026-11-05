"""Fake time, a deterministic workload and scripted ECS and Cloud Run clients."""

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


def ecs_container(store, workload):
    def run(environ):
        outcome = run_container(store, environ, workload)
        if outcome is Killed:
            return stopped(
                137, "SpotInterruption", "Your Spot Task was interrupted"
            )
        return stopped(outcome)

    return run


def run_v2_container(store, workload):
    def run(environ):
        outcome = run_container(store, environ, workload)
        if outcome is Killed:
            return {
                "failed_count": 1,
                "completion_time": "2026-01-01T00:00:00Z",
                "conditions": [
                    {"reason": "Preempted", "message": "SIGTERM received"}
                ],
            }
        if outcome == 0:
            return {"succeeded_count": 1}
        return {
            "failed_count": 1,
            "conditions": [{"reason": "Failed", "message": "exit(1)"}],
        }

    return run


class FakeEcs:
    """Scripted ECS: each `run_task` runs the container in process."""

    def __init__(self, store, containers, clock=None):
        self.store = store
        self.containers = list(containers)
        self.clock = clock
        self.requests = []
        self.stopped = []
        self.tasks = {}
        self.placement_failures = []

    def run_task(self, **request):
        self.requests.append(request)
        if self.placement_failures:
            return {
                "tasks": [],
                "failures": [{"reason": self.placement_failures.pop(0)}],
            }
        arn = f"arn:aws:ecs:task/{len(self.requests)}"
        env = {
            item["name"]: item["value"]
            for item in request["overrides"]["containerOverrides"][0][
                "environment"
            ]
        }
        container = self.containers.pop(0)
        self.tasks[arn] = {"env": env, "container": container, "polls": 0}
        return {"tasks": [{"taskArn": arn}], "failures": []}

    def describe_tasks(self, cluster, tasks):
        task = self.tasks[tasks[0]]
        task["polls"] += 1
        if task["polls"] == 1:
            return {"tasks": [{"lastStatus": "PENDING"}]}
        if task["polls"] == 2:
            return {"tasks": [{"lastStatus": "RUNNING"}]}
        if "done" not in task:
            task["done"] = task["container"](task["env"])
        return {"tasks": [task["done"]]}

    def stop_task(self, cluster, task, reason):
        self.stopped.append((task, reason))
        self.tasks[task]["done"] = {
            "lastStatus": "STOPPED",
            "stopCode": "UserInitiated",
            "stoppedReason": reason,
            "containers": [{"exitCode": 137}],
        }


def stopped(exit_code=0, code="EssentialContainerExited", reason=""):
    return {
        "lastStatus": "STOPPED",
        "stopCode": code,
        "stoppedReason": reason,
        "containers": [{"exitCode": exit_code}],
    }


class FakeCloudRun:
    """Scripted Cloud Run jobs client."""

    def __init__(self, containers):
        self.containers = list(containers)
        self.started = []
        self.cancelled = []
        self.executions = {}

    def run_job(self, name, overrides):
        self.started.append((name, overrides))
        execution = f"{name}/executions/{len(self.started)}"
        env = {
            item["name"]: item["value"]
            for item in overrides["container_overrides"][0]["env"]
        }
        self.executions[execution] = {
            "env": env,
            "container": self.containers.pop(0),
            "polls": 0,
        }
        return execution

    def get_execution(self, name):
        item = self.executions[name]
        item["polls"] += 1
        if item["polls"] == 1:
            return {"running_count": 0}
        if item["polls"] == 2:
            return {"running_count": 1}
        if "done" not in item:
            item["done"] = item["container"](item["env"])
        return item["done"]

    def cancel_execution(self, name):
        self.cancelled.append(name)
        self.executions[name]["done"] = {"cancelled_count": 1}
