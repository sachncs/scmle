"""AWS Step Functions and GCP Cloud Workflows definitions, as data.

`to_step_functions` and `to_cloud_workflows` turn a `Workflow` into the JSON
documents those services accept, and `validate_step_functions` and
`validate_cloud_workflows` check the structure of such a document without
calling any service. They have not been deployed, so validity here means
"well-formed against the documented structure", not "accepted by the
service".

What is expressed and what is not:

* Dependencies become layers (a `Parallel` state or a `Workflows` `parallel`
  block per layer of independent steps; a layer of one is a plain task).
* Retries, with backoff and attempts, and timeouts are native fields.
* An approval step waits for a callback (`waitForTaskToken` in Step
  Functions, `events.await_callback` in Cloud Workflows) bounded by its
  expiry, and a rejection or timeout goes to a failure state, so the
  approved steps after it never run.
* The idempotency key is the execution name: Step Functions refuses a second
  execution with the same name and input, and the Cloud Workflows documents
  take it as the `execution_key` argument for the caller to enforce.
* Concurrency limits, cost budgets and groups cannot be written into either
  language. They are returned beside the definition under ``"controls"`` for
  the infrastructure (scheduler concurrency, billing alarms) to enforce; the
  documents themselves do not enforce them.
"""

from collections.abc import Mapping
from collections.abc import Sequence
from typing import Any

from scmle.contracts import errors
from scmle.workflow import definition

ASL_STATE_TYPES = frozenset(
    {
        "Task",
        "Pass",
        "Choice",
        "Wait",
        "Succeed",
        "Fail",
        "Parallel",
        "Map",
    }
)
TASK_ARN = "arn:aws:states:::ecs:runTask.sync"
CALLBACK_ARN = "arn:aws:states:::sqs:sendMessage.waitForTaskToken"
RETRY_FIELDS = frozenset(
    {
        "ErrorEquals",
        "IntervalSeconds",
        "MaxAttempts",
        "BackoffRate",
        "MaxDelaySeconds",
    }
)
FAIL_STATE = "RunFailed"
SUCCESS_STATE = "RunSucceeded"


def retry_clause(step: definition.Step) -> dict[str, Any]:
    """The Step Functions `Retry` entry for a step's policy."""
    policy = step.retries
    entry: dict[str, Any] = {
        "ErrorEquals": ["States.TaskFailed", "States.Timeout"],
        "IntervalSeconds": max(1, int(policy.base_delay_seconds)),
        "MaxAttempts": policy.max_attempts - 1,
        "BackoffRate": policy.factor,
    }
    if policy.max_delay_seconds:
        entry["MaxDelaySeconds"] = max(1, int(policy.max_delay_seconds))
    return entry


def task_state(
    step: definition.Step,
    resources: Mapping[str, str],
    next_state: str | None,
    *,
    catch: bool = True,
) -> dict[str, Any]:
    """One Step Functions state for a step.

    Args:
        step: The step.
        resources: ARNs by name (see `to_step_functions`).
        next_state: The following state, or None to end the scope.
        catch: Whether to route failures to the failure state; a state in a
            `Parallel` branch cannot name a state outside its branch, so the
            enclosing `Parallel` state catches instead.
    """
    if step.kind is definition.StepKind.APPROVAL:
        state: dict[str, Any] = {
            "Type": "Task",
            "Resource": CALLBACK_ARN,
            "Comment": "waits for a signed approval; approvers: "
            + ", ".join(step.approvers),
            "Parameters": {
                "QueueUrl": resources.get(step.name, "APPROVAL_QUEUE_URL"),
                "MessageBody": {
                    "step": step.name,
                    "token.$": "$$.Task.Token",
                    "execution.$": "$$.Execution.Name",
                },
            },
            "TimeoutSeconds": int(step.expires_seconds),
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": FAIL_STATE,
                    "ResultPath": "$.error",
                }
            ],
        }
    else:
        state = {
            "Type": "Task",
            "Resource": TASK_ARN,
            "Comment": f"{step.kind.value}; effect: {step.effect}",
            "Parameters": {
                "Cluster": resources.get("cluster", "CLUSTER_ARN"),
                "TaskDefinition": resources.get(
                    step.name, "TASK_DEFINITION_ARN"
                ),
                "Overrides": {
                    "ContainerOverrides": [
                        {
                            "Name": "step",
                            "Environment": [
                                {"Name": "SCMLE_STEP", "Value": step.name},
                                {
                                    "Name": "SCMLE_IDEMPOTENCY_KEY",
                                    "Value.$": "$$.Execution.Name",
                                },
                            ],
                        }
                    ]
                },
            },
            "TimeoutSeconds": int(step.timeout_seconds),
            "Retry": [retry_clause(step)],
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": FAIL_STATE,
                    "ResultPath": "$.error",
                }
            ],
        }
    if not catch:
        state.pop("Catch", None)
    if next_state is None:
        state["End"] = True
    else:
        state["Next"] = next_state
    return state


def to_step_functions(
    workflow: definition.Workflow, resources: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The Step Functions document and the controls it cannot carry.

    Args:
        workflow: The definition.
        resources: Optional ARNs: ``cluster`` and, per step name, the task
            definition (or queue URL for approvals). Placeholders otherwise.

    Returns:
        ``{"definition": <ASL>, "schedule": <expression>|None,
        "controls": {...}}``.
    """
    resources = resources or {}
    layers = workflow.layers()
    states: dict[str, Any] = {}
    names = [f"Layer{index}" for index in range(len(layers))]
    for index, layer in enumerate(layers):
        following = (
            names[index + 1] if index + 1 < len(layers) else SUCCESS_STATE
        )
        if len(layer) == 1:
            step = workflow.step(layer[0])
            states[names[index]] = task_state(step, resources, following)
            continue
        branches = []
        for name in layer:
            step = workflow.step(name)
            branches.append(
                {
                    "StartAt": name,
                    "States": {
                        name: task_state(step, resources, None, catch=False)
                    },
                }
            )
        states[names[index]] = {
            "Type": "Parallel",
            "Branches": branches,
            "Next": following,
            "Catch": [
                {
                    "ErrorEquals": ["States.ALL"],
                    "Next": FAIL_STATE,
                    "ResultPath": "$.error",
                }
            ],
        }
    states[SUCCESS_STATE] = {"Type": "Succeed"}
    states[FAIL_STATE] = {
        "Type": "Fail",
        "Error": "WorkflowFailed",
        "Cause": "a step failed, was rejected or expired",
    }
    return {
        "definition": {
            "Comment": f"{workflow.name} v{workflow.version} "
            f"({workflow.digest()[:12]}); the execution name is the "
            "idempotency key",
            "StartAt": names[0],
            "States": states,
        },
        "schedule": None
        if workflow.schedule is None
        else eventbridge_rate(workflow.schedule.interval_seconds),
        "controls": controls_of(workflow),
    }


def controls_of(workflow: definition.Workflow) -> dict[str, Any]:
    """What the infrastructure must enforce because the language cannot."""
    return {
        "max_concurrent_runs": workflow.max_concurrent_runs,
        "run_cost_budget": workflow.cost_budget,
        "step_cost_budgets": {
            s.name: s.cost_budget
            for s in workflow.steps
            if s.cost_budget is not None
        },
        "concurrency_groups": {
            s.name: [
                s.concurrency_group,
                workflow.group_limit(s.concurrency_group),
            ]
            for s in workflow.steps
            if s.concurrency_group
        },
        "approved_steps": [
            s.name for s in workflow.steps if s.effect == definition.APPROVED
        ],
    }


def eventbridge_rate(seconds: float) -> str:
    """An EventBridge Scheduler `rate()` expression (whole minutes or more).

    Raises:
        ConfigError: If the interval is not a whole number of minutes.
    """
    if seconds < 60 or seconds % 60:
        raise errors.ConfigError("EventBridge rates are whole minutes")
    minutes = int(seconds // 60)
    for unit, size in (("day", 1440), ("hour", 60), ("minute", 1)):
        if minutes % size == 0:
            count = minutes // size
            return f"rate({count} {unit}{'' if count == 1 else 's'})"
    raise errors.ConfigError("unreachable")


def cron_for(seconds: float) -> str:
    """A Cloud Scheduler cron expression for an interval.

    Raises:
        ConfigError: If the interval cannot be written as one cron line
            (it must divide the minute, hour or day field evenly).
    """
    if seconds < 60 or seconds % 60:
        raise errors.ConfigError("cron schedules are whole minutes")
    minutes = int(seconds // 60)
    if minutes < 60 and 60 % minutes == 0:
        return f"*/{minutes} * * * *"
    if minutes % 60 == 0 and minutes < 1440 and 24 % (minutes // 60) == 0:
        return f"0 */{minutes // 60} * * *"
    if minutes % 1440 == 0 and minutes // 1440 <= 28:
        return f"0 0 */{minutes // 1440} * *" if minutes > 1440 else "0 0 * * *"
    raise errors.ConfigError(
        f"{seconds}s cannot be written as a single cron expression"
    )


def validate_step_functions(
    document: Mapping[str, Any], *, nested: bool = False
) -> list[str]:
    """Structural problems in a Step Functions definition (empty if none).

    Checks: required fields, state types, that `StartAt`, `Next`, `Default`
    and `Catch` targets exist in the same scope, that every state is
    reachable, that a non-terminal state has `Next` or `End`, task fields,
    retry entries, and branches of parallel states recursively.
    """
    problems: list[str] = []
    states = document.get("States")
    start = document.get("StartAt")
    if not isinstance(states, dict) or not states:
        return ["States must be a non-empty object"]
    if start not in states:
        problems.append(f"StartAt {start!r} is not a state")
    reached: set[str] = set()
    pending: list[str] = [str(start)] if start in states else []
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        state = states[name]
        pending.extend(
            target
            for target in successors(state)
            if isinstance(target, str) and target in states
        )
    for name, state in states.items():
        label = f"state {name}"
        if (
            not isinstance(state, dict)
            or state.get("Type") not in ASL_STATE_TYPES
        ):
            problems.append(f"{label}: unknown or missing Type")
            continue
        kind = state["Type"]
        for target in successors(state):
            if target not in states:
                problems.append(f"{label}: target {target!r} does not exist")
        if name not in reached:
            problems.append(f"{label}: unreachable")
        terminal = kind in ("Succeed", "Fail", "Choice")
        if not terminal and ("Next" in state) == bool(state.get("End")):
            problems.append(f"{label}: needs exactly one of Next and End")
        if kind == "Task" and not str(state.get("Resource", "")).startswith(
            "arn:"
        ):
            problems.append(f"{label}: Task needs an ARN Resource")
        timeout = state.get("TimeoutSeconds", 1)
        if kind == "Task" and (not isinstance(timeout, int) or timeout < 1):
            problems.append(
                f"{label}: TimeoutSeconds must be a positive integer"
            )
        for entry in state.get("Retry", []):
            problems += retry_problems(label, entry)
        if kind == "Parallel":
            for index, branch in enumerate(state.get("Branches", [])):
                problems += [
                    f"{label} branch {index}: {p}"
                    for p in validate_step_functions(branch, nested=True)
                ]
            if not state.get("Branches"):
                problems.append(f"{label}: Parallel needs Branches")
    if not nested and not any(
        s.get("Type") in ("Succeed", "Fail") for s in states.values()
    ):
        problems.append("no terminal Succeed or Fail state")
    return problems


def successors(state: Mapping[str, Any]) -> list[Any]:
    """Names a state can transfer to in its own scope."""
    found: list[Any] = []
    for field in ("Next", "Default"):
        if field in state:
            found.append(state[field])
    for choice in state.get("Choices", []):
        if "Next" in choice:
            found.append(choice["Next"])
    for catch in state.get("Catch", []):
        if "Next" in catch:
            found.append(catch["Next"])
    return found


def retry_problems(label: str, entry: Mapping[str, Any]) -> list[str]:
    """Problems in one `Retry` entry."""
    problems = []
    unknown = set(entry) - RETRY_FIELDS
    if unknown:
        problems.append(f"{label}: unknown Retry fields {sorted(unknown)}")
    if not entry.get("ErrorEquals"):
        problems.append(f"{label}: Retry needs ErrorEquals")
    if entry.get("MaxAttempts", 0) < 0:
        problems.append(f"{label}: MaxAttempts must not be negative")
    if entry.get("IntervalSeconds", 1) < 1:
        problems.append(f"{label}: IntervalSeconds must be at least 1")
    if entry.get("BackoffRate", 1.0) < 1.0:
        problems.append(f"{label}: BackoffRate must be at least 1.0")
    return problems


# ---- Cloud Workflows -------------------------------------------------------

JOBS_RUN = "googleapis.run.v2.projects.locations.jobs.run"
AWAIT_CALLBACK = "events.await_callback"
PUBSUB_PUBLISH = "googleapis.pubsub.v1.projects.topics.publish"
CREATE_CALLBACK = "events.create_callback_endpoint"
WORKFLOW_KEYS = frozenset(
    {
        "call",
        "assign",
        "switch",
        "parallel",
        "try",
        "raise",
        "return",
        "for",
        "steps",
    }
)


def retry_block(step: definition.Step) -> dict[str, Any]:
    """The Cloud Workflows `retry` block for a step's policy."""
    policy = step.retries
    return {
        "predicate": "${retry_predicate}",
        "max_retries": policy.max_attempts - 1,
        "backoff": {
            "initial_delay": max(1.0, policy.base_delay_seconds),
            "max_delay": max(1.0, policy.max_delay_seconds),
            "multiplier": policy.factor,
        },
    }


def workflow_step(
    step: definition.Step, resources: Mapping[str, str]
) -> dict[str, Any]:
    """The Cloud Workflows step (a `{name: body}` mapping) for a step."""
    if step.kind is definition.StepKind.APPROVAL:
        reply = f"{step.name}_reply"
        callback_url = f"${{{step.name}_callback.url}}"
        rejected = f'${{{reply}.http_request.body.decision != "approve"}}'
        return {
            step.name: {
                "steps": [
                    {
                        f"{step.name}_endpoint": {
                            "call": CREATE_CALLBACK,
                            "args": {"http_callback_method": "POST"},
                            "result": f"{step.name}_callback",
                        }
                    },
                    {
                        f"{step.name}_notify": {
                            "call": PUBSUB_PUBLISH,
                            "args": {
                                "topic": resources.get(step.name, "TOPIC"),
                                "body": {
                                    "messages": [
                                        {
                                            "attributes": {
                                                "callback": callback_url,
                                                "step": step.name,
                                            }
                                        }
                                    ]
                                },
                            },
                        }
                    },
                    {
                        f"{step.name}_wait": {
                            "call": AWAIT_CALLBACK,
                            "args": {
                                "callback": f"${{{step.name}_callback}}",
                                "timeout": int(step.expires_seconds),
                            },
                            "result": reply,
                        }
                    },
                    {
                        f"{step.name}_decide": {
                            "switch": [
                                {
                                    "condition": rejected,
                                    "raise": "approval rejected",
                                }
                            ]
                        }
                    },
                ]
            }
        }
    call = {
        "call": JOBS_RUN,
        "args": {
            "name": resources.get(step.name, "JOB_RESOURCE_NAME"),
            "body": {
                "overrides": {
                    "containerOverrides": [
                        {
                            "env": [
                                {"name": "SCMLE_STEP", "value": step.name},
                                {
                                    "name": "SCMLE_IDEMPOTENCY_KEY",
                                    "value": "${execution_key}",
                                },
                            ]
                        }
                    ],
                    "timeout": f"{int(step.timeout_seconds)}s",
                }
            },
        },
        "result": f"{step.name}_result",
    }
    return {
        step.name: {
            "try": {"steps": [{f"{step.name}_call": call}]},
            "retry": retry_block(step),
            "except": {
                "as": "error",
                "steps": [{f"{step.name}_failed": {"raise": "${error}"}}],
            },
        }
    }


def to_cloud_workflows(
    workflow: definition.Workflow, resources: Mapping[str, str] | None = None
) -> dict[str, Any]:
    """The Cloud Workflows document and the controls it cannot carry.

    Returns:
        ``{"definition": {...}, "schedule": <cron>|None, "controls": {...}}``.
    """
    resources = resources or {}
    steps: list[dict[str, Any]] = [
        {"init": {"assign": [{"execution_key": "${args.execution_key}"}]}}
    ]
    for layer in workflow.layers():
        if len(layer) == 1:
            steps.append(workflow_step(workflow.step(layer[0]), resources))
            continue
        branches = [
            {
                f"branch_{name}": {
                    "steps": [workflow_step(workflow.step(name), resources)]
                }
            }
            for name in layer
        ]
        steps.append(
            {"layer_" + "_".join(layer): {"parallel": {"branches": branches}}}
        )
    steps.append({"finish": {"return": "${execution_key}"}})
    return {
        "definition": {"main": {"params": ["args"], "steps": steps}},
        "schedule": None
        if workflow.schedule is None
        else cron_for(workflow.schedule.interval_seconds),
        "controls": controls_of(workflow),
    }


def step_names(steps: Sequence[Mapping[str, Any]]) -> list[str]:
    """Every step name in a list of Cloud Workflows steps, nested included."""
    names: list[str] = []
    for item in steps:
        for name, body in item.items():
            names.append(name)
            if not isinstance(body, dict):
                continue
            names += step_names(body.get("steps", []))
            if isinstance(body.get("try"), dict):
                names += step_names(body["try"].get("steps", []))
            if isinstance(body.get("except"), dict):
                names += step_names(body["except"].get("steps", []))
            for branch in (body.get("parallel") or {}).get("branches", []):
                names += step_names([branch])
    return names


def validate_cloud_workflows(document: Mapping[str, Any]) -> list[str]:
    """Structural problems in a Cloud Workflows definition (empty if none).

    Checks: a `main` workflow with `steps`, each step a single-key mapping
    whose body uses a known keyword, unique step names, `next` and
    `switch` targets that exist, `parallel` branches that hold steps, retry
    values that are positive and `call` steps that name a function.
    """
    main = document.get("main")
    if not isinstance(main, dict) or not isinstance(main.get("steps"), list):
        return ["main.steps must be a list"]
    problems: list[str] = []
    names = step_names(main["steps"])
    for name in sorted({n for n in names if names.count(n) > 1}):
        problems.append(f"step name {name!r} is used more than once")
    known = set(names)
    problems += check_workflow_steps(main["steps"], known)
    return problems


def check_workflow_steps(
    steps: Sequence[Mapping[str, Any]], known: set[str]
) -> list[str]:
    """Problems in a list of steps, recursively."""
    problems: list[str] = []
    for item in steps:
        if not isinstance(item, dict) or len(item) != 1:
            problems.append("each step must be a mapping with one name")
            continue
        ((name, body),) = item.items()
        if not isinstance(body, dict):
            problems.append(f"step {name}: body must be a mapping")
            continue
        if not WORKFLOW_KEYS & set(body):
            problems.append(
                f"step {name}: no call, assign, switch, ... keyword"
            )
        if "call" in body and not (
            isinstance(body["call"], str) and body["call"]
        ):
            problems.append(f"step {name}: call must name a function")
        for target in [body.get("next")] + [
            c.get("next") for c in body.get("switch", []) if isinstance(c, dict)
        ]:
            if (
                target is not None
                and target not in known
                and target not in ("end",)
            ):
                problems.append(f"step {name}: next {target!r} does not exist")
        retry = body.get("retry")
        if isinstance(retry, dict):
            if retry.get("max_retries", 0) < 0:
                problems.append(
                    f"step {name}: max_retries must not be negative"
                )
            backoff = retry.get("backoff", {})
            if (
                backoff.get("multiplier", 1) < 1
                or backoff.get("initial_delay", 1) <= 0
            ):
                problems.append(f"step {name}: backoff values must be positive")
        if "parallel" in body:
            branches = body["parallel"].get("branches")
            if not branches:
                problems.append(f"step {name}: parallel needs branches")
            for branch in branches or []:
                for branch_name, branch_body in branch.items():
                    inner = branch_body.get("steps")
                    if not inner:
                        problems.append(f"branch {branch_name}: needs steps")
                    else:
                        problems += check_workflow_steps(inner, known)
        for nested in (
            body.get("steps"),
            (body.get("try") or {}).get("steps")
            if isinstance(body.get("try"), dict)
            else None,
            (body.get("except") or {}).get("steps")
            if isinstance(body.get("except"), dict)
            else None,
        ):
            if nested:
                problems += check_workflow_steps(nested, known)
    return problems
