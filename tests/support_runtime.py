"""Stateful stand-ins for the ECS and Cloud Run APIs, built on recorded JSON.

The recorded answers in tests/fixtures/cloud are shaped by the documented
responses of DescribeServices, DescribeTasks, DescribeTaskDefinition and the
Cloud Run Admin API v2. These fakes change those answers when asked to deploy
or move traffic, settle after a number of reads (or never), and can run a
different image than the one requested. They are not a model of either
service; they let the control-plane logic be tested for what it does when the
platform lags, refuses or disagrees.
"""

# ruff: noqa: N803  (argument names follow the AWS SDK)

import copy
import dataclasses
import json
import pathlib

from scmle.deploy import control
from scmle.deploy import ecs
from scmle.governance import registry
from tests import support_release as sr

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "cloud"
OLD = "sha256:" + "a" * 64
NEW = "sha256:" + "b" * 64
OTHER = "sha256:" + "9" * 64


def load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


NO_WAIT = control.ConfirmPolicy(attempts=4, delay=0, sleep=lambda seconds: None)


class FakeEcs:
    """A one-service ECS cluster that rolls deployments."""

    def __init__(self, settle_after=2, never_settles=False, runs=None):
        self.services = load("ecs-describe-services")
        self.template = load("ecs-describe-task-definition")
        self.tasks = load("ecs-describe-tasks")
        self.family = "scmle-staging-serve"
        self.digests = {12: OLD}
        self.primary = 12
        self.active = None
        self.settled = True
        self.reads = 0
        self.settle_after = settle_after
        self.never_settles = never_settles
        self.runs = runs
        self.calls = []

    def arn(self, number):
        return (
            self.template["taskDefinition"]["taskDefinitionArn"].rsplit(":", 1)[
                0
            ]
            + f":{number}"
        )

    def number(self, reference):
        return int(reference.rsplit(":", 1)[-1])

    def tick(self):
        if self.settled:
            return
        self.reads += 1
        if self.reads >= self.settle_after and not self.never_settles:
            self.settled = True
            self.active = None

    def describe_services(self, **kwargs):
        self.calls.append("describe_services")
        self.tick()
        document = copy.deepcopy(self.services)
        service = document["services"][0]
        base = service["deployments"][0]

        def deployment(status, number, running, state):
            item = copy.deepcopy(base)
            item.update(
                status=status,
                taskDefinition=self.arn(number),
                runningCount=running,
                pendingCount=0 if running == 2 else 2 - running,
                rolloutState=state,
            )
            return item

        if self.settled:
            service["deployments"] = [
                deployment("PRIMARY", self.primary, 2, "COMPLETED")
            ]
        else:
            service["deployments"] = [
                deployment("PRIMARY", self.primary, 1, "IN_PROGRESS"),
                deployment("ACTIVE", self.active, 1, "COMPLETED"),
            ]
        service["taskDefinition"] = self.arn(self.primary)
        service["runningCount"] = 2
        return document

    def running_numbers(self):
        if self.settled:
            return [self.primary]
        return [self.primary, self.active]

    def list_tasks(self, **kwargs):
        self.calls.append("list_tasks")
        return {"taskArns": [f"task-{n}" for n in self.running_numbers()]}

    def describe_tasks(self, **kwargs):
        self.calls.append("describe_tasks")
        out = []
        for number in self.running_numbers():
            task = copy.deepcopy(self.tasks["tasks"][0])
            task["taskDefinitionArn"] = self.arn(number)
            digest = self.digests[number]
            task["containers"][0]["imageDigest"] = digest
            out.append(task)
        return {"tasks": out, "failures": []}

    def describe_task_definition(self, taskDefinition):
        self.calls.append("describe_task_definition")
        number = self.number(taskDefinition)
        body = copy.deepcopy(self.template)
        body["taskDefinition"]["taskDefinitionArn"] = self.arn(number)
        body["taskDefinition"]["revision"] = number
        image = body["taskDefinition"]["containerDefinitions"][0]["image"]
        body["taskDefinition"]["containerDefinitions"][0]["image"] = (
            image.split("@")[0] + "@" + self.digests[number]
        )
        return body

    def register_task_definition(self, **request):
        self.calls.append("register_task_definition")
        number = max(self.digests) + 1
        image = request["containerDefinitions"][0]["image"]
        requested = image.split("@", 1)[1]
        self.digests[number] = self.runs or requested
        return {"taskDefinition": {"taskDefinitionArn": self.arn(number)}}

    def update_service(self, cluster, service, taskDefinition):
        self.calls.append(f"update_service {taskDefinition}")
        number = self.number(taskDefinition)
        if number == self.primary:
            return {}
        self.active = self.primary
        self.primary = number
        self.settled = False
        self.reads = 0
        return {}


class FakeCloudRun:
    """A Cloud Run service whose served traffic lags the requested traffic."""

    def __init__(self, settle_after=2, never_settles=False, runs=None):
        self.service = load("cloudrun-service")
        self.revisions = load("cloudrun-revisions")
        self.name = self.service["name"]
        self.requested = copy.deepcopy(self.service["traffic"])
        self.pending = False
        self.reads = 0
        self.settle_after = settle_after
        self.never_settles = never_settles
        self.runs = runs
        self.count = 11
        self.calls = []
        self.ready_revisions = set(self.revisions)

    def tick(self):
        if not self.pending:
            return
        self.reads += 1
        if self.reads >= self.settle_after and not self.never_settles:
            self.pending = False
            self.service["trafficStatuses"] = copy.deepcopy(self.requested)
            self.service["traffic"] = copy.deepcopy(self.requested)
            self.ready_revisions = set(self.revisions)
            created = self.service["latestCreatedRevision"].rsplit("/", 1)[-1]
            self.service["latestReadyRevision"] = (
                f"{self.name}/revisions/{created}"
            )

    def get_service(self, name):
        self.calls.append("get_service")
        self.tick()
        return copy.deepcopy(self.service)

    def get_revision(self, name):
        self.calls.append("get_revision")
        short = name.rsplit("/", 1)[-1]
        revision = copy.deepcopy(self.revisions[short])
        state = (
            "CONDITION_SUCCEEDED"
            if short in self.ready_revisions
            else "CONDITION_PENDING"
        )
        revision["conditions"] = [{"type": "Ready", "state": state}]
        return revision

    def update_service(self, name, body, update_mask):
        self.calls.append(f"update_service {update_mask}")
        if "template" in body:
            self.count += 1
            short = f"scmle-staging-serve-{self.count:05d}-new"
            image = body["template"]["containers"][0]["image"]
            if self.runs:
                image = image.split("@")[0] + "@" + self.runs
            revision = copy.deepcopy(
                self.revisions["scmle-staging-serve-00011-abc"]
            )
            revision["name"] = f"{self.name}/revisions/{short}"
            revision["containers"][0]["image"] = image
            self.revisions[short] = revision
            self.ready_revisions.discard(short)
            self.service["latestCreatedRevision"] = revision["name"]
            self.service["template"] = copy.deepcopy(body["template"])
        self.requested = copy.deepcopy(body["traffic"])
        self.pending = True
        self.reads = 0


def make_registry(root, versions=(1, 2, 3), signed=True):
    """A registry directory with the given versions (no bundles needed)."""
    root.mkdir(parents=True, exist_ok=True)
    entries = {}
    for number in versions:
        record = registry.VersionRecord(
            model="churn",
            version=number,
            registered_at=sr.NOW,
            metrics={},
            config={},
            run_id=f"run-{number}",
            path=f"churn/v{number}",
            content_id=str(number) * 64,
            signed=signed,
        )
        entries[str(number)] = dataclasses.asdict(record)
    (root / "registry.json").write_text(
        json.dumps({"models": {"churn": {"versions": entries, "aliases": {}}}}),
        encoding="utf-8",
    )
    return registry.ModelRegistryStore(root)


def approval_for(number, **changes):
    return sr.approval(version=number, content_id=str(number) * 64, **changes)


def revision_name(arn_or_short):
    return ecs.revision_of(arn_or_short)
