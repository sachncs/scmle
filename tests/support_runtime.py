"""An in-memory control plane for testing release logic.

`MemoryPlane` implements `scmle.deploy.control.ControlPlane` and nothing
else. It settles after a number of reads (or never), can run a different
image than the one requested, and can refuse a traffic split. It is not a
model of any platform; it lets the release logic be tested for what it does
when the platform lags, refuses or disagrees.
"""

import dataclasses
import json

from scmle.contracts import errors
from scmle.deploy import contract
from scmle.deploy import control
from scmle.deploy import observation
from scmle.governance import registry
from tests import support_release as sr

OLD = "sha256:" + "a" * 64
NEW = "sha256:" + "b" * 64
OTHER = "sha256:" + "9" * 64

NO_WAIT = control.ConfirmPolicy(attempts=4, delay=0, sleep=lambda seconds: None)


class MemoryPlane(control.ControlPlane):
    """A one-service plane whose served state lags what was asked.

    Args:
        split: Whether traffic can be split between revisions and a new
            revision starts without traffic. When false, a deploy sends all
            traffic to the new revision and only a one-revision allocation
            is accepted.
        settle_after: Reads of the plane before a change is served.
        never_settles: Whether a change is never served.
        runs: The digest every new revision runs instead of the requested one.
    """

    provider = "memory"
    service = "serve"

    def __init__(
        self, split=True, settle_after=2, never_settles=False, runs=None
    ):
        self.split = split
        self.settle_after = settle_after
        self.never_settles = never_settles
        self.runs = runs
        self.digests = {"serve-1": OLD}
        self.served = {"serve-1": 100}
        self.requested = dict(self.served)
        self.pending = False
        self.reads = 0
        self.calls = []

    def request(self, allocation):
        self.requested = {k: v for k, v in allocation.items() if v}
        self.pending = True
        self.reads = 0

    def observe(self):
        self.calls.append("observe")
        if self.pending:
            self.reads += 1
            if self.reads >= self.settle_after and not self.never_settles:
                self.pending = False
                self.served = dict(self.requested)
        primary = max(self.served, key=self.served.get)
        return observation.Observation(
            provider=self.provider,
            service=self.service,
            revision=primary,
            image_digest=self.digests[primary],
            ready=not self.pending,
            traffic=tuple(
                observation.Traffic(name, percent)
                for name, percent in self.served.items()
            ),
            observed_at=sr.NOW,
            revisions=tuple(self.digests.items()),
        )

    def deploy(self, image_digest):
        contract.require_digest(image_digest, "image_digest")
        self.calls.append("deploy")
        name = f"serve-{len(self.digests) + 1}"
        self.digests[name] = self.runs or image_digest
        if self.split:
            self.request(self.served)
        else:
            self.request({name: 100})
        return name

    def set_traffic(self, allocation):
        self.calls.append("set_traffic")
        shares = {k: v for k, v in allocation.items() if v}
        if not self.split and len(shares) != 1:
            raise control.Unsupported("this plane cannot split traffic")
        if sum(shares.values()) != 100:
            raise errors.ConfigError("traffic shares must sum to 100")
        self.request(shares)


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
