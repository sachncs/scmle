"""Recorded responses and a release candidate for the gate and live tests."""

import copy
import json
import pathlib

from scmle.deploy import gate
from scmle.deploy import livetests
from scmle.deploy import observation
from scmle.governance import registry
from scmle.inference import parity

ROOT = pathlib.Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "deploy" / "local" / "golden"
RECORDED = ROOT / "tests" / "fixtures" / "cloud" / "staging-http.json"
KEY = "release-test-signing-key"
COMMIT = "a1b2c3d4e5f60718293a4b5c6d7e8f9012345678"
CONTAINER = "sha256:" + "c" * 64
OTHER_CONTAINER = "sha256:" + "d" * 64
CONFIG = "sha256:" + "e" * 64
SCAN_REPORT = "sha256:" + "f" * 64
NOW = "2026-11-05T10:00:00+00:00"


def recorded(model_digest=None):
    text = RECORDED.read_text(encoding="utf-8")
    document = json.loads(text)
    if model_digest is not None:
        text = text.replace(document["model_digest"], model_digest)
    return copy.deepcopy(json.loads(text))


def fetch_from(document):
    """A fetch that answers from the recorded exchanges and nothing else."""
    labels = {f"Bearer {v}": k for k, v in document["tokens"].items()}
    table = {
        (e["method"], e["path"], e["auth"]): e for e in document["exchanges"]
    }

    def fetch(method, path, headers, body):
        auth = labels.get(headers.get("Authorization", ""), "none")
        found = table.get((method, path, auth))
        if found is None:
            return livetests.Response(404, {}, b"{}")
        raw = found["body"]
        data = raw if isinstance(raw, bytes) else json.dumps(raw).encode()
        return livetests.Response(found["status"], found["headers"], data)

    return fetch


def target(document=None, **changes):
    document = document or recorded()
    values = {
        "environment": "staging",
        "source_commit": COMMIT,
        "container_digest": CONTAINER,
        "model_digest": document["model_digest"],
        "token": document["tokens"]["valid"],
        "base_url": document["base_url"],
        "read_only_token": document["tokens"]["readonly"],
    }
    values.update(changes)
    return livetests.Target(**values)


def golden():
    loaded = parity.load_golden(GOLDEN)
    return loaded, parity.read_json(GOLDEN / "expected.json")


def run_live(document=None, **changes):
    document = document or recorded()
    loaded, expected = golden()
    return livetests.run_checks(
        fetch_from(document), target(document, **changes), loaded, expected
    )


def version(signed=True):
    return registry.VersionRecord(
        model="churn",
        version=3,
        registered_at=NOW,
        metrics={},
        config={},
        run_id="run-1",
        path="churn/v3",
        content_id="9" * 64,
        signed=signed,
    )


def approval(**changes):
    values = {
        "model": "churn",
        "version": 3,
        "content_id": "9" * 64,
        "reviewer": "reviewer-1",
        "decision": registry.PromotionDecision.APPROVED,
        "reason": "evidence reviewed",
        "evidence": ("ticket-17",),
        "approved_at": NOW,
        "independent": True,
    }
    values.update(changes)
    return registry.PromotionApproval(**values).sign(KEY)


def seen(**changes):
    values = {
        "provider": "aws",
        "service": "serve",
        "revision": "serve:12",
        "image_digest": CONTAINER,
        "ready": True,
        "traffic": (observation.Traffic("serve:12", 100),),
        "observed_at": NOW,
    }
    values.update(changes)
    return observation.Observation(**values)


def candidate(environment="staging", **changes):
    document = recorded()
    checks = tuple(run_live(document, environment=environment))
    values = {
        "environment": environment,
        "provider": "aws",
        "source_commit": COMMIT,
        "model": version(),
        "model_digest": document["model_digest"],
        "approval": approval(),
        "container_digest": CONTAINER,
        "config_digest": CONFIG,
        "requested_by": "author",
        "approver": "release-manager",
        "approved_at": NOW,
        "scan": gate.ImageScan(CONTAINER, True, SCAN_REPORT),
        "checks": checks,
        "observation": seen(),
    }
    values.update(changes)
    return gate.Candidate(**values)


def staging_record():
    decision = gate.certify(candidate("staging"), KEY, NOW)
    assert decision.allowed, decision.reasons
    return decision.record
