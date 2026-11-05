"""A complete, valid Phase 17 evidence tree to seed failures into."""

import dataclasses
import hashlib
import json

from scmle.operations import claim
from scmle.operations import recovery
from tests import support_release as sr

PIPELINE_KEY = sr.KEY
REVIEWER_KEY = "independent-reviewer-key-held-by-a-person"
COMMIT = sr.COMMIT
STAMP = "2026-11-05T10:00:00+00:00"


def acceptance():
    return {
        "items": {
            name: {"status": "passed", "evidence": ["sha256:" + "a" * 64]}
            for name in claim.ACCEPTANCE
        }
    }


def deployment(provider):
    staged = dataclasses.replace(sr.staging_record(), provider=provider)
    return {"record": staged.signed(PIPELINE_KEY).to_dict()}


def recovery_report(provider):
    results = []
    for fault in recovery.CATALOGUE:
        results.append(
            {
                "fault_id": fault.id,
                "status": "passed",
                "environment": f"{provider}:staging",
                "measurements": [
                    {"component": c, "rto_seconds": 1.0, "rpo_lost": 0}
                    for c in fault.components
                ],
            }
        )
    return {"results": results}


def observability():
    return {
        "injected_failure": {"verdict": "infrastructure"},
        "controlled_drift": {"verdict": "data_quality"},
        "alerts_delivered": True,
        "telemetry_retained": True,
        "operator_actions": ["incident_acknowledged", "incident_resolved"],
    }


def continuous():
    return {
        "proposal": "recommend",
        "escalation_id": "esc-7",
        "gate_allowed": True,
        "replacement_confirmed": True,
        "post_promotion_check": {"overall": "inconclusive"},
    }


def write(folder, kind, provider, body, **changes):
    values = {
        "kind": kind,
        "provider": provider,
        "commit": COMMIT,
        "recorded_at": STAMP,
        "body": body,
        "key": PIPELINE_KEY,
    }
    values.update(changes)
    record = claim.envelope(**values)
    (folder / f"{kind}.json").write_text(json.dumps(record), encoding="utf-8")


def sign_off(folder, provider, **body_changes):
    covers = {
        kind: "sha256:"
        + hashlib.sha256((folder / f"{kind}.json").read_bytes()).hexdigest()
        for kind in claim.KINDS
        if (folder / f"{kind}.json").exists()
    }
    body = {
        "reviewer": "reviewer-1",
        "role": "release reviewer",
        "independent": True,
        "statement": "I reviewed the staging evidence for this commit.",
        "covers": covers,
    }
    body.update(body_changes)
    write(folder, "signoff", provider, body, key=REVIEWER_KEY)


def build(root):
    """Writes a complete tree for both providers under ``root``."""
    bodies = {
        "acceptance": lambda p: acceptance(),
        "deployment": deployment,
        "recovery": recovery_report,
        "observability": lambda p: observability(),
        "continuous-evaluation": lambda p: continuous(),
    }
    for provider in claim.PROVIDERS:
        folder = root / provider
        folder.mkdir(parents=True)
        for kind, make in bodies.items():
            write(folder, kind, provider, make(provider))
        sign_off(folder, provider)
    return root
