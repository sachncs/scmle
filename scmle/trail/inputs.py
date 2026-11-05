"""The statements people make about a model: KPIs, acceptance, thresholds.

``inputs.json`` has two signed sections. The stakeholder's holds the business
KPIs mapped to metrics, the acceptance criteria and the approved expectation
of the observed outcome. The reviewer's holds the domain quality thresholds
and the evidence of the review, which names the records read and their
digests. Each section is signed with that person's own HMAC key
(``scmle.contracts.signing``); the signature proves possession of the key,
not identity.
"""

from collections.abc import Mapping
import hmac
from typing import Any

from scmle.contracts import signing
from scmle.trail import records

SECTIONS = ("stakeholder", "reviewer")
EXPECTATION_FIELDS = ("kpi", "metric", "low", "high", "level", "basis")


def section_payload(record: Mapping[str, Any], role: str) -> dict[str, Any]:
    """The bytes a section's signature covers, as a mapping."""
    body = {
        key: value
        for key, value in dict(record[role]).items()
        if key != "signature"
    }
    return {
        "role": role,
        "model": record["model"],
        "version": record["version"],
        "charter_hash": record["charter_hash"],
        "body": body,
    }


def sign_inputs(
    record: Mapping[str, Any], keys: Mapping[str, str | bytes]
) -> dict[str, Any]:
    """Returns a copy of ``record`` with both sections signed.

    Args:
        record: The inputs, without signatures.
        keys: Signing key by role (``stakeholder``, ``reviewer``).
    """
    signed = {
        key: dict(value) if isinstance(value, Mapping) else value
        for key, value in record.items()
    }
    for role in SECTIONS:
        signed[role]["signature"] = signing.sign_payload(
            section_payload(signed, role), keys[role]
        )
    return signed


def signature_problem(
    record: Mapping[str, Any], role: str, key: str | bytes | None
) -> str | None:
    """Why a section's signature is not valid, or None when it is."""
    section = record.get(role)
    if not isinstance(section, Mapping):
        return f"the {role} section is absent"
    signature = section.get("signature")
    if not signature:
        return f"the {role} section is unsigned"
    if key is None:
        return f"no {role} key was supplied to check the signature"
    expected = signing.sign_payload(section_payload(record, role), key)
    if not hmac.compare_digest(str(signature), expected):
        return f"the {role} signature does not verify"
    return None


def structure_problems(record: Mapping[str, Any]) -> list[str]:
    """Missing or malformed fields in the inputs."""
    found: list[str] = []
    for name in ("model", "version", "charter_hash", "builder"):
        if name not in record:
            found.append(f"missing {name}")
    stakeholder = record.get("stakeholder", {})
    reviewer = record.get("reviewer", {})
    for name in ("name", "kpis", "acceptance", "expectation"):
        if name not in stakeholder:
            found.append(f"stakeholder section lacks {name}")
    for name in ("name", "thresholds", "evidence"):
        if name not in reviewer:
            found.append(f"reviewer section lacks {name}")
    for kpi in stakeholder.get("kpis", []):
        for name in ("kpi", "owner", "metric", "direction"):
            if name not in kpi:
                found.append(f"a KPI lacks {name}")
    for item in [
        *stakeholder.get("acceptance", []),
        *reviewer.get("thresholds", []),
    ]:
        for name in ("id", "path", "op", "value"):
            if name not in item:
                found.append(f"a criterion lacks {name}")
        if item.get("op") not in records.OPERATORS:
            found.append(f"criterion {item.get('id')} has a bad operator")
    expectation = stakeholder.get("expectation", {})
    for name in EXPECTATION_FIELDS:
        if name not in expectation:
            found.append(f"the expectation lacks {name}")
    return found
