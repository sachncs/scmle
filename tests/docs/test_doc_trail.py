"""Guarantees of docs/TRAIL.md that no other test enforced."""

import shutil

import pytest

from scmle.contracts import signing
from scmle.trail import trace
from tests.unit import test_trail as trail


@pytest.fixture
def folder(tmp_path):
    """A writable copy of the committed trail folder."""
    target = tmp_path / "trail"
    shutil.copytree(trail.EVIDENCE / "trail", target)
    return target


@pytest.mark.parametrize(
    "change", [{"decision": "rejected"}, {"independent": False}]
)
def test_a_rejected_or_dependent_approval_is_an_unapproved_step(folder, change):
    """Enforces docs/TRAIL.md:66 (rejected, not independent)."""
    path = folder / "retained/registry/audit.jsonl"
    rows = trail.audit_rows(folder)
    record = trail.promotion_row(rows)["approval_record"]
    record.update(change)
    record.pop("signature")
    record["signature"] = signing.mac(record, trail.KEYS["reviewer"])
    trail.write_rows(path, rows)
    report = trail.reseal(folder)
    assert trace.UNAPPROVED in trail.kinds(report)
    assert trace.UNSIGNED not in trail.kinds(report)
