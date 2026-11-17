"""Guarantees of docs/TRAIL.md that no other test enforced."""

import shutil

import pytest

from scmle.contracts import signing
from scmle.trail import trace
from tests.unit import test_trail


@pytest.fixture
def folder(tmp_path):
    """A writable copy of the committed trail folder."""
    target = tmp_path / "trail"
    shutil.copytree(test_trail.EVIDENCE / "trail", target)
    return target


@pytest.mark.parametrize(
    "change", [{"decision": "rejected"}, {"independent": False}]
)
def test_a_rejected_or_dependent_approval_is_an_unapproved_step(folder, change):
    """Enforces docs/TRAIL.md:66 (rejected, not independent)."""
    path = folder / "retained/registry/audit.jsonl"
    rows = test_trail.audit_rows(folder)
    record = test_trail.promotion_row(rows)["approval_record"]
    record.update(change)
    record.pop("signature")
    record["signature"] = signing.mac(record, test_trail.KEYS["reviewer"])
    test_trail.write_rows(path, rows)
    report = test_trail.reseal(folder)
    assert trace.UNAPPROVED in test_trail.kinds(report)
    assert trace.UNSIGNED not in test_trail.kinds(report)
