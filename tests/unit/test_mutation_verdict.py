"""The exact record of a claims-audit verification."""

from scmle.artifacts.claims_audit import verdict


def test_a_verification_writes_every_key():
    findings = (
        verdict.Finding("hash", verdict.PASS, "matches"),
        verdict.Finding("plot", verdict.SKIPPED, "absent"),
        verdict.Finding("score", verdict.FAIL, "differs"),
    )

    record = verdict.Verification("runs/a", findings).to_dict()

    assert record == {
        "directory": "runs/a",
        "ok": False,
        "passed": 1,
        "findings": [
            {"name": "hash", "status": "pass", "detail": "matches"},
            {"name": "plot", "status": "skipped", "detail": "absent"},
            {"name": "score", "status": "fail", "detail": "differs"},
        ],
    }


def test_a_verification_without_failures_is_ok():
    findings = (verdict.Finding("hash", verdict.PASS, "matches"),)

    record = verdict.Verification("runs/b", findings).to_dict()

    assert record["ok"] is True
    assert record["passed"] == 1
