"""Tests for independent release review evidence."""

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scmle.governance import review


def claim(kind: review.Kind, criterion: str) -> review.Recomputation:
    """Build a deterministic independent claim."""
    return review.Recomputation(
        criterion=criterion,
        kind=kind,
        passed=True,
        observed="0.91",
        reference=">=0.90",
        uri=f"evidence/{criterion}.json",
        notes="Recomputed from the clean checkout.",
    )


def make_review() -> review.Signoff:
    """Build a complete review fixture."""
    return review.Signoff(
        release="beta-2026-10-20",
        commit="abc1234",
        digests={
            "clone": "sha256:clone",
            "environment": "sha256:environment",
            "artifacts": {"wheel": "sha256:wheel", "evidence": "sha256:index"},
        },
        reviewer="reviewer@example.com",
        implementers=("engineer@example.com",),
        commands=("uv run pytest -q", "uv run python -m benchmarks"),
        claims=(
            claim(review.Kind.METRIC, "metric.roc_auc"),
            claim(review.Kind.FAIRNESS, "fairness.equalized_odds"),
            claim(review.Kind.EXPLANATION, "explanation.fidelity"),
            claim(review.Kind.RELIABILITY, "reliability.faults"),
            claim(review.Kind.SECURITY, "security.provenance"),
        ),
        observed="2026-10-20T12:00:00+00:00",
        decision=review.Verdict.APPROVED,
    )


def test_review_round_trips_and_accepts_exact_signed_proof():
    """A signed independent review passes only for the exact commit."""
    original = make_review().sign("release-key")
    restored = review.Signoff.from_dict(original.to_dict())

    assert restored == original
    assert restored.verify("release-key")
    assert restored.accepted(
        "abc1234",
        (
            review.Kind.METRIC,
            review.Kind.FAIRNESS,
            review.Kind.EXPLANATION,
        ),
        "release-key",
    )
    assert not restored.accepted("different", (), "release-key")
    assert not restored.verify("wrong-key")


def test_signed_review_writes_atomically_and_cli_verifies_exact_commit(
    tmp_path: Path,
):
    """A reviewer can verify persisted evidence without importing internals."""
    path = tmp_path / "independent-review.json"
    make_review().sign("release-key").write(path)
    restored = review.Signoff.read(path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_release_review.py",
            str(path),
            "--source-commit",
            "abc1234",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "SCMLE_REVIEW_SIGNING_KEY": "release-key"},
    )

    assert restored == make_review().sign("release-key")
    assert result.returncode == 0
    assert json.loads(result.stdout)["accepted"] is True
    assert not list(tmp_path.glob("*.tmp"))

    wrong_commit = subprocess.run(
        [
            sys.executable,
            "scripts/verify_release_review.py",
            str(path),
            "--source-commit",
            "different",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "SCMLE_REVIEW_SIGNING_KEY": "release-key"},
    )
    assert wrong_commit.returncode == 1


def test_review_fails_closed_for_blocked_or_failed_claims():
    """A signature cannot turn a blocked or failed review into approval.

    Also enforces website/src/pages/docs/evidence.astro:86.
    """
    blocked_record = {**make_review().to_dict(), "decision": "blocked"}
    blocked = review.Signoff.from_dict(blocked_record).sign("release-key")
    assert not blocked.accepted("abc1234", (review.Kind.METRIC,), "release-key")

    failed_claim = claim(review.Kind.METRIC, "metric.roc_auc")
    failed_claim = review.Recomputation(
        **{**failed_claim.to_dict(), "passed": False}
    )
    failed_record = {
        **make_review().to_dict(),
        "claims": [failed_claim.to_dict()],
    }
    failed = review.Signoff.from_dict(failed_record).sign("release-key")
    assert not failed.accepted("abc1234", (review.Kind.METRIC,), "release-key")


def test_cli_requires_reliability_and_security_claims_by_default(
    tmp_path: Path,
):
    """The default CLI gate covers every critical review claim kind."""
    path = tmp_path / "incomplete-review.json"
    complete = make_review()
    incomplete = dataclasses.replace(complete, claims=complete.claims[:3])
    incomplete.sign("release-key").write(path)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/verify_release_review.py",
            str(path),
            "--source-commit",
            "abc1234",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**os.environ, "SCMLE_REVIEW_SIGNING_KEY": "release-key"},
    )

    assert result.returncode == 1
    assert "acceptance gate" in result.stderr


def test_review_rejects_conflicted_identity_and_incomplete_evidence():
    """Reviewer independence, timestamps, and required evidence are strict.

    Also enforces website/src/pages/docs/evidence.astro:86.
    """
    with pytest.raises(ValueError, match="independent"):
        review.Signoff.from_dict(
            {
                **make_review().to_dict(),
                "implementers": ["reviewer@example.com"],
            }
        )
    with pytest.raises(ValueError, match="timezone"):
        review.Signoff.from_dict(
            {**make_review().to_dict(), "observed": "2026-10-20T12:00:00"}
        )
    with pytest.raises(ValueError, match="duplicate"):
        review.Signoff.from_dict(
            {
                **make_review().to_dict(),
                "claims": [
                    claim(review.Kind.METRIC, "metric.roc_auc").to_dict(),
                    claim(review.Kind.METRIC, "metric.roc_auc").to_dict(),
                ],
            }
        )
