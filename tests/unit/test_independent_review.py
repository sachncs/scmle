"""Tests for independent release review evidence."""

import dataclasses
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scml.governance import review


def claim(
    kind: review.ReviewClaimKind, criterion_id: str
) -> review.ReviewClaim:
    """Build a deterministic independent claim."""
    return review.ReviewClaim(
        criterion_id=criterion_id,
        kind=kind,
        passed=True,
        observed="0.91",
        reference=">=0.90",
        evidence_uri=f"evidence/{criterion_id}.json",
        notes="Recomputed from the clean checkout.",
    )


def make_review() -> review.IndependentReleaseReview:
    """Build a complete review fixture."""
    return review.IndependentReleaseReview(
        release_id="beta-2026-10-20",
        source_commit="abc1234",
        clean_clone_digest="sha256:clone",
        reviewer_id="reviewer@example.com",
        implementer_ids=("engineer@example.com",),
        environment_digest="sha256:environment",
        commands=("uv run pytest -q", "uv run python -m benchmarks"),
        claims=(
            claim(review.ReviewClaimKind.METRIC, "metric.roc_auc"),
            claim(review.ReviewClaimKind.FAIRNESS, "fairness.equalized_odds"),
            claim(review.ReviewClaimKind.EXPLANATION, "explanation.fidelity"),
            claim(review.ReviewClaimKind.RELIABILITY, "reliability.faults"),
            claim(review.ReviewClaimKind.SECURITY, "security.provenance"),
        ),
        artifact_digests={"wheel": "sha256:wheel", "evidence": "sha256:index"},
        observed_at="2026-10-20T12:00:00+00:00",
        decision=review.ReviewDecision.APPROVED,
    )


def test_review_round_trips_and_accepts_exact_signed_proof():
    """A signed independent review passes only for the exact commit."""
    original = make_review().sign("release-key")
    restored = review.IndependentReleaseReview.from_dict(original.to_dict())

    assert restored == original
    assert restored.verify_signature("release-key")
    assert restored.accepted(
        "abc1234",
        (
            review.ReviewClaimKind.METRIC,
            review.ReviewClaimKind.FAIRNESS,
            review.ReviewClaimKind.EXPLANATION,
        ),
        "release-key",
    )
    assert not restored.accepted("different", (), "release-key")
    assert not restored.verify_signature("wrong-key")


def test_signed_review_writes_atomically_and_cli_verifies_exact_commit(
    tmp_path: Path,
):
    """A reviewer can verify persisted evidence without importing internals."""
    path = tmp_path / "independent-review.json"
    make_review().sign("release-key").write(path)
    restored = review.IndependentReleaseReview.read(path)

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
        env={**os.environ, "SCML_REVIEW_SIGNING_KEY": "release-key"},
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
        env={**os.environ, "SCML_REVIEW_SIGNING_KEY": "release-key"},
    )
    assert wrong_commit.returncode == 1


def test_review_fails_closed_for_blocked_or_failed_claims():
    """A signature cannot turn a blocked or failed review into approval."""
    blocked_record = {**make_review().to_dict(), "decision": "blocked"}
    blocked = review.IndependentReleaseReview.from_dict(blocked_record).sign(
        "release-key"
    )
    assert not blocked.accepted(
        "abc1234", (review.ReviewClaimKind.METRIC,), "release-key"
    )

    failed_claim = claim(review.ReviewClaimKind.METRIC, "metric.roc_auc")
    failed_claim = review.ReviewClaim(
        **{**failed_claim.to_dict(), "passed": False}
    )
    failed_record = {
        **make_review().to_dict(),
        "claims": [failed_claim.to_dict()],
    }
    failed = review.IndependentReleaseReview.from_dict(failed_record).sign(
        "release-key"
    )
    assert not failed.accepted(
        "abc1234", (review.ReviewClaimKind.METRIC,), "release-key"
    )


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
        env={**os.environ, "SCML_REVIEW_SIGNING_KEY": "release-key"},
    )

    assert result.returncode == 1
    assert "acceptance gate" in result.stderr


def test_review_rejects_conflicted_identity_and_incomplete_evidence():
    """Reviewer independence, timestamps, and required evidence are strict."""
    with pytest.raises(ValueError, match="independent"):
        review.IndependentReleaseReview.from_dict(
            {
                **make_review().to_dict(),
                "implementer_ids": ["reviewer@example.com"],
            }
        )
    with pytest.raises(ValueError, match="timezone"):
        review.IndependentReleaseReview.from_dict(
            {**make_review().to_dict(), "observed_at": "2026-10-20T12:00:00"}
        )
    with pytest.raises(ValueError, match="duplicate"):
        review.IndependentReleaseReview.from_dict(
            {
                **make_review().to_dict(),
                "claims": [
                    claim(
                        review.ReviewClaimKind.METRIC, "metric.roc_auc"
                    ).to_dict(),
                    claim(
                        review.ReviewClaimKind.METRIC, "metric.roc_auc"
                    ).to_dict(),
                ],
            }
        )
