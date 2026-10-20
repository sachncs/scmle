"""Tests for independent release review evidence."""

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
