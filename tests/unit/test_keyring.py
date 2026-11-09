"""Tests for signing-key lifecycle policy and metadata persistence."""

import datetime

import pytest

from scmle.governance import keyring

CREATED = "2026-10-20T00:00:00+00:00"
ROTATED = "2026-10-21T00:00:00+00:00"
OBSERVED = "2026-10-21T01:00:00+00:00"


def record(
    key_id: str = "release-2026-10",
    state: keyring.KeyState = keyring.KeyState.ACTIVE,
    created_at: str = CREATED,
    **kwargs: str,
) -> keyring.SigningKeyRecord:
    """Build representative key metadata."""
    return keyring.SigningKeyRecord(
        key_id=key_id,
        owner="release-engineering",
        created_at=created_at,
        state=state,
        **kwargs,
    )


def test_active_key_signs_and_verifies_without_persisting_secret():
    """The registry binds identity while secret custody stays external."""
    registry = keyring.SigningKeyRegistry((record(),))
    payload = {"release": "candidate-1", "digest": "abc"}
    envelope = registry.sign(payload, "release-2026-10", "secret", OBSERVED)

    assert registry.verify(
        payload,
        envelope,
        {"release-2026-10": "secret"},
        OBSERVED,
    )
    assert "secret" not in registry.to_json()
    assert envelope.to_dict()["key_id"] == "release-2026-10"


def test_retirement_stops_signing_but_allows_explicit_historical_verification():
    """Rotation preserves old evidence without permitting new signatures."""
    registry = keyring.SigningKeyRegistry((record(), record("release-2026-11")))
    payload = {"release": "candidate-1"}
    envelope = registry.sign(
        payload,
        "release-2026-10",
        "old",
        "2026-10-20T12:00:00+00:00",
    )
    registry.retire(
        "release-2026-10",
        ROTATED,
        "scheduled rotation",
        replacement_key_id="release-2026-11",
    )

    assert not registry.verify(
        payload,
        envelope,
        {"release-2026-10": "old"},
        OBSERVED,
    )
    assert registry.verify(
        payload,
        envelope,
        {"release-2026-10": "old"},
        OBSERVED,
        historical=True,
    )
    with pytest.raises(ValueError, match="cannot sign"):
        registry.sign(payload, "release-2026-10", "old", OBSERVED)


def test_revocation_invalidates_even_historical_signatures():
    """Compromise response is immediate and cannot be bypassed."""
    registry = keyring.SigningKeyRegistry((record(),))
    payload = {"release": "candidate-1"}
    envelope = registry.sign(payload, "release-2026-10", "secret", OBSERVED)
    registry.revoke("release-2026-10", ROTATED, "credential compromise")

    assert not registry.verify(
        payload,
        envelope,
        {"release-2026-10": "secret"},
        OBSERVED,
        historical=True,
    )


def test_expired_key_can_only_verify_pre_expiry_historical_evidence():
    """Expiry blocks current acceptance while preserving explicit history."""
    expires = "2026-10-21T00:30:00+00:00"
    registry = keyring.SigningKeyRegistry((record(expires_at=expires),))
    payload = {"release": "candidate-1"}
    envelope = registry.sign(
        payload,
        "release-2026-10",
        "secret",
        "2026-10-20T12:00:00+00:00",
    )

    assert not registry.verify(
        payload,
        envelope,
        {"release-2026-10": "secret"},
        OBSERVED,
    )
    assert registry.verify(
        payload,
        envelope,
        {"release-2026-10": "secret"},
        OBSERVED,
        historical=True,
    )
    with pytest.raises(ValueError, match="cannot sign"):
        registry.sign(payload, "release-2026-10", "secret", OBSERVED)


def test_registry_round_trip_is_deterministic_and_atomic(tmp_path):
    """The persisted registry contains only reviewable public metadata."""
    registry = keyring.SigningKeyRegistry((record(),))
    path = tmp_path / "signing-keys.json"
    registry.write(path)
    restored = keyring.SigningKeyRegistry.read(path)

    assert restored.to_json() == registry.to_json()
    assert restored.get("release-2026-10") == record()
    assert not list(tmp_path.glob(".signing-keys.json.*.tmp"))


def test_key_metadata_rejects_naive_or_invalid_transitions():
    """Key state is not a free-form label that can weaken verification."""
    with pytest.raises(ValueError, match="timezone"):
        record(created_at="2026-10-20T00:00:00")
    with pytest.raises(ValueError, match="state_changed_at"):
        record(state=keyring.KeyState.REVOKED)
    with pytest.raises(ValueError, match="after created_at"):
        record(expires_at=CREATED)


def test_signature_envelope_rejects_tampering_and_wrong_algorithm():
    """Envelope parsing fails before cryptographic verification."""
    with pytest.raises(ValueError, match="unsupported signature algorithm"):
        keyring.SignatureEnvelope(
            key_id="release-2026-10",
            signed_at=OBSERVED,
            value="signature",
            algorithm="none",
        )
    with pytest.raises(ValueError, match="key_id"):
        keyring.SignatureEnvelope(
            key_id="",
            signed_at=OBSERVED,
            value="signature",
        )


def test_timestamp_parser_accepts_zulu_and_offset_forms():
    """Both common UTC spellings normalize to offset-aware timestamps."""
    assert keyring.parse_timestamp("2026-10-21T00:00:00Z", "timestamp").tzinfo
    assert isinstance(
        keyring.parse_timestamp(OBSERVED, "timestamp"), datetime.datetime
    )


def test_a_retired_key_found_compromised_can_still_be_revoked():
    """Retirement keeps historical proof alive, so revocation must stay open."""
    registry = keyring.SigningKeyRegistry((record(),))
    payload = {"release": "candidate-1"}
    envelope = registry.sign(
        payload, "release-2026-10", "secret", "2026-10-20T12:00:00+00:00"
    )
    registry.retire("release-2026-10", ROTATED, "rotation")
    assert registry.verify(
        payload,
        envelope,
        {"release-2026-10": "secret"},
        OBSERVED,
        historical=True,
    )
    registry.revoke("release-2026-10", OBSERVED, "credential compromise")

    assert not registry.verify(
        payload,
        envelope,
        {"release-2026-10": "secret"},
        OBSERVED,
        historical=True,
    )
    with pytest.raises(ValueError, match="already revoked"):
        registry.revoke("release-2026-10", OBSERVED, "again")
