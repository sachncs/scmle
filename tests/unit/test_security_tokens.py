"""Token verification against the mock identity provider."""

import json

import pytest

from scmle.contracts import errors
from scmle.security import tokens
from tests import support_identity as idp


def verifier(clock, *keys, revocations=None, **options):
    return tokens.TokenVerifier(
        tokens.KeySet([tokens.VerificationKey.from_jwk(k) for k in keys]),
        idp.ISSUER,
        idp.AUDIENCE,
        revocations=revocations,
        clock=clock,
        **options,
    )


def hs_token(clock, kid="hs-1", header=None, **claims):
    return tokens.sign_hs256(
        idp.HS_SECRET, kid, idp.claims_at(clock.now, **claims)
    )


def refused(check, token):
    with pytest.raises(tokens.TokenError) as raised:
        check.verify(token)
    return raised.value.reason


def test_a_valid_hs256_token_gives_a_principal():
    clock = idp.Clock()
    check = verifier(clock, idp.hs_jwk("hs-1"))
    principal = check.verify(hs_token(clock))
    assert principal.subject == "alice" and principal.tenant == "acme"
    assert principal.scopes == {"model:read", "model:predict"}
    assert principal.key_id == "hs-1" and principal.token_id == "tok-1"


def test_a_valid_rs256_token_gives_a_principal():
    clock = idp.Clock()
    check = verifier(clock, idp.rsa_jwk("k1", "rs-1"))
    token = idp.rs256_token("k1", "rs-1", idp.claims_at(clock.now))
    assert check.verify(token).key_id == "rs-1"


def test_scopes_may_arrive_as_a_list():
    clock = idp.Clock()
    check = verifier(clock, idp.hs_jwk("hs-1"))
    token = tokens.sign_hs256(
        idp.HS_SECRET,
        "hs-1",
        idp.claims_at(clock.now, scope=None, scp=["model:read"]),
    )
    assert check.verify(token).scopes == {"model:read"}


def test_a_signature_from_another_key_is_refused():
    clock = idp.Clock()
    check = verifier(clock, idp.rsa_jwk("k1", "rs-1"))
    forged = idp.rs256_token("k2", "rs-1", idp.claims_at(clock.now))
    assert refused(check, forged) == tokens.BAD_SIGNATURE


def test_a_tampered_payload_is_refused():
    clock = idp.Clock()
    check = verifier(clock, idp.hs_jwk("hs-1"))
    head, body, signature = hs_token(clock).split(".")
    claims = json.loads(tokens.decode_segment(body))
    claims["scope"] = "model:admin"
    changed = tokens.encode_segment(json.dumps(claims).encode())
    assert refused(check, f"{head}.{changed}.{signature}") == (
        tokens.BAD_SIGNATURE
    )


def test_the_none_algorithm_and_confusion_attacks_are_refused():
    clock = idp.Clock()
    check = verifier(clock, idp.rsa_jwk("k1", "rs-1"), idp.hs_jwk("hs-1"))
    claims = idp.claims_at(clock.now)
    unsigned = idp.compact(
        {"alg": "none", "kid": "rs-1"}, claims, lambda data: b"x"
    )
    assert refused(check, unsigned) == tokens.BAD_ALGORITHM
    # An HS256 token naming the RSA key's kid: the key's algorithm must win.
    confused = tokens.sign_hs256(idp.HS_SECRET, "rs-1", claims)
    assert refused(check, confused) == tokens.BAD_ALGORITHM
    # A token without a kid cannot choose a key.
    anonymous = idp.compact({"alg": "HS256"}, claims, lambda data: b"x")
    assert refused(check, anonymous) == tokens.UNKNOWN_KEY


@pytest.mark.parametrize(
    "token",
    ["", "a.b", "a.b.c.d", "...", "not a token", "x" * 9000],
)
def test_malformed_tokens_are_refused(token):
    check = verifier(idp.Clock(), idp.hs_jwk("hs-1"))
    assert refused(check, token) in (tokens.MALFORMED, tokens.BAD_ALGORITHM)


def test_expired_and_future_tokens_are_refused():
    clock = idp.Clock()
    check = verifier(clock, idp.hs_jwk("hs-1"), leeway_seconds=0)
    old = hs_token(clock, exp=int(clock.now) - 1)
    assert refused(check, old) == tokens.EXPIRED
    later = hs_token(clock, nbf=int(clock.now) + 60)
    assert refused(check, later) == tokens.NOT_YET_VALID
    clock.now += 10_000
    assert refused(check, hs_token(idp.Clock())) == tokens.EXPIRED


def test_the_leeway_is_bounded():
    clock = idp.Clock()
    check = verifier(clock, idp.hs_jwk("hs-1"), leeway_seconds=30)
    assert check.verify(hs_token(clock, exp=int(clock.now) - 10))
    assert refused(check, hs_token(clock, exp=int(clock.now) - 31)) == (
        tokens.EXPIRED
    )


def test_wrong_issuer_audience_and_missing_claims_are_refused():
    clock = idp.Clock()
    check = verifier(clock, idp.hs_jwk("hs-1"))
    assert refused(check, hs_token(clock, iss="https://evil")) == (
        tokens.BAD_ISSUER
    )
    assert refused(check, hs_token(clock, aud="other")) == tokens.BAD_AUDIENCE
    assert check.verify(hs_token(clock, aud=["other", idp.AUDIENCE]))
    assert refused(check, hs_token(clock, sub=None)) == tokens.MISSING_CLAIM
    assert refused(check, hs_token(clock, exp=None)) == tokens.MISSING_CLAIM
    assert refused(check, hs_token(clock, exp="soon")) == tokens.MALFORMED


def test_revoked_tokens_subjects_and_keys_are_refused():
    clock = idp.Clock()
    revocations = tokens.RevocationList(
        tokens=["tok-bad"], subjects=["mallory"], key_ids=["hs-old"]
    )
    check = verifier(
        clock, idp.hs_jwk("hs-1"), idp.hs_jwk("hs-old"), revocations=revocations
    )
    assert check.verify(hs_token(clock))
    assert refused(check, hs_token(clock, jti="tok-bad")) == tokens.REVOKED
    assert refused(check, hs_token(clock, sub="mallory")) == tokens.REVOKED
    assert refused(check, hs_token(clock, kid="hs-old")) == tokens.REVOKED
    revocations.tokens.add("tok-1")
    assert refused(check, hs_token(clock)) == tokens.REVOKED


def test_the_revocation_file_is_read_on_every_check_and_fails_closed(tmp_path):
    clock = idp.Clock()
    path = tmp_path / "revoked.json"
    path.write_text(json.dumps({"tokens": []}))
    check = verifier(
        clock,
        idp.hs_jwk("hs-1"),
        revocations=lambda: tokens.RevocationList.from_file(path),
    )
    assert check.verify(hs_token(clock))
    path.write_text(json.dumps({"tokens": ["tok-1"]}))
    assert refused(check, hs_token(clock)) == tokens.REVOKED
    path.write_text("{ not json")
    assert refused(check, hs_token(clock)) == tokens.REVOCATION_UNAVAILABLE


def test_an_expired_key_stops_verifying_and_a_future_key_is_not_yet_valid():
    clock = idp.Clock()
    window = idp.hs_jwk("hs-1", nbf=clock.now - 100, exp=clock.now + 100)
    check = verifier(clock, window)
    token = hs_token(clock)
    assert check.verify(token)
    clock.now += 200
    fresh = hs_token(clock)
    assert refused(check, fresh) == tokens.KEY_EXPIRED
    early = verifier(clock, idp.hs_jwk("hs-1", nbf=clock.now + 50))
    assert refused(early, fresh) == tokens.KEY_NOT_YET_VALID


def test_rotation_keeps_old_tokens_until_the_old_key_expires():
    clock = idp.Clock()
    check = verifier(clock, idp.rsa_jwk("k1", "rs-old"))
    old = idp.rs256_token("k1", "rs-old", idp.claims_at(clock.now))
    assert check.verify(old)
    both = tokens.KeySet(
        [
            tokens.VerificationKey.from_jwk(idp.rsa_jwk("k1", "rs-old")),
            tokens.VerificationKey.from_jwk(idp.rsa_jwk("k2", "rs-new")),
        ]
    )
    check.keys = both
    new = idp.rs256_token("k2", "rs-new", idp.claims_at(clock.now))
    assert check.verify(old) and check.verify(new)
    check.keys = tokens.KeySet(
        [tokens.VerificationKey.from_jwk(idp.rsa_jwk("k2", "rs-new"))]
    )
    assert refused(check, old) == tokens.UNKNOWN_KEY
    assert check.verify(new)


def test_weak_or_malformed_keys_are_refused():
    with pytest.raises(errors.ConfigError, match="32 bytes"):
        tokens.VerificationKey("k", tokens.HS256, secret=b"short")
    with pytest.raises(errors.ConfigError, match="2048"):
        tokens.VerificationKey(
            "k", tokens.RS256, modulus=2**1024 + 1, exponent=3
        )
    with pytest.raises(errors.ConfigError):
        tokens.VerificationKey("k", "ES256")
    with pytest.raises(errors.ConfigError):
        tokens.VerificationKey.from_jwk({"kty": "EC", "kid": "k"})
    with pytest.raises(errors.ConfigError):
        tokens.KeySet([])
    twin = tokens.VerificationKey("k", tokens.HS256, secret=idp.HS_SECRET)
    with pytest.raises(errors.ConfigError, match="unique"):
        tokens.KeySet([twin, twin])


def test_a_key_set_loads_from_a_jwks_file(tmp_path):
    path = tmp_path / "jwks.json"
    path.write_text(
        json.dumps({"keys": [idp.rsa_jwk("k1", "rs-1"), idp.hs_jwk("hs-1")]})
    )
    assert set(tokens.KeySet.from_file(path).keys) == {"rs-1", "hs-1"}
    with pytest.raises(errors.ConfigError):
        tokens.KeySet.from_file(tmp_path / "missing.json")


def test_a_short_hs256_secret_cannot_mint():
    with pytest.raises(errors.ConfigError):
        tokens.sign_hs256(b"short", "k", {})


def test_a_non_numeric_validity_bound_is_a_config_error():
    with pytest.raises(errors.ConfigError):
        tokens.VerificationKey.from_jwk({**idp.hs_jwk("hs-1"), "exp": "soon"})


@pytest.mark.parametrize("document", ["[]", '"keys"', "5", "null"])
def test_a_key_set_file_that_is_not_an_object_is_a_config_error(
    tmp_path, document
):
    path = tmp_path / "jwks.json"
    path.write_text(document)
    with pytest.raises(errors.ConfigError):
        tokens.KeySet.from_file(path)
