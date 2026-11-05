"""A mock identity provider: mints, rotates and revokes test tokens."""

import hashlib
import json
import pathlib

from scmle.security import tokens

KEYS = json.loads(
    (
        pathlib.Path(__file__).parent / "fixtures" / "rsa_test_keys.json"
    ).read_text(encoding="utf-8")
)["keys"]
ISSUER = "https://idp.test"
AUDIENCE = "scmle-service"
HS_SECRET = b"mock-idp-hs256-secret-0123456789abcdef"
DIGEST_INFO = tokens.SHA256_DIGEST_INFO


def rsa_signature(name, signing_input):
    private = KEYS[name]
    modulus = int(private["n"], 16)
    width = (modulus.bit_length() + 7) // 8
    digest = hashlib.sha256(signing_input).digest()
    pad = b"\xff" * (width - len(DIGEST_INFO) - 32 - 3)
    encoded = b"\x00\x01" + pad + b"\x00" + DIGEST_INFO + digest
    value = pow(int.from_bytes(encoded, "big"), int(private["d"], 16), modulus)
    return value.to_bytes(width, "big")


def rsa_jwk(name, kid, **window):
    private = KEYS[name]
    modulus = int(private["n"], 16)
    return {
        "kty": "RSA",
        "kid": kid,
        "n": tokens.encode_segment(modulus.to_bytes(256, "big")),
        "e": tokens.encode_segment((65537).to_bytes(3, "big")),
        **window,
    }


def hs_jwk(kid, secret=HS_SECRET, **window):
    return {
        "kty": "oct",
        "kid": kid,
        "k": tokens.encode_segment(secret),
        **window,
    }


def compact(header, claims, signature_for):
    parts = [
        tokens.encode_segment(json.dumps(part, sort_keys=True).encode())
        for part in (header, claims)
    ]
    signing_input = ".".join(parts).encode()
    signature = tokens.encode_segment(signature_for(signing_input))
    return ".".join([*parts, signature])


def rs256_token(name, kid, claims, header=None):
    head = {"alg": "RS256", "typ": "JWT", "kid": kid, **(header or {})}
    return compact(head, claims, lambda data: rsa_signature(name, data))


def claims_at(now, **extra):
    base = {
        "iss": ISSUER,
        "aud": AUDIENCE,
        "sub": "alice",
        "iat": int(now) - 5,
        "exp": int(now) + 600,
        "jti": "tok-1",
        "scope": "model:read model:predict",
        "tenant": "acme",
    }
    base.update(extra)
    return {k: v for k, v in base.items() if v is not None}


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now
