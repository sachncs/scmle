"""Bearer-token verification: JWT with HS256 or RS256, key ids and rotation.

Only the standard library is used. HS256 is an HMAC; RS256 is verified with
the RSASSA-PKCS1-v1_5 check of RFC 8017 (public exponentiation and an exact
comparison with the encoded digest), so only public keys are ever needed to
verify. Tokens are refused unless every one of these holds:

* the header names an allowed algorithm (never `none`) and a `kid`;
* the `kid` names a key in the key set, the key's algorithm equals the
  header's (no algorithm confusion) and the key is inside its validity window
  (`not_before` to `expires`); a retired key is rotated out by setting its
  `expires`, so tokens signed with it stop verifying then;
* the signature is correct;
* `iss` and `aud` match, `sub` is present, `exp` is in the future and `nbf` /
  `iat` are not in the future (a small leeway is allowed);
* the token, its `jti` or its subject is not revoked.

The failure reason is a short code for the audit log; the caller is never
told which check failed (see `scmle.security.access`).
"""

import base64
import binascii
from collections.abc import Callable
from collections.abc import Mapping
from collections.abc import Sequence
import dataclasses
import hashlib
import hmac
import json
import pathlib
import time
from typing import Any
from typing import Self

from scmle.contracts import errors

HS256 = "HS256"
RS256 = "RS256"
ALGORITHMS = (HS256, RS256)
MAX_TOKEN_BYTES = 8192
MIN_RSA_BITS = 2048
MIN_HMAC_BYTES = 32
MAX_KID_LENGTH = 128
MIN_RSA_EXPONENT = 3
TOKEN_PARTS = 3  # header, payload and signature
DEFAULT_LEEWAY_SECONDS = 30.0
SHA256_DIGEST_INFO = bytes.fromhex("3031300d060960864801650304020105000420")

# Failure codes (audit reasons).
MALFORMED = "malformed"
BAD_ALGORITHM = "bad_algorithm"
UNKNOWN_KEY = "unknown_key"
KEY_NOT_YET_VALID = "key_not_yet_valid"
KEY_EXPIRED = "key_expired"
BAD_SIGNATURE = "bad_signature"
BAD_ISSUER = "bad_issuer"
BAD_AUDIENCE = "bad_audience"
EXPIRED = "expired"
NOT_YET_VALID = "not_yet_valid"
MISSING_CLAIM = "missing_claim"
REVOKED = "revoked"
REVOCATION_UNAVAILABLE = "revocation_unavailable"


class TokenError(errors.ScmleError):
    """A bearer token was refused.

    Attributes:
        reason: Short code naming the failed check (for logs and audit only).
    """

    category = "auth"

    def __init__(self, reason: str, message: str = "") -> None:
        """Initializes the error.

        Args:
            reason: One of the failure codes in this module.
            message: Optional detail, such as a claim name; it never contains
                the token. Defaults to `reason`.
        """
        super().__init__(message or reason)
        self.reason = reason


def decode_segment(text: str) -> bytes:
    """Decodes one base64url segment; padding is optional.

    Args:
        text: The segment.

    Returns:
        The decoded bytes.

    Raises:
        TokenError: If the segment is not base64url.
    """
    try:
        padded = text + "=" * (-len(text) % 4)
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (binascii.Error, UnicodeError, ValueError) as error:
        raise TokenError(MALFORMED) from error


def encode_segment(raw: bytes) -> str:
    """Encodes bytes as an unpadded base64url segment.

    Args:
        raw: The bytes.

    Returns:
        The segment.
    """
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def int_from_segment(text: str) -> int:
    """Decodes a big-endian unsigned integer from a base64url segment.

    Args:
        text: The segment.

    Returns:
        The integer. The `TokenError` that `decode_segment` raises for a
        segment that is not base64url is not caught.
    """
    return int.from_bytes(decode_segment(text), "big")


@dataclasses.dataclass(frozen=True, slots=True)
class Key:
    """One key that may verify tokens.

    Attributes:
        kid: Key id named by token headers.
        algorithm: `HS256` or `RS256`.
        secret: HMAC secret (HS256 only).
        modulus: RSA modulus (RS256 only).
        exponent: RSA public exponent (RS256 only).
        not_before: Epoch seconds before which the key is not yet valid.
        expires: Epoch seconds from which the key no longer verifies.
    """

    kid: str
    algorithm: str
    secret: bytes = b""
    modulus: int = 0
    exponent: int = 0
    not_before: float | None = None
    expires: float | None = None

    def __post_init__(self) -> None:
        """Validates the key id, algorithm and key material.

        Raises:
            ConfigError: If the `kid` is empty or over 128 characters, the
                algorithm is not `HS256` or `RS256`, an HS256 secret is under
                32 bytes, an RS256 modulus is under 2048 bits, or the RSA
                exponent is even or below 3.
        """
        if not self.kid or len(self.kid) > MAX_KID_LENGTH:
            raise errors.ConfigError("a key needs a kid of 1-128 characters")
        if self.algorithm == HS256:
            if len(self.secret) < MIN_HMAC_BYTES:
                raise errors.ConfigError(
                    f"an HS256 secret needs at least {MIN_HMAC_BYTES} bytes"
                )
        elif self.algorithm == RS256:
            if self.modulus.bit_length() < MIN_RSA_BITS:
                raise errors.ConfigError(
                    f"an RS256 modulus needs at least {MIN_RSA_BITS} bits"
                )
            if self.exponent < MIN_RSA_EXPONENT or self.exponent % 2 == 0:
                raise errors.ConfigError("the RSA exponent must be odd, >= 3")
        else:
            raise errors.ConfigError(
                f"algorithm must be one of {ALGORITHMS}, not {self.algorithm!r}"
            )

    @classmethod
    def from_jwk(cls, entry: Mapping[str, Any]) -> Self:
        """Builds a key from a JWK object.

        `kty` `oct` (field `k`) gives HS256, `RSA` (fields `n`, `e`) gives
        RS256. The extension fields `nbf` and `exp` (epoch seconds) bound the
        key's validity.

        Args:
            entry: The JWK object.

        Returns:
            The key.

        Raises:
            ConfigError: If the entry lacks `kid` or `kty`, a key field is
                missing or not base64url, the type is neither `oct` nor `RSA`,
                a non-numeric `nbf` or `exp`, or the key is too weak.
        """
        try:
            kid = str(entry["kid"])
            kind = entry["kty"]
            not_before = optional_float(entry.get("nbf"))
            expires = optional_float(entry.get("exp"))
            if kind == "oct":
                return cls(
                    kid,
                    HS256,
                    secret=decode_segment(entry["k"]),
                    not_before=not_before,
                    expires=expires,
                )
            if kind == "RSA":
                return cls(
                    kid,
                    RS256,
                    modulus=int_from_segment(entry["n"]),
                    exponent=int_from_segment(entry["e"]),
                    not_before=not_before,
                    expires=expires,
                )
        except (KeyError, TypeError, ValueError, TokenError) as error:
            raise errors.ConfigError("malformed JWK entry") from error
        raise errors.ConfigError(f"unsupported JWK type {kind!r}")

    def check_window(self, now: float) -> None:
        """Raises if the key is outside its validity window.

        Args:
            now: Epoch seconds.

        Raises:
            TokenError: `key_not_yet_valid` before `not_before`, `key_expired`
                from `expires` on.
        """
        if self.not_before is not None and now < self.not_before:
            raise TokenError(KEY_NOT_YET_VALID)
        if self.expires is not None and now >= self.expires:
            raise TokenError(KEY_EXPIRED)

    def verify(self, signing_input: bytes, signature: bytes) -> bool:
        """Checks a signature.

        Args:
            signing_input: The signed bytes (`header.payload`).
            signature: The signature bytes.

        Returns:
            True if `signature` is valid for `signing_input` under this key.
        """
        if self.algorithm == HS256:
            expected = hmac.new(self.secret, signing_input, "sha256").digest()
            return hmac.compare_digest(expected, signature)
        width = (self.modulus.bit_length() + 7) // 8
        if len(signature) != width:
            return False
        value = int.from_bytes(signature, "big")
        if value >= self.modulus:
            return False
        recovered = pow(value, self.exponent, self.modulus).to_bytes(
            width, "big"
        )
        digest = hashlib.sha256(signing_input).digest()
        padding = b"\xff" * (width - len(SHA256_DIGEST_INFO) - 32 - 3)
        expected = b"\x00\x01" + padding + b"\x00" + SHA256_DIGEST_INFO + digest
        return hmac.compare_digest(recovered, expected)


def optional_float(value: Any) -> float | None:
    """Converts a value to a float unless it is absent.

    Args:
        value: A number or None.

    Returns:
        The float, or None when `value` is None.
    """
    return None if value is None else float(value)


class KeySet:
    """The keys currently trusted, by `kid`; replaced as a whole on rotation.

    Attributes:
        keys: The trusted keys by `kid`.
    """

    def __init__(self, keys: Sequence[Key]) -> None:
        """Initializes the set.

        Args:
            keys: The keys; `kid` values must be unique.

        Raises:
            ConfigError: If a `kid` repeats or there is no key.
        """
        if not keys:
            raise errors.ConfigError("a key set needs at least one key")
        by_kid = {key.kid: key for key in keys}
        if len(by_kid) != len(keys):
            raise errors.ConfigError("kid values must be unique")
        self.keys = by_kid

    @classmethod
    def from_jwks(cls, document: Mapping[str, Any]) -> Self:
        """Builds a set from a JWKS document (`{"keys": [...]}`).

        Args:
            document: The parsed JWKS document.

        Returns:
            The key set.

        Raises:
            ConfigError: If the document is not an object with a `keys`
                list. A malformed entry is refused with the same error.
        """
        parsed: object = document  # may be any parsed JSON value
        entries = parsed.get("keys") if isinstance(parsed, Mapping) else None
        if not isinstance(entries, list):
            raise errors.ConfigError("a JWKS document needs a keys list")
        return cls([Key.from_jwk(entry) for entry in entries])

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> Self:
        """Reads a JWKS file.

        Args:
            path: The file.

        Returns:
            The key set.

        Raises:
            ConfigError: If the file is unreadable, is not JSON, or holds no
                valid `keys` list (a JSON document that is not an object
                has none).
        """
        try:
            text = pathlib.Path(path).read_text(encoding="utf-8")
            return cls.from_jwks(json.loads(text))
        except (OSError, ValueError) as error:
            raise errors.ConfigError(f"cannot read key set {path}") from error

    def get(self, kid: str) -> Key:
        """Looks up the key with this id.

        Args:
            kid: The key id.

        Returns:
            The key.

        Raises:
            TokenError: If no such key is trusted.
        """
        key = self.keys.get(kid)
        if key is None:
            raise TokenError(UNKNOWN_KEY)
        return key


class RevocationList:
    """Tokens (`jti`), subjects and keys (`kid`) that are no longer accepted.

    Attributes:
        tokens: Revoked `jti` values.
        subjects: Revoked `sub` values.
        key_ids: Revoked `kid` values.
    """

    def __init__(
        self,
        tokens: Sequence[str] = (),
        subjects: Sequence[str] = (),
        key_ids: Sequence[str] = (),
    ) -> None:
        """Initializes the list.

        Args:
            tokens: Revoked `jti` values.
            subjects: Revoked `sub` values (every token of the subject).
            key_ids: Revoked `kid` values (every token the key signed).
        """
        self.tokens = set(tokens)
        self.subjects = set(subjects)
        self.key_ids = set(key_ids)

    @classmethod
    def from_file(cls, path: str | pathlib.Path) -> Self:
        """Reads `{"tokens": [], "subjects": [], "key_ids": []}`.

        Args:
            path: The file.

        Returns:
            The list.

        Raises:
            ConfigError: If the file is unreadable or malformed.
        """
        try:
            document = json.loads(pathlib.Path(path).read_text("utf-8"))
            return cls(
                document.get("tokens", ()),
                document.get("subjects", ()),
                document.get("key_ids", ()),
            )
        except (OSError, ValueError, AttributeError) as error:
            raise errors.ConfigError(
                f"cannot read revocations {path}"
            ) from error

    def revokes(
        self, header: Mapping[str, Any], claims: Mapping[str, Any]
    ) -> bool:
        """Reports whether the token is revoked.

        Args:
            header: The token's header.
            claims: The token's claims.

        Returns:
            True if its `kid` or `sub` is revoked, or its `jti` is.
        """
        return (
            header.get("kid") in self.key_ids
            or claims.get("sub") in self.subjects
            or (claims.get("jti") is not None and claims["jti"] in self.tokens)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class Principal:
    """A verified caller.

    Attributes:
        subject: The token's `sub`.
        tenant: The token's tenant claim if it is a non-empty string, else
            None.
        scopes: Granted scopes.
        token_id: The token's `jti`, or None.
        key_id: The key that verified it.
        expires: Token expiry, epoch seconds.
    """

    subject: str
    tenant: str | None
    scopes: frozenset[str]
    token_id: str | None
    key_id: str
    expires: float


class Verifier:
    """Verifies bearer tokens against a key set and a revocation list.

    Attributes:
        keys: Trusted keys; replace to rotate.
        issuer: Required `iss`.
        audience: Required member of `aud`.
        revocations: A list, a callable returning the current list, or None.
        tenant_claim: Name of the tenant claim.
        leeway: Clock skew tolerated, in seconds.
        clock: Source of epoch seconds.
    """

    def __init__(
        self,
        keys: KeySet,
        issuer: str,
        audience: str,
        *,
        revocations: RevocationList
        | Callable[[], RevocationList]
        | None = None,
        tenant_claim: str = "tenant",
        leeway_seconds: float = DEFAULT_LEEWAY_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        """Initializes the verifier.

        Args:
            keys: Trusted keys. Replace `keys` to rotate.
            issuer: Required `iss`.
            audience: Required member of `aud`.
            revocations: A list, or a callable returning the current list
                (so a file can be re-read).
            tenant_claim: Name of the tenant claim.
            leeway_seconds: Clock skew tolerated on `exp`, `nbf`, `iat`.
            clock: Source of epoch seconds (injectable for tests).

        Raises:
            ConfigError: If issuer or audience is empty or leeway negative.
        """
        if not issuer or not audience or leeway_seconds < 0:
            raise errors.ConfigError(
                "issuer and audience are required; leeway cannot be negative"
            )
        self.keys = keys
        self.issuer = issuer
        self.audience = audience
        self.revocations = revocations
        self.tenant_claim = tenant_claim
        self.leeway = leeway_seconds
        self.clock = clock

    def current_revocations(self) -> RevocationList | None:
        """Returns the revocation list now in force.

        Returns:
            The list, or None if no revocations are configured.

        Raises:
            TokenError: `revocation_unavailable` if the callable raises
                `ConfigError`.
        """
        if callable(self.revocations):
            try:
                return self.revocations()
            except errors.ConfigError as error:
                raise TokenError(REVOCATION_UNAVAILABLE) from error
        return self.revocations

    def verify(self, token: str) -> Principal:
        """Verifies a token.

        Args:
            token: The compact JWT, without the `Bearer ` prefix.

        Returns:
            The verified caller.

        Raises:
            TokenError: If any check fails; `reason` names it.
        """
        if not token or len(token) > MAX_TOKEN_BYTES:
            raise TokenError(MALFORMED)
        parts = token.split(".")
        if len(parts) != TOKEN_PARTS or not all(parts):
            raise TokenError(MALFORMED)
        header = self.parse_object(parts[0])
        algorithm = header.get("alg")
        kid = header.get("kid")
        if algorithm not in ALGORITHMS:
            raise TokenError(BAD_ALGORITHM)
        if not isinstance(kid, str) or not kid:
            raise TokenError(UNKNOWN_KEY)
        key = self.keys.get(kid)
        if key.algorithm != algorithm:
            raise TokenError(BAD_ALGORITHM)
        now = self.clock()
        key.check_window(now)
        signing_input = f"{parts[0]}.{parts[1]}".encode("ascii", "replace")
        if not key.verify(signing_input, decode_segment(parts[2])):
            raise TokenError(BAD_SIGNATURE)
        claims = self.parse_object(parts[1])
        return self.check_claims(header, claims, key, now)

    def parse_object(self, segment: str) -> dict[str, Any]:
        """Parses a JSON object from a base64url segment.

        Args:
            segment: A token segment.

        Returns:
            The JSON object.

        Raises:
            TokenError: `malformed` if the segment is not a JSON object.
        """
        try:
            value = json.loads(decode_segment(segment))
        except (ValueError, UnicodeError) as error:
            raise TokenError(MALFORMED) from error
        if not isinstance(value, dict):
            raise TokenError(MALFORMED)
        return value

    def check_claims(
        self,
        header: Mapping[str, Any],
        claims: Mapping[str, Any],
        key: Key,
        now: float,
    ) -> Principal:
        """Checks the claims of a token whose signature is valid.

        The checks run in order: issuer, audience, subject, expiry, `nbf` and
        `iat`, then revocation.

        Args:
            header: The token's header.
            claims: The token's claims.
            key: The key that verified the signature.
            now: Epoch seconds.

        Returns:
            The verified caller.

        Raises:
            TokenError: If the issuer or audience differs, `sub` or `exp` is
                missing, `exp`, `nbf` or `iat` is not a number, the token is
                expired or not yet valid, it is revoked, or the revocation
                list is unavailable.
        """
        if claims.get("iss") != self.issuer:
            raise TokenError(BAD_ISSUER)
        audience = claims.get("aud")
        audiences = [audience] if isinstance(audience, str) else audience
        if not isinstance(audiences, list) or self.audience not in audiences:
            raise TokenError(BAD_AUDIENCE)
        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject:
            raise TokenError(MISSING_CLAIM, "sub")
        expiry = number_claim(claims, "exp")
        if expiry is None:
            raise TokenError(MISSING_CLAIM, "exp")
        if now >= expiry + self.leeway:
            raise TokenError(EXPIRED)
        for name in ("nbf", "iat"):
            value = number_claim(claims, name)
            if value is not None and value > now + self.leeway:
                raise TokenError(NOT_YET_VALID, name)
        revocations = self.current_revocations()
        if revocations is not None and revocations.revokes(header, claims):
            raise TokenError(REVOKED)
        tenant = claims.get(self.tenant_claim)
        scopes = scopes_of(claims)
        return Principal(
            subject=subject,
            tenant=tenant if isinstance(tenant, str) and tenant else None,
            scopes=scopes,
            token_id=str(claims["jti"]) if "jti" in claims else None,
            key_id=key.kid,
            expires=float(expiry),
        )


def number_claim(claims: Mapping[str, Any], name: str) -> float | None:
    """Reads an optional numeric claim.

    Args:
        claims: The token's claims.
        name: The claim name.

    Returns:
        The claim as a float, or None if absent.

    Raises:
        TokenError: `malformed` if the claim is not a number.
    """
    if name not in claims:
        return None
    value = claims[name]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TokenError(MALFORMED, name)
    return float(value)


def scopes_of(claims: Mapping[str, Any]) -> frozenset[str]:
    """Collects scopes from `scope` (space separated) or `scp` (list).

    Args:
        claims: The token's claims.

    Returns:
        The union of both; empty if neither is present.
    """
    found: set[str] = set()
    text = claims.get("scope")
    if isinstance(text, str):
        found.update(text.split())
    listed = claims.get("scp")
    if isinstance(listed, list):
        found.update(item for item in listed if isinstance(item, str))
    return frozenset(found)


def sign_hs256(secret: bytes, kid: str, claims: Mapping[str, Any]) -> str:
    """Mints an HS256 token (for the local demo and for tests).

    Args:
        secret: HMAC secret of at least `MIN_HMAC_BYTES` bytes.
        kid: Key id to put in the header.
        claims: The claims.

    Returns:
        The compact JWT.

    Raises:
        ConfigError: If the secret is too short.
    """
    if len(secret) < MIN_HMAC_BYTES:
        raise errors.ConfigError(
            f"an HS256 secret needs at least {MIN_HMAC_BYTES} bytes"
        )
    header = {"alg": HS256, "typ": "JWT", "kid": kid}
    body = ".".join(
        encode_segment(json.dumps(part, sort_keys=True).encode())
        for part in (header, dict(claims))
    )
    signature = hmac.new(secret, body.encode("ascii"), "sha256").digest()
    return f"{body}.{encode_segment(signature)}"
