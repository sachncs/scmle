"""Authorization for the prediction service: scopes, tenant, rate limits.

`Gateway.check` is the one decision point. It runs before a request body is
read and answers with a `Principal` or raises `AccessDenied`. Whatever the
cause, the body a caller sees is the one `AccessDenied.body` returns: a fixed
message per status (401, 403, 429), with no reason, key id, claim or token
fragment in it. The precise reason goes to the audit log, which the caller
cannot read.

Probe paths (`/livez`, `/readyz`) are public by design: they disclose only
liveness and readiness. Every other path needs a token with the scope the
route table names; a path missing from the table is denied.

Tenant isolation: see `docs/TENANT_ISOLATION.md`.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import hashlib
import ipaddress
import threading
import time
from typing import Any
from typing import NoReturn

from scmle.contracts import errors
from scmle.security import audit as audit_module
from scmle.security import tokens

PUBLIC_PATHS = frozenset({"/livez", "/readyz"})
SCOPE_READ = "model:read"
SCOPE_PREDICT = "model:predict"
SCOPE_ADMIN = "model:admin"
DEFAULT_ROUTES: Mapping[tuple[str, str], str] = {
    ("GET", "/health"): SCOPE_READ,
    ("GET", "/schema"): SCOPE_READ,
    ("POST", "/predict"): SCOPE_PREDICT,
    ("POST", "/v1/predict"): SCOPE_PREDICT,
}
SAFE_MESSAGES = {
    401: "authentication required",
    403: "forbidden",
    429: "too many requests",
}
SAFE_CATEGORIES = {401: "unauthenticated", 403: "forbidden", 429: "overloaded"}
MAX_TRACKED_CLIENTS = 10_000


def is_loopback(host: str) -> bool:
    """Reports whether a bind address is reachable only from this machine.

    Args:
        host: A host name or IP address.

    Returns:
        True for `localhost` and for loopback IP addresses; False for any other
        name or address, including a string that is not an IP address.
    """
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def require_authentication_for(host: str, gateway: "Gateway | None") -> None:
    """Refuses to expose a backend without authentication.

    A server may run without a gateway only on a loopback address.

    Args:
        host: The address the server binds.
        gateway: The configured gateway, if any.

    Raises:
        ConfigError: If `host` is reachable from other machines and there is
            no gateway.
    """
    if gateway is None and not is_loopback(host):
        raise errors.ConfigError(
            f"refusing to listen on {host!r} without authentication: set "
            "SCMLE_AUTH_KEYS_FILE (or SCMLE_AUTH_HS256_SECRET) with "
            "SCMLE_AUTH_ISSUER and SCMLE_AUTH_AUDIENCE, or bind a loopback "
            "address"
        )


class AccessDenied(errors.ScmleError):
    """A request was refused before it reached the model.

    Attributes:
        status: HTTP status (401, 403 or 429).
        reason: Precise cause for the audit log; never sent to the caller.
        retry_after: Seconds to wait, for 429.
        principal: The verified caller, if authentication had succeeded.
    """

    category = "auth"

    def __init__(
        self,
        status: int,
        reason: str,
        retry_after: float | None = None,
        principal: tokens.Principal | None = None,
    ) -> None:
        """Initializes the refusal.

        Args:
            status: HTTP status.
            reason: Cause for the audit log.
            retry_after: Seconds until the caller may retry.
            principal: The verified caller, when the refusal came after
                authentication (for the audit row).

        Raises:
            KeyError: If ``status`` is not 401, 403 or 429.
        """
        super().__init__(SAFE_MESSAGES[status])
        self.principal = principal
        self.status = status
        self.reason = reason
        self.retry_after = retry_after

    @property
    def body(self) -> dict[str, Any]:
        """The fixed, caller-safe error body."""
        return {
            "error": SAFE_MESSAGES[self.status],
            "category": SAFE_CATEGORIES[self.status],
        }

    @property
    def headers(self) -> dict[str, str]:
        """The response headers for the refusal.

        A 401 carries the bearer challenge, a 429 a ``Retry-After`` of
        ``retry_after`` rounded up to whole seconds (at least 1), and a 403
        none.
        """
        if self.status == 401:
            return {"WWW-Authenticate": 'Bearer realm="scmle"'}
        if self.status == 429:
            wait = max(1, int(-(-(self.retry_after or 1.0) // 1)))
            return {"Retry-After": str(wait)}
        return {}


class RateLimiter:
    """A token bucket per key, with an injectable clock.

    Each key may make `rate` requests per second on average and bursts of up to
    `burst`. State is kept in memory for at most `MAX_TRACKED_CLIENTS` keys;
    when full the least recently refilled half is dropped, which can only make
    the limiter more lenient to those keys, never lock one out.

    Attributes:
        rate: Requests per second refilled.
        burst: Bucket capacity.
        clock: Monotonic seconds.
        buckets: Level and last-refill time of each tracked key.
        lock: Guards the buckets.
    """

    def __init__(
        self,
        rate: float,
        burst: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initializes the limiter.

        Args:
            rate: Requests per second refilled.
            burst: Bucket capacity.
            clock: Monotonic seconds.

        Raises:
            ConfigError: If rate is not positive or burst is less than 1.
        """
        if rate <= 0 or burst < 1:
            raise errors.ConfigError("rate and burst must be positive")
        self.rate = rate
        self.burst = burst
        self.clock = clock
        self.buckets: dict[str, tuple[float, float]] = {}
        self.lock = threading.Lock()

    def take(self, key: str) -> float | None:
        """Takes one token from the bucket of a key.

        A key seen for the first time starts with a full bucket. A refused
        call takes nothing.

        Args:
            key: The bucket to take from (for example a subject or client key).

        Returns:
            None if allowed, otherwise seconds until a token is available.
        """
        now = self.clock()
        with self.lock:
            level, stamp = self.buckets.get(key, (float(self.burst), now))
            level = min(float(self.burst), level + (now - stamp) * self.rate)
            if level >= 1.0:
                self.buckets[key] = (level - 1.0, now)
                self.trim()
                return None
            self.buckets[key] = (level, now)
            return (1.0 - level) / self.rate

    def trim(self) -> None:
        """Bounds memory by dropping the stalest half of the keys.

        Nothing is dropped while at most `MAX_TRACKED_CLIENTS` keys are tracked.
        """
        if len(self.buckets) <= MAX_TRACKED_CLIENTS:
            return
        stale = sorted(self.buckets, key=lambda name: self.buckets[name][1])
        for name in stale[: len(stale) // 2]:
            del self.buckets[name]


def bearer_token(header: str | None) -> str | None:
    """Extracts the token of an ``Authorization: Bearer <token>`` header.

    Args:
        header: The header value, or None.

    Returns:
        The token, or None if the header is absent, is not a bearer credential,
        or has an empty token.
    """
    if header is None:
        return None
    scheme, separator, value = header.partition(" ")
    if scheme.lower() != "bearer" or not separator or not value.strip():
        return None
    return value.strip()


def client_key(remote: str) -> str:
    """Derives a fixed-size key for a remote address.

    Args:
        remote: The peer address.

    Returns:
        The first 16 hex characters of its SHA-256.
    """
    return hashlib.sha256(remote.encode()).hexdigest()[:16]


class Gateway:
    """Authenticates, authorizes and rate-limits requests, and audits them.

    Attributes:
        verifier: Verifies bearer tokens.
        tenant: The one tenant this deployment serves, or None.
        routes: Scope needed per `(method, path)`.
        limiter: Per-subject request limit.
        failure_limiter: Per-client limit on failed authentication attempts.
        audit: Where decisions are recorded, or None.
    """

    def __init__(
        self,
        verifier: tokens.TokenVerifier,
        *,
        tenant: str | None = None,
        routes: Mapping[tuple[str, str], str] = DEFAULT_ROUTES,
        limiter: RateLimiter | None = None,
        failure_limiter: RateLimiter | None = None,
        audit: audit_module.AuditLog | None = None,
    ) -> None:
        """Initializes the gateway.

        Args:
            verifier: Verifies bearer tokens.
            tenant: The one tenant this deployment serves. When set, a token
                whose tenant claim differs, or is absent, is refused (403).
            routes: Scope needed per `(method, path)`.
            limiter: Per-subject request limit (default 20/s, burst 40).
            failure_limiter: Per-client limit on failed authentication
                attempts (missing or invalid credentials), so that guessing
                credentials is slowed (default 2/s, burst 10).
            audit: Where decisions are recorded.
        """
        self.verifier = verifier
        self.tenant = tenant
        self.routes = dict(routes)
        self.limiter = limiter or RateLimiter(20.0, 40)
        self.failure_limiter = failure_limiter or RateLimiter(2.0, 10)
        self.audit = audit

    def check(
        self,
        method: str,
        path: str,
        authorization: str | None,
        remote: str,
        request_id: str,
    ) -> tokens.Principal | None:
        """Decides whether a request may proceed.

        Args:
            method: HTTP method.
            path: Request path; anything after a ``?`` is ignored.
            authorization: The ``Authorization`` header value, if any.
            remote: The peer address.
            request_id: Request id for the audit row.

        Returns:
            The verified principal, or None for a ``GET`` of a public path
            (which is not audited). Every other decision, allowed or
            denied, is recorded in the audit log when there is one.

        Raises:
            AccessDenied: With status 401, 403 or 429.
        """
        route = path.split("?", 1)[0]
        if method == "GET" and route in PUBLIC_PATHS:
            return None
        try:
            principal = self.decide(method, route, authorization, remote)
        except AccessDenied as denied:
            if self.audit is not None:
                self.audit.access(
                    denied.principal,
                    route,
                    audit_module.DENIED,
                    denied.status,
                    denied.reason,
                    request_id,
                )
            raise
        if self.audit is not None:
            self.audit.access(
                principal, route, audit_module.ALLOWED, 200, "ok", request_id
            )
        return principal

    def decide(
        self,
        method: str,
        route: str,
        authorization: str | None,
        remote: str,
    ) -> tokens.Principal:
        """Decides without auditing.

        Checks in order: the credentials, the route table, the scope, the
        tenant and the caller's rate limit.

        Args:
            method: HTTP method.
            route: Request path without query string.
            authorization: The ``Authorization`` header value, if any.
            remote: The peer address.

        Returns:
            The verified caller.

        Raises:
            AccessDenied: With status 401 for missing or invalid credentials,
                403 for a route not in the table, a missing scope or a tenant
                mismatch, and 429 when the caller is rate-limited or the
                client has made too many failed attempts.
        """
        token = bearer_token(authorization)
        if token is None:
            self.refuse(401, "no_credentials", remote)
        try:
            principal = self.verifier.verify(token or "")
        except tokens.TokenError as error:
            self.refuse(401, error.reason, remote)
        required = self.routes.get((method, route))
        if required is None:
            raise AccessDenied(403, "route_not_allowed", None, principal)
        if required not in principal.scopes:
            raise AccessDenied(
                403, f"missing_scope:{required}", None, principal
            )
        if self.tenant is not None and principal.tenant != self.tenant:
            raise AccessDenied(403, "tenant_mismatch", None, principal)
        wait = self.limiter.take(principal.subject)
        if wait is not None:
            raise AccessDenied(429, "rate_limited", wait, principal)
        return principal

    def refuse(self, status: int, reason: str, remote: str) -> NoReturn:
        """Counts a refused attempt, then raises (429 when guessing).

        Args:
            status: The status to refuse with.
            reason: The precise cause, for the audit log.
            remote: The peer address, counted against the failure limit.

        Raises:
            AccessDenied: With the given status, or with 429 if the client has
                made too many refused attempts.
        """
        wait = self.failure_limiter.take(client_key(remote))
        if wait is not None:
            raise AccessDenied(429, "too_many_failures", wait)
        raise AccessDenied(status, reason)

    def rotate(self, keys: tokens.KeySet, actor: str, reason: str) -> None:
        """Replaces the verifier's trusted keys and audits the change.

        The audit row, written only when there is an audit log, lists the
        key ids before and after.

        Args:
            keys: The new key set.
            actor: Who rotated.
            reason: Why (free text for the audit row).
        """
        previous = sorted(self.verifier.keys.keys)
        self.verifier.keys = keys
        if self.audit is not None:
            self.audit.admin(
                actor,
                "key_rotation",
                reason=reason,
                kids_before=previous,
                kids_after=sorted(keys.keys),
            )


@dataclasses.dataclass(frozen=True, slots=True)
class AuthSettings:
    """Authentication settings read from the environment.

    Attributes:
        keys_file: JWKS file path, or empty.
        hs256_secret: Inline HS256 secret (local or small deployments).
        issuer: Required issuer.
        audience: Required audience.
        tenant: The tenant this deployment serves, or empty.
        revoked_file: Revocation file path, or empty.
        audit_file: Audit ledger path, or empty for memory only.
    """

    keys_file: str
    hs256_secret: str
    issuer: str
    audience: str
    tenant: str
    revoked_file: str
    audit_file: str

    @classmethod
    def from_environment(cls, environ: Mapping[str, str]) -> "AuthSettings":
        """Reads the authentication variables of an environment.

        These are `SCMLE_AUTH_KEYS_FILE`, `SCMLE_AUTH_HS256_SECRET`,
        `SCMLE_AUTH_ISSUER`, `SCMLE_AUTH_AUDIENCE`, `SCMLE_TENANT`,
        `SCMLE_AUTH_REVOKED_FILE` and `SCMLE_AUDIT_FILE`.

        Args:
            environ: The environment; absent variables read as empty.

        Returns:
            The settings.
        """
        names = {
            "keys_file": "SCMLE_AUTH_KEYS_FILE",
            "hs256_secret": "SCMLE_AUTH_HS256_SECRET",  # nosec B105 - a name
            "issuer": "SCMLE_AUTH_ISSUER",
            "audience": "SCMLE_AUTH_AUDIENCE",
            "tenant": "SCMLE_TENANT",
            "revoked_file": "SCMLE_AUTH_REVOKED_FILE",
            "audit_file": "SCMLE_AUDIT_FILE",
        }
        return cls(**{k: environ.get(v, "") for k, v in names.items()})

    @property
    def configured(self) -> bool:
        """Whether any key source is configured."""
        return bool(self.keys_file or self.hs256_secret)


def gateway_from_environment(environ: Mapping[str, str]) -> Gateway | None:
    """Builds the gateway the environment asks for.

    A keys file takes precedence over an inline HS256 secret. The revocation
    file, if named, is re-read on each token verification rather than here.
    The audit log is opened from the audit file (in memory if none is named).

    Args:
        environ: The environment (`SCMLE_AUTH_*`, `SCMLE_TENANT`,
            `SCMLE_AUDIT_FILE`).

    Returns:
        The gateway, or None if no key source is configured.

    Raises:
        ConfigError: If keys are configured without an issuer and audience,
            or the key file or inline secret is unusable (for example an
            HS256 secret under 32 bytes).
    """
    settings = AuthSettings.from_environment(environ)
    if not settings.configured:
        return None
    if not settings.issuer or not settings.audience:
        raise errors.ConfigError(
            "SCMLE_AUTH_ISSUER and SCMLE_AUTH_AUDIENCE are required with "
            "authentication keys"
        )
    if settings.keys_file:
        keys = tokens.KeySet.from_file(settings.keys_file)
    else:
        keys = tokens.KeySet(
            [
                tokens.VerificationKey(
                    "env", tokens.HS256, secret=settings.hs256_secret.encode()
                )
            ]
        )
    revocation_source: Callable[[], tokens.RevocationList] | None = None
    if settings.revoked_file:
        path = settings.revoked_file

        def revocations() -> tokens.RevocationList:
            """Reads the revocation list from its file.

            Returns:
                The revocations currently in the file.
            """
            return tokens.RevocationList.from_file(path)

        revocation_source = revocations

    verifier = tokens.TokenVerifier(
        keys, settings.issuer, settings.audience, revocations=revocation_source
    )
    log = audit_module.AuditLog.open(settings.audit_file or None)
    return Gateway(verifier, tenant=settings.tenant or None, audit=log)
