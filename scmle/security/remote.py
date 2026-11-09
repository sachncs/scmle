"""Keys and revocations fetched over the network.

`tokens.KeySet.from_file` and `RevocationList.from_file` read files. This
module adds the pull side, with the standard library only:

* `discover` reads an OpenID Connect discovery document and checks that its
  `issuer` is exactly the issuer asked for;
* `RemoteKeySet` is a key set backed by a JWKS URL: cached for `ttl`
  seconds, refetched once for an unknown `kid` (at most every `min_refresh`
  seconds, so a stream of forged `kid` values cannot drive fetches), served
  stale for at most `max_stale` seconds when the source is down, and after
  that trusting nothing (`KEYS_UNAVAILABLE`);
* `RemoteRevocations` is a callable for `TokenVerifier(revocations=...)`:
  every replica that pulls the same URL sees a new revocation within `ttl`
  seconds plus its fetch time. When the source stays down past `max_stale`
  seconds the verifier raises `REVOCATION_UNAVAILABLE` and accepts no token
  (fail closed).

Limits. Only `https` URLs are fetched unless `allow_http` is set (tests, a
local mock identity provider). The documents are not signed: their integrity
is that of the TLS connection to the origin, so the origin must be trusted.
Distribution is by pull, so a revocation reaches a replica after at most the
cache lifetime, never instantly. Nothing here ran against a real identity
provider or inside a cloud.
"""

from collections.abc import Callable
import json
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from scmle.contracts import errors
from scmle.security import tokens

KEYS_UNAVAILABLE = "keys_unavailable"
TIMEOUT_SECONDS = 5.0
MAX_BYTES = 1_000_000
DEFAULT_TTL = 300.0
DEFAULT_MIN_REFRESH = 10.0
DEFAULT_MAX_STALE = 3600.0
DISCOVERY_PATH = "/.well-known/openid-configuration"


def fetch_json(
    url: str,
    *,
    timeout: float = TIMEOUT_SECONDS,
    max_bytes: int = MAX_BYTES,
    allow_http: bool = False,
) -> Any:
    """Fetches and parses one JSON document.

    Args:
        url: Where to fetch it.
        timeout: Seconds to wait.
        max_bytes: Largest body accepted.
        allow_http: Permit plain http (tests only).

    Returns:
        The parsed JSON value.

    Raises:
        ConfigError: For a disallowed scheme, a failed request, an oversized
            body or a body that is not JSON.
    """
    scheme = urllib.parse.urlparse(url).scheme
    if scheme != "https" and not (allow_http and scheme == "http"):
        raise errors.ConfigError(f"only https URLs are fetched, not {url!r}")
    request = urllib.request.Request(
        url, headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(  # nosec B310
            request, timeout=timeout
        ) as response:
            body = response.read(max_bytes + 1)
    except (urllib.error.URLError, OSError, ValueError) as error:
        raise errors.ConfigError(f"cannot fetch {url}") from error
    if len(body) > max_bytes:
        raise errors.ConfigError(f"{url} is larger than {max_bytes} bytes")
    try:
        return json.loads(body.decode("utf-8"))
    except ValueError as error:
        raise errors.ConfigError(f"{url} is not JSON") from error


def discover(
    issuer: str,
    *,
    allow_http: bool = False,
    fetch: Callable[..., Any] = fetch_json,
) -> dict[str, Any]:
    """Fetches the discovery document of `issuer`, checked to name that issuer.

    Args:
        issuer: The issuer URL.
        allow_http: Permit plain http (tests only).
        fetch: The JSON fetcher.

    Returns:
        The discovery document.

    Raises:
        ConfigError: If it cannot be fetched, is not an object, names another
            issuer or has no `jwks_uri`.
    """
    document = fetch(issuer.rstrip("/") + DISCOVERY_PATH, allow_http=allow_http)
    if not isinstance(document, dict):
        raise errors.ConfigError("the discovery document is not an object")
    if document.get("issuer") != issuer:
        raise errors.ConfigError("the discovery document names another issuer")
    if not isinstance(document.get("jwks_uri"), str):
        raise errors.ConfigError("the discovery document has no jwks_uri")
    return document


class RemoteKeySet(tokens.KeySet):
    """A key set that refreshes itself from a JWKS URL.

    Attributes:
        uri: Where the JWKS document is served.
        ttl: Seconds a fetched set is used before it is refetched.
        min_refresh: Least seconds between fetches caused by an unknown `kid`.
        max_stale: Seconds a set may be served after its refresh began to fail.
        clock: Epoch seconds.
        allow_http: Whether plain http is permitted.
        fetcher: The JSON fetcher.
        fetched_at: Clock reading of the last successful fetch.
    """

    def __init__(
        self,
        jwks_uri: str,
        *,
        ttl: float = DEFAULT_TTL,
        min_refresh: float = DEFAULT_MIN_REFRESH,
        max_stale: float = DEFAULT_MAX_STALE,
        clock: Callable[[], float] = time.time,
        allow_http: bool = False,
        fetch: Callable[..., Any] = fetch_json,
    ) -> None:
        """Fetches the key set once; it must succeed.

        A first fetch that fails, or a document that is bad, raises
        `ConfigError`.

        Args:
            jwks_uri: Where the JWKS document is served.
            ttl: Seconds a fetched set is used before it is refetched.
            min_refresh: Least seconds between fetches caused by an unknown
                `kid`.
            max_stale: Seconds a set may be served after its refresh began to
                fail; beyond that no key is trusted.
            clock: Epoch seconds (injectable for tests).
            allow_http: Permit plain http (tests only).
            fetch: The JSON fetcher (injectable).
        """
        self.uri = jwks_uri
        self.ttl = ttl
        self.min_refresh = min_refresh
        self.max_stale = max_stale
        self.clock = clock
        self.allow_http = allow_http
        self.fetcher = fetch
        self.fetched_at = 0.0
        self.refresh_started = False
        loaded = self.load()
        super().__init__(list(loaded.keys.values()))
        self.fetched_at = self.clock()

    def load(self) -> tokens.KeySet:
        """Fetches and parses the JWKS document.

        Returns:
            The key set in the document.
        """
        return tokens.KeySet.from_jwks(
            self.fetcher(self.uri, allow_http=self.allow_http)
        )

    def refresh(self) -> bool:
        """Replaces the keys from the JWKS document.

        Returns:
            True if the keys were replaced; False, with the keys unchanged, if
            the fetch failed.
        """
        try:
            loaded = self.load()
        except errors.ConfigError:
            return False
        self.keys = loaded.keys
        self.fetched_at = self.clock()
        return True

    def get(self, kid: str) -> tokens.VerificationKey:
        """Looks up the key with this id, refreshing the set when it is due.

        The set is refetched when older than `ttl`, and also for an unknown
        `kid` when older than `min_refresh`.

        Args:
            kid: The key id.

        Returns:
            The key.

        Raises:
            TokenError: `unknown_key` for a key the set does not hold, or
                `keys_unavailable` once the set is too stale to trust.
        """
        age = self.clock() - self.fetched_at
        if age >= self.ttl:
            self.refresh()
            age = self.clock() - self.fetched_at
            if age >= self.ttl + self.max_stale:
                raise tokens.TokenError(KEYS_UNAVAILABLE)
        elif kid not in self.keys and age >= self.min_refresh:
            self.refresh()
        return super().get(kid)


class RemoteRevocations:
    """A pulled revocation list, usable as `TokenVerifier(revocations=...)`.

    Attributes:
        url: Where the revocation document is served.
        ttl: Seconds a fetched list is used.
        max_stale: Seconds a list may be served after its refresh began to fail.
        clock: Epoch seconds.
        allow_http: Whether plain http is permitted.
        fetcher: The JSON fetcher.
        current: The list in force, or None before the first fetch.
        fetched_at: Clock reading of the last successful fetch.
    """

    def __init__(
        self,
        url: str,
        *,
        ttl: float = 30.0,
        max_stale: float = DEFAULT_MAX_STALE,
        clock: Callable[[], float] = time.time,
        allow_http: bool = False,
        fetch: Callable[..., Any] = fetch_json,
    ) -> None:
        """Creates the puller; the first call fetches.

        Args:
            url: Where `{"tokens": [], "subjects": [], "key_ids": []}` is
                served.
            ttl: Seconds a fetched list is used.
            max_stale: Seconds a list may be served after its refresh began
                to fail.
            clock: Epoch seconds.
            allow_http: Permit plain http (tests only).
            fetch: The JSON fetcher.
        """
        self.url = url
        self.ttl = ttl
        self.max_stale = max_stale
        self.clock = clock
        self.allow_http = allow_http
        self.fetcher = fetch
        self.current: tokens.RevocationList | None = None
        self.fetched_at = 0.0

    def __call__(self) -> tokens.RevocationList:
        """Returns the list in force, refreshing it when due.

        Returns:
            The cached list while it is younger than `ttl`; otherwise a freshly
            fetched list, or the cached one if the fetch failed and it is
            younger than `ttl + max_stale`.

        Raises:
            ConfigError: If the list is malformed, or cannot be refreshed and is
                older than `ttl + max_stale`.
        """
        now = self.clock()
        if self.current is not None and now - self.fetched_at < self.ttl:
            return self.current
        try:
            document = self.fetcher(self.url, allow_http=self.allow_http)
            if not isinstance(document, dict):
                raise errors.ConfigError("the revocation list is not an object")
            fresh = tokens.RevocationList(
                document.get("tokens", ()),
                document.get("subjects", ()),
                document.get("key_ids", ()),
            )
        except errors.ConfigError:
            if (
                self.current is not None
                and now - self.fetched_at < self.ttl + self.max_stale
            ):
                return self.current
            raise
        self.current, self.fetched_at = fresh, now
        return fresh
