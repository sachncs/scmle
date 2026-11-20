"""Keys and revocations fetched over the network.

`tokens.Keystore.read` and `Revocations.read` read files. This
module adds the pull side, with the standard library only:

* `discover` reads an OpenID Connect discovery document and checks that its
  `issuer` is exactly the issuer asked for;
* `Keystore` is a key set backed by a JWKS URL: cached for `ttl`
  seconds, refetched once for an unknown `kid` (at most every `min_refresh`
  seconds, so a stream of forged `kid` values cannot drive fetches), served
  stale for at most `max_stale` seconds when the source is down, and after
  that trusting nothing (`KEYS_UNAVAILABLE`);
* `Revocations` is a callable for `Verifier(revocations=...)`:
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
import logging
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from scmle.contracts import errors
from scmle.security import tokens

LOGGER = logging.getLogger(__name__)

KEYS_UNAVAILABLE = "keys_unavailable"
TIMEOUT_SECONDS = 5.0
MAX_BYTES = 1_000_000
DEFAULT_TTL = 300.0
DEFAULT_MIN_REFRESH = 10.0
DEFAULT_MAX_STALE = 3600.0
DISCOVERY_PATH = "/.well-known/openid-configuration"


def retrieve(
    url: str,
    *,
    timeout: float = TIMEOUT_SECONDS,
    max_bytes: int = MAX_BYTES,
    allow_http: bool = False,
) -> Any:
    """Fetches and parses one JSON document.

    Args:
        url: Where to fetch it; only ``https`` unless ``allow_http`` is set.
        timeout: Seconds to wait for the connection and for each read.
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
    request = urllib.request.Request(  # noqa: S310 -- the scheme was checked above
        url, headers={"Accept": "application/json"}
    )
    try:
        with urllib.request.urlopen(  # noqa: S310 -- the scheme was checked above  # nosec B310
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
    fetch: Callable[..., Any] = retrieve,
) -> dict[str, Any]:
    """Fetches the discovery document of `issuer`, checked to name that issuer.

    The document is read from the issuer URL (without a trailing slash) plus
    ``/.well-known/openid-configuration``.

    Args:
        issuer: The issuer URL.
        allow_http: Permit plain http (tests only).
        fetch: The JSON fetcher.

    Returns:
        The discovery document.

    Raises:
        ConfigError: If it cannot be fetched (raised by the fetcher), is not
            an object, names an issuer that is not exactly `issuer`, or has no
            text `jwks_uri`.
    """
    document = fetch(issuer.rstrip("/") + DISCOVERY_PATH, allow_http=allow_http)
    if not isinstance(document, dict):
        raise errors.ConfigError("the discovery document is not an object")
    if document.get("issuer") != issuer:
        raise errors.ConfigError("the discovery document names another issuer")
    if not isinstance(document.get("jwks_uri"), str):
        raise errors.ConfigError("the discovery document has no jwks_uri")
    return document


class Keystore(tokens.Keystore):
    """A key set that refreshes itself from a JWKS URL.

    Attributes:
        uri: Where the JWKS document is served.
        ttl: Seconds a fetched set is used before it is refetched.
        cooldown: Least seconds between fetches caused by an unknown `kid`.
        staleness: Seconds a set may be served after its refresh began to fail.
        clock: Epoch seconds.
        http: Whether plain http is permitted.
        fetcher: The JSON fetcher.
        fetched: Clock reading of the last successful fetch.
    """

    def __init__(
        self,
        jwks_uri: str,
        *,
        ttl: float = DEFAULT_TTL,
        cooldown: float = DEFAULT_MIN_REFRESH,
        staleness: float = DEFAULT_MAX_STALE,
        clock: Callable[[], float] = time.time,
        http: bool = False,
        fetch: Callable[..., Any] = retrieve,
    ) -> None:
        """Fetches the key set once; it must succeed.

        Args:
            jwks_uri: Where the JWKS document is served.
            ttl: Seconds a fetched set is used before it is refetched.
            cooldown: Least seconds between fetches caused by an unknown
                `kid`.
            staleness: Seconds a set may be served after its refresh began to
                fail; beyond that no key is trusted.
            clock: Epoch seconds (injectable for tests).
            http: Permit plain http (tests only).
            fetch: The JSON fetcher (injectable).

        Raises:
            ConfigError: If the first fetch fails or the document is not a
                valid JWKS with at least one key.
        """
        self.uri = jwks_uri
        self.ttl = ttl
        self.cooldown = cooldown
        self.staleness = staleness
        self.clock = clock
        self.http = http
        self.fetcher = fetch
        self.fetched = 0.0
        self.started = False
        loaded = self.load()
        super().__init__(list(loaded.keys.values()))
        self.fetched = self.clock()

    def load(self) -> tokens.Keystore:
        """Fetches and parses the JWKS document.

        Returns:
            The key set in the document.

        Raises:
            ConfigError: If the fetch fails or the document is not a valid
                JWKS.
        """
        return tokens.Keystore.jwks(
            self.fetcher(self.uri, allow_http=self.http)
        )

    def refresh(self) -> bool:
        """Replaces the keys from the JWKS document.

        Returns:
            True if the keys were replaced; False, with the keys unchanged, if
            the fetch failed.
        """
        try:
            loaded = self.load()
        except errors.ConfigError as error:
            LOGGER.warning(
                "WARNING key set %s did not refresh; the cached keys stay in "
                "use: %s",
                self.uri,
                error,
            )
            return False
        self.keys = loaded.keys
        self.fetched = self.clock()
        return True

    def get(self, kid: str) -> tokens.Key:
        """Looks up the key with this id, refreshing the set when it is due.

        The set is refetched when it is `ttl` seconds old or more, and for
        an unknown `kid` when it is `cooldown` seconds old or more. If the
        refetch fails the old set is still used until it is `ttl` plus
        `staleness` seconds old.

        Args:
            kid: The key id.

        Returns:
            The key.

        Raises:
            TokenError: `unknown_key` for a key the set does not hold, or
                `keys_unavailable` once the set is too stale to trust.
        """
        age = self.clock() - self.fetched
        if age >= self.ttl:
            self.refresh()
            age = self.clock() - self.fetched
            if age >= self.ttl + self.staleness:
                raise tokens.TokenError(KEYS_UNAVAILABLE)
        elif kid not in self.keys and age >= self.cooldown:
            self.refresh()
        return super().get(kid)


def revoked(document: Any) -> tokens.Revocations:
    """Builds a revocation list from a fetched document.

    Args:
        document: The parsed JSON; an object with optional `tokens`,
            `subjects` and `key_ids`.

    Returns:
        The revocation list.

    Raises:
        ConfigError: If the document is not an object.
    """
    if not isinstance(document, dict):
        raise errors.ConfigError("the revocation list is not an object")
    return tokens.Revocations(
        document.get("tokens", ()),
        document.get("subjects", ()),
        document.get("key_ids", ()),
    )


class Revocations:
    """A pulled revocation list, usable as `Verifier(revocations=...)`.

    Attributes:
        url: Where the revocation document is served.
        ttl: Seconds a fetched list is used.
        staleness: Seconds a list may be served after its refresh began to fail.
        clock: Epoch seconds.
        http: Whether plain http is permitted.
        fetcher: The JSON fetcher.
        current: The list in force, or None before the first fetch.
        fetched: Clock reading of the last successful fetch.
    """

    def __init__(
        self,
        url: str,
        *,
        ttl: float = 30.0,
        staleness: float = DEFAULT_MAX_STALE,
        clock: Callable[[], float] = time.time,
        http: bool = False,
        fetch: Callable[..., Any] = retrieve,
    ) -> None:
        """Creates the puller; the first call fetches.

        Args:
            url: Where `{"tokens": [], "subjects": [], "key_ids": []}` is
                served.
            ttl: Seconds a fetched list is used.
            staleness: Seconds a list may be served after its refresh began
                to fail.
            clock: Epoch seconds.
            http: Permit plain http (tests only).
            fetch: The JSON fetcher.
        """
        self.url = url
        self.ttl = ttl
        self.staleness = staleness
        self.clock = clock
        self.http = http
        self.fetcher = fetch
        self.current: tokens.Revocations | None = None
        self.fetched = 0.0

    def __call__(self) -> tokens.Revocations:
        """Returns the list in force, refreshing it when due.

        Returns:
            The cached list while it is younger than `ttl`; otherwise a freshly
            fetched list, or the cached one if the fetch failed or the
            document was malformed and the cached list is younger than
            `ttl + max_stale` (each call then tries again to fetch).

        Raises:
            ConfigError: If the fetch fails or the list is malformed and
                there is no cached list younger than `ttl + max_stale`
                (including on the first call).
        """
        now = self.clock()
        if self.current is not None and now - self.fetched < self.ttl:
            return self.current
        try:
            fresh = revoked(self.fetcher(self.url, allow_http=self.http))
        except errors.ConfigError as error:
            if (
                self.current is not None
                and now - self.fetched < self.ttl + self.staleness
            ):
                LOGGER.warning(
                    "WARNING revocation list %s did not refresh; the cached "
                    "list is used until it is %ss stale: %s",
                    self.url,
                    self.staleness,
                    error,
                )
                return self.current
            raise
        self.current, self.fetched = fresh, now
        return fresh
