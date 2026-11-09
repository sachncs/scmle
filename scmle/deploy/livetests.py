"""Live checks of a deployed service: health, schema, inference, security.

The checks take a ``fetch`` function so the same code runs against a real
HTTPS endpoint (`http_fetch`) and against recorded responses in tests. Each
check returns a `CheckResult` that names the environment, the source commit
and the container digest it was run for, and the digest of the exchange it is
based on, so the promotion gate can refuse evidence that belongs to another
build.

What passing means: the recorded or live answers had the documented shape and
values. It does not mean the service is correct for every input, secure
against every attack, or that a recorded transcript came from a real cloud.
"""

from collections.abc import Callable
from collections.abc import Mapping
import dataclasses
import hashlib
import json
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

from scmle.contracts import errors
from scmle.contracts import serving
from scmle.inference import parity

CHECK_NAMES = ("health", "schema", "inference", "security")
LOOPBACK = ("localhost", "127.0.0.1", "::1")
DENIED = (401, 403)


@dataclasses.dataclass(frozen=True, slots=True)
class Response:
    """One HTTP answer: status, headers (lower-case names) and body.

    Attributes:
        status: HTTP status code.
        headers: Response headers, with lower-case names.
        body: Raw response body.
    """

    status: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> dict[str, Any]:
        """Parses the body as JSON.

        Returns:
            The body as a JSON object, or an empty dict if it is not one.
        """
        try:
            value = json.loads(self.body)
        except (ValueError, UnicodeError):
            return {}
        return value if isinstance(value, dict) else {}


Fetch = Callable[[str, str, Mapping[str, str], bytes | None], Response]
"""``fetch(method, path, headers, body) -> Response``."""


@dataclasses.dataclass(frozen=True, slots=True)
class CheckResult:
    """The outcome of one check, bound to the build it ran against.

    Attributes:
        name: Check name (one of `CHECK_NAMES`).
        passed: Whether the check found no problem.
        problems: What the check found wrong; empty if it passed.
        evidence_digest: Digest of the HTTP exchanges the result rests on.
        environment: Environment the check ran against.
        source_commit: Source commit of the build checked.
        container_digest: Digest of the image checked.
        model_digest: Digest of the model checked.
    """

    name: str
    passed: bool
    problems: tuple[str, ...]
    evidence_digest: str
    environment: str
    source_commit: str
    container_digest: str
    model_digest: str

    def to_dict(self) -> dict[str, Any]:
        """Builds a JSON-ready record.

        Returns:
            A dict of all fields, with `problems` as a list.
        """
        record = dataclasses.asdict(self)
        record["problems"] = list(self.problems)
        return record

    @classmethod
    def from_dict(cls, record: Mapping[str, Any]) -> "CheckResult":
        """Rebuilds a result from `to_dict`.

        Args:
            record: A record produced by `to_dict`.

        Returns:
            The result.

        Raises:
            ConfigError: If the record's fields are not exactly the result's.
        """
        names = {f.name for f in dataclasses.fields(cls)}
        if set(record) != names:
            raise errors.ConfigError("unexpected or missing check fields")
        values = dict(record)
        values["problems"] = tuple(values["problems"])
        return cls(**values)


@dataclasses.dataclass(frozen=True, slots=True)
class Target:
    """What the checks are run for.

    Attributes:
        environment: Environment name.
        source_commit: Source commit of the build.
        container_digest: Digest of the deployed image.
        model_digest: Digest of the deployed model.
        token: Bearer token of a caller allowed to use the service.
        base_url: Base URL of the service.
        read_only_token: Token of a caller that must not reach `/predict`; None
            skips that check.
    """

    environment: str
    source_commit: str
    container_digest: str
    model_digest: str
    token: str
    base_url: str = "https://localhost"
    read_only_token: str | None = None


class Recorder:
    """Wraps ``fetch`` and keeps a digest of every exchange.

    Attributes:
        fetch: The wrapped request function.
        exchanges: One record per request made so far.
    """

    def __init__(self, fetch: Fetch) -> None:
        """Starts with no exchanges.

        Args:
            fetch: The request function to wrap.
        """
        self.fetch = fetch
        self.exchanges: list[dict[str, Any]] = []

    def call(
        self,
        method: str,
        path: str,
        headers: Mapping[str, str] | None = None,
        body: bytes | None = None,
    ) -> Response:
        """Sends a request and remembers what came back (never the token).

        Args:
            method: HTTP method.
            path: Request path.
            headers: Request headers; none if None.
            body: Request body; none if None.

        Returns:
            The response.
        """
        response = self.fetch(method, path, dict(headers or {}), body)
        self.exchanges.append(
            {
                "method": method,
                "path": path,
                "status": response.status,
                "response_sha256": hashlib.sha256(response.body).hexdigest(),
            }
        )
        return response

    def digest(self) -> str:
        """Digests the exchanges so far.

        Returns:
            `sha256:` followed by the hex SHA-256 of the recorded method, path,
            status and response hash of every exchange.
        """
        payload = json.dumps(self.exchanges, sort_keys=True).encode()
        return "sha256:" + hashlib.sha256(payload).hexdigest()


def bearer(token: str | None) -> dict[str, str]:
    """Builds the Authorization header for ``token``, or none.

    Args:
        token: The bearer token, or None.

    Returns:
        A one-entry header dict, or an empty dict if ``token`` is None.
    """
    return {} if token is None else {"Authorization": f"Bearer {token}"}


def json_headers(token: str | None) -> dict[str, str]:
    """Builds the headers of a JSON request.

    Args:
        token: The bearer token, or None for an anonymous request.

    Returns:
        A dict with the JSON content type and, if given, the Authorization
        header.
    """
    return {"Content-Type": "application/json", **bearer(token)}


def finish(
    name: str, target: Target, wire: Recorder, problems: list[str]
) -> CheckResult:
    """Builds the result of one check.

    Args:
        name: Check name.
        target: What the check was run for.
        wire: The recorder holding the check's exchanges.
        problems: What the check found wrong.

    Returns:
        A result that passes exactly when ``problems`` is empty.
    """
    return CheckResult(
        name,
        not problems,
        tuple(problems),
        wire.digest(),
        target.environment,
        target.source_commit,
        target.container_digest,
        target.model_digest,
    )


def check_health(fetch: Fetch, target: Target) -> CheckResult:
    """Checks `/livez`, `/readyz` and `/health` and the model identity.

    Args:
        fetch: The request function.
        target: What the check is run for.

    Returns:
        The result; it fails if `/livez` or `/readyz` does not answer 200 or the
        service is not ready, `/health` is not ok, or the reported artifact
        digest is not the target's model digest.
    """
    wire = Recorder(fetch)
    problems: list[str] = []
    live = wire.call("GET", "/livez")
    if live.status != 200:
        problems.append(f"/livez answered {live.status}")
    ready = wire.call("GET", "/readyz")
    body = ready.json()
    if ready.status != 200 or body.get("ready") is not True:
        problems.append(f"/readyz answered {ready.status} and is not ready")
    health = wire.call("GET", "/health", bearer(target.token))
    info = health.json()
    if health.status != 200 or info.get("status") != "ok":
        problems.append(f"/health answered {health.status}")
    elif info.get("artifact_digest") != target.model_digest:
        problems.append("the service runs a different model than the release")
    return finish("health", target, wire, problems)


def check_schema(fetch: Fetch, target: Target) -> CheckResult:
    """Checks that `/schema` serves the contract, digest and model identity.

    Args:
        fetch: The request function.
        target: What the check is run for.

    Returns:
        The result; it fails if `/schema` does not answer 200, lacks a schema
        version, digest or request/response schema, or names a different model
        digest.
    """
    wire = Recorder(fetch)
    problems: list[str] = []
    reply = wire.call("GET", "/schema", bearer(target.token))
    body = reply.json()
    if reply.status != 200:
        problems.append(f"/schema answered {reply.status}")
    else:
        for key in (
            "schema_version",
            "schema_digest",
            "request_schema",
            "response_schema",
        ):
            if key not in body:
                problems.append(f"/schema lacks {key}")
        if body.get("artifact_digest") != target.model_digest:
            problems.append("/schema names a different model digest")
    return finish("schema", target, wire, problems)


def check_inference(
    fetch: Fetch,
    target: Target,
    golden: parity.GoldenSet,
    expected: dict[str, Any],
) -> CheckResult:
    """Checks that the golden records score within the declared bound.

    Args:
        fetch: The request function.
        target: What the check is run for.
        golden: The golden records and their cross-environment tolerances.
        expected: The reference predictions and probabilities to compare with.

    Returns:
        The result; it fails if `/predict` does not answer 200, the response
        lacks required keys, the scores differ beyond the bound, or the
        answering model is not the target's.
    """
    wire = Recorder(fetch)
    problems: list[str] = []
    request = json.dumps({"records": golden.records}).encode()
    reply = wire.call("POST", "/predict", json_headers(target.token), request)
    body = reply.json()
    if reply.status != 200:
        problems.append(f"/predict answered {reply.status}")
        return finish("inference", target, wire, problems)
    missing = [k for k in serving.RESPONSE_KEYS if k not in body]
    if missing:
        problems.append(f"response lacks {missing}")
    else:
        report = parity.compare(
            expected,
            body["predictions"],
            body["probabilities"],
            golden.cross_environment,
        )
        problems.extend(report.mismatches)
    if body.get("artifact_digest") != target.model_digest:
        problems.append("the answering model is not the released model")
    return finish("inference", target, wire, problems)


def check_security(fetch: Fetch, target: Target) -> CheckResult:
    """Checks that anonymous and forged callers are refused, over TLS.

    Args:
        fetch: The request function.
        target: What the check is run for.

    Returns:
        The result; it fails if the service is reached over plain HTTP outside
        loopback, a request without a token or with a forged token is not
        answered 401 or 403 or leaks a traceback, a 401 lacks a WWW-Authenticate
        header, or the read-only token (when given) is not answered 403 on
        `/predict`.
    """
    wire = Recorder(fetch)
    problems: list[str] = []
    host = urllib.parse.urlparse(target.base_url)
    if host.scheme != "https" and host.hostname not in LOOPBACK:
        problems.append("the service is not reached over HTTPS")
    request = json.dumps({"records": []}).encode()
    for label, token in (
        ("no token", None),
        ("forged token", "forged.token.x"),
    ):
        reply = wire.call("POST", "/predict", json_headers(token), request)
        if reply.status not in DENIED:
            problems.append(f"/predict with {label} answered {reply.status}")
        if b"Traceback" in reply.body:
            problems.append(f"the refusal with {label} leaks a traceback")
    anonymous = wire.call("GET", "/schema")
    if anonymous.status not in DENIED:
        problems.append(f"/schema without a token answered {anonymous.status}")
    challenge = wire.call("POST", "/predict", json_headers(None), request)
    if challenge.status == 401 and "www-authenticate" not in {
        k.lower() for k in challenge.headers
    }:
        problems.append("a 401 carries no WWW-Authenticate header")
    if target.read_only_token is not None:
        scoped = wire.call(
            "POST", "/predict", json_headers(target.read_only_token), request
        )
        if scoped.status != 403:
            problems.append(
                f"a read-only token reached /predict: {scoped.status}"
            )
    return finish("security", target, wire, problems)


def run_checks(
    fetch: Fetch,
    target: Target,
    golden: parity.GoldenSet,
    expected: dict[str, Any],
) -> list[CheckResult]:
    """Runs the four checks in order; a failing check does not stop the rest.

    Args:
        fetch: The request function.
        target: What the checks are run for.
        golden: The golden records and their cross-environment tolerances.
        expected: The reference predictions and probabilities for the inference
            check.

    Returns:
        The health, schema, inference and security results, in that order.
    """
    return [
        check_health(fetch, target),
        check_schema(fetch, target),
        check_inference(fetch, target, golden, expected),
        check_security(fetch, target),
    ]


def http_fetch(base_url: str, timeout: float = 30.0) -> Fetch:
    """Builds a `Fetch` over HTTP(S) to ``base_url`` (no redirects followed).

    Args:
        base_url: Absolute http(s) URL of the service.
        timeout: Seconds to wait for each request.

    Returns:
        A function that sends a request and returns the response; HTTP error
        statuses are returned as responses, not raised.

    Raises:
        ConfigError: If the URL is not an absolute http(s) URL.
    """
    parsed = urllib.parse.urlparse(base_url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise errors.ConfigError("base_url must be an absolute http(s) URL")
    root = base_url.rstrip("/")

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        """Redirects are answers, not something to follow with a token."""

        def redirect_request(self, *arguments: Any, **keywords: Any) -> None:
            """Refuses to follow a redirect.

            Args:
                *arguments: Ignored.
                **keywords: Ignored.

            Returns:
                None, so the redirect response is returned to the caller.
            """
            return None

    opener = urllib.request.build_opener(NoRedirect)

    def fetch(
        method: str,
        path: str,
        headers: Mapping[str, str],
        body: bytes | None,
    ) -> Response:
        """Sends one request to the service.

        Args:
            method: HTTP method.
            path: Request path, appended to the base URL.
            headers: Request headers.
            body: Request body, or None.

        Returns:
            The response, with lower-case header names. A non-2xx status is
            returned, not raised.
        """
        request = urllib.request.Request(
            root + path, data=body, headers=dict(headers), method=method
        )
        try:
            with opener.open(request, timeout=timeout) as reply:  # nosec B310
                return Response(
                    reply.status,
                    {k.lower(): v for k, v in reply.headers.items()},
                    reply.read(),
                )
        except urllib.error.HTTPError as error:
            return Response(
                error.code,
                {k.lower(): v for k, v in error.headers.items()},
                error.read(),
            )

    return fetch
