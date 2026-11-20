r"""Base images pinned by digest, and resolving a tag to its digest.

A Dockerfile never trusts a tag. `PINNED` records the digest of the
multi-architecture index (so amd64 and arm64 builds use the same base) of
each Python base image scmle builds on, resolved on `PINNED_AT`; generated
Dockerfiles default to it. The placeholder digest stays refused by every
build, for a base that is not pinned. `resolve` asks the registry again (to
refresh a pin or to pin another base), `argument` prints the `--build-arg`
value of a lock, `check` fails if a lock is missing, malformed or still the
placeholder, and `pinned` prints a recorded pin.

    python -m scmle.deploy.base_image resolve --image python:3.12-slim \
        --output base-image.lock.json
    docker build --build-arg BASE_IMAGE="$(python -m scmle.deploy.base_image \
        argument --lock base-image.lock.json)" .

Resolving speaks the registry protocol (distribution spec v2): an anonymous
`HEAD` of the manifest, and on a `401` the bearer token its
`WWW-Authenticate` challenge names. A registry on `localhost` is spoken to
over plain HTTP, as Docker does.
"""

import argparse
from collections.abc import Callable
import datetime
import http
import json
import pathlib
import re
import sys
import urllib.error
import urllib.parse
import urllib.request

from scmle.contracts import errors
from scmle.deploy import contract

PLACEHOLDER = "sha256:" + "0" * 64
INDEX_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
)
DEFAULT_REGISTRY = "registry-1.docker.io"
LOCAL_HOSTS = ("localhost", "127.0.0.1", "[::1]")
CHALLENGE = re.compile(r'(\w+)="([^"]*)"')
PINNED_AT = "2026-11-19"
PINNED = {
    "python:3.11-slim": "sha256:"
    "07d4824036af339fa1b6ef769cc99d9c10a141769538b52b3b597c84f7a7b535",
    "python:3.12-slim": "sha256:"
    "5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b",
    "python:3.13-slim": "sha256:"
    "e48f59cdba69a3cbd9ed2cf0f348d90d45a8d350d98134cf1070bc106ee9ed90",
    "python:3.14-slim": "sha256:"
    "9aaa474097d094040649f2162a627c0049b408f52138d980916788d80f04802f",
}
IMAGE_PATTERN = re.compile(
    r"(?:(?P<registry>[a-z0-9.-]+(?::[0-9]+)?)/)?"
    r"(?P<repository>[a-z0-9][a-z0-9._/-]*):(?P<tag>[A-Za-z0-9._-]+)"
)
Fetcher = Callable[[str, str], str]


def decompose(image: str) -> tuple[str, str, str]:
    """Splits `repo:tag` into (registry, repository, tag).

    Docker Hub is assumed when no registry is given, and official images get the
    `library/` prefix.

    Args:
        image: Image reference in the form `[registry/]repository:tag`.

    Returns:
        The registry host, the repository and the tag.

    Raises:
        ConfigError: If the reference is not `repository:tag` (a digest or a
            missing tag is refused).
    """
    match = IMAGE_PATTERN.fullmatch(image)
    if match is None or "@" in image:
        raise errors.ConfigError(f"expected repository:tag, got {image!r}")
    registry = match["registry"]
    repository = match["repository"]
    if registry is None or ("." not in registry and ":" not in registry):
        if registry is not None:
            repository = f"{registry}/{repository}"
        registry = DEFAULT_REGISTRY
    if registry == DEFAULT_REGISTRY and "/" not in repository:
        repository = f"library/{repository}"
    return registry, repository, match["tag"]


def pinned(image: str) -> str:
    """Returns the recorded digest of a base image.

    Args:
        image: `repository:tag`.

    Returns:
        The digest recorded in `PINNED`, or `PLACEHOLDER` (which every build
        refuses) for a base that is not pinned.
    """
    decompose(image)
    return PINNED.get(image, PLACEHOLDER)


def origin(registry: str) -> str:
    """Builds the base URL of a registry.

    Args:
        registry: The registry host, with an optional port.

    Returns:
        `http://` for a registry on this machine, otherwise `https://`.
    """
    host = registry.rsplit(":", 1)[0] if "]" not in registry else registry
    local = host in LOCAL_HOSTS or registry.startswith("[::1]")
    return f"{'http' if local else 'https'}://{registry}"


def authorize(challenge: str, base: str, timeout: float) -> str:
    """Fetches the bearer token a registry's challenge asks for.

    Args:
        challenge: The `WWW-Authenticate` header of the `401`.
        base: The registry's base URL; the token realm must use its scheme
            or `https`.
        timeout: Seconds to wait.

    Returns:
        The token.

    Raises:
        ArtifactError: If the challenge is not a bearer challenge, its realm
            is not HTTP(S), or the reply holds no token.
    """
    scheme, rest = [*challenge.split(" ", 1), ""][:2]
    fields = dict(CHALLENGE.findall(rest))
    realm = fields.pop("realm", "")
    allowed = {"https", urllib.parse.urlsplit(base).scheme}
    if scheme.lower() != "bearer" or urllib.parse.urlsplit(
        realm
    ).scheme not in (allowed):
        raise errors.ArtifactError(
            f"unsupported registry challenge {challenge!r}"
        )
    address = f"{realm}?{urllib.parse.urlencode(fields)}" if fields else realm
    request = urllib.request.Request(address)  # noqa: S310 -- the scheme is checked above
    try:
        with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 -- the scheme is checked above  # nosec B310
            body = json.loads(reply.read())
        token: str = body.get("token") or body["access_token"]
    except (OSError, json.JSONDecodeError, KeyError, AttributeError) as error:
        raise errors.ArtifactError(f"no token from {realm}") from error
    return token


def head(address: str, headers: dict[str, str], timeout: float) -> str:
    """Asks for a manifest's media type and digest.

    Args:
        address: The manifest URL.
        headers: Request headers.
        timeout: Seconds to wait.

    Returns:
        `<media type> <digest>`.
    """
    request = urllib.request.Request(  # noqa: S310 -- http(s) by construction
        address, method="HEAD", headers=headers
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:  # noqa: S310 -- http(s) by construction  # nosec B310
        kind = reply.headers.get("Content-Type", "").split(";")[0]
        return f"{kind} {reply.headers.get('Docker-Content-Digest', '')}"


def query(image: str, timeout: float = 30.0) -> str:
    """Asks the registry for the digest of the index behind `image`.

    Args:
        image: Image reference as `[registry/]repository:tag`.
        timeout: Seconds to wait for each registry request.

    Returns:
        The `Docker-Content-Digest` of the image index.

    Raises:
        ArtifactError: If the registry cannot be reached, refuses the
            request, or answers with something other than a
            multi-architecture index.
    """
    registry, repository, tag = decompose(image)
    base = origin(registry)
    address = f"{base}/v2/{repository}/manifests/{tag}"
    headers = {"Accept": ", ".join(INDEX_TYPES)}
    try:
        try:
            answer = head(address, headers, timeout)
        except urllib.error.HTTPError as error:
            challenge = error.headers.get("WWW-Authenticate", "")
            if error.code != http.HTTPStatus.UNAUTHORIZED or not challenge:
                raise
            token = authorize(challenge, base, timeout)
            headers["Authorization"] = f"Bearer {token}"
            answer = head(address, headers, timeout)
    except urllib.error.HTTPError as error:
        raise errors.ArtifactError(
            f"{image}: the registry answered {error.code}"
        ) from error
    except OSError as error:
        raise errors.ArtifactError(f"{image}: {error}") from error
    kind, digest = answer.split(" ", 1)
    if kind not in INDEX_TYPES:
        raise errors.ArtifactError(
            f"{image} is not a multi-architecture index ({kind or 'unknown'})"
        )
    return digest


def resolve(
    image: str, fetch: Callable[[str], str] | None = None
) -> dict[str, str]:
    """Returns the lock record for `image`.

    Args:
        image: `repository:tag`.
        fetch: Replaces the registry call (tests).

    Returns:
        A dict with `image`, `digest` and `resolved_at` (an ISO-8601 UTC
        timestamp).

    Raises:
        ConfigError: If the registry returns the placeholder digest. A reply
            that is not a `sha256:` digest, or an image that is not
            `repository:tag`, is refused with the same error by the helpers.
    """
    decompose(image)
    digest = (fetch or query)(image)
    contract.require(digest, "resolved digest")
    if digest == PLACEHOLDER:
        raise errors.ConfigError("the registry returned the placeholder")
    return {
        "image": image,
        "digest": digest,
        "resolved_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def peruse(path: str | pathlib.Path) -> dict[str, str]:
    """Reads and validates a lock; refuses the placeholder.

    Args:
        path: Path of the lock file written by `resolve`.

    Returns:
        A dict with the lock's `image` and `digest`.

    Raises:
        ArtifactError: If the lock is missing, unreadable or unresolved.
    """
    try:
        record = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
        digest = record["digest"]
        image = record["image"]
    except (
        OSError,
        UnicodeError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
    ) as error:
        raise errors.ArtifactError(
            f"cannot read base image lock {path}"
        ) from error
    contract.require(digest, "base image lock digest")
    if digest == PLACEHOLDER:
        raise errors.ArtifactError(
            "the base image digest is still the placeholder: run "
            "`python -m scmle.deploy.base_image resolve` first"
        )
    decompose(image)
    return {"image": image, "digest": digest}


def main(arguments: list[str] | None = None) -> int:
    """Runs the command line (`resolve`, `argument`, `check` or `pinned`).

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        0 on success, 1 if the action fails (the error goes to stderr).
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]  # pyright: ignore[reportOptionalMemberAccess]  # the module docstring is always present
    )
    actions = parser.add_subparsers(dest="action", required=True)
    resolve_parser = actions.add_parser("resolve", help="ask the registry")
    resolve_parser.add_argument("--image", required=True)
    resolve_parser.add_argument("--output", required=True)
    for name in ("argument", "check"):
        action = actions.add_parser(name)
        action.add_argument("--lock", required=True)
    actions.add_parser("pinned", help="print a recorded pin").add_argument(
        "--image", required=True
    )
    options = parser.parse_args(arguments)
    try:
        if options.action == "pinned":
            digest = pinned(options.image)
            if digest == PLACEHOLDER:
                raise errors.ConfigError(f"{options.image} is not pinned")
            print(f"{options.image.rsplit(':', 1)[0]}@{digest}")  # noqa: T201 -- command-line output
            return 0
        if options.action == "resolve":
            record = resolve(options.image)
            pathlib.Path(options.output).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"{options.image} -> {record['digest']}")  # noqa: T201 -- command-line output
            return 0
        lock = peruse(options.lock)
        if options.action == "argument":
            repository = lock["image"].rsplit(":", 1)[0]
            print(f"{repository}@{lock['digest']}")  # noqa: T201 -- command-line output
        else:
            print(f"ok: {lock['image']} pinned to {lock['digest']}")  # noqa: T201 -- command-line output
    except (errors.ScmleError, OSError, KeyError, ValueError) as error:
        # handled: returned as failed finding/problem/refusal with error text
        print(f"error: {error}", file=sys.stderr)  # noqa: T201 -- command-line output
        return 1
    else:
        return 0


if __name__ == "__main__":
    sys.exit(main())
