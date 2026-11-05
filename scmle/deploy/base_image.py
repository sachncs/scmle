r"""Resolve a base image tag to the digest it points at, at build time.

A generated Dockerfile never trusts a tag: it carries a placeholder digest and
refuses to build until the placeholder is replaced by the digest of the
multi-architecture index (so amd64 and arm64 builds use the same base).
`resolve` asks the registry; `argument` prints the `--build-arg` value;
`check` fails if a lock is missing, malformed or still the placeholder.

    python -m scmle.deploy.base_image resolve --image python:3.12-slim \\
        --output base-image.lock.json
    docker build --build-arg BASE_IMAGE="$(python -m scmle.deploy.base_image \\
        argument --lock base-image.lock.json)" .

Resolving needs network access to the registry; everything else here is
offline and tested with a fake registry.
"""

import argparse
from collections.abc import Callable
import datetime
import json
import pathlib
import re
import sys
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
AUTH_ENDPOINT = (
    "https://auth.docker.io/token?service=registry.docker.io"
    "&scope=repository:{repository}:pull"
)
IMAGE_PATTERN = re.compile(
    r"(?:(?P<registry>[a-z0-9.-]+(?::[0-9]+)?)/)?"
    r"(?P<repository>[a-z0-9][a-z0-9._/-]*):(?P<tag>[A-Za-z0-9._-]+)"
)
Fetcher = Callable[[str, str], str]


def split_image(image: str) -> tuple[str, str, str]:
    """Splits `repo:tag` into (registry, repository, tag).

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


def fetch_registry_digest(image: str, timeout: float = 30.0) -> str:
    """Asks the registry for the digest of the index behind `image`.

    Only Docker Hub-style token authentication is implemented.

    Raises:
        ArtifactError: If the registry answers with something other than a
            multi-architecture index.
    """
    registry, repository, tag = split_image(image)
    token_request = urllib.request.Request(
        AUTH_ENDPOINT.format(
            repository=urllib.parse.quote(repository, safe="/")
        )
    )
    with urllib.request.urlopen(token_request, timeout=timeout) as reply:  # nosec B310
        token = json.loads(reply.read())["token"]
    request = urllib.request.Request(
        f"https://{registry}/v2/{repository}/manifests/{tag}",
        method="HEAD",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": ", ".join(INDEX_TYPES),
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as reply:  # nosec B310
        kind = reply.headers.get("Content-Type", "").split(";")[0]
        digest = reply.headers.get("Docker-Content-Digest", "")
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

    Raises:
        ConfigError: If the registry returns something that is not a digest
            or the placeholder.
    """
    split_image(image)
    digest = (fetch or fetch_registry_digest)(image)
    contract.require_digest(digest, "resolved digest")
    if digest == PLACEHOLDER:
        raise errors.ConfigError("the registry returned the placeholder")
    return {
        "image": image,
        "digest": digest,
        "resolved_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }


def read_lock(path: str | pathlib.Path) -> dict[str, str]:
    """Reads and validates a lock; refuses the placeholder.

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
    contract.require_digest(digest, "base image lock digest")
    if digest == PLACEHOLDER:
        raise errors.ArtifactError(
            "the base image digest is still the placeholder: run "
            "`python -m scmle.deploy.base_image resolve` first"
        )
    split_image(image)
    return {"image": image, "digest": digest}


def build_argument(lock: dict[str, str]) -> str:
    """The value for `--build-arg BASE_IMAGE=`."""
    repository = lock["image"].rsplit(":", 1)[0]
    return f"{repository}@{lock['digest']}"


def main(arguments: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    actions = parser.add_subparsers(dest="action", required=True)
    resolve_parser = actions.add_parser("resolve", help="ask the registry")
    resolve_parser.add_argument("--image", required=True)
    resolve_parser.add_argument("--output", required=True)
    for name in ("argument", "check"):
        action = actions.add_parser(name)
        action.add_argument("--lock", required=True)
    options = parser.parse_args(arguments)
    try:
        if options.action == "resolve":
            record = resolve(options.image)
            pathlib.Path(options.output).write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
                newline="\n",
            )
            print(f"{options.image} -> {record['digest']}")
            return 0
        lock = read_lock(options.lock)
        if options.action == "argument":
            print(build_argument(lock))
        else:
            print(f"ok: {lock['image']} pinned to {lock['digest']}")
        return 0
    except (errors.ScmleError, OSError, KeyError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
