"""The runtime dependency closure of a service, as exact pins.

`environment.lock` lists everything installed where the model was built,
tests and linters included. A service image needs only what the model and the
framework import. `runtime_closure` walks the declared requirements of the
roots through the installed metadata and returns just that set; the image
installs exactly it with `--no-deps`, so pip resolves nothing at build time.
"""

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Mapping
from importlib import metadata
import re

from scmle.contracts import errors

NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def canonical(name: str) -> str:
    """The normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def marker_applies(marker: str) -> bool:
    """Whether an environment marker holds here (extras never do)."""
    if "extra" in marker:
        return False
    try:
        from packaging import markers
    except ImportError:
        return False
    return bool(markers.Marker(marker).evaluate({"extra": ""}))


def requirement_name(line: str) -> tuple[str, str]:
    """Splits a `Requires-Dist` line into (name, marker)."""
    parts = line.partition(";")
    match = NAME_PATTERN.match(parts[0].strip())
    if match is None:
        raise errors.ArtifactError(f"cannot read requirement {line!r}")
    return canonical(match.group(0)), parts[2].strip()


def runtime_closure(
    roots: Iterable[str],
    versions: Mapping[str, str] | None = None,
    requires: Callable[[str], list[str]] | None = None,
    installed: Callable[[str], str] | None = None,
    unresolved: list[str] | None = None,
) -> dict[str, str]:
    """Returns `name -> version` for the roots and everything they need.

    Args:
        roots: Distribution names to start from.
        versions: Preferred versions (the build environment recorded in the
            bundle); the installed version is used for names not listed.
        requires: Replaces `importlib.metadata.requires` (tests).
        installed: Replaces `importlib.metadata.version` (tests).
        unresolved: If given, receives the names whose own requirements
            could not be read because they are not installed here; they are
            still pinned (from `versions`) but their dependencies are not
            followed.

    Raises:
        ArtifactError: If a distribution is neither installed nor in
            `versions`.
    """
    look_up = requires or (lambda name: metadata.requires(name) or [])
    find = installed or metadata.version
    preferred = {canonical(k): v for k, v in (versions or {}).items()}
    closure: dict[str, str] = {}
    pending = [canonical(name) for name in roots]
    while pending:
        name = pending.pop()
        if name in closure:
            continue
        try:
            version = preferred.get(name) or find(name)
        except metadata.PackageNotFoundError as error:
            raise errors.ArtifactError(
                f"{name} is required but neither installed nor recorded"
            ) from error
        try:
            declared = look_up(name)
        except metadata.PackageNotFoundError:
            declared = []
            if unresolved is not None:
                unresolved.append(name)
        closure[name] = version
        for line in declared:
            dependency, marker = requirement_name(line)
            if not marker or marker_applies(marker):
                pending.append(dependency)
    return dict(sorted(closure.items()))


def render(closure: Mapping[str, str], header: Iterable[str] = ()) -> str:
    """Formats a closure as a pinned requirements file."""
    lines = [f"# {line}" for line in header]
    lines += [f"{name}=={version}" for name, version in closure.items()]
    return "\n".join(lines) + "\n"
