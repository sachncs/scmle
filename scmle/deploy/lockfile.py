"""The runtime dependency closure of a service, as exact pins.

`environment.lock` lists everything installed where the model was built,
tests and linters included. A service image needs only what the model and the
framework import. `transitive` walks the declared requirements of the
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
    """Normalizes a distribution name.

    Args:
        name: Distribution name as written in a requirement.

    Returns:
        The name lower-cased with each run of `-`, `_` and `.` replaced by `-`.
    """
    return re.sub(r"[-_.]+", "-", name).lower()


def applies(marker: str) -> bool:
    """Reports whether an environment marker holds here (extras never do).

    Args:
        marker: A PEP 508 environment marker.

    Returns:
        True if the marker holds for this environment; False if it mentions an
        extra.

    Raises:
        ArtifactError: If the `packaging` library is not installed: the
            marker cannot be evaluated, and treating it as false dropped the
            dependency from the lock without a word (ERR-12).
    """
    if "extra" in marker:
        return False
    try:
        from packaging import markers  # noqa: PLC0415 -- guarded import
    except ImportError as error:
        raise errors.ArtifactError(
            f"cannot evaluate the environment marker {marker!r}: install "
            "'packaging' to build a complete runtime lock"
        ) from error
    return bool(markers.Marker(marker).evaluate({"extra": ""}))


def transitive(
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
        unresolved: If given, receives the names whose own requirements could
            not be read because they are not installed here; they are still
            pinned (from `versions`) but their dependencies are not followed.
            Without it such a name is an error, never silently cut short.

    Returns:
        The pinned versions of the roots and of their dependencies whose markers
        hold here, sorted by name.

    Raises:
        ArtifactError: If a distribution is neither installed nor in
            `versions`, or its requirements cannot be read and no
            `unresolved` list was given to record it (ERR-12).
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
        except metadata.PackageNotFoundError as error:
            if unresolved is None:
                raise errors.ArtifactError(
                    f"the requirements of {name} cannot be read here, so the "
                    "lock would miss its dependencies"
                ) from error
            # handled: recorded in the caller's `unresolved` list
            declared = []
            unresolved.append(name)
        closure[name] = version
        for line in declared:
            parts = line.partition(";")
            match = NAME_PATTERN.match(parts[0].strip())
            if match is None:
                raise errors.ArtifactError(f"cannot read requirement {line!r}")
            marker = parts[2].strip()
            if not marker or applies(marker):
                pending.append(canonical(match.group(0)))
    return dict(sorted(closure.items()))


def render(closure: Mapping[str, str], header: Iterable[str] = ()) -> str:
    """Formats a closure as a pinned requirements file.

    Args:
        closure: Mapping from distribution name to version.
        header: Lines written first as comments.

    Returns:
        The file text: the commented header, then one `name==version` line per
        entry, ending with a newline.
    """
    lines = [f"# {line}" for line in header]
    lines += [f"{name}=={version}" for name, version in closure.items()]
    return "\n".join(lines) + "\n"
