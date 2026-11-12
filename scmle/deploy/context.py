"""What goes into a container build context, and whether a secret does.

The build context is every file of the service folder that `.dockerignore`
does not exclude. `scan` lists anything in it that looks like a credential: a
key or environment file, key material inside a file, a configured secret's
value, or a Dockerfile instruction that bakes one in. It cannot prove there
is no secret (a password in a pickle is a password), so it is a tripwire for
the common mistakes, run before every build.
"""

import argparse
from collections.abc import Iterable
import fnmatch
import pathlib
import re
import sys

SECRET_FILE_PATTERNS = (
    ".env",
    ".env.*",
    "*.pem",
    "*.key",
    "*.p12",
    "*.pfx",
    "id_rsa*",
    "id_ed25519*",
    ".netrc",
    "credentials*",
    "secrets*",
)
IGNORED = (
    ".git",
    ".git/*",
    ".aws",
    ".aws/*",
    *SECRET_FILE_PATTERNS,
    "__pycache__",
    "*.pyc",
    "build-metadata.json",
    "scan.json",
    "image.id",
)
KEY_MARKERS = (
    b"-----BEGIN PRIVATE KEY",
    b"-----BEGIN RSA PRIVATE KEY",
    b"-----BEGIN OPENSSH PRIVATE KEY",
    b"-----BEGIN EC PRIVATE KEY",
    b"-----BEGIN PGP PRIVATE KEY",
)
INSTRUCTION = re.compile(r"^\s*(ENV|ARG)\s+([A-Za-z0-9_]+)(?:[= ]+(\S+))?")
SECRET_WORDS = ("SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "PRIVATE", "KEY")
MAX_SCANNED_BYTES = 64 * 1024 * 1024


def dockerignore_text() -> str:
    """Builds the `.dockerignore` written beside a generated Dockerfile.

    Returns:
        One pattern per line from `IGNORED`, ending with a newline.
    """
    return "\n".join(IGNORED) + "\n"


def patterns(root: pathlib.Path) -> list[str]:
    """Reads the patterns of the folder's `.dockerignore`.

    Args:
        root: The folder to be built.

    Returns:
        The non-blank, non-comment lines; empty if there is no `.dockerignore`.
    """
    path = root / ".dockerignore"
    if not path.exists():
        return []
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]


def excluded(relative: str, ignore: Iterable[str]) -> bool:
    """Reports whether `.dockerignore` removes `relative` from the context.

    Args:
        relative: POSIX path relative to the build folder.
        ignore: Glob patterns, matched against the whole path and against each
            path component.

    Returns:
        True if any pattern matches.
    """
    parts = relative.split("/")
    for pattern in ignore:
        if fnmatch.fnmatch(relative, pattern):
            return True
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
    return False


def included_files(root: str | pathlib.Path) -> list[pathlib.Path]:
    """Lists the files that would be sent to the builder.

    Args:
        root: The folder to be built.

    Returns:
        Paths relative to `root`, sorted, for every file that the folder's
        `.dockerignore` does not exclude.
    """
    base = pathlib.Path(root)
    ignore = patterns(base)
    return sorted(
        path.relative_to(base)
        for path in base.rglob("*")
        if path.is_file()
        and not excluded(path.relative_to(base).as_posix(), ignore)
    )


def scan(
    root: str | pathlib.Path, secret_values: Iterable[str] = ()
) -> list[str]:
    """Lists secrets-looking content in the build context.

    Args:
        root: The folder to be built.
        secret_values: Secret strings that must not appear anywhere in it.

    Returns:
        One finding per problem; empty means nothing suspicious was found.
    """
    base = pathlib.Path(root)
    values = [value.encode() for value in secret_values if value]
    findings = []
    if not (base / ".dockerignore").exists():
        findings.append(".dockerignore is missing: everything would be sent")
    for relative in included_files(base):
        name = relative.as_posix()
        if any(
            fnmatch.fnmatch(part, pattern)
            for part in relative.parts
            for pattern in SECRET_FILE_PATTERNS
        ):
            findings.append(f"{name}: looks like a credential file")
        path = base / relative
        if path.stat().st_size > MAX_SCANNED_BYTES:
            findings.append(f"{name}: too large to scan")
            continue
        content = path.read_bytes()
        if any(marker in content for marker in KEY_MARKERS):
            findings.append(f"{name}: contains private key material")
        findings += [
            f"{name}: contains a configured secret value"
            for value in values
            if value in content
        ]
        if path.name == "Dockerfile":
            for line in content.decode("utf-8", "replace").splitlines():
                match = INSTRUCTION.match(line)
                if (
                    match
                    and match.group(3)
                    and any(w in match.group(2).upper() for w in SECRET_WORDS)
                ):
                    findings.append(
                        f"{name}: {match.group(1)} {match.group(2)} sets a "
                        "value that looks secret"
                    )
    return findings


def main(arguments: list[str] | None = None) -> int:
    """Runs the command line: `python -m scmle.deploy.context check DIR`.

    Args:
        arguments: Command-line arguments; None reads `sys.argv`.

    Returns:
        1 if the scan reports findings (printed to stderr), otherwise 0.
    """
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0]  # pyright: ignore[reportOptionalMemberAccess]  # the module docstring is always present
    )
    parser.add_argument("action", choices=["check"])
    parser.add_argument("folder", type=pathlib.Path)
    options = parser.parse_args(arguments)
    findings = scan(options.folder)
    for finding in findings:
        print(finding, file=sys.stderr)  # noqa: T201 -- command-line output
    if not findings:
        print(  # noqa: T201 -- command-line output
            f"{len(included_files(options.folder))} files in the context; "
            "nothing suspicious found"
        )
    return 1 if findings else 0


if __name__ == "__main__":
    sys.exit(main())
