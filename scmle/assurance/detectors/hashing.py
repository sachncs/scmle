"""Hashes of detector code and corpora, the keys of detector certificates.

A certificate is keyed by the hash of the code it measured, so a code change
without re-certification leaves it stale. The hash is taken over source text
with comments, docstrings and blank lines removed.
"""

import ast
from collections.abc import Iterable
import hashlib
import inspect
import io
import re
import textwrap
import tokenize
from typing import Any

from scmle.contracts import errors

SHA256_HEX = re.compile(r"[0-9a-f]{64}")
DOCSTRING_OWNERS = (
    ast.Module,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def documentation(tree: ast.Module) -> set[int]:
    """Finds the lines occupied only by a docstring statement.

    Args:
        tree: Parsed module.

    Returns:
        The 1-based line numbers of every docstring that shares its lines
        with no other statement.
    """
    statements = [n for n in ast.walk(tree) if isinstance(n, ast.stmt)]
    lines: set[int] = set()
    for owner in ast.walk(tree):
        if not isinstance(owner, DOCSTRING_OWNERS) or not owner.body:
            continue
        first = owner.body[0]
        if not (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            continue
        span = range(first.lineno, (first.end_lineno or first.lineno) + 1)
        shared = any(
            other is not first and other.lineno in span for other in statements
        )
        if not shared:
            lines.update(span)
    return lines


def hash_source(source: str) -> str:
    """SHA-256 of source text with comments, docstrings and blanks removed.

    The text itself is hashed, not a syntax tree, so the value does not
    depend on the Python version that computes it. Reformatting the code
    changes the hash, which is deliberate: any edit needs re-certification.
    Source that is not valid Python raises `SyntaxError`.

    Args:
        source: Python source for a function, class or module.

    Returns:
        64 lowercase hexadecimal characters.
    """
    text = textwrap.dedent(source)
    tree = ast.parse(text)
    lines = text.splitlines()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.COMMENT:
            row, column = token.start
            lines[row - 1] = lines[row - 1][:column]
    dropped = documentation(tree)
    kept = [
        line.rstrip()
        for number, line in enumerate(lines, start=1)
        if line.strip() and number not in dropped
    ]
    return hashlib.sha256("\n".join(kept).encode("utf-8")).hexdigest()


def code_hash(*objects: Any) -> str:
    """Combined hash of the source of functions, classes or modules.

    Only the given objects' own source is hashed, not what they call, so a
    detector's certificate must list every callable that affects its result.

    Args:
        *objects: Objects whose source `inspect` can read.

    Returns:
        64 lowercase hexadecimal characters; order matters.

    Raises:
        ValueError: If no object is given.
    """
    if not objects:
        raise errors.InputError("code_hash needs at least one object")
    parts = [hash_source(inspect.getsource(item)) for item in objects]
    return hashlib.sha256("\n".join(parts).encode("ascii")).hexdigest()


def corpus_hash(parts: Iterable[str]) -> str:
    """Hashes the corpus file hashes a run used, ignoring their order.

    Args:
        parts: SHA-256 hex digests of the corpus files.

    Returns:
        64 lowercase hexadecimal characters.

    Raises:
        ValueError: If no part is given or a part is not a SHA-256 digest.
    """
    members = sorted(parts)
    if not members:
        raise errors.InputError("corpus_hash needs at least one part")
    for part in members:
        require_digest(part, "corpus part")
    return hashlib.sha256("\n".join(members).encode("ascii")).hexdigest()


def require_digest(value: Any, name: str) -> str:
    """Returns `value` if it is a lowercase SHA-256 hex digest.

    Args:
        value: The value to check.
        name: Name of the value, used in the error message.

    Returns:
        `value`, unchanged.

    Raises:
        ValueError: If `value` is not a 64-character lowercase hex string.
    """
    if not isinstance(value, str) or not SHA256_HEX.fullmatch(value):
        raise errors.InputError(
            f"{name} must be a 64-character lowercase SHA-256"
        )
    return value
