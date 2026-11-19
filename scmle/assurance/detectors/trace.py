"""Binds a certificate to the code it measured, and only that (task 0.2).

`digest` hashes the source of the entry functions or classes and of every
function, class and module constant of the package they reach by name. A
change on that path, a helper or a threshold, changes the digest; a change
elsewhere in the same module does not, so a module can be refactored without
staling a certificate for code the certificate did not measure.
"""

import ast
from collections.abc import Mapping
import dataclasses
import enum
import hashlib
import inspect
import textwrap
from typing import Any

from scmle.assurance.detectors import certificates
from scmle.contracts import errors

PACKAGE = "scmle"
CONSTANTS = (int, float, str, bool, bytes, tuple, frozenset, enum.Enum)


def steady(value: Any) -> bool:
    """Whether a module value is a constant whose text can be hashed.

    Numbers, text, enums, tuples and frozen sets, mappings and lists of
    them, and dataclass instances (such as a settings object) qualify; so
    does None.
    """
    if value is None or isinstance(value, CONSTANTS):
        return True
    if isinstance(value, Mapping | list):
        return True
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def owned(value: Any) -> bool:
    """Whether a function or class is defined inside the package."""
    module = getattr(value, "__module__", "") or ""
    return (inspect.isfunction(value) or inspect.isclass(value)) and (
        module == PACKAGE or module.startswith(PACKAGE + ".")
    )


def references(tree: ast.AST) -> set[tuple[str, ...]]:
    """Every dotted name a piece of source reads from its module.

    Names the source binds itself (assignments, parameters, loop and
    comprehension targets, class fields) are left out, so a local that
    shares a module function's name does not pull that function in.

    Args:
        tree: Parsed source of a function or class.

    Returns:
        The names, as tuples of their dotted parts.
    """
    bound = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load)
    }
    bound |= {a.arg for a in ast.walk(tree) if isinstance(a, ast.arg)}
    found: set[tuple[str, ...]] = set()
    for node in ast.walk(tree):
        parts: list[str] = []
        current = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            continue
        if current.id in bound or not isinstance(current.ctx, ast.Load):
            continue
        if parts or isinstance(node, ast.Name):
            found.add((current.id, *reversed(parts)))
    return found


def resolve(scope: Mapping[str, Any], dotted: tuple[str, ...]) -> list[Any]:
    """The objects a dotted name passes through, modules left out.

    Attributes are followed only through modules and package classes, so
    `module.Class.MEMBER` yields the class and the member, and an attribute
    of anything else (an instance, a library object) ends the walk.

    Args:
        scope: The module namespace the name is read in.
        dotted: The name's parts.

    Returns:
        The objects reached, in order.
    """
    reached = []
    value = scope.get(dotted[0])
    for name in dotted[1:]:
        if not (inspect.ismodule(value) or owned(value)):
            break
        if not inspect.ismodule(value):
            reached.append(value)
        value = getattr(value, name, None)
    if not inspect.ismodule(value):
        reached.append(value)
    return reached


def digest(*entries: Any) -> str:
    """Hash of the code a result depends on: the entries and all they reach.

    Starting from the entry functions or classes, every function or class
    of the package that their source reads by name (through attributes too)
    is followed, and every module constant they read is included by its
    text. So a change to anything on the path from the entries, a helper or
    a threshold, changes the hash, and a change elsewhere in the same module
    does not. This is what lets a module be refactored without staling a
    certificate for code it did not measure (register task 0.2).

    Args:
        *entries: Functions or classes of the package, or constants the
            result depends on that the code does not read itself (such as
            the settings a harness passes in), included by their text.

    Returns:
        64 lowercase hexadecimal characters.

    Raises:
        ValueError: If no entry is given, or an entry is neither package
            code nor a constant.
    """
    unwrapped = [inspect.unwrap(e) if callable(e) else e for e in entries]
    if not unwrapped or not all(owned(e) or steady(e) for e in unwrapped):
        raise errors.InputError(
            "digest needs entries of package code or constants"
        )
    queue = [e for e in unwrapped if owned(e)]
    parts = {
        f"entry:{index}": repr(e)
        for index, e in enumerate(unwrapped)
        if not owned(e)
    }
    while queue:
        item = queue.pop()
        key = f"{item.__module__}.{item.__qualname__}"
        if key in parts:
            continue
        source = textwrap.dedent(inspect.getsource(item))
        parts[key] = certificates.digest(source)
        scope = vars(inspect.getmodule(item))
        for dotted in sorted(references(ast.parse(source))):
            for value in resolve(scope, dotted):
                target = inspect.unwrap(value) if callable(value) else value
                if owned(target):
                    queue.append(target)
                elif steady(target):
                    name = ".".join(dotted)
                    parts[f"{item.__module__}:{name}"] = repr(target)
    text = "\n".join(f"{k}={v}" for k, v in sorted(parts.items()))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
