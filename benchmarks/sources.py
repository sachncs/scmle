"""Fingerprints that cover all the product code a measurement runs.

A stored measurement is current only while the code it ran is unchanged.
Listing the functions by hand misses what they call, so a result can keep
passing after the code it measured has changed. `digest` hashes the given
functions or modules and every module of the product (`scmle`) that their
modules import, followed transitively through the import statements of
each file. The import graph is read from the source without importing it,
so the closure is an over-approximation: a module that is imported but not
called still counts, which makes a result go stale too often, never too
rarely.
"""

import ast
from collections.abc import Iterable
import hashlib
import inspect
from pathlib import Path
import types
from typing import Any

from scmle.assurance.detectors import hashing

ROOT = Path(__file__).resolve().parents[1]
PRODUCT = "scmle"


def locate(name: str, root: Path = ROOT) -> Path | None:
    """The source file of a module name under `root`, if it has one.

    Args:
        name: Dotted module name.
        root: Directory holding the top-level packages.

    Returns:
        `a/b.py` or `a/b/__init__.py`, or None when neither exists.
    """
    base = root.joinpath(*name.split("."))
    for path in (base.with_suffix(".py"), base / "__init__.py"):
        if path.is_file():
            return path
    return None


def imports(path: Path, name: str, root: Path = ROOT) -> set[str]:
    """Module names of the product that one file imports.

    Args:
        path: The file.
        name: Its dotted module name (for relative imports).
        root: Directory holding the top-level packages.

    Returns:
        Every product module the file's import statements load, with their
        parent packages (importing `a.b.c` runs `a` and `a.b` too).
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    package = name if path.name == "__init__.py" else name.rpartition(".")[0]
    found: set[str] = set()
    for node in ast.walk(tree):
        targets: list[str] = []
        if isinstance(node, ast.Import):
            targets = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            stem = node.module or ""
            if node.level:
                parts = package.split(".")
                parts = parts[: len(parts) - node.level + 1]
                stem = ".".join([*parts, stem] if stem else parts)
            targets = [stem]
            targets += [f"{stem}.{alias.name}" for alias in node.names]
        for target in targets:
            if target.split(".")[0] != PRODUCT:
                continue
            pieces = target.split(".")
            for end in range(1, len(pieces) + 1):
                prefix = ".".join(pieces[:end])
                if locate(prefix, root) is not None:
                    found.add(prefix)
    return found


def closure(names: Iterable[str], root: Path = ROOT) -> list[str]:
    """Every product module reachable by import from the given modules.

    Args:
        names: Dotted names of the starting modules (any package).
        root: Directory holding the top-level packages.

    Returns:
        The sorted product modules reached, with any product module among
        the starting ones.
    """
    seen: set[str] = set()
    queue = list(names)
    while queue:
        name = queue.pop()
        if name in seen:
            continue
        seen.add(name)
        path = locate(name, root)
        if path is not None:
            queue += sorted(imports(path, name, root) - seen)
    return sorted(n for n in seen if n.split(".")[0] == PRODUCT)


def name(module: types.ModuleType) -> str:
    """A module's import name, also when it runs as `python -m`.

    Args:
        module: The module.

    Returns:
        The name in its import spec (`__main__` names no file to follow),
        else its `__name__`.
    """
    spec = module.__spec__
    return spec.name if spec is not None else module.__name__


def digest(*objects: Any, root: Path = ROOT) -> str:
    """Hash of the given objects and of all product code their modules load.

    Args:
        *objects: Functions, classes or modules the measurement runs.
        root: Directory holding the top-level packages.

    Returns:
        64 lowercase hexadecimal characters: the code hash of the objects,
        then the name and source hash of each module in their closure.
    """
    names = set()
    for item in objects:
        module = item if isinstance(item, types.ModuleType) else None
        module = module or inspect.getmodule(item)
        if module is not None:
            names.add(name(module))
    parts = [hashing.code_hash(*objects)]
    for found in closure(names, root):
        path = locate(found, root)
        if path is None:
            continue
        parts.append(found)
        parts.append(hashing.hash_source(path.read_text(encoding="utf-8")))
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
