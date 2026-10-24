"""Enforces the project naming rule: no underscore-prefixed identifiers.

Dunder names (`__init__`, `__all__`, ...) are language protocol and exempt.
Everything else, in the package and in the tests, must be a public name.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCES = sorted(
    path
    for folder in ("scmle", "tests")
    for path in (ROOT / folder).rglob("*.py")
)


def is_private(name: str) -> bool:
    """Whether `name` starts with an underscore and is not a dunder."""
    is_dunder = name.startswith("__") and name.endswith("__") and len(name) > 4
    return name.startswith("_") and not is_dunder


def identifiers(tree: ast.AST) -> list[tuple[int, str]]:
    """Collects every identifier the module defines or references."""
    found = []
    for node in ast.walk(tree):
        names = []
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            names.append(node.name)
        elif isinstance(node, ast.Name):
            names.append(node.id)
        elif isinstance(node, ast.Attribute):
            names.append(node.attr)
        elif isinstance(node, ast.arg) or (
            isinstance(node, ast.keyword) and node.arg
        ):
            names.append(node.arg)
        elif isinstance(node, ast.alias):
            names.extend([node.name.split(".")[-1], node.asname or ""])
        elif isinstance(node, ast.Global | ast.Nonlocal):
            names.extend(node.names)
        found += [(getattr(node, "lineno", 0), n) for n in names if n]
    return found


@pytest.mark.parametrize(
    "path", SOURCES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_no_private_names(path):
    offenders = [
        f"{path.name}:{line} {name}"
        for line, name in identifiers(ast.parse(path.read_text()))
        if is_private(name)
    ]
    assert not offenders, offenders


@pytest.mark.parametrize(
    "path", SOURCES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_no_private_file_names(path):
    assert not is_private(path.stem), path


def test_checker_detects_violations():
    tree = ast.parse("def _a(): pass\nclass _B: pass\nx = y._z\n_ = 1")
    private = {entry[1] for entry in identifiers(tree) if is_private(entry[1])}
    assert private == {
        "_a",
        "_B",
        "_z",
        "_",
    }
    assert not is_private("__init__")
