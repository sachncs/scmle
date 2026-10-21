"""Tests for repository-wide Python naming conventions."""

import ast
from collections.abc import Iterator
import pathlib


def python_identifiers(path: pathlib.Path) -> Iterator[tuple[int, str]]:
    """Yield source identifiers and their line numbers from one module."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(
            node,
            (
                ast.Name,
                ast.arg,
                ast.Attribute,
                ast.AsyncFunctionDef,
                ast.ClassDef,
                ast.FunctionDef,
            ),
        ):
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.arg):
                name = node.arg
            elif isinstance(node, ast.Attribute):
                name = node.attr
            else:
                name = node.name
            yield node.lineno, name
        elif isinstance(node, ast.alias) and node.asname is not None:
            yield node.lineno, node.asname
        elif isinstance(node, ast.ExceptHandler) and node.name is not None:
            yield node.lineno, node.name


def test_package_does_not_define_semi_private_identifiers():
    """Public package code uses names that are either public or dunder-only."""
    package_root = pathlib.Path(__file__).parents[2] / "scml"
    violations = [
        f"{path}:{line}: {name}"
        for path in sorted(package_root.rglob("*.py"))
        for line, name in python_identifiers(path)
        if name.startswith("_") and not name.startswith("__") and name != "_"
    ]

    assert violations == []
