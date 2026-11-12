"""Enforces the project naming rule: no underscore-prefixed identifiers.

Dunder names (`__init__`, `__all__`, ...) are language protocol and exempt.
Everything else, in the package and in the tests, must be a public name.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
# `type X = ...` statements exist only from Python 3.12; on 3.11 none can occur.
TYPE_ALIAS_NODES = (ast.TypeAlias,) if hasattr(ast, "TypeAlias") else ()
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
        for line, name in identifiers(
            ast.parse(path.read_text(encoding="utf-8"))
        )
        if is_private(name)
    ]
    assert not offenders, offenders


@pytest.mark.parametrize(
    "path", SOURCES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_no_private_file_names(path):
    assert not is_private(path.stem), path


ALIAS_SOURCES = sorted(
    path
    for folder in ("scmle", "tests", "benchmarks", "scripts", "examples")
    for path in (ROOT / folder).rglob("*.py")
    if "mutants" not in path.parts
)


def chain(node):
    """Whether an expression is a plain name or an attribute chain on one."""
    while isinstance(node, ast.Attribute):
        node = node.value
    return isinstance(node, ast.Name)


def aliases(tree):
    """Lists every second name for an existing thing, by line.

    Covers `import x as y`, `from m import n as a`, module-level
    `NAME = other.name` assignments, and type aliases. Class-level
    assignments are attribute configuration, not aliases, and are not listed.
    """
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.alias) and node.asname:
            found.append((node.lineno, f"import {node.name} as {node.asname}"))
        elif isinstance(node, TYPE_ALIAS_NODES):
            found.append((node.lineno, f"type alias {node.name.id}"))
        elif isinstance(node, ast.AnnAssign) and "TypeAlias" in ast.unparse(
            node.annotation
        ):
            found.append((node.lineno, "TypeAlias annotation"))
    found.extend(
        (
            node.lineno,
            f"{node.targets[0].id} = {ast.unparse(node.value)}",
        )
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Name | ast.Attribute)
        and chain(node.value)
        and ast.unparse(node.value) not in ("True", "False", "None")
    )
    return found


@pytest.mark.parametrize(
    "path", ALIAS_SOURCES, ids=lambda p: str(p.relative_to(ROOT))
)
def test_no_aliases(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))

    assert not aliases(tree), [f"{path.name}:{n} {d}" for n, d in aliases(tree)]


def test_alias_checker_detects_violations():
    source = (
        "import numpy as np\n"
        "from os import path as p\n"
        "X = np.pi\n"
        "Y: TypeAlias = int | str\n"
        "type Z = list[int]\n"
        "class K:\n"
        "    kind = X\n"
        "ok = None\n"
        "def f():\n"
        "    inner = X\n"
    )

    found = [item[1] for item in aliases(ast.parse(source))]

    assert sorted(found) == [
        "TypeAlias annotation",
        "X = np.pi",
        "import numpy as np",
        "import path as p",
        "type alias Z",
    ]


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
