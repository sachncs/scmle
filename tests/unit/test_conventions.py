"""Enforces the project conventions: naming, aliases and the run board.

Dunder names (`__init__`, `__all__`, ...) are language protocol and exempt.
Everything else, in the package and in the tests, must be a public name.

The run context (`scmle/pipeline/context.py`) is on its way out: stages
read the products the board holds instead of its fields. `CONTEXTUAL` lists
the package modules that still import it; the list may only shrink, and
the context is deleted when it is empty.
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
        "class K:\n"
        "    kind = X\n"
        "ok = None\n"
        "def f():\n"
        "    inner = X\n"
    )
    expected = [
        "TypeAlias annotation",
        "X = np.pi",
        "import numpy as np",
        "import path as p",
    ]
    if TYPE_ALIAS_NODES:
        source += "type Z = list[int]\n"
        expected.append("type alias Z")

    found = [item[1] for item in aliases(ast.parse(source))]

    assert sorted(found) == sorted(expected)


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


CONTEXT = "scmle.pipeline.context"
CONTEXTUAL = frozenset(
    {
        "scmle/pipeline/ensemble_search.py",
        "scmle/pipeline/hyperband.py",
        "scmle/pipeline/orchestrator.py",
        "scmle/pipeline/policies.py",
        "scmle/pipeline/strategy_search.py",
        "scmle/pipeline/trials.py",
        "scmle/pipeline/tuning.py",
        *(
            f"scmle/pipeline/stages/{name}.py"
            for name in (
                "admission",
                "assessment",
                "bracketing",
                "calibration",
                "cost",
                "diagnosis",
                "documentation",
                "examination",
                "exploration",
                "identification",
                "impounding",
                "ingestion",
                "interface",
                "labelling",
                "packaging",
                "planning",
                "profiling",
                "qualification",
                "relabelling",
                "releasing",
                "remedy",
                "reporting",
                "reservation",
                "scoring",
                "screening",
                "selecting",
                "summary",
                "surveying",
                "tracing",
                "training",
            )
        ),
    }
)


def imported(tree):
    """Every module an AST imports, at any depth, by its dotted name."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
    return found


def test_only_the_recorded_modules_still_import_the_run_context():
    importers = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "scmle").rglob("*.py")
        if CONTEXT in imported(ast.parse(path.read_text(encoding="utf-8")))
    } - {"scmle/pipeline/context.py"}
    assert importers <= CONTEXTUAL, sorted(importers - CONTEXTUAL)


def test_the_import_checker_finds_nested_and_from_imports():
    tree = ast.parse(
        "def f():\n    import scmle.pipeline.context\n"
        "from scmle.pipeline import context\n"
    )
    assert CONTEXT in imported(tree)
    assert CONTEXT not in imported(
        ast.parse("from scmle.pipeline import kinds")
    )
