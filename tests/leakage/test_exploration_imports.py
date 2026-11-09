"""Detectors, admissibility and robustness cannot import custody.

Every module under `scmle/data/detectors`, `scmle/data/admissibility` and
`scmle/data/robustness` is parsed. None may import the custody package, the
vault, the splitters module (which defines the locked test set) or the
orchestrator, directly or through any scmle module it imports. A sealed set
reaches a module of these packages only as an argument handed over by the
orchestrator.
"""

import ast
import importlib.util
import pathlib

import pytest

import scmle

FORBIDDEN = (
    "scmle.data.custody",
    "scmle.governance.vault",
    "scmle.data.splitting.splitters",
    "scmle.core.stages",
)
PACKAGES = (
    "scmle.data.detectors",
    "scmle.data.admissibility",
    "scmle.data.robustness",
)
ROOT = pathlib.Path(scmle.__file__).parent.parent


def source_of(module):
    try:
        spec = importlib.util.find_spec(module)
    except (ImportError, AttributeError, ValueError):
        return None
    if spec is None or spec.origin is None or spec.origin == "built-in":
        return None
    return pathlib.Path(spec.origin)


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def forbidden(name):
    return any(name == f or name.startswith(f + ".") for f in FORBIDDEN)


def closure(start):
    """Every scmle module reachable by imports, with the path to it."""
    reached = {}
    stack = [(start, (start,))]
    while stack:
        module, trail = stack.pop()
        if module in reached:
            continue
        reached[module] = trail
        path = source_of(module)
        if path is None:
            continue
        stack.extend(
            (name, (*trail, name))
            for name in imported_modules(path)
            if name.startswith("scmle.")
            and name not in reached
            and (source_of(name) is not None or forbidden(name))
        )
    return reached


def package_modules(package):
    folder = ROOT / package.replace(".", "/")
    return sorted(
        f"{package}.{path.stem}"
        for path in folder.glob("*.py")
        if path.stem != "__init__"
    )


@pytest.mark.parametrize("package", PACKAGES)
def test_no_module_imports_custody_directly(package):
    modules = package_modules(package)
    assert len(modules) >= 4
    for module in modules:
        for name in imported_modules(source_of(module)):
            assert not forbidden(name), f"{module} imports {name}"


@pytest.mark.parametrize("package", PACKAGES)
def test_no_module_reaches_custody_through_its_imports(package):
    for module in package_modules(package):
        for name, trail in closure(module).items():
            assert not forbidden(name), " -> ".join(trail)


def test_the_checker_sees_an_import_and_matches_whole_names(tmp_path):
    sample = tmp_path / "bad.py"
    sample.write_text("from scmle.governance import vault\n")
    assert "scmle.governance.vault" in imported_modules(sample)
    assert forbidden("scmle.data.custody.sealed")
    assert not forbidden("scmle.data.custody_like")


def test_the_closure_follows_imports_to_the_vault():
    reached = closure("scmle.data.custody.pipeline")
    assert "scmle.governance.vault" in reached
    trail = reached["scmle.governance.vault"]
    assert trail[0] == "scmle.data.custody.pipeline" and len(trail) >= 3
