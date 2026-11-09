"""Strings that name a class by its dotted path must name a class that exists.

The deep-learning estimators start their engines in a worker process from a
string such as `scmle.models.deeplearning.engines.Classifier`. A rename that
changes the class but not the string passes every test that does not start a
worker, and those tests are skipped when `torch` is not installed. This test
reads the target file instead of importing it, so it needs no optional
dependency.
"""

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
PATH = re.compile(r"scmle(?:\.[a-z_0-9]+)+\.[A-Za-z_][A-Za-z_0-9]*")


def defined(tree):
    """Names a module defines or imports at its top level."""
    names = set()
    for node in tree.body:
        if isinstance(
            node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id
                for target in node.targets
                if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            names.add(node.target.id)
        elif isinstance(node, ast.ImportFrom | ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
    return names


def module_file(parts):
    """The source file of a dotted module path, or None if it is not one."""
    base = ROOT.joinpath(*parts)
    if base.with_suffix(".py").exists():
        return base.with_suffix(".py")
    if (base / "__init__.py").exists():
        return base / "__init__.py"
    return None


def test_every_dotted_class_path_in_a_string_resolves():
    unresolved = []
    for path in sorted((ROOT / "scmle").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and PATH.fullmatch(node.value)
            ):
                continue
            if module_file(node.value.split(".")) is not None:
                continue  # the whole string is a module, not a class
            *module, name = node.value.split(".")
            target = module_file(module)
            if target is None:
                continue
            source = ast.parse(target.read_text(encoding="utf-8"))
            if name not in defined(source):
                unresolved.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} {node.value}"
                )
    assert unresolved == []


def test_the_check_finds_a_class_that_was_renamed():
    tree = ast.parse("class Trainer:\n    pass\n")

    assert "Trainer" in defined(tree)
    assert "TabularEngine" not in defined(tree)
