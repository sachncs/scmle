"""A module constant must not be silently assigned twice.

A second plain assignment to the same name replaces the first for every reader
in the module, so a threshold can change without any line that uses it
changing. A name may be assigned again only when the new value is built from
the old one, as the capability tables in `scmle.contracts.evaluation` do.
"""

import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
FOLDERS = ("scmle", "benchmarks", "scripts", "examples")


def names_in(node):
    """Every name an expression reads."""
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def targets(node):
    """The names a top-level assignment binds, with the value assigned."""
    if isinstance(node, ast.Assign):
        return [
            (target.id, node.value)
            for target in node.targets
            if isinstance(target, ast.Name)
        ]
    if (
        isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.value is not None
    ):
        return [(node.target.id, node.value)]
    return []


def test_no_module_constant_is_reassigned_without_reading_its_old_value():
    offenders = []
    for folder in FOLDERS:
        for path in sorted((ROOT / folder).rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            bound = set()
            for node in tree.body:
                for name, value in targets(node):
                    if name in bound and name not in names_in(value):
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno} {name}"
                        )
                    bound.add(name)
    assert offenders == []
