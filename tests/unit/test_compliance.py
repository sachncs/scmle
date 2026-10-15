"""The traceability matrix may only cite files and tests that exist."""

import ast
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[2]
MATRIX = ROOT / "docs" / "COMPLIANCE.md"
CITATION = re.compile(
    r"`([\w./-]+\.(?:py|md|toml|yml|json|ya?ml)|LICENSE)(?:::(\w+))?`"
)


def citations():
    found = []
    for number, line in enumerate(MATRIX.read_text().splitlines(), 1):
        if line.startswith("|"):
            found += [
                (number, path, name) for path, name in CITATION.findall(line)
            ]
    return found


def test_every_cited_path_exists():
    missing = [
        f"line {number}: {path}"
        for number, path, name in citations()
        if not (ROOT / path).exists()
    ]
    assert not missing, missing


def defined_names(path):
    """Names a Python file really defines (functions, classes, assignments)."""
    tree = ast.parse((ROOT / path).read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        ):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(
            node.target, ast.Name
        ):
            names.add(node.target.id)
    return names


def test_every_cited_symbol_is_defined_in_its_file():
    missing = []
    for number, path, name in citations():
        if not name:
            continue
        if path.endswith(".py"):
            found = name in defined_names(path)
        else:
            found = name in (ROOT / path).read_text()
        if not found:
            missing.append(f"line {number}: {path}::{name}")
    assert not missing, missing


def test_every_cited_test_file_contains_tests_with_assertions():
    weak = []
    for citation in citations():
        number, path = citation[0], citation[1]
        if not (path.startswith("tests/") and path.endswith(".py")):
            continue
        tree = ast.parse((ROOT / path).read_text())
        tests = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name.startswith("test_")
        ]
        asserting = [
            t
            for t in tests
            if any(
                isinstance(n, ast.Assert)
                or (
                    isinstance(n, ast.With)
                    and any("raises" in ast.dump(i) for i in n.items)
                )
                for n in ast.walk(t)
            )
        ]
        if not asserting:
            weak.append(f"line {number}: {path}")
    assert not weak, weak


def test_matrix_is_substantial_and_labels_every_row():
    rows = [
        line
        for line in MATRIX.read_text().splitlines()
        if line.startswith("|")
        and not line.startswith("| ---")
        and "Requirement" not in line
        and "| Risk" not in line
    ]
    statuses = ("Implemented", "Partial", "Extension point")
    labelled = [
        row for row in rows if any(status in row for status in statuses)
    ]
    assert len(rows) >= 80
    risk_rows = [row for row in rows if row not in labelled]
    assert len(risk_rows) <= 8  # only the risk table lacks a status column


def test_no_requirement_is_left_partial():
    """Every spec requirement is Implemented; reintroducing a gap must be
    a deliberate edit of this test, not a silent drift."""
    partial = [
        line
        for line in MATRIX.read_text().splitlines()
        if "| Partial |" in line
    ]
    assert partial == []
