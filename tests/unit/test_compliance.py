"""The traceability matrix may only cite files and tests that exist."""

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


def test_every_cited_symbol_exists_in_its_file():
    missing = []
    for number, path, name in citations():
        if name and name not in (ROOT / path).read_text():
            missing.append(f"line {number}: {path}::{name}")
    assert not missing, missing


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


def test_partial_rows_state_what_is_missing():
    partial = [
        line
        for line in MATRIX.read_text().splitlines()
        if "| Partial |" in line
    ]
    assert partial and all(
        ";" in row or " no " in row or "only" in row for row in partial
    )
