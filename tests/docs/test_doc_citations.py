"""Test citations of doc lines stay on the statements they name (C12).

`benchmarks/evidence/guarantees.py` links a guarantee to a test that cites
its line. An edit above a cited line moves the line and silently breaks or
misdirects every such link, so these tests fail as soon as a citation is
stale (`scripts/recite.py --check` would move it) or lands where no
statement is.
"""

import importlib.util
import pathlib
import subprocess

import pytest

from benchmarks import scorecard
from benchmarks.evidence import guarantees
from benchmarks.evidence import links
from benchmarks.evidence import pages

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "recite.py"


def recite():
    """Imports scripts/recite.py as a module."""
    spec = importlib.util.spec_from_file_location("recite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def history():
    """Whether the repository root has its git history."""
    done = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    return done.returncode == 0


def test_recite_names_the_docs_the_guarantee_scan_reads():
    assert recite().SOURCE == links.SOURCE


@pytest.mark.skipif(not history(), reason="needs the git history of the repo")
def test_no_citation_is_stale_or_lost():
    module = recite()
    moved, files, lost = module.survey(module.Atlas(ROOT), write=False)

    message = "run `python scripts/recite.py`, then re-cite what it calls lost"
    assert (moved, files, lost) == (0, [], []), message


def test_every_cited_line_holds_a_statement():
    paths = guarantees.collect(ROOT)
    docs = {p.relative_to(ROOT).as_posix() for p in paths}
    records = guarantees.survey(ROOT, docs)[1]
    spans = {}
    for path in paths:
        name = path.relative_to(ROOT).as_posix()
        text = path.read_text(encoding="utf-8")
        units = (
            guarantees.chunk(text)
            if name.endswith(".md")
            else pages.units(text, name)
        )
        spans[name] = [(u["start"], u["end"]) for u in units]

    stray = [
        f"{r['test']} cites {r['doc']}:{r['lines'][0]}-{r['lines'][1]}"
        for r in records
        if "lines" in r
        and not any(
            start <= r["lines"][1] and r["lines"][0] <= end
            for start, end in spans[r["doc"]]
        )
    ]

    assert stray == []


def test_ci_fails_on_stale_evidence_not_on_an_unlinked_guarantee(tmp_path):
    """Enforces docs/preregistration/c12.md:89-90."""
    (tmp_path / "docs").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "docs" / "X.md").write_text("It never fails.\n", "utf-8")
    document = guarantees.measure(tmp_path)

    assert document["counts"]["unlinked"] == 1
    assert guarantees.problems(document, tmp_path) == []
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text("utf-8")
    assert "guarantees" in workflow
    assert 'benchmarks.evidence.${module}" --check' in workflow
    (row,) = [r for r in scorecard.rows() if r.claim.startswith("C12 ")]
    assert not any("audit" in clause.criterion for clause in row.clauses)
