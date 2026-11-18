"""Tests for scripts/recite.py, which moves doc-line citations in tests."""

import importlib.util
import pathlib

SCRIPT = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "recite.py"


def load():
    """Imports the script as a module."""
    spec = importlib.util.spec_from_file_location("recite", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_lines_follow_an_insertion_and_a_reworded_line_keeps_its_place():
    module = load()
    before = ["a", "b", "c", "d"]
    after = ["a", "new", "new", "b", "C", "d"]

    found = module.align(before, after)

    # "c" was reworded in place, so line 3 maps to its new position.
    assert found == {1: 1, 2: 4, 3: 5, 4: 6}


def test_a_citation_names_the_doc_and_the_line():
    module = load()

    # Built from parts, so recite.py does not move this sample citation.
    sample = "# Enforces docs/LEAKAGE" + ".md:172 and README" + ".md:9."
    found = module.CITATION.findall(sample)

    assert found == [("docs/LEAKAGE.md", "172"), ("README.md", "9")]
