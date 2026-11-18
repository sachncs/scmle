"""Tests for scripts/keys.py, which writes the stored-record key map."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "keys.py"


def load():
    """Imports the script as a module."""
    spec = importlib.util.spec_from_file_location("keys", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_a_vocabulary_table_becomes_renames_per_record_kind():
    module = load()
    text = (
        "### `data/custody/manifest.py` `Manifest`\n\n"
        "| Old | New | Confidence | Reason |\n"
        "|---|---|---|---|\n"
        "| `charter_hash` | `digests.charter` | judgement | x |\n"
        "| `schema_version` | keep `schema_version` | judgement | kept |\n"
        "| `row_roots` | `roots` | clear |  |\n"
    )

    found = module.parse(text)

    assert found == {
        "data/custody/manifest.py Manifest": {
            "charter_hash": "digests.charter",
            "row_roots": "roots",
        }
    }


def test_the_published_markdown_matches_the_published_map():
    module = load()
    kinds = json.loads(module.TABLE.read_text(encoding="utf-8"))["kinds"]

    assert module.DOC.read_text(encoding="utf-8") == module.render(kinds)
    assert kinds["data/custody/manifest.py Manifest"]["data_hash"] == (
        "digests.data"
    )
