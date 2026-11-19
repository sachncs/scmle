"""The phase 6 lane maps are well formed before any of them is applied.

`scripts/renames/<lane>.toml` holds the renames rename.py applies (the
top-level entries) and a `[review]` table of names no single word fits
without ambiguity. These checks are static: rename.py's dry run is the
proof that a map resolves and collides with nothing.
"""

import builtins
import collections
import importlib.util
import keyword
import pathlib
import sys
import tomllib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
MAPS = SCRIPTS / "renames"
LANES = {
    "data_assurance": ("data", "assurance"),
    "models": ("evaluation", "models", "inference", "serving"),
    "pipeline": ("pipeline", "runtime", "workflow", "operations", "monitoring"),
    "rest": (),
}
SHADOWS = set(dir(builtins)) | set(keyword.kwlist) | set(keyword.softkwlist)


def structure():
    """Imports `scripts/structure.py`, which owns the naming rule."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "structure", SCRIPTS / "structure.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def document(lane):
    """A lane map, split into its entries and its review table."""
    found = tomllib.loads((MAPS / f"{lane}.toml").read_text("utf-8"))
    review = found.pop("review", {})
    return found, review


def owner(path):
    """The lane a dotted path belongs to."""
    package = path.split(".")[1]
    named = [lane for lane, packages in LANES.items() if package in packages]
    return named[0] if named else "rest"


@pytest.mark.parametrize("lane", sorted(LANES))
def test_every_new_name_is_one_word(lane):
    rule = structure()
    entries = document(lane)[0]
    bad = {
        old: new
        for old, new in entries.items()
        if not isinstance(new, str)
        or not new.isidentifier()
        or new.startswith("_")
        or rule.compound(new)
        or new in SHADOWS
    }
    assert not bad


@pytest.mark.parametrize("lane", sorted(LANES))
def test_every_old_name_is_multiword_and_not_exempt(lane):
    rule = structure()
    entries, review = document(lane)
    bad = [
        old
        for old in [*entries, *review]
        if old.rpartition(".")[2] in rule.EXEMPT
        or not rule.compound(old.rpartition(".")[2])
    ]
    assert not bad


@pytest.mark.parametrize("lane", sorted(LANES))
def test_entries_stay_in_their_lane(lane):
    entries, review = document(lane)
    assert all(path.startswith("scmle.") for path in [*entries, *review])
    assert {owner(path) for path in [*entries, *review]} <= {lane}


@pytest.mark.parametrize("lane", sorted(LANES))
def test_no_two_entries_take_one_name_in_a_scope(lane):
    entries = document(lane)[0]
    seen = collections.Counter(
        (old.rpartition(".")[0], new) for old, new in entries.items()
    )
    assert [key for key, count in seen.items() if count > 1] == []


@pytest.mark.parametrize("lane", sorted(LANES))
def test_review_lists_two_or_three_candidates(lane):
    review = document(lane)[1]
    bad = {
        old: options
        for old, options in review.items()
        if not isinstance(options, list)
        or not 2 <= len(options) <= 3
        or not all(isinstance(item, str) for item in options)
    }
    assert not bad


def test_a_name_is_in_one_lane_once():
    places = collections.Counter(
        path for lane in LANES for part in document(lane) for path in part
    )
    assert [path for path, count in places.items() if count > 1] == []
