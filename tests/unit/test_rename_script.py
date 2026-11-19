"""The rename tool changes exactly the sites that resolve to a definition."""

import importlib.util
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"

STORE = '''"""Shelves, and `build`, which makes one."""


class Shelf:
    """Holds items; `Shelf.build` counts them."""

    def __init__(self) -> None:
        self.size = 0

    def build(self) -> int:
        """Counts the items, unlike `build` at module level."""
        return self.size


class Crate(Shelf):
    """A shelf with a lid."""

    def build(self) -> int:
        """One more than `Shelf.build`."""
        return super().build() + 1


def build() -> int:
    """Makes a shelf; unlike `Shelf.build`, it takes no receiver."""
    return Shelf().build()
'''

INIT = '''"""A fixture package."""

from pkg.store import build

__all__ = ["build"]
'''

USE = '''"""Uses the store."""

from pkg import store
from pkg import store as shelves
from pkg.store import Shelf, build


def local() -> int:
    build = 3
    return build


def run(shelf: Shelf) -> int:
    # `store.build` makes a shelf; `Shelf.build` counts one.
    return shelf.build() + store.build() + shelves.build() + build()


PATH = "pkg.store.build"
METHOD = "pkg.store.Shelf.build"
'''

TEST = """from pkg import build


def test_build():
    assert build() == 0
"""

GUIDE = """# Guide

Call `pkg.store.build` or `store.build`; `Shelf.build` counts.
A bare `build` is ambiguous.
"""


def load():
    """Imports the script as a module; it imports `structure` beside it."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "rename", SCRIPTS / "rename.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tree(tmp_path):
    """A small repository: a package, a test and a doc."""
    files = {
        "pkg/__init__.py": INIT,
        "pkg/store.py": STORE,
        "pkg/use.py": USE,
        "tests/test_use.py": TEST,
        "docs/guide.md": GUIDE,
    }
    for name, text in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return tmp_path


def run(module, root, entries, *flags):
    """Writes a map and runs the tool on the fixture."""
    path = root.parent / f"{root.name}.toml"
    path.write_text(
        "".join(f'"{old}" = "{new}"\n' for old, new in entries.items()),
        encoding="utf-8",
    )
    arguments = ["--map", str(path), "--root", str(root), "--package", "pkg"]
    return module.main([*arguments, *flags])


def text(root, name):
    """A fixture file's content."""
    return (root / name).read_text(encoding="utf-8")


def test_a_function_rename_changes_only_its_references(tree):
    module = load()
    assert run(module, tree, {"pkg.store.build": "make"}) == 0
    store = text(tree, "pkg/store.py")
    assert '"""Shelves, and `make`, which makes one."""' in store
    assert "def make() -> int:" in store
    assert "return Shelf().build()" in store
    assert store.count("def build(self)") == 2
    assert '"""Counts the items, unlike `build` at module level."""' in store
    assert store.count("`Shelf.build`") == 3
    assert text(tree, "pkg/__init__.py") == INIT.replace("build", "make")
    use = text(tree, "pkg/use.py")
    assert "from pkg.store import Shelf, make" in use
    assert "    build = 3\n    return build\n" in use
    assert (
        "return shelf.build() + store.make() + shelves.make() + make()" in use
    )
    assert "# `store.make` makes a shelf; `Shelf.build` counts one." in use
    assert 'PATH = "pkg.store.make"' in use
    assert 'METHOD = "pkg.store.Shelf.build"' in use
    assert text(tree, "tests/test_use.py") == (
        "from pkg import make\n\n\ndef test_build():\n    assert make() == 0\n"
    )
    guide = text(tree, "docs/guide.md")
    assert "Call `pkg.store.make` or `store.make`; `Shelf.build` counts." in (
        guide
    )
    assert "A bare `build` is ambiguous." in guide


def test_a_method_rename_follows_receivers_and_overrides(tree):
    module = load()
    assert run(module, tree, {"pkg.store.Shelf.build": "count"}) == 0
    store = text(tree, "pkg/store.py")
    assert store.count("def count(self) -> int:") == 2
    assert "return super().count() + 1" in store
    assert "return Shelf().count()" in store
    assert "def build() -> int:" in store
    assert store.count("`Shelf.count`") == 3
    assert '"""Shelves, and `build`, which makes one."""' in store
    use = text(tree, "pkg/use.py")
    assert (
        "return shelf.count() + store.build() + shelves.build() + build()"
        in use
    )
    assert "# `store.build` makes a shelf; `Shelf.count` counts one." in use
    assert 'METHOD = "pkg.store.Shelf.count"' in use
    assert 'PATH = "pkg.store.build"' in use
    assert text(tree, "pkg/__init__.py") == INIT


def test_a_dry_run_prints_the_edits_and_writes_nothing(tree, capsys):
    module = load()
    before = text(tree, "pkg/use.py")
    assert run(module, tree, {"pkg.store.build": "make"}, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "pkg/use.py:5: build -> make  (import)" in out
    assert "review: docs/guide.md: unresolved citation `build`" in out
    assert text(tree, "pkg/use.py") == before


@pytest.mark.parametrize(
    ("entries", "reason"),
    [
        ({"pkg.store.build": "make_shelf"}, "is not one word"),
        ({"pkg.store.build": "_make"}, "leading underscore"),
        ({"pkg.store.build": "local"}, "local is already bound here"),
        ({"pkg.store.build": "Shelf"}, "pkg.store already binds Shelf"),
        ({"pkg.store.Shelf.build": "size"}, "Shelf already binds size"),
        ({"pkg.store.Crate.build": "size"}, "pkg.store.Crate has size"),
        ({"pkg.store.Shelf.get_params": "params"}, "exempt"),
        ({"pkg.store.missing": "gone"}, "no such definition"),
    ],
)
def test_a_colliding_map_is_refused(tree, capsys, entries, reason):
    module = load()
    before = {name: text(tree, name) for name in ("pkg/use.py", "pkg/store.py")}
    assert run(module, tree, entries) == 1
    assert reason in capsys.readouterr().out
    assert {name: text(tree, name) for name in before} == before


def test_verify_passes_once_the_map_is_applied(tree, capsys):
    module = load()
    entries = {"pkg.store.Shelf.build": "count", "pkg.store.build": "make"}
    assert run(module, tree, entries) == 0
    capsys.readouterr()
    assert run(module, tree, entries, "--verify") == 0, capsys.readouterr()


def test_verify_catches_a_missed_reference(tree, capsys):
    module = load()
    entries = {"pkg.store.Shelf.build": "count"}
    assert run(module, tree, entries) == 0
    path = tree / "pkg" / "use.py"
    path.write_text(
        text(tree, "pkg/use.py")
        + "\n\ndef again(shelf: Shelf) -> int:\n    return shelf.build()\n",
        encoding="utf-8",
    )
    capsys.readouterr()
    assert run(module, tree, entries, "--verify") == 1
    out = capsys.readouterr().out
    assert "pkg/use.py:23: still names an old definition" in out
    assert "mypy failed" in out


def test_a_module_rename_moves_the_file_and_its_imports(tree):
    module = load()
    assert run(module, tree, {"pkg.store": "depot"}) == 0
    assert not (tree / "pkg" / "store.py").exists()
    assert (tree / "pkg" / "depot.py").exists()
    use = text(tree, "pkg/use.py")
    assert "from pkg import depot\n" in use
    assert "from pkg import depot as shelves\n" in use
    assert "from pkg.depot import Shelf, build\n" in use
    assert "store.build()" not in use
    assert "depot.build()" in use
    assert 'PATH = "pkg.depot.build"' in use
    assert text(tree, "pkg/__init__.py") == INIT.replace("store", "depot")
