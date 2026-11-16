"""A traced hash follows the code a result reaches, and only that (task 0.2).

Each test writes a small module that pretends to live in the package, so
`trace.digest` follows it, then edits one part and checks whether the hash moved.
"""

import ast
import importlib.util
import inspect
import sys
import textwrap

import pytest

from scmle.data.detectors import trace

BASE = """
LIMIT = 3


def helper(x):
    return x + LIMIT


def entry(x):
    total = helper(x)
    return total


def elsewhere(x):
    return x * 100


def family(x):
    return x
"""


def build(folder, source, name="scmle.tracedemo"):
    """Imports `source` as a package module and returns it."""
    path = folder / "tracedemo.py"
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def traced(tmp_path):
    """Hash of `entry` for a version of the module."""
    count = iter(range(1000))

    def run(source):
        folder = tmp_path / str(next(count))
        folder.mkdir()
        module = build(folder, source)
        try:
            return trace.digest(module.entry)
        finally:
            sys.modules.pop("scmle.tracedemo", None)

    return run


def reach(*entries):
    """Qualified names of the package code `trace.digest` follows from `entries`."""
    reached = set()
    queue = list(entries)
    while queue:
        item = queue.pop()
        name = f"{item.__module__}.{item.__qualname__}"
        if name in reached:
            continue
        reached.add(name)
        tree = ast.parse(textwrap.dedent(inspect.getsource(item)))
        scope = vars(inspect.getmodule(item))
        for dotted in trace.references(tree):
            for value in trace.resolve(scope, dotted):
                target = inspect.unwrap(value) if callable(value) else value
                if trace.owned(target):
                    queue.append(target)
    return reached


def test_the_hash_is_stable(traced):
    assert traced(BASE) == traced(BASE)


def test_editing_unreached_code_does_not_move_the_hash(traced):
    edited = BASE.replace("return x * 100", "return x * 200")
    assert traced(edited) == traced(BASE)


def test_editing_a_reached_helper_moves_the_hash(traced):
    edited = BASE.replace("return x + LIMIT", "return x - LIMIT")
    assert traced(edited) != traced(BASE)


def test_changing_a_constant_it_reads_moves_the_hash(traced):
    assert traced(BASE.replace("LIMIT = 3", "LIMIT = 4")) != traced(BASE)


def test_comments_and_docstrings_do_not_move_the_hash(traced):
    edited = BASE.replace(
        "def helper(x):\n", 'def helper(x):\n    """Adds."""  # note\n'
    )
    assert traced(edited) == traced(BASE)


def test_a_local_named_like_a_module_function_is_not_followed(traced):
    shadowing = BASE.replace(
        "    total = helper(x)\n    return total",
        "    family = helper(x)\n    return family",
    )
    edited = shadowing.replace(
        "def family(x):\n    return x", "def family(x):\n    return -x"
    )
    assert traced(edited) == traced(shadowing)


def test_a_constant_entry_is_part_of_the_hash(tmp_path):
    module = build(tmp_path, BASE)
    try:
        one = trace.digest(module.entry, {"ceiling": 0.95})
        two = trace.digest(module.entry, {"ceiling": 0.90})
    finally:
        sys.modules.pop("scmle.tracedemo", None)
    assert one != two


def test_entries_outside_the_package_are_refused():
    with pytest.raises(ValueError, match="package code or constants"):
        trace.digest(textwrap.dedent)
    with pytest.raises(ValueError, match="package code or constants"):
        trace.digest()


def test_the_screen_digest_reaches_the_procedure_and_not_its_wiring():
    from scmle.data.admissibility import plants
    from scmle.data.admissibility import screen

    reached = reach(screen.screen, plants.plant)
    wiring = {"screen_table", "probe", "disclose", "evidence", "shipped_store"}
    assert not {n.rsplit(".", 1)[1] for n in reached} & wiring
    assert "scmle.data.admissibility.screen.scan_frame" in reached
    assert "scmle.data.admissibility.plants.Numbers" in reached
