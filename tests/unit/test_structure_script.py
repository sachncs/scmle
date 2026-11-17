"""The structure report counts what the plan budgets and catches regressions."""

import importlib.util
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "structure.py"


def load():
    """Imports the script as a module, since `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("structure", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compound_names_are_snake_or_camel_joins():
    module = load()
    assert module.compound("load_store")
    assert module.compound("LoadedStore")
    assert not module.compound("load")
    assert not module.compound("Store")
    assert not module.compound("__init__")
    assert not module.compound("F5")
    assert not module.compound("ConflictError")
    assert module.compound("LeaseLostError")


def test_the_limits_come_from_the_thresholds_file():
    assert load().limits() == {"module": 800, "class": 300, "function": 80}


def test_the_tree_is_within_its_recorded_budget():
    module = load()
    budget = json.loads(module.BUDGET.read_text(encoding="utf-8"))
    assert module.compare(module.measure(), budget) == []
    assert module.main([]) == 0


def test_a_worse_count_fails(monkeypatch, tmp_path):
    module = load()
    found = module.measure()
    budget = {**found, "compound_functions": found["compound_functions"] - 1}
    path = tmp_path / "structure.json"
    path.write_text(json.dumps(budget), encoding="utf-8")
    monkeypatch.setattr(module, "BUDGET", path)
    assert module.main([]) == 1
    assert module.compare(found, {}) != []


def test_a_planted_long_function_is_counted(monkeypatch, tmp_path):
    module = load()
    package = tmp_path / "scmle"
    package.mkdir()
    body = "\n".join(f"    x{i} = {i}" for i in range(90))
    (package / "planted.py").write_text(
        f"def long_name():\n{body}\n\nclass Short:\n    pass\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "PACKAGE", package)
    found = module.measure()
    assert found["functions_over"] == 1
    assert found["compound_functions"] == 1
    assert found["compound_classes"] == 0


def plant(root, files):
    """Writes a synthetic package tree under `root/scmle`."""
    package = root / "scmle"
    for relative, text in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return package


def write(root, layers, moves):
    """Writes a layer map with one step holding every move."""
    rows = ", ".join(f'"{row}"' for row in layers)
    pairs = ", ".join(f'["{old}", "{new}"]' for old, new in moves)
    path = root / "layers.toml"
    path.write_text(
        f'layers = [{rows}]\n\n[[step]]\nname = "one"\nmoves = [{pairs}]\n',
        encoding="utf-8",
    )
    return path


TREE = {
    "__init__.py": "",
    "top/__init__.py": "",
    "top/run.py": "from scmle.low import base\n",
    "low/__init__.py": "",
    "low/base.py": "def build():\n    from scmle.top import run\n",
    "low/plain.py": "",
    "side/__init__.py": "",
    "side/tool.py": (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n    from scmle.other import thing\n"
    ),
    "other/__init__.py": "",
    "other/thing.py": "",
}


def test_simulate_finds_a_cycle_and_a_move_removes_it(monkeypatch, tmp_path):
    module = load()
    monkeypatch.setattr(module, "PACKAGE", plant(tmp_path, TREE))
    layout = module.load(write(tmp_path, ["top", "low", "side | other"], []))
    breaks, largest = module.simulate(layout)
    assert breaks == [
        "scmle.low.base -> scmle.top.run (low -> top, upward)",
        "scmle.side.tool -> scmle.other.thing (side -> other, sibling)",
    ]
    assert largest == 2
    moves = [
        ("scmle.low.base", "scmle.top.base"),
        ("scmle.other", "scmle.low.other"),
    ]
    layout = module.load(write(tmp_path, ["top", "low", "side"], moves))
    breaks, largest = module.simulate(layout)
    assert breaks == [
        "scmle.side.tool -> scmle.other.thing (side -> low, upward)"
    ]
    assert largest == 1


def test_an_unplaced_package_breaks_the_order(monkeypatch, tmp_path):
    module = load()
    monkeypatch.setattr(module, "PACKAGE", plant(tmp_path, TREE))
    layout = module.load(write(tmp_path, ["top", "low"], []))
    breaks = module.simulate(layout)[0]
    assert (
        "scmle.side.tool -> scmle.other.thing (side -> other, unplaced)"
        in breaks
    )


def test_the_longest_move_wins():
    moves = {
        "scmle.alpha": "scmle.beta",
        "scmle.alpha.retry": "scmle.gamma.retry",
    }
    module = load()
    assert module.rename("scmle.alpha.retry", moves) == "scmle.gamma.retry"
    assert module.rename("scmle.alpha.stages", moves) == "scmle.beta.stages"
    assert module.rename("scmle.alphax", moves) == "scmle.alphax"
