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
