"""Exploration code cannot reach the confirmation set through imports.

The selection of survivors is made by `online_fdr`, `bivariate`, `explore`,
`controls`, `multiplicity` and `ledger`. None of them, and not the
confirmation module itself, may import custody, the vault or the splitters:
the sealed set reaches `confirm` only as an argument handed over by the
orchestrator.
"""

import ast
import pathlib

import scmle.data.ledger as package

FORBIDDEN = (
    "scmle.data.custody",
    "scmle.governance.vault",
    "scmle.data.splitting",
    "scmle.core.stages",
)
ROOT = pathlib.Path(package.__file__).parent


def imported_modules(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def test_no_ledger_module_imports_custody_or_the_vault():
    modules = sorted(ROOT.glob("*.py"))
    assert len(modules) >= 8
    for path in modules:
        for name in imported_modules(path):
            for forbidden in FORBIDDEN:
                assert not (
                    name == forbidden or name.startswith(forbidden + ".")
                ), f"{path.name} imports {name}"


def test_the_checker_sees_an_import_when_there_is_one(tmp_path):
    sample = tmp_path / "bad.py"
    sample.write_text("from scmle.data.custody import sealed\n")
    assert "scmle.data.custody.sealed" in imported_modules(sample)
