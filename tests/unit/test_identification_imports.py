"""No estimator module imports the custody or confirmation APIs."""

import ast
import pathlib

import scmle.causal.identification

FORBIDDEN = (
    "scmle.assurance.custody",
    "scmle.assurance.ledger.confirm",
    "scmle.assurance.ledger.governed",
    "scmle.assurance.ledger.ledger",
)


def imports(path: pathlib.Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
            found.update(f"{node.module}.{a.name}" for a in node.names)
    return found


def test_the_identification_modules_import_no_custody_or_confirmation_api():
    folder = pathlib.Path(scmle.causal.identification.__file__).parent
    files = sorted(folder.glob("*.py"))
    assert len(files) >= 10
    for path in files:
        for name in imports(path):
            assert not name.startswith(FORBIDDEN), (path.name, name)
