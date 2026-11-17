"""The layers contract places every package and admits no exceptions."""

import pathlib
import tomllib

ROOT = pathlib.Path(__file__).resolve().parents[2]
NAME = "Each package imports only the layers below it"


def contract():
    document = tomllib.loads((ROOT / "pyproject.toml").read_text("utf-8"))
    contracts = document["tool"]["importlinter"]["contracts"]
    return next(item for item in contracts if item["name"] == NAME)


def test_the_layers_contract_is_exhaustive_and_has_no_exceptions():
    # Enforces docs/ARCHITECTURE.md:15.
    found = contract()
    assert found["type"] == "layers"
    assert found["containers"] == ["scmle"]
    assert found["exhaustive"] is True
    assert "ignore_imports" not in found


def test_every_package_sits_in_exactly_one_layer():
    placed = [
        name.strip() for row in contract()["layers"] for name in row.split("|")
    ]
    package = ROOT / "scmle"
    children = {
        path.stem
        for path in package.iterdir()
        if (path.suffix == ".py" and path.stem != "__init__")
        or (path / "__init__.py").exists()
    }
    assert len(placed) == len(set(placed))
    assert set(placed) == children
