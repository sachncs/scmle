"""New qualification code must draw randomness only from keyed streams."""

import ast
from pathlib import Path
import textwrap

import pytest

ROOT = Path(__file__).parents[2]
PACKAGES = (
    "charter",
    "custody",
    "ledger",
    "admissibility",
    "detectors",
    "robustness",
)
ALLOWED_NUMPY_RANDOM = {"Generator", "SeedSequence", "BitGenerator", "PCG64"}


def violations(source):
    """Returns the unseeded or global random-number uses in `source`."""
    found = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found += [
                f"line {node.lineno}: import {alias.name}"
                for alias in node.names
                if alias.name.split(".")[0] == "random"
            ]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.split(".")[0] == "random":
                found.append(f"line {node.lineno}: from {module} import")
            if module == "numpy.random":
                found += [
                    f"line {node.lineno}: from numpy.random import {a.name}"
                    for a in node.names
                    if a.name not in ALLOWED_NUMPY_RANDOM | {"default_rng"}
                ]
        elif isinstance(node, ast.Attribute):
            owner = node.value
            if (
                isinstance(owner, ast.Attribute)
                and owner.attr == "random"
                and isinstance(owner.value, ast.Name)
                and owner.value.id in {"np", "numpy"}
                and node.attr not in ALLOWED_NUMPY_RANDOM | {"default_rng"}
            ):
                found.append(f"line {node.lineno}: np.random.{node.attr}")
        elif isinstance(node, ast.Call):
            function = node.func
            name = getattr(function, "attr", getattr(function, "id", ""))
            if name == "default_rng" and not node.args and not node.keywords:
                found.append(f"line {node.lineno}: unseeded default_rng()")
    return found


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nx = np.random.rand(3)\n",
        "import numpy\nnumpy.random.seed(1)\n",
        "import random\n",
        "from random import choice\n",
        "from numpy.random import rand\n",
        "import numpy as np\nrng = np.random.default_rng()\n",
        "from numpy.random import default_rng\nrng = default_rng()\n",
    ],
)
def test_scanner_flags_unseeded_randomness(source):
    assert violations(source)


@pytest.mark.parametrize(
    "source",
    [
        "import numpy as np\nrng = np.random.default_rng(seq)\n",
        "import numpy as np\nrng = np.random.default_rng(seed=3)\n",
        "import numpy as np\ndef f(r: np.random.Generator) -> None: ...\n",
        "import numpy as np\nseq = np.random.SeedSequence(3)\n",
        textwrap.dedent(
            """
            def f(rng):
                return rng.random()
            """
        ),
    ],
)
def test_scanner_accepts_keyed_generators(source):
    assert violations(source) == []


def test_qualification_packages_use_no_unseeded_randomness():
    problems = []
    for package in PACKAGES:
        directory = ROOT / "scmle" / "data" / package
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            problems += [
                f"{path.relative_to(ROOT)}: {item}"
                for item in violations(path.read_text(encoding="utf-8"))
            ]
    assert problems == []
