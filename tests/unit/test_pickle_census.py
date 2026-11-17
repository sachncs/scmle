"""A bundle's pickles name only modules pinned with the bundle format.

A bundle stores the module path of every class it pickles, so moving one
of those modules makes older bundles unloadable. This test builds a binary,
a multiclass and a regression bundle, records the scmle modules their
pickled members name, and compares them with
`tests/fixtures/pickled_modules.json`. Moving a pinned module fails here
until `BUNDLE_FORMAT` is bumped and the census is rewritten (register task
3.4b): `python tests/unit/test_pickle_census.py --write`.
"""

import importlib.util
import json
import pathlib
import sys
import tempfile
from unittest import mock

from joblib import numpy_pickle
import numpy
import pandas
import pytest

import scmle.artifacts.bundle
from scmle.contracts import data
from scmle.contracts import tasks
import scmle.models.registry
from scmle.preparation import preprocessing

CENSUS = pathlib.Path(__file__).parents[1] / "fixtures" / "pickled_modules.json"
PLAN = data.Blueprint(("num",), ("cat",))
ROWS = 120


def build(task, folder):
    """Saves one small random-forest bundle for a task."""
    rng = numpy.random.default_rng(0)
    features = pandas.DataFrame(
        {"num": rng.normal(size=ROWS), "cat": rng.choice(["a", "b"], ROWS)}
    )
    if task is tasks.Task.REGRESSION:
        target = features.num * 2
    elif task is tasks.Task.MULTICLASS:
        target = pandas.Series(numpy.digitize(features.num, [-0.5, 0.5]))
    else:
        target = (features.num > 0).astype(int)
    registry = scmle.models.registry.Catalog.stock()
    fitted = preprocessing.instantiate(
        PLAN, registry.get("random_forest"), task, 0
    ).fit(features, target)
    bundle = scmle.artifacts.bundle.Bundle.create(
        fitted, PLAN, features, "y", task.value, "roc_auc", "rf"
    )
    return bundle.save(folder / task.value)


def record(folder):
    """The scmle modules the pickled members of a bundle folder name."""
    seen = set()
    original = numpy_pickle.NumpyUnpickler.find_class

    def find(self, module, name):
        if module.split(".")[0] == "scmle":
            seen.add(module)
        return original(self, module, name)

    with mock.patch.object(numpy_pickle.NumpyUnpickler, "find_class", find):
        for path in sorted(folder.glob("*.joblib")):
            numpy_pickle.load(path)
    return seen


def census(root):
    """Every scmle module the three bundles pickle."""
    found = set()
    for task in (
        tasks.Task.BINARY,
        tasks.Task.MULTICLASS,
        tasks.Task.REGRESSION,
    ):
        found |= record(build(task, root))
    return sorted(found)


def pinned():
    return json.loads(CENSUS.read_text(encoding="utf-8"))


def test_every_pinned_module_imports_while_the_format_is_unchanged():
    stored = pinned()
    if stored["bundle_format"] != scmle.artifacts.bundle.BUNDLE_FORMAT:
        pytest.fail("BUNDLE_FORMAT changed: rewrite the census with --write")
    missing = [
        name
        for name in stored["modules"]
        if importlib.util.find_spec(name) is None
    ]
    assert not missing, f"bump BUNDLE_FORMAT: pickled modules moved {missing}"


def test_bundles_pickle_only_pinned_modules(tmp_path):
    found = census(tmp_path)
    stored = pinned()["modules"]
    assert set(found) <= set(stored), (
        "bump BUNDLE_FORMAT and rewrite the census: bundles now pickle "
        f"{sorted(set(found) - set(stored))}"
    )
    assert found, "the census recorded no scmle module"


if __name__ == "__main__" and sys.argv[1:] == ["--write"]:
    with tempfile.TemporaryDirectory() as scratch:
        modules = census(pathlib.Path(scratch))
    document = {
        "bundle_format": scmle.artifacts.bundle.BUNDLE_FORMAT,
        "modules": modules,
    }
    CENSUS.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
