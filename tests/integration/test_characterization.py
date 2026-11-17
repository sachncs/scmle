"""Characterization goldens for whole runs (register task 0.3).

The restructure (plan stage B) must not change what a run produces except
where a commit says so. For one run per task type this test records the run
directory's file list, the key structure and value types of every JSON file
in it, and the headline metrics, and compares them with
`tests/fixtures/run_goldens.json`.

A deliberate change regenerates the goldens in the same commit, naming the
register rows it closes (plan section 8, the rewrite rule):

    SCMLE_WRITE_GOLDENS=1 uv run pytest tests/integration/test_characterization.py
"""

import json
import math
import os
import pathlib

import numpy
import pandas
import pytest
import support

from scmle.contracts import tasks

GOLDENS = (
    pathlib.Path(__file__).resolve().parents[1]
    / "fixtures"
    / "run_goldens.json"
)
WRITE = os.environ.get("SCMLE_WRITE_GOLDENS") == "1"
# Metrics may move by this much without a change in behaviour: seeds fix
# the data and the splits, and only floating-point order differs between
# platforms. Policy: far below any difference a model change would make.
TOLERANCE = 1e-6
# Models an optional extra adds; excluded so the goldens hold whichever
# extras are installed (the CI jobs install different sets).
OPTIONAL = ["lightgbm", "xgboost", "catboost"]
# Keys whose mapping lists the installed packages or the loaded math
# libraries (`threads`: OpenMP, OpenBLAS, MKL ...), which vary by environment.
INSTALLED = frozenset(
    {"environment", "libraries", "packages", "dependencies", "threads"}
)


def table(task, rows=240, seed=7):
    """A seeded synthetic table with a learnable target."""
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "x0": rng.normal(size=rows),
            "x1": rng.normal(size=rows),
            "x2": rng.choice(["a", "b", "c"], rows),
        }
    )
    signal = frame["x0"] + 0.5 * frame["x1"] + rng.normal(scale=0.5, size=rows)
    if task == "binary":
        frame["y"] = (signal > 0).astype(int)
    elif task == "multiclass":
        frame["y"] = pandas.cut(
            signal, 3, labels=["low", "mid", "high"]
        ).astype(str)
    else:
        frame["y"] = signal
    return frame


def shape(value):
    """The structure of a JSON value: key trees with value type names.

    A mapping listed under a key in `INSTALLED` names the packages of the
    environment the run happened in, so only its presence is recorded.
    """
    if isinstance(value, dict):
        return {
            str(k): "installed" if k in INSTALLED else shape(v)
            for k, v in sorted(value.items())
        }
    if isinstance(value, list):
        return [shape(value[0])] if value else []
    return type(value).__name__


def headline(run_dir):
    """The metrics the goldens hold: selection and test scores."""
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    found = {"selected_cv_score": evaluation.get("selected_cv_score")}
    for name, score in sorted(evaluation.get("test_metrics", {}).items()):
        if isinstance(score, int | float):
            found[f"test.{name}"] = score
    return found


def capture(run_dir):
    """Everything a golden records about one run directory."""
    files = sorted(
        p.relative_to(run_dir).as_posix()
        for p in run_dir.rglob("*")
        if p.is_file()
    )
    structure = {}
    for name in files:
        if name.endswith(".json"):
            try:
                structure[name] = shape(
                    json.loads((run_dir / name).read_text())
                )
            except json.JSONDecodeError:
                structure[name] = "unreadable"
    return {
        "files": files,
        "structure": structure,
        "metrics": headline(run_dir),
    }


@pytest.mark.parametrize("task", ["binary", "multiclass", "regression"])
def test_a_run_produces_what_its_golden_records(task, tmp_path):
    options = {"task": "regression"} if task == "regression" else {}
    models = tasks.Candidates(
        exclude_families=support.FAST_MODELS.exclude_families,
        exclude_models=OPTIONAL,
    )
    result = support.make_project(
        table(task), "y", output_dir=tmp_path, models=models, **options
    ).run()
    found = capture(pathlib.Path(result.run_dir))
    goldens = json.loads(GOLDENS.read_text()) if GOLDENS.exists() else {}
    if WRITE:
        goldens[task] = found
        GOLDENS.write_text(json.dumps(goldens, indent=1, sort_keys=True) + "\n")
        return
    expected = goldens[task]
    assert found["files"] == expected["files"]
    assert found["structure"] == expected["structure"]
    assert set(found["metrics"]) == set(expected["metrics"])
    for name, value in expected["metrics"].items():
        actual = found["metrics"][name]
        if value is None or actual is None:
            assert actual == value, name
        else:
            assert math.isclose(
                actual, value, rel_tol=TOLERANCE, abs_tol=TOLERANCE
            ), name
