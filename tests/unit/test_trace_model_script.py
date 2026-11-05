"""A reviewer traces one model with only the retained records.

The script runs in an isolated interpreter, with a working directory that
holds nothing but a copy of the trail folder, so it cannot read the
repository, the code that built the model, or the original run.
"""

import ast
import json
import pathlib
import shutil
import subprocess  # nosec B404
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "trace_model.py"
EVIDENCE = ROOT / "plan" / "evidence" / "phase-18" / "local"


@pytest.fixture
def reviewer_desk(tmp_path):
    """A folder holding only the retained files and the keys."""
    desk = tmp_path / "desk"
    shutil.copytree(EVIDENCE / "trail", desk / "trail")
    shutil.copyfile(EVIDENCE / "keys.json", desk / "keys.json")
    shutil.copyfile(SCRIPT, desk / "trace_model.py")
    return desk


def walk(desk, *extra):
    """Runs the copied script from the desk."""
    return subprocess.run(  # nosec B603
        [sys.executable, "-I", "trace_model.py", "trail", *extra],
        cwd=desk,
        capture_output=True,
        text=True,
        check=False,
    )


def test_the_script_imports_nothing_from_the_project():
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            modules.add((node.module or "").split(".")[0])
    assert "scmle" not in modules
    assert modules <= set(sys.stdlib_module_names)


def test_a_reviewer_traces_the_model_from_the_retained_records(reviewer_desk):
    done = walk(reviewer_desk, "--keys", "keys.json")
    assert done.returncode == 0, done.stdout + done.stderr
    for phrase in (
        "The business question",
        "Framing and the decision-cost model",
        "Domain quality thresholds",
        "the registered bundle is the bundle the run wrote",
        "the approval: signature made with the reviewer key",
        "Observed outcomes against the approved expectation",
        "every check passed",
    ):
        assert phrase in done.stdout
    assert "[FAIL]" not in done.stdout


def test_without_keys_signatures_are_not_claimed_as_checked(reviewer_desk):
    done = walk(reviewer_desk)
    assert done.returncode == 2
    assert "not checked" in done.stdout


def test_a_changed_record_fails_the_walk(reviewer_desk):
    path = reviewer_desk / "trail/retained/run/evaluation.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["test_metrics"]["roc_auc"] = 0.99
    path.write_text(json.dumps(document), encoding="utf-8")
    done = walk(reviewer_desk, "--keys", "keys.json")
    assert done.returncode == 1
    assert "evaluation.json is the sealed file" in done.stdout


def test_a_missing_record_fails_the_walk(reviewer_desk):
    (reviewer_desk / "trail/retained/registry/audit.jsonl").unlink()
    done = walk(reviewer_desk, "--keys", "keys.json")
    assert done.returncode == 1
    assert "[FAIL]" in done.stdout


def test_a_wrong_key_fails_the_walk(reviewer_desk):
    keys = json.loads((reviewer_desk / "keys.json").read_text())
    keys["reviewer"] = "someone-else"
    (reviewer_desk / "keys.json").write_text(json.dumps(keys))
    done = walk(reviewer_desk, "--keys", "keys.json")
    assert done.returncode == 1
    assert "the approval: signature made with the reviewer key" in done.stdout
