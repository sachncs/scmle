"""The evidence runner re-measures only stale stores, in order, in a clone."""

import importlib.util
import json
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "evidence.py"
# A store module: `--check` fails unless its JSON says "fresh"; `--write`
# makes it fresh and logs the call to `order.log` (ignored by git).
STORE = '''"""Store."""
import json, pathlib, sys
PATH = pathlib.Path(__file__).with_suffix(".json")
LOG = pathlib.Path(__file__).resolve().parents[2] / "order.log"
if "--check" in sys.argv:
    fresh = PATH.exists() and json.loads(PATH.read_text())["state"] == "fresh"
    sys.exit(0 if fresh else 1)
if "--write" in sys.argv:
    with LOG.open("a") as log:
        log.write(PATH.stem + "\\n")
    PATH.write_text(json.dumps({"state": "fresh"}))
'''
BROKEN = '''"""Store."""
import sys
sys.exit(2 if "--write" in sys.argv else 1 if "--check" in sys.argv else 0)
'''
CERTIFY = '''"""Certify."""
import json, pathlib, sys
PATH = pathlib.Path(__file__).parents[2] / "scmle/certificates/toy.json"
if "--check" in sys.argv:
    sys.exit(0 if PATH.exists() else 1)
if "--write" in sys.argv:
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps({"state": "fresh"}))
'''
SCORECARD = '''"""Scorecard."""
print("C1 pass")
'''


def load():
    """Imports the script as a module, since `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location("evidence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def git(root, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def stale(name):
    return json.dumps({"state": "stale" if name else "fresh"})


@pytest.fixture
def repo(tmp_path):
    """A committed toy repository with stale and fresh stores."""
    root = tmp_path / "repo"
    files = {
        ".gitignore": "order.log\n__pycache__/\n",
        "benchmarks/__init__.py": "",
        "benchmarks/scorecard.py": SCORECARD,
        "benchmarks/evidence/__init__.py": "",
        "benchmarks/evidence/helpers.py": '"""Not a store."""\n',
        "benchmarks/qualification/__init__.py": "",
        "benchmarks/qualification/toy_certify.py": CERTIFY,
    }
    for name in ("reproducibility", "leakage", "custody", "quality", "budget"):
        files[f"benchmarks/evidence/{name}.py"] = STORE
        files[f"benchmarks/evidence/{name}.json"] = stale(name != "custody")
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    git(root, "init", "-q")
    git(root, "add", ".")
    git(root, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "s")
    return root


def run(module, repo, *extra):
    clone = repo.parent / "clone"
    code = module.main(
        [
            "--repo",
            str(repo),
            "--clone",
            str(clone),
            "--commit",
            git(repo, "rev-parse", "HEAD"),
            "--python",
            sys.executable,
            *extra,
        ]
    )
    return code, clone


def logged(clone):
    return (clone / "order.log").read_text(encoding="utf-8").split()


def state(root, name):
    path = root / "benchmarks/evidence" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))["state"]


def test_discover_finds_checkable_stores_and_certificates(repo):
    found = {job.module: job.timing for job in load().discover(repo)}
    assert found == {
        "benchmarks.evidence.budget": True,
        "benchmarks.evidence.custody": False,
        "benchmarks.evidence.leakage": False,
        "benchmarks.evidence.quality": True,
        "benchmarks.evidence.reproducibility": False,
        "benchmarks.qualification.toy_certify": False,
    }


def test_stale_untimed_stores_are_measured_reproducibility_first(repo, capsys):
    code, clone = run(load(), repo)
    order = logged(clone)
    assert code == 0
    assert order[0] == "reproducibility"
    assert sorted(order) == ["leakage", "reproducibility"]
    assert state(repo, "leakage") == "fresh"
    assert state(repo, "quality") == "stale"
    assert (repo / "scmle/certificates/toy.json").exists()
    assert "C1 pass" in capsys.readouterr().out


def test_timing_stores_wait_for_quiet_machine(repo, monkeypatch):
    module = load()
    readings = [9.0, 9.0]
    monkeypatch.setattr(
        module.os,
        "getloadavg",
        lambda: ((readings.pop() if readings else 0.1),) * 3,
    )
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    code, clone = run(module, repo, "--timing", "--load", "1", "--wait", "60")
    assert code == 0
    assert sorted(logged(clone)[-2:]) == ["budget", "quality"]
    assert state(repo, "quality") == "fresh"


def test_busy_machine_raises_typed_error(repo, monkeypatch):
    module = load()
    monkeypatch.setattr(module.os, "getloadavg", lambda: (9.0,) * 3)
    monkeypatch.setattr(module.time, "sleep", lambda seconds: None)
    with pytest.raises(module.LoadError):
        run(module, repo, "--timing", "--load", "1", "--wait", "0")


def test_failed_measurement_raises_and_copies_nothing(repo):
    module = load()
    path = repo / "benchmarks/evidence/leakage.py"
    path.write_text(BROKEN, encoding="utf-8")
    git(
        repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qam", "b"
    )
    with pytest.raises(module.MeasureError):
        run(module, repo)
    assert state(repo, "reproducibility") == "stale"


def test_dirty_clone_is_refused(repo):
    module = load()
    clone = repo.parent / "clone"
    git(repo.parent, "clone", "-q", str(repo), str(clone))
    (clone / "benchmarks/evidence/leakage.json").write_text("{}")
    with pytest.raises(module.CloneError):
        run(module, repo)


def test_clone_at_another_commit_is_refused(repo):
    module = load()
    clone = repo.parent / "clone"
    git(repo.parent, "clone", "-q", str(repo), str(clone))
    (clone / "extra.txt").write_text("x")
    git(clone, "add", ".")
    git(
        clone, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "x"
    )
    with pytest.raises(module.CloneError):
        run(module, repo)
