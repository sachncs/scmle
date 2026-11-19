"""Claim C10 evidence: a test's stray child no longer leaves a module undecided.

mutmut's default fork runner reaps any child of its process and looks the
pid up among its own workers. A kill-set test that leaves a helper process
unreaped (the sealed test set's governance helper) made it die with
`KeyError: <pid>` on every attempt, so `scmle/evaluation/selection.py`
stayed undecided. The fork-server runner ignores children it did not start.
"""

import json

from benchmarks.evidence import mutation

STRAY = (
    "Traceback (most recent call last):\n"
    '  File "/x/site-packages/mutmut/__main__.py", line 1007, in drain\n'
    "    result = runner.wait_for_result()\n"
    '  File "/x/site-packages/mutmut/workers/isolation.py", line 889, in '
    "wait_for_result\n"
    "    worker = self._running.pop(pid)\n"
    "KeyError: 20420\n"
)
BROKEN = (
    "Traceback (most recent call last):\n"
    '  File "/x/site-packages/mutmut/workers/isolation.py", line 70, in '
    "send_message\n"
    "BrokenPipeError: [Errno 32] Broken pipe\n"
)


def test_a_stray_child_crash_is_told_apart_from_other_pool_crashes():
    assert mutation.stray(STRAY)
    assert not mutation.stray(BROKEN)
    assert not mutation.stray(STRAY.replace("workers/isolation", "core"))
    assert not mutation.stray("KeyError: 'name'\n")


def test_the_configuration_names_its_isolation(tmp_path):
    (tmp_path / "scmle").mkdir()
    mutation.configure(tmp_path, "scmle/a.py", ["tests/unit/test_a.py"])
    text = (tmp_path / "setup.cfg").read_text(encoding="utf-8")
    assert f"process_isolation = {mutation.ISOLATION[0]}" in text
    mutation.configure(tmp_path, "scmle/a.py", [], "forkserver")
    text = (tmp_path / "setup.cfg").read_text(encoding="utf-8")
    assert "process_isolation = forkserver" in text


def test_a_forked_child_may_initialise_objective_c_classes():
    # macOS aborts (SIGABRT, exit -6) a forked child that initialises an
    # Objective-C class another thread was initialising at the fork.
    assert mutation.environment()["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] == (
        "YES"
    )


def fake(tmp_path, monkeypatch, crash, crashes):
    """A tested module whose mutmut run crashes `crashes` times first."""
    package = tmp_path / "scmle" / "part"
    package.mkdir(parents=True)
    (tmp_path / "tests" / "unit").mkdir(parents=True)
    (package / "used.py").write_text("def f(x):\n    return x + 1\n")
    (tmp_path / "tests" / "unit" / "test_used.py").write_text(
        "from scmle.part import used\n"
    )
    path = "scmle/part/used.py"
    meta = tmp_path / "mutants" / f"{path}.meta"
    names = [f"scmle.part.used.x_f__mutmut_{i}" for i in range(1, 4)]
    isolations: list[str] = []

    def call(work, command, limit):
        meta.parent.mkdir(parents=True, exist_ok=True)
        if command[0] == "-c":
            meta.write_text(
                json.dumps({"exit_code_by_key": dict.fromkeys(names)})
            )
            return 0, ""
        config = (work / "setup.cfg").read_text(encoding="utf-8")
        isolations.append(config.split("process_isolation = ")[1].split()[0])
        if len(isolations) <= crashes:
            return 1, crash
        stored = {names[0]: 1, names[1]: 1, names[2]: 0}
        meta.write_text(json.dumps({"exit_code_by_key": stored}))
        return 0, "done"

    monkeypatch.setattr(mutation, "call", call)
    return isolations


def test_a_stray_child_crash_is_retried_under_the_fork_server(
    tmp_path, monkeypatch
):
    isolations = fake(tmp_path, monkeypatch, STRAY, crashes=1)
    found = mutation.examine(tmp_path, "scmle/part/used.py", 0, 60.0)
    assert isolations == ["fork", "forkserver"]
    assert found["retries"] == [
        {"workers": 2, "isolation": "fork", "exception": "KeyError: 20420"}
    ]
    assert not found["undecided"]
    assert found["decided"] == 3
    assert found["isolation"] == "forkserver"


def test_another_pool_crash_keeps_the_fork_runner(tmp_path, monkeypatch):
    isolations = fake(tmp_path, monkeypatch, BROKEN, crashes=1)
    found = mutation.examine(tmp_path, "scmle/part/used.py", 0, 60.0)
    assert isolations == ["fork", "fork"]
    assert found["isolation"] == "fork"
    assert found["decided"] == 3


def test_a_module_crashing_under_both_runners_stays_undecided(
    tmp_path, monkeypatch
):
    isolations = fake(tmp_path, monkeypatch, STRAY, crashes=100)
    found = mutation.examine(tmp_path, "scmle/part/used.py", 0, 60.0)
    assert len(isolations) == mutation.attempts()
    assert found["undecided"]
    assert found["decided"] == 0
    assert found["counts"]["killed"] == 0
