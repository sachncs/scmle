"""Claim C1 evidence: a gate behind an optional extra is measured with it.

The ONNX divergence gate needs the `onnx` extra, which has no wheel for the
free-threaded interpreter the suite runs on. Its paths then run in an
interpreter that has the extra instead of ending `failed` on the import.
"""

import json
import subprocess
from typing import Any

import pytest

from benchmarks.evidence import faults


@pytest.mark.parametrize("part", faults.PARTS)
def test_the_onnx_gate_declares_its_extra(part: str) -> None:
    injection = faults.injections(part)["export.onnx.divergence"]
    assert injection.extra == "onnx"
    assert set(faults.EXTRAS["onnx"]) >= {"skl2onnx", "onnxruntime"}


def test_every_declared_extra_is_known() -> None:
    for part in faults.PARTS:
        for injection in faults.injections(part).values():
            assert not injection.extra or injection.extra in faults.EXTRAS


def test_an_extra_is_installed_only_when_every_module_imports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(faults.EXTRAS, "probe", ("json", "no_such_module_x"))
    assert not faults.installed("probe")
    monkeypatch.setitem(faults.EXTRAS, "probe", ("json",))
    assert faults.installed("probe")


def finished(stdout: str, code: int = 0) -> Any:
    """A completed process that printed `stdout`."""
    return subprocess.CompletedProcess([], code, stdout=stdout, stderr="boom")


def test_a_path_without_its_extra_runs_where_the_extra_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []
    inner = {
        "gate": "export.onnx.divergence",
        "path": "fault",
        "part": "real",
        "level": "run",
        "outcome": "blocked",
        "by": "export.onnx.divergence",
    }

    def launch(command: list[str], **options: Any) -> Any:
        seen.append(command)
        return finished("noise\n" + json.dumps(inner) + "\n")

    monkeypatch.setattr(faults, "installed", lambda extra: False)
    monkeypatch.setattr(faults.subprocess, "run", launch)
    record = faults.run("export.onnx.divergence", "fault", "real")
    assert record["outcome"] == "blocked"
    assert record["interpreter"] == faults.INTERPRETER
    assert record["extra"] == "onnx"
    assert faults.satisfies(record)
    (command,) = seen
    assert command[command.index("--extra") + 1] == "onnx"
    assert command[command.index("--python") + 1] == faults.INTERPRETER
    tail = command[command.index("--attempt") :]
    assert tail == ["--attempt", "export.onnx.divergence", "fault", "real"]


@pytest.mark.parametrize(
    "outcome",
    [
        finished("", 2),
        finished("not json\n"),
        subprocess.TimeoutExpired(["uv"], 1.0),
        FileNotFoundError("uv"),
    ],
)
def test_a_delegated_path_that_cannot_run_is_failed_not_satisfied(
    monkeypatch: pytest.MonkeyPatch, outcome: Any
) -> None:
    def launch(command: list[str], **options: Any) -> Any:
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(faults, "installed", lambda extra: False)
    monkeypatch.setattr(faults.subprocess, "run", launch)
    record = faults.run("export.onnx.divergence", "override", "simulated")
    assert record["outcome"] == "failed"
    assert record["message"]
    assert not faults.satisfies(record)


def test_a_path_with_its_extra_installed_runs_here(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called: list[tuple[str, str, str]] = []

    def attempt(name: str, path: str, part: str = "simulated") -> Any:
        called.append((name, path, part))
        return {"outcome": "blocked"}

    def refuse(*arguments: Any, **options: Any) -> Any:
        raise AssertionError("nothing is delegated")

    monkeypatch.setattr(faults, "installed", lambda extra: True)
    monkeypatch.setattr(faults, "attempt", attempt)
    monkeypatch.setattr(faults.subprocess, "run", refuse)
    faults.run("export.onnx.divergence", "control", "simulated")
    assert called == [("export.onnx.divergence", "control", "simulated")]


def test_the_attempt_mode_prints_the_record_as_its_last_line(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = faults.main(
        ["--attempt", "ingest.stage.run", "control", "simulated"]
    )
    assert code == 0
    lines = capsys.readouterr().out.strip().splitlines()
    record = json.loads(lines[-1])
    assert (record["gate"], record["outcome"]) == ("ingest.stage.run", "passed")
