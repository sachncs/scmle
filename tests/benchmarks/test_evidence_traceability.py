"""Claim C6 traceability evidence: verdicts, merging and the controls."""

import copy
import json
import math
import shutil
import subprocess
from typing import Any

import pytest

from benchmarks.evidence import reproducibility
from benchmarks.evidence import traceability
from scmle.contracts import errors
from scmle.trail import origin
from scmle.trail import replica
from tests.unit import test_trail_origin

FIELDS = ["data hash", "configuration", "commit", "environment"]
# Stands for the code hash: the real one hashes the whole package, which
# other work may change between two calls, and takes seconds each time.
CODE = "f" * 64


@pytest.fixture(autouse=True)
def code(monkeypatch):
    monkeypatch.setattr(traceability, "fingerprint", lambda: CODE)


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    return test_trail_origin.train(tmp_path_factory.mktemp("trained"))


def traced(
    *failing: str, integrity: bool = False, signed: bool = True
) -> dict[str, Any]:
    return {
        "integrity": ["broken"] if integrity else [],
        "signed": signed,
        "checks": {
            name: {"verified": name not in failing, "problems": []}
            for name in FIELDS
        },
        "traced": not failing and not integrity,
    }


def rebuilt(identical: bool = True, relative: float = 0.0) -> dict[str, Any]:
    tolerance = traceability.tolerance()
    return {
        "identical": identical,
        "gap": {"absolute": relative, "relative": relative},
        "reproduced": identical or relative <= tolerance,
    }


def document(runner: bool = True, **changes: Any) -> dict[str, Any]:
    found = {
        "version": 1,
        "claim": "C6",
        "tolerance": {
            "name": traceability.TOLERANCE,
            "value": traceability.tolerance(),
        },
        "code": CODE,
        "traced": {kind: traced() for kind in reproducibility.TASKS},
        "controls": {
            name: {"expected": pair[1], "failed": [pair[1]], "caught": True}
            for name, pair in traceability.CONTROLS.items()
        },
        "local": {kind: rebuilt() for kind in reproducibility.TASKS},
        "runner": {
            "code": CODE,
            "traced": {kind: traced() for kind in reproducibility.TASKS},
            "rebuilt": {kind: rebuilt() for kind in reproducibility.TASKS},
        }
        if runner
        else None,
    }
    found.update(changes)
    return found


def test_every_bundle_traced_and_every_control_caught_passes():
    value, verdict = traceability.summary(document())
    assert verdict == "pass"
    assert value == (
        "4/4 bundles from real runs trace to data hash, configuration, "
        "commit and environment, each verified; 7/7 tamperings caught by "
        "exactly the field they break"
    )


def test_a_field_that_fails_fails_the_trace_clause_and_is_named():
    found = document()
    found["traced"]["binary"] = traced("commit")
    value, verdict = traceability.summary(found)
    assert verdict == "fail"
    assert value.endswith("; binary fails commit")


def test_a_broken_manifest_fails_the_trace_clause():
    found = document()
    found["traced"]["regression"] = traced(integrity=True)
    assert traceability.summary(found)[1] == "fail"
    assert traceability.failing(traced(integrity=True)) == ["integrity"]


def test_an_unsigned_bundle_fails_the_trace_clause():
    found = document()
    found["traced"]["binary"] = traced(signed=False)
    value, verdict = traceability.summary(found)
    assert verdict == "fail"
    assert value.endswith("; binary fails integrity")
    assert traceability.failing({"integrity": [], "checks": {}}) == [
        "integrity"
    ]


def test_a_missed_control_fails_the_trace_clause():
    found = document()
    found["controls"]["unknown commit"]["caught"] = False
    value, verdict = traceability.summary(found)
    assert verdict == "fail"
    assert value.endswith("; missed: unknown commit")


def test_a_dropped_control_fails_the_trace_clause():
    found = document()
    del found["controls"]["unknown commit"]
    assert traceability.summary(found)[1] == "fail"


def test_rebuilds_on_both_sides_pass_the_rebuild_clause():
    value, verdict = traceability.summary(document(), traceability.REBUILD)
    assert verdict == "pass"
    assert value == (
        "this machine: 4/4 bundles rebuilt from the manifest alone reproduce "
        "(4 byte-identical; max metric relative difference 0, tol 1e-06); "
        "a clean runner: 4/4 bundles rebuilt from the manifest alone "
        "reproduce (4 byte-identical; max metric relative difference 0, "
        "tol 1e-06)"
    )


def test_without_a_clean_runner_the_rebuild_clause_is_not_measured():
    value, verdict = traceability.summary(
        document(runner=False), traceability.REBUILD
    )
    assert verdict == "not measured"
    assert value.endswith("; clean runner not measured")


def test_a_rebuild_beyond_the_tolerance_fails_even_without_a_runner():
    found = document(runner=False)
    beyond = 10 * traceability.tolerance()
    found["local"]["binary"] = rebuilt(identical=False, relative=beyond)
    assert traceability.summary(found, traceability.REBUILD)[1] == "fail"


def test_a_rebuild_within_the_tolerance_reproduces():
    found = document()
    within = traceability.tolerance() / 10
    found["runner"]["rebuilt"]["binary"] = rebuilt(False, within)
    value, verdict = traceability.summary(found, traceability.REBUILD)
    assert verdict == "pass"
    assert "4/4 bundles rebuilt from the manifest alone reproduce (3 " in value


def test_a_runner_missing_a_task_fails_the_rebuild_clause():
    found = document()
    del found["runner"]["rebuilt"]["binary"]
    assert traceability.summary(found, traceability.REBUILD)[1] == "fail"


def test_a_runner_bundle_that_does_not_trace_fails_the_rebuild_clause():
    found = document()
    found["runner"]["traced"]["interaction"] = traced("environment")
    value, verdict = traceability.summary(found, traceability.REBUILD)
    assert verdict == "fail"
    assert value.endswith("; a clean runner: interaction do not trace there")
    missing = document()
    del missing["runner"]["traced"]["binary"]
    assert traceability.summary(missing, traceability.REBUILD)[1] == "fail"


def test_a_rebuild_that_could_not_run_fails_the_rebuild_clause():
    found = document()
    failed = traceability.unbuilt(errors.ArtifactError("uv is not installed"))
    found["runner"]["rebuilt"]["binary"] = failed
    value, verdict = traceability.summary(found, traceability.REBUILD)
    assert verdict == "fail"
    assert "a clean runner: 3/4 bundles" in value
    assert "max metric relative difference inf" in value


def test_stale_or_malformed_documents_are_not_measured():
    stale = document(code="0" * 64)
    assert traceability.problems(stale) == [
        "traceability: measurements are for other code"
    ]
    for clause in traceability.CLAUSES:
        assert traceability.summary(stale, clause)[1] == "not measured"
    moved = copy.deepcopy(document())
    moved["tolerance"]["value"] = 1.0
    assert traceability.problems(moved) == [
        "traceability: cross_platform_tol differs from acceptance.toml"
    ]
    assert traceability.problems({"code": "x"}) == [
        "traceability: the document has no trace records"
    ]
    short = document()
    del short["traced"]["binary"]
    assert traceability.problems(short) == [
        "traceability: tasks not traced: binary"
    ]


def test_an_unknown_clause_is_refused():
    with pytest.raises(errors.InputError, match="unknown clause"):
        traceability.summary(document(), "other")


def test_merge_stores_a_runner_record_for_the_same_code(tmp_path, monkeypatch):
    path = tmp_path / "traceability.json"
    path.write_text(json.dumps(document(runner=False)), encoding="utf-8")
    monkeypatch.setattr(traceability, "PATH", path)
    record = tmp_path / "rebuilt.json"
    runner = document()["runner"]
    record.write_text(json.dumps(runner), encoding="utf-8")
    assert traceability.merge(record)["runner"] == runner
    record.write_text(json.dumps({**runner, "code": "x"}), encoding="utf-8")
    with pytest.raises(errors.InputError, match="other code"):
        traceability.merge(record)


@pytest.mark.parametrize("missing", ["traced", "rebuilt"])
def test_merge_refuses_a_record_without_both_sections(
    tmp_path, monkeypatch, missing
):
    path = tmp_path / "traceability.json"
    path.write_text(json.dumps(document(runner=False)), encoding="utf-8")
    monkeypatch.setattr(traceability, "PATH", path)
    runner = document()["runner"]
    del runner[missing]
    record = tmp_path / "rebuilt.json"
    record.write_text(json.dumps(runner), encoding="utf-8")
    with pytest.raises(errors.InputError, match="no trace and rebuild"):
        traceability.merge(record)
    record.write_text("[]", encoding="utf-8")
    with pytest.raises(errors.InputError, match="no trace and rebuild"):
        traceability.merge(record)


def test_main_merge_writes_the_document(tmp_path, monkeypatch, capsys):
    path = tmp_path / "traceability.json"
    path.write_text(json.dumps(document(runner=False)), encoding="utf-8")
    monkeypatch.setattr(traceability, "PATH", path)
    record = tmp_path / "rebuilt.json"
    record.write_text(json.dumps(document()["runner"]), encoding="utf-8")
    assert traceability.main(["--merge", str(record)]) == 0
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored["runner"] == document()["runner"]
    printed = [
        json.loads(line) for line in capsys.readouterr().out.splitlines()
    ]
    assert [line["verdict"] for line in printed] == ["pass", "pass"]


def test_main_check_is_zero_when_current_and_one_when_stale(
    tmp_path, monkeypatch, capsys
):
    path = tmp_path / "traceability.json"
    monkeypatch.setattr(traceability, "PATH", path)
    path.write_text(json.dumps(document()), encoding="utf-8")
    assert traceability.main(["--check"]) == 0
    assert json.loads(capsys.readouterr().out) == {"ok": True, "problems": []}
    path.write_text(json.dumps(document(code="0" * 64)), encoding="utf-8")
    assert traceability.main(["--check"]) == 1
    assert json.loads(capsys.readouterr().out)["problems"] == [
        "traceability: measurements are for other code"
    ]
    path.unlink()
    assert traceability.main(["--check"]) == 1


def test_main_rebuild_needs_a_record(tmp_path, capsys):
    with pytest.raises(SystemExit) as exited:
        traceability.main(["--rebuild", str(tmp_path)])
    assert exited.value.code == 2
    assert "--rebuild needs --record" in capsys.readouterr().err


def dirty(monkeypatch):
    def git(repository, *arguments):
        return subprocess.CompletedProcess(arguments, 0, " M scmle/x.py\n", "")

    monkeypatch.setattr(origin, "git", git)


def test_a_dirty_tree_is_refused_before_training(tmp_path, monkeypatch):
    dirty(monkeypatch)
    trains = []
    monkeypatch.setattr(traceability, "train", lambda *a: trains.append(a))
    assert traceability.clean() is False
    for call in (
        lambda: traceability.build(tmp_path),
        lambda: traceability.main(["--build", str(tmp_path)]),
        traceability.measure,
    ):
        with pytest.raises(errors.InputError, match="uncommitted changes"):
            call()
    assert trains == []


def test_a_failing_git_status_is_an_error(monkeypatch):
    monkeypatch.setattr(
        origin,
        "git",
        lambda *a: subprocess.CompletedProcess(a, 128, "", "not a repo"),
    )
    with pytest.raises(errors.ArtifactError, match="git status failed"):
        traceability.clean()


def test_build_trains_every_task_in_its_own_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(traceability, "clean", lambda: True)
    monkeypatch.setattr(
        traceability, "train", lambda kind, folder: folder / "runs" / kind
    )
    assert traceability.build(tmp_path) == {
        kind: tmp_path / kind / "runs" / kind for kind in reproducibility.TASKS
    }


def test_train_runs_the_command_line_with_the_signing_key(
    trained, tmp_path, monkeypatch
):
    calls = []

    def run(command, cwd, env, **options):
        calls.append((command, env["SCMLE_SIGNING_KEY"]))
        shutil.copytree(trained / "runs", cwd / "runs")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    found = traceability.train("binary", tmp_path / "binary")
    assert found == traceability.run(tmp_path / "binary")
    assert (tmp_path / "binary" / traceability.DATA).is_file()
    command, key = calls[0]
    assert command[1:5] == ["-m", "scmle", "run", traceability.DATA]
    assert command[command.index("--seed") + 1] == str(reproducibility.SEED)
    assert command[-2:] == ["--output-dir", "runs"]
    assert key == traceability.SIGNING_KEY


def test_train_reports_a_failing_command(tmp_path, monkeypatch):
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **options: subprocess.CompletedProcess(
            command, 1, "", "boom"
        ),
    )
    with pytest.raises(errors.ArtifactError, match="training binary failed"):
        traceability.train("binary", tmp_path / "binary")


def outcome(bundle) -> replica.Rebuilt:
    return replica.Rebuilt(
        str(bundle),
        "Python 3",
        1,
        "c" * 40,
        ("model_manifest.json",),
        (),
        {"auc": 0.9},
        {"auc": 0.9},
        {"absolute": 0.0, "relative": 0.0},
        replica.TOLERANCE,
        1.0,
    )


@pytest.fixture
def bundles(trained, tmp_path, monkeypatch):
    """A `--build` folder holding a copy of the shared run per task."""
    monkeypatch.setattr(traceability, "SIGNING_KEY", test_trail_origin.KEY)
    folder = tmp_path / "bundles"
    for kind in reproducibility.TASKS:
        shutil.copytree(trained, folder / kind)
    return folder


def test_main_rebuild_writes_a_record_and_passes(
    bundles, tmp_path, monkeypatch
):
    monkeypatch.setattr(replica, "rebuild", lambda bundle, **o: outcome(bundle))
    record = tmp_path / "rebuilt.json"
    assert (
        traceability.main(["--rebuild", str(bundles), "--record", str(record)])
        == 0
    )
    found = json.loads(record.read_text(encoding="utf-8"))
    assert found["code"] == CODE
    assert sorted(found["rebuilt"]) == sorted(reproducibility.TASKS)
    assert all(item["reproduced"] for item in found["rebuilt"].values())
    assert all(item["traced"] for item in found["traced"].values())


def test_main_rebuild_records_a_tampered_bundle_and_exits_one(
    bundles, tmp_path, monkeypatch
):
    real = replica.rebuild

    def rebuild(bundle, **options):
        if "binary" in bundle.parts:
            return real(bundle, **options)
        return outcome(bundle)

    monkeypatch.setattr(replica, "rebuild", rebuild)
    traceability.corrupt(
        traceability.run(bundles / "binary") / "model" / "pipeline.joblib"
    )
    shutil.rmtree(bundles / "regression" / "runs")
    record = tmp_path / "rebuilt.json"
    assert (
        traceability.main(["--rebuild", str(bundles), "--record", str(record)])
        == 1
    )
    found = json.loads(record.read_text(encoding="utf-8"))
    binary = found["rebuilt"]["binary"]
    assert binary["reproduced"] is False
    assert binary["gap"]["relative"] == math.inf
    assert "does not trace" in binary["error"]
    assert traceability.failing(found["traced"]["binary"]) == ["integrity"]
    assert "holds 0 runs" in found["rebuilt"]["regression"]["error"]
    assert found["traced"]["regression"]["traced"] is False
    assert found["rebuilt"]["multiclass"]["reproduced"] is True


def test_measure_keeps_a_runner_record_for_the_same_code(tmp_path, monkeypatch):
    path = tmp_path / "traceability.json"
    path.write_text(json.dumps(document()), encoding="utf-8")
    monkeypatch.setattr(traceability, "PATH", path)
    monkeypatch.setattr(
        traceability,
        "build",
        lambda root: {kind: root / kind for kind in reproducibility.TASKS},
    )
    monkeypatch.setattr(traceability, "check", lambda bundle: traced())
    monkeypatch.setattr(
        traceability, "control", lambda folder, scratch: document()["controls"]
    )
    monkeypatch.setattr(
        traceability,
        "rebuild",
        lambda runs: {kind: rebuilt() for kind in runs},
    )
    found = traceability.measure()
    assert found["runner"] == document()["runner"]
    assert found["traced"] == document()["traced"]
    assert found["local"] == document()["local"]
    assert traceability.summary(found) == traceability.summary(document())
    path.write_text(json.dumps(document(code="0" * 64)), encoding="utf-8")
    assert traceability.measure()["runner"] is None


def test_every_control_is_caught_by_exactly_its_field(
    trained,
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(traceability, "SIGNING_KEY", test_trail_origin.KEY)
    folder = tmp_path / "task"
    shutil.copytree(trained, folder)
    outcomes = traceability.control(folder, tmp_path / "controls")
    assert {name: item["failed"] for name, item in outcomes.items()} == {
        name: [pair[1]] for name, pair in traceability.CONTROLS.items()
    }
    assert all(item["caught"] for item in outcomes.values())
    clean = traceability.check(traceability.run(folder) / "model")
    assert traceability.failing(clean) == []
