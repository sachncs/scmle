"""`scmle.trail.replica` and the `python -m scmle.trail` command line.

The end-to-end test rebuilds the shared real run of `test_trail_origin` in
a new virtual environment built from its manifest alone and retrains it. It
installs from PyPI (marked ``rebuild``) and checks out HEAD, so it runs only
when the tracked files match HEAD.
"""

import json
import math
import shutil
import subprocess
import sys

import pytest

from scmle.contracts import errors
from scmle.trail import __main__
from scmle.trail import origin
from scmle.trail import replica
from tests.unit import test_trail_origin


def dirty():
    """Whether tracked files differ from HEAD, which a rebuild checks out."""
    status = origin.git(
        test_trail_origin.ROOT, "status", "--porcelain", "--untracked-files=no"
    )
    return status.returncode != 0 or bool(status.stdout.strip())


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    return test_trail_origin.train(tmp_path_factory.mktemp("trained"))


def rebuilt(**changes):
    fields = {
        "bundle": "b",
        "python": "Python 3.14.7",
        "pins": 2,
        "commit": "c" * 40,
        "members": ("a.json",),
        "differing": (),
        "original": {"auc": 0.9},
        "replayed": {"auc": 0.9},
        "gap": {"absolute": 0.0, "relative": 0.0},
        "tolerance": replica.TOLERANCE,
        "seconds": 1.0,
    }
    return replica.Rebuilt(**{**fields, **changes})


def test_requirements_pin_everything_but_scmle_sorted():
    manifest = {
        "environment": {"scmle": "0.1.0", "pandas": "3.0.6", "numpy": "2.5.4"}
    }
    assert replica.requirements(manifest) == ["numpy==2.5.4", "pandas==3.0.6"]


def test_requirements_need_an_environment():
    with pytest.raises(errors.ArtifactError, match="no environment"):
        replica.requirements({"environment": {}})


def test_gap_is_zero_for_equal_metrics():
    assert replica.gap({"a": 0.5, "b": 0.0}, {"a": 0.5, "b": 0.0}) == {
        "absolute": 0.0,
        "relative": 0.0,
    }


def test_gap_is_relative_to_the_larger_magnitude():
    assert replica.gap({"a": 0.5}, {"a": 0.4}) == pytest.approx(
        {"absolute": 0.1, "relative": 0.2}
    )


def test_gap_is_infinite_for_a_metric_one_side_lacks():
    assert replica.gap({"a": 1.0}, {"b": 1.0})["relative"] == math.inf


def test_gap_skips_a_metric_undefined_on_both_sides():
    nan = float("nan")
    assert replica.gap({"a": nan}, {"a": nan})["relative"] == 0.0
    assert replica.gap({"a": nan}, {"a": 1.0})["relative"] == math.inf


def test_members_add_the_manifest_and_a_present_signature(tmp_path):
    manifest = {"files": {"pipeline.joblib": "x"}}
    assert replica.members(tmp_path, manifest) == [
        "model_manifest.json",
        "pipeline.joblib",
    ]
    (tmp_path / "signature.json").write_text("{}", encoding="utf-8")
    assert replica.members(tmp_path, manifest)[-1] == "signature.json"


def test_threads_follow_the_manifest_and_drop_the_callers_path(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/elsewhere")
    monkeypatch.setenv("OMP_NUM_THREADS", "8")
    monkeypatch.delenv("OPENBLAS_NUM_THREADS", raising=False)
    variables = replica.threads({"threads": {"openmp": 1, "other": 3}})
    assert variables["OMP_NUM_THREADS"] == "1"
    assert "PYTHONPATH" not in variables
    assert "OPENBLAS_NUM_THREADS" not in variables


def test_interpreter_is_inside_the_environment(tmp_path):
    found = replica.interpreter(tmp_path)
    assert found.parent.parent == tmp_path
    assert found.name in {"python", "python.exe"}


def report(data):
    return origin.Report("b", "r", str(data), (), (), True)


def test_stage_copies_a_relative_source_into_the_workspace(tmp_path):
    data = tmp_path / "verified.csv"
    data.write_text("a\n1\n", encoding="utf-8")
    work = tmp_path / "work"
    replica.stage(report(data), "inputs/data.csv", work)
    assert (work / "inputs" / "data.csv").read_text("utf-8") == "a\n1\n"


def test_stage_refuses_an_in_memory_run(tmp_path):
    with pytest.raises(errors.InputError, match="in-memory frame"):
        replica.stage(report(tmp_path / "x.csv"), "", tmp_path)


def test_stage_refuses_an_absolute_source_elsewhere(tmp_path):
    with pytest.raises(errors.InputError, match="place the verified data"):
        replica.stage(report(tmp_path / "x.csv"), "/data/y.csv", tmp_path)


def test_stage_keeps_an_absolute_source_that_is_the_verified_file(tmp_path):
    data = tmp_path / "x.csv"
    data.write_text("a\n", encoding="utf-8")
    replica.stage(report(data), str(data), tmp_path / "work")
    assert not (tmp_path / "work").exists()


def test_checkout_clones_the_commit_and_leaves_the_repository(tmp_path):
    git = shutil.which("git")
    source = tmp_path / "source"
    source.mkdir()
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.com"]

    def run(*arguments):
        return subprocess.run(
            [git, "-C", str(source), *identity, *arguments],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

    run("init", "--quiet")
    (source / "f.txt").write_text("one", encoding="utf-8")
    run("add", "f.txt")
    run("commit", "--quiet", "-m", "one")
    first = run("rev-parse", "HEAD")
    (source / "f.txt").write_text("two", encoding="utf-8")
    run("commit", "--quiet", "-am", "two")
    target = replica.checkout(first, source, tmp_path / "clone")
    assert (target / "f.txt").read_text("utf-8") == "one"
    assert (source / "f.txt").read_text("utf-8") == "two"


def test_checkout_of_an_unknown_commit_fails(tmp_path):
    with pytest.raises(errors.ArtifactError, match="checking out"):
        replica.checkout("0" * 40, test_trail_origin.ROOT, tmp_path / "clone")


def test_compare_names_nested_members_and_extras(tmp_path):
    original, rebuilt = tmp_path / "original", tmp_path / "rebuilt"
    for side in (original, rebuilt):
        (side / "parts").mkdir(parents=True)
        (side / "a.json").write_text("1", encoding="utf-8")
        (side / "parts" / "b.json").write_text("2", encoding="utf-8")
    (rebuilt / "parts" / "b.json").write_text("3", encoding="utf-8")
    (rebuilt / "parts" / "extra.bin").write_bytes(b"x")
    names = ["a.json", "parts/b.json"]
    assert replica.compare(original, rebuilt, names) == (
        "parts/b.json",
        "parts/extra.bin",
    )
    (rebuilt / "parts" / "b.json").write_text("2", encoding="utf-8")
    (rebuilt / "parts" / "extra.bin").unlink()
    assert replica.compare(original, rebuilt, names) == ()


def test_reproduced_needs_identical_bytes_or_metrics_within_tolerance():
    assert rebuilt().reproduced
    moved = {"absolute": 1e-3, "relative": 1e-3}
    assert not rebuilt(differing=("a.json",), gap=moved).reproduced
    assert rebuilt(differing=("a.json",)).reproduced
    record = rebuilt(differing=("a.json",), gap=moved).dump()
    assert record["identical"] is False
    assert record["within"] is False
    assert record["reproduced"] is False
    assert record["differing"] == ["a.json"]


def test_an_untraced_bundle_is_not_rebuilt(tmp_path):
    (tmp_path / "model_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(errors.ArtifactError, match="does not trace") as caught:
        replica.rebuild(tmp_path, repository=test_trail_origin.ROOT)
    assert "data hash: absent" in str(caught.value)


@pytest.mark.rebuild
@pytest.mark.skipif(
    dirty(), reason="rebuild compares against HEAD; tree is dirty"
)
def test_a_real_bundle_rebuilds_byte_identical(trained, tmp_path):
    bundle = next((trained / "runs").glob("run_*")) / "model"
    result = replica.rebuild(
        bundle,
        repository=test_trail_origin.ROOT,
        key=test_trail_origin.KEY,
        workspace=tmp_path / "rebuild",
    )
    assert result.differing == ()
    assert result.identical
    assert result.gap == {"absolute": 0.0, "relative": 0.0}
    assert result.original == result.replayed
    assert result.commit == origin.load(bundle)["commit"]
    assert result.pins == len(origin.load(bundle)["environment"]) - 1
    assert "model_manifest.json" in result.members
    assert "signature.json" in result.members


def test_the_command_line_traces_a_bundle(trained, capsys, monkeypatch):
    bundle = next((trained / "runs").glob("run_*")) / "model"
    monkeypatch.setenv("SCMLE_SIGNING_KEY", test_trail_origin.KEY)
    code = __main__.main(
        ["trace", str(bundle), "--repository", str(test_trail_origin.ROOT)]
    )
    assert code == 0
    assert json.loads(capsys.readouterr().out)["traced"] is True


def test_the_command_line_reports_an_error_with_code_two(tmp_path, capsys):
    assert __main__.main(["trace", str(tmp_path)]) == 2
    assert "no readable" in capsys.readouterr().err


def test_the_command_line_exits_one_for_an_untraced_bundle(
    trained, tmp_path, capsys
):
    copy = tmp_path / "copy"
    shutil.copytree(trained, copy)
    (copy / "data.csv").unlink()
    bundle = next((copy / "runs").glob("run_*")) / "model"
    assert (
        __main__.main(
            ["trace", str(bundle), "--repository", str(test_trail_origin.ROOT)]
        )
        == 1
    )
    assert json.loads(capsys.readouterr().out)["traced"] is False


def test_the_module_runs_as_a_program(tmp_path):
    done = subprocess.run(
        [sys.executable, "-m", "scmle.trail", "trace", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
        cwd=test_trail_origin.ROOT,
    )
    assert done.returncode == 2
    assert done.stderr.startswith("error: ")
