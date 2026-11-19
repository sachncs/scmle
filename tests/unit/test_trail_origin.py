"""`scmle.trail.origin`: a bundle traces to data, configuration, commit, env.

One real run, made by the command line in a fresh process from a CSV file,
is shared; each test copies it before changing anything.
"""

import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy
import pandas
import pytest

from scmle.artifacts.bundle import layout
from scmle.artifacts.bundle import seal
from scmle.contracts import errors
from scmle.trail import origin

ROOT = pathlib.Path(__file__).resolve().parents[2]
KEY = "origin-test-key"
FIELDS = ["data hash", "configuration", "commit", "environment"]


def train(folder):
    """Trains a small binary task with the command line inside `folder`."""
    rng = numpy.random.default_rng(3)
    frame = pandas.DataFrame(
        {"a": rng.normal(size=300), "b": rng.normal(size=300)}
    )
    frame["y"] = (frame.a + rng.normal(scale=0.5, size=300) > 0).astype(int)
    frame.to_csv(folder / "data.csv", index=False)
    done = subprocess.run(
        [
            sys.executable,
            "-m",
            "scmle",
            "run",
            "data.csv",
            "--target",
            "y",
            "--seed",
            "3",
            "--budget",
            "600",
            "--exclude-families",
            "bagging",
            "deep_learning",
            "foundation",
            "boosting",
            "--output-dir",
            "runs",
        ],
        cwd=folder,
        env={**os.environ, "SCMLE_SIGNING_KEY": KEY, "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    return folder


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    return train(tmp_path_factory.mktemp("trained"))


@pytest.fixture
def folder(trained, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(trained, copy)
    return copy


def bundle(folder):
    return next((folder / "runs").glob("run_*")) / "model"


def check(folder, **kwargs):
    kwargs.setdefault("repository", ROOT)
    kwargs.setdefault("key", KEY)
    return origin.trace(bundle(folder), **kwargs)


def resign(path, manifest):
    target = path / layout.MANIFEST_FILE
    target.write_text(json.dumps(manifest), encoding="utf-8")
    signature = json.loads((path / seal.SIGNATURE_FILE).read_text("utf-8"))
    signature["signature"] = seal.sign(target.read_bytes(), KEY.encode())
    (path / seal.SIGNATURE_FILE).write_text(json.dumps(signature), "utf-8")


def failed(report):
    return [check.name for check in report.checks if not check.verified]


def test_a_real_bundle_traces_to_all_four_fields(folder):
    report = check(folder)
    assert [c.name for c in report.checks] == FIELDS
    assert report.problems() == []
    assert report.traced
    assert report.signed
    assert report.data == str(folder / "data.csv")
    assert report.run == str(bundle(folder).parent)
    record = report.dump()
    assert record["traced"] is True
    assert set(record["checks"]) == set(FIELDS)
    assert record["checks"]["environment"]["value"].endswith(" pins")


def test_the_run_directory_defaults_to_the_bundles_parent(folder):
    copy = folder.parent / "saved"
    shutil.copytree(bundle(folder), copy)
    alone = origin.trace(copy, repository=ROOT, key=KEY)
    assert alone.run == ""
    assert failed(alone) == ["data hash", "configuration"]
    given = origin.trace(
        copy, run=bundle(folder).parent, repository=ROOT, key=KEY
    )
    assert given.traced


def test_a_changed_data_value_fails_only_the_data_hash(folder):
    path = folder / "data.csv"
    lines = path.read_text(encoding="utf-8").splitlines()
    cells = lines[1].split(",")
    cells[0] = "123.5"
    lines[1] = ",".join(cells)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report = check(folder)
    assert failed(report) == ["data hash"]
    problems = report.checks[0].problems
    assert len(problems) == 2
    assert "raw source" in problems[0]
    assert "rehashes to" in problems[1]


def test_a_missing_data_file_fails_the_data_hash(folder):
    (folder / "data.csv").unlink()
    report = check(folder)
    assert failed(report) == ["data hash"]
    assert report.checks[0].problems == (
        "the training data (data.csv) cannot be found",
    )


def test_a_moved_data_file_can_be_given(folder):
    moved = folder.parent / "moved.csv"
    (folder / "data.csv").rename(moved)
    assert check(folder, data=moved).traced


def test_a_changed_configuration_fails_only_the_configuration(folder):
    path = bundle(folder).parent / "config.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    config["validation"]["seed"] += 1
    path.write_text(json.dumps(config), encoding="utf-8")
    report = check(folder)
    assert failed(report) == ["configuration"]
    assert "rehashes to" in report.checks[1].problems[0]


def test_an_unknown_commit_fails_only_the_commit(folder):
    manifest = origin.load(bundle(folder))
    manifest["commit"] = "0" * 40
    resign(bundle(folder), manifest)
    report = check(folder)
    assert failed(report) == ["commit"]
    assert report.checks[2].problems == (f"commit {'0' * 40} is not in {ROOT}",)


def test_a_short_commit_is_not_a_commit(folder):
    manifest = origin.load(bundle(folder))
    manifest["commit"] = manifest["commit"][:12]
    resign(bundle(folder), manifest)
    assert check(folder).checks[2].problems == (
        f"{manifest['commit']!r} is not a full commit id",
    )


def test_a_version_other_than_the_commits_fails_the_commit(folder):
    manifest = origin.load(bundle(folder))
    manifest["scmle"] = "9.9.9"
    manifest["environment"]["scmle"] = "9.9.9"
    resign(bundle(folder), manifest)
    report = check(folder)
    assert failed(report) == ["commit", "environment"]
    assert report.checks[2].problems[0].endswith(", the bundle 9.9.9")


def test_a_pin_that_disagrees_fails_only_the_environment(folder):
    manifest = origin.load(bundle(folder))
    manifest["environment"]["scikit-learn"] = "0.0.1"
    resign(bundle(folder), manifest)
    report = check(folder)
    assert failed(report) == ["environment"]
    assert report.checks[3].problems == (
        f"scikit-learn is pinned at '0.0.1', recorded {manifest['sklearn']!r}",
        "the lock in provenance.json differs from the pins",
    )


def test_an_absent_field_is_not_present(folder):
    manifest = origin.load(bundle(folder))
    manifest["commit"] = ""
    resign(bundle(folder), manifest)
    report = check(folder)
    commit = report.checks[2]
    assert not commit.present
    assert commit.problems == ()
    assert not report.traced
    assert report.problems() == ["commit: absent"]


def test_an_unsigned_edit_breaks_the_integrity(folder):
    manifest = origin.load(bundle(folder))
    manifest["dataset"] = "0" * 16
    (bundle(folder) / layout.MANIFEST_FILE).write_text(json.dumps(manifest))
    report = check(folder)
    assert not report.signed
    assert len(report.integrity) == 1
    assert "signature is invalid" in report.integrity[0]
    assert not report.traced


def test_a_corrupted_member_breaks_the_integrity(folder):
    path = bundle(folder) / "pipeline.joblib"
    content = bytearray(path.read_bytes())
    content[-1] ^= 0xFF
    path.write_bytes(bytes(content))
    report = check(folder)
    assert report.integrity == (
        "pipeline.joblib does not match its manifest digest",
    )
    assert report.signed


def test_a_folder_without_a_manifest_is_refused(tmp_path):
    with pytest.raises(errors.ArtifactError, match="no readable"):
        origin.trace(tmp_path)


def test_a_missing_run_is_refused(folder):
    with pytest.raises(errors.InputError, match="does not exist"):
        check(folder, run=folder / "nowhere")


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("scikit_learn", "scikit-learn"),
        ("imblearn", "imbalanced-learn"),
        ("sklearn", "scikit-learn"),
        ("jaraco.classes", "jaraco-classes"),
        ("PyYAML", "pyyaml"),
    ],
)
def test_normalize(name, expected):
    assert origin.normalize(name) == expected
