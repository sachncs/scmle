"""Claim C6 across platforms: merging records and the verdict."""

import copy
import json
import pathlib
from typing import Any

import pytest

from benchmarks.evidence import platforms
from benchmarks.evidence import reproducibility
from scmle.contracts import errors


def task(auc: float = 0.9, differing: list[str] | None = None) -> dict:
    return {
        "selected": ["m", "m"],
        "metrics": {"roc_auc": auc, "accuracy": 0.8},
        "gap": {"absolute": 0.0, "relative": 0.0},
        "members": 3,
        "differing": differing or [],
        "hashes": {},
        "untraced": [],
        "trials": [1, 1],
        "seconds": [1.0, 1.0],
        "threads": [],
    }


def record(name: str, **tasks: dict[str, Any]) -> dict[str, Any]:
    return {
        "machine": {"python": "3.14.0", "platform": name},
        "seed": reproducibility.SEED,
        "rows": reproducibility.ROWS,
        "tasks": tasks or {"binary": task(), "regression": task()},
        "seconds": 1.0,
    }


def document(**records: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "claim": "C6",
        "tolerance": {
            "name": reproducibility.TOLERANCE,
            "value": reproducibility.tolerance(),
        },
        "code": reproducibility.fingerprint(),
        "platforms": records,
    }


def save(folder: pathlib.Path, name: str, found: dict) -> pathlib.Path:
    path = folder / f"{name}.json"
    path.write_text(json.dumps(found), encoding="utf-8")
    return path


def test_merge_unites_platforms_and_tolerates_identical_copies():
    shared = record("mac")
    merged = platforms.merge(
        [
            document(linux=record("linux"), mac=shared),
            document(windows=record("windows"), mac=copy.deepcopy(shared)),
        ]
    )
    assert list(merged["platforms"]) == ["linux", "mac", "windows"]
    assert merged["code"] == reproducibility.fingerprint()


@pytest.mark.parametrize(
    "change",
    [
        lambda d: d.update(code="other"),
        lambda d: d.update(tolerance={"name": "x", "value": 1.0}),
        lambda d: d["platforms"]["mac"]["tasks"].update(binary=task(0.5)),
    ],
)
def test_merge_refuses_inconsistent_documents(change):
    second = document(mac=record("mac"))
    change(second)
    with pytest.raises(errors.InputError):
        platforms.merge([document(mac=record("mac")), second])


def test_merge_refuses_nothing():
    with pytest.raises(errors.InputError):
        platforms.merge([])


def test_pairs_cover_every_pair_and_shared_task():
    gaps = platforms.pairs(
        {
            "a": record("a"),
            "b": record("b", binary=task(0.9 * (1 + 1e-3))),
            "c": record("c"),
        }
    )
    assert [(g["left"], g["right"], g["task"]) for g in gaps] == [
        ("a", "b", "binary"),
        ("a", "c", "binary"),
        ("a", "c", "regression"),
        ("b", "c", "binary"),
    ]
    assert gaps[0]["task"] == "binary"
    assert gaps[0]["relative"] == pytest.approx(1e-3 / (1 + 1e-3))
    assert gaps[1]["relative"] == 0.0


def test_agreeing_platforms_pass(tmp_path):
    paths = [
        save(tmp_path, "linux", document(linux=record("linux"))),
        save(tmp_path, "windows", document(windows=record("windows"))),
    ]
    found = platforms.measure(paths)
    assert platforms.problems(found) == []
    value, verdict = platforms.summary(found)
    assert verdict == "pass"
    assert "across 2 platforms" in value


def test_metrics_beyond_tolerance_fail(tmp_path):
    limit = reproducibility.tolerance()
    moved = record("windows", binary=task(0.9 * (1 + 10 * limit)))
    found = platforms.measure(
        [
            save(tmp_path, "linux", document(linux=record("linux"))),
            save(tmp_path, "windows", document(windows=moved)),
        ]
    )
    value, verdict = platforms.summary(found)
    assert verdict == "fail"
    assert "binary differs between linux and windows" in value


def test_one_platform_is_never_a_pass(tmp_path):
    found = platforms.measure(
        [save(tmp_path, "linux", document(linux=record("linux")))]
    )
    assert "none can pass below 2" in platforms.problems(found)[0]
    assert platforms.summary(found)[1] == "not measured"


def test_one_failing_platform_fails_even_alone(tmp_path):
    broken = record("linux", binary=task(differing=["pipeline.joblib"]))
    path = save(tmp_path, "linux", document(linux=broken))
    found = platforms.measure([path])
    assert platforms.summary(found)[1] == "fail"


def test_stale_records_are_not_measured(tmp_path):
    found = platforms.measure(
        [
            save(tmp_path, "a", document(a=record("a"))),
            save(tmp_path, "b", document(b=record("b"))),
        ]
    )
    found["merged"]["code"] = "other"
    value, verdict = platforms.summary(found)
    assert verdict == "not measured"
    assert "other code" in value
    assert platforms.summary({"pairs": []}) == (
        "no measurement",
        "not measured",
    )
    assert platforms.problems({"pairs": []})


def test_main_gate_and_merged_output(tmp_path, monkeypatch):
    monkeypatch.setattr(platforms, "PATH", tmp_path / "platforms.json")
    linux = save(tmp_path, "linux", document(linux=record("linux")))
    windows = save(tmp_path, "windows", document(windows=record("windows")))
    into = tmp_path / "merged.json"
    code = platforms.main(
        ["--write", str(linux), str(windows), "--into", str(into), "--gate"]
    )
    assert code == 0
    assert len(json.loads(into.read_text())["platforms"]) == 2
    assert platforms.main(["--check"]) == 0
    assert platforms.main(["--write", str(linux), "--gate"]) == 1
    assert platforms.main(["--check"]) == 1


def test_load_refuses_a_missing_file(tmp_path):
    with pytest.raises(errors.InputError):
        platforms.load([tmp_path / "absent.json"])


def test_merge_reads_every_record_below_a_folder(tmp_path, monkeypatch):
    monkeypatch.setattr(platforms, "PATH", tmp_path / "platforms.json")
    downloads = tmp_path / "records"
    for name in ("ubuntu", "windows", "macos"):
        folder = downloads / f"reproducibility-{name}"
        folder.mkdir(parents=True)
        (folder / "reproducibility.json").write_text(
            json.dumps(document(**{name: record(name)})), encoding="utf-8"
        )
    (downloads / "notes.json").write_text("{}", encoding="utf-8")
    into = tmp_path / "merged.json"
    code = platforms.main(
        ["--merge", str(downloads), "--into", str(into), "--gate"]
    )
    assert code == 0
    stored = json.loads(platforms.PATH.read_text(encoding="utf-8"))
    assert list(stored["merged"]["platforms"]) == [
        "macos",
        "ubuntu",
        "windows",
    ]
    assert len(stored["sources"]) == 3
    assert json.loads(into.read_text())["platforms"].keys() == {
        "macos",
        "ubuntu",
        "windows",
    }


def test_merge_refuses_a_folder_without_records(tmp_path):
    with pytest.raises(errors.InputError):
        platforms.collect(tmp_path)
    with pytest.raises(errors.InputError):
        platforms.collect(tmp_path / "absent")


def test_merge_takes_a_folder_not_files(tmp_path):
    with pytest.raises(SystemExit):
        platforms.main(["--merge", str(tmp_path), str(tmp_path / "a.json")])
