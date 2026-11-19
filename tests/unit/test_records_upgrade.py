"""`scmle records upgrade`: stored records move to the current schema.

The old-version folders are assembled from the version-1 fixtures in
`tests/fixtures/records` (a signed charter, a sealed split manifest, a
dataset manifest, a feasibility record and a cost model) and from the whole
version-1 charter run in `trail_sample` (with its hash-chained ledger and
custody log).
"""

import json
import pathlib
import shutil

import pytest

from scmle import cli
from scmle.assurance.charter import sealing
from scmle.cli import maintenance
from scmle.contracts import records

FIXTURES = pathlib.Path(__file__).parents[1] / "fixtures"
VERSION_ONE = FIXTURES / "records"
RUN = FIXTURES / "trail_sample" / "trail" / "retained" / "run"
COPIES = {
    "v1/custody/charter.json": "analysis_charter.json",
    "v1/custody/manifest.json": "split_manifest.json",
    "v1/custody/feasibility.json": "feasibility.json",
    "contracts/dataset_manifest.json": "dataset_manifest.json",
    "governed/tariff.json": "framing/decision_cost_model.json",
}
KEYS = {"records": "r1-fixture-key", "run": "phase18-demonstration-analyst-key"}


def assemble(kind, root):
    """Copies an old-version folder into `root` and writes its key file."""
    folder = root / kind
    if kind == "run":
        shutil.copytree(RUN, folder)
    else:
        for source, target in COPIES.items():
            (folder / target).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(VERSION_ONE / source, folder / target)
    key = root / f"{kind}.key"
    key.write_text(KEYS[kind], encoding="utf-8")
    return folder, key


def snapshot(folder):
    """The bytes of every file under `folder`, by relative path."""
    return {
        str(p.relative_to(folder)): p.read_bytes()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def upgrade(folder, key, *extra):
    """Runs the command and returns its exit status."""
    return cli.main(
        ["records", "upgrade", str(folder), "--key-file", str(key), *extra]
    )


@pytest.mark.parametrize("kind", sorted(KEYS))
def test_an_old_folder_upgrades_and_still_verifies(kind, tmp_path):
    folder, key = assemble(kind, tmp_path)
    before = snapshot(folder)
    assert upgrade(folder, key) == cli.EXIT_OK
    rewritten = maintenance.inventory(folder)
    assert rewritten
    for path in rewritten:
        backup = path.with_name(path.name + maintenance.BACKUP)
        assert backup.read_bytes() == before[str(path.relative_to(folder))]
    manifest = json.loads((folder / "split_manifest.json").read_text("utf-8"))
    assert manifest[records.KEY] == records.ORIGIN + 1
    assert "digests" in manifest
    charter = sealing.read(
        folder / "analysis_charter.json",
        key.read_text("utf-8"),
        require_signature=True,
    )
    assert charter.signed
    assert maintenance.verify(folder, key.read_text("utf-8")) == []
    again = snapshot(folder)
    assert upgrade(folder, key) == cli.EXIT_OK
    assert snapshot(folder) == again


@pytest.mark.parametrize("kind", sorted(KEYS))
def test_dry_run_changes_nothing(kind, tmp_path, capsys):
    folder, key = assemble(kind, tmp_path)
    before = snapshot(folder)
    assert upgrade(folder, key, "--dry-run") == cli.EXIT_OK
    assert snapshot(folder) == before
    assert "would upgrade split_manifest.json" in capsys.readouterr().out


def test_a_tampered_record_fails_and_changes_nothing(tmp_path):
    folder, key = assemble("records", tmp_path)
    path = folder / "split_manifest.json"
    stored = json.loads(path.read_text("utf-8"))
    stored["confirmation_ids"] = [3]
    path.write_text(json.dumps(stored), encoding="utf-8")
    before = snapshot(folder)
    assert upgrade(folder, key) == cli.EXIT_FAILED
    assert snapshot(folder) == before


def test_a_wrong_key_fails_and_changes_nothing(tmp_path):
    folder, key = assemble("run", tmp_path)
    key.write_text("another-key", encoding="utf-8")
    before = snapshot(folder)
    assert upgrade(folder, key) == cli.EXIT_FAILED
    assert snapshot(folder) == before


def test_a_failure_after_the_rewrite_restores_every_file(tmp_path, monkeypatch):
    folder, key = assemble("run", tmp_path)
    before = snapshot(folder)
    verify = maintenance.verify
    calls = []

    def flaky(directory, signing_key):
        calls.append(snapshot(directory))
        found = verify(directory, signing_key)
        return found if len(calls) == 1 else [*found, "tampered on disk"]

    monkeypatch.setattr(maintenance, "verify", flaky)
    assert upgrade(folder, key) == cli.EXIT_FAILED
    assert calls[1] != before
    assert snapshot(folder) == before


def test_a_missing_folder_fails(tmp_path):
    key = tmp_path / "key"
    key.write_text("k", encoding="utf-8")
    assert upgrade(tmp_path / "absent", key) == cli.EXIT_FAILED
