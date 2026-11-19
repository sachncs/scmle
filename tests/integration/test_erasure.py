"""Forget, retire and delete on real runs, registries and prediction logs."""

import json
import pathlib
import shutil
import sqlite3
import types

import numpy
import pandas
import pytest

from scmle import cli
from scmle.api import erasure
from scmle.api.erasure import steps
from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.data.readers import tabular
from scmle.deploy import release_state
from scmle.governance import registry
from scmle.serving import server
from tests import support

KEY = "erasure-test-key"
ROWS = 200


def table(rows=ROWS, seed=0):
    rng = numpy.random.default_rng(seed)
    frame = pandas.DataFrame(
        {
            "id": [f"P-{n:04d}" for n in range(rows)],
            "a": rng.normal(size=rows).round(6),
            "b": rng.normal(size=rows).round(6),
        }
    )
    frame["y"] = (frame.a + rng.normal(size=rows) > 0).astype(int)
    return frame


@pytest.fixture(scope="module")
def template(tmp_path_factory):
    """One real project: a trained run, a second run on other data."""
    base = tmp_path_factory.mktemp("template")
    table().to_csv(base / "people.csv", index=False)
    table(seed=7).to_csv(base / "other.csv", index=False)
    for name in ("people.csv", "other.csv"):
        support.make_project(
            base / name,
            "y",
            id_columns=["id"],
            output_dir=base / "project",
            time_budget=30,
        ).run()
    return base


@pytest.fixture
def world(template, tmp_path):
    """A private copy of the template project for one test."""
    shutil.copytree(template, tmp_path, dirs_exist_ok=True)
    project = tmp_path / "project"
    runs = sorted(p for p in project.iterdir() if p.name.startswith("run_"))
    people, other = sorted(
        runs,
        key=lambda run: json.loads(
            (run / "dataset_manifest.json").read_text("utf-8")
        )["uri"].endswith("other.csv"),
    )
    return types.SimpleNamespace(
        base=tmp_path, project=project, people=people, other=other
    )


def tracked(project):
    with sqlite3.connect(project / "experiments.sqlite") as connection:
        return sorted(
            row[0] for row in connection.execute("SELECT run_id FROM runs")
        )


def test_forgetting_a_dataset_erases_its_run_and_keeps_custody(world):
    base, project, people, other = (
        world.base,
        world.project,
        world.people,
        world.other,
    )
    before = sorted(p.name for p in other.rglob("*"))

    receipt = erasure.Forgetting(
        [project],
        erasure.Dataset.of(base / "people.csv"),
        base / "receipts",
        "ticket 118",
    ).execute(KEY)

    kept = {p.name for p in people.iterdir()}
    assert kept == {"model"} | (erasure.CUSTODY & kept)
    assert {"provenance.json", "dataset_manifest.json", "events.jsonl"} <= kept
    assert sorted(p.name for p in (people / "model").iterdir()) == [
        "model_manifest.json",
        "retired.json",
    ]
    assert (project / "custody" / f"{people.name}.jsonl").is_file()
    assert tracked(project) == [other.name]
    assert sorted(p.name for p in other.rglob("*")) == before
    assert receipt.verify(KEY)
    assert receipt.kind == "forget"
    assert receipt.tally()["retire"] == 1
    assert receipt.tally()["unlist"] == 1
    assert all(step["result"] for step in receipt.steps)
    with pytest.raises(erasure.RetiredError, match="ticket 118"):
        bundle.Bundle.load(people / "model", trusted=True)


def test_forgetting_finds_registry_copies_and_prediction_logs(world):
    base, project, people = world.base, world.project, world.people
    store = registry.ModelRegistryStore(base / "registry")
    release = store.register("churn", people / "model", trusted=True)
    loaded = bundle.Bundle.load(people / "model", trusted=True)
    log = base / "scores" / "scored.csv"
    log.parent.mkdir()
    rows = table(20, seed=3).drop(columns="y")
    pandas.DataFrame({"prediction": loaded.predict(rows)}).to_csv(log)
    # the manifest `scmle batch` writes beside its output names the bundle
    # by the digest the serving adapter reports
    log.with_name("scored.csv.manifest.json").write_text(
        json.dumps({"artifact_digest": server.Batcher(loaded).artifact_digest}),
        encoding="utf-8",
    )
    unrelated = base / "scores" / "kept.csv"
    unrelated.write_text("x\n1\n", encoding="utf-8")

    receipt = erasure.Forgetting(
        [project, base / "registry", base / "scores"],
        erasure.Dataset.of(base / "people.csv"),
        base / "receipts",
        "ticket 119",
    ).execute(KEY)

    assert not log.exists()
    assert not log.with_name("scored.csv.manifest.json").exists()
    assert unrelated.is_file()
    machine = release_state.ReleaseRegistry(store)
    assert machine.lookup("churn", release.version) is (
        release_state.ReleaseState.RETIRED
    )
    audit = (base / "registry" / "audit.jsonl").read_text("utf-8")
    assert receipt.identifier in audit
    with pytest.raises(erasure.RetiredError):
        store.load("churn", release.version)
    assert receipt.tally()["retire"] == 2


def test_forgetting_a_subject_purges_the_rows_and_retrains(world):
    base, project, other = world.base, world.project, world.other
    source = base / "people.csv"
    original = tabular.read(source)

    receipt = erasure.Forgetting(
        [project],
        erasure.Subject(source, "id", ["P-0003", "P-0150"]),
        base / "receipts",
        "erasure request 2026-118",
        retrain=True,
    ).execute(KEY)

    purged = tabular.read(source)
    assert len(purged) == ROWS - 2
    assert not purged.id.isin(["P-0003", "P-0150"]).any()
    assert purged.equals(
        original[~original.id.isin(["P-0003", "P-0150"])].reset_index(drop=True)
    )
    retrained = pathlib.Path(
        next(s for s in receipt.steps if s["action"] == "retrain")["result"][
            "run"
        ]
    )
    manifest = json.loads(
        (retrained / "dataset_manifest.json").read_text("utf-8")
    )
    assert manifest["rows"] == ROWS - 2
    assert bundle.Bundle.load(retrained / "model", trusted=True).predict(
        purged.head(3)
    ).shape == (3,)
    assert sorted(tracked(project)) == sorted([other.name, retrained.name])
    stored = (base / "receipts" / f"{receipt.identifier}.json").read_text(
        "utf-8"
    )
    assert "P-0003" not in stored
    assert steps.fingerprint("P-0003") in stored
    assert receipt.steps[0]["action"] == "purge"
    assert receipt.steps[0]["result"]["rows"] == 2


def test_an_interrupted_erasure_resumes_and_signs_once(world, monkeypatch):
    base, people = world.base, world.people
    job = erasure.Deletion([people], base / "receipts", "retention")
    calls = []
    perform = steps.Removal.perform

    def flaky(self, order):
        calls.append(self.target)
        if len(calls) == 3:
            raise OSError("disk went away")
        return perform(self, order)

    monkeypatch.setattr(steps.Removal, "perform", flaky)
    with pytest.raises(OSError, match="disk went away"):
        job.execute(KEY)
    folder = base / "receipts" / job.identifier
    plan = (folder / "plan.json").read_text("utf-8")
    journaled = (folder / "journal.jsonl").read_text("utf-8").splitlines()
    # the model's retirement mark, then the two removals that finished
    assert [json.loads(r)["step"] for r in journaled] == [0, 1, 2]
    assert json.loads(journaled[0])["result"]["marked"] is True
    assert not (base / "receipts" / f"{job.identifier}.json").exists()
    monkeypatch.setattr(steps.Removal, "perform", perform)

    receipt = job.execute(KEY)

    assert (folder / "plan.json").read_text("utf-8") == plan
    rows = (folder / "journal.jsonl").read_text("utf-8").splitlines()
    assert len(rows) == len(receipt.steps)
    assert [json.loads(r)["step"] for r in rows] == list(
        range(len(receipt.steps))
    )
    assert receipt.verify(KEY)
    assert job.execute(KEY) == receipt


def test_deleting_a_project_keeps_custody_and_foreign_files(world):
    base, project, people, other = (
        world.base,
        world.project,
        world.people,
        world.other,
    )
    note = project / "notes.txt"
    note.write_text("mine\n", encoding="utf-8")

    receipt = erasure.Deletion([project], base / "receipts", "closed").execute(
        KEY
    )

    assert not (project / "experiments.sqlite").exists()
    assert note.is_file()
    assert sorted(p.name for p in (project / "custody").iterdir()) == sorted(
        [f"{people.name}.jsonl", f"{other.name}.jsonl"]
    )
    for run in (people, other):
        assert not (run / "leaderboard.parquet").exists()
        assert (run / "provenance.json").is_file()
    assert receipt.tally()["keep"] >= 1


def test_retiring_a_registered_version_refuses_serving(world):
    base, other = world.base, world.other
    store = registry.ModelRegistryStore(base / "registry")
    release = store.register("churn", other / "model", trusted=True)
    job = erasure.Retirement(
        base / "registry", base / "receipts", "drift", "churn", release.version
    )

    receipt = job.execute(KEY)

    copy = base / "registry" / release.path
    assert (copy / "pipeline.joblib").is_file()
    mark = json.loads((copy / "retired.json").read_text("utf-8"))
    assert mark["reason"] == "drift"
    assert mark["operation"] == receipt.identifier
    assert mark["content"] == release.content_id
    code = cli.main(
        ["predict", str(copy), str(base / "other.csv"), "--trusted"]
    )
    assert code == cli.EXIT_BLOCKED
    assert job.execute(KEY) == receipt


def test_the_cli_forgets_verifies_and_refuses_a_tampered_receipt(world, capsys):
    base, project, people = world.base, world.project, world.people
    key = base / "key.txt"
    key.write_text(KEY + "\n", encoding="utf-8")
    digest = json.loads((people / "dataset_manifest.json").read_text("utf-8"))[
        "digests"
    ]["data"]
    arguments = [
        "forget",
        "dataset",
        str(project),
        "--digest",
        digest,
        "--reason",
        "ticket 120",
        "--receipts",
        str(base / "receipts"),
        "--key-file",
        str(key),
    ]

    assert cli.main([*arguments, "--dry-run"]) == cli.EXIT_OK
    assert (people / "leaderboard.parquet").is_file()
    assert "would remove" in capsys.readouterr().out
    assert cli.main(arguments) == cli.EXIT_OK
    written = next((base / "receipts").glob("forget-*.json"))
    assert f"receipt: {written}" in capsys.readouterr().out
    verify = ["forget", "verify", str(written), "--key-file", str(key)]
    assert cli.main(verify) == cli.EXIT_OK
    stored = json.loads(written.read_text("utf-8"))
    stored["reason"] = "nothing happened"
    written.write_text(json.dumps(stored), encoding="utf-8")
    assert cli.main(verify) == cli.EXIT_FAILED


def test_refusals_are_typed_and_change_nothing(world, monkeypatch):
    base, project, people = world.base, world.project, world.people
    monkeypatch.delenv("SCMLE_SIGNING_KEY", raising=False)
    monkeypatch.delenv("SCMLE_SIGNING_KEY_FILE", raising=False)
    job = erasure.Deletion([people], base / "receipts", "retention")
    with pytest.raises(errors.ConfigError, match="signed"):
        job.execute()
    assert (people / "leaderboard.parquet").is_file()
    with pytest.raises(errors.InputError, match="no row"):
        erasure.Forgetting(
            [project],
            erasure.Subject(base / "people.csv", "id", ["nobody"]),
            base / "receipts",
            "x",
        ).execute(KEY)
    with pytest.raises(errors.InputError, match="inside"):
        erasure.Deletion(
            [people], people / "diagnostics" / "receipts", "x"
        ).execute(KEY)
    with pytest.raises(errors.InputError, match="neither"):
        erasure.Deletion([base], base / "receipts", "x").execute(KEY)
    with pytest.raises(errors.InputError, match="only a subject"):
        erasure.Forgetting(
            [project], erasure.Dataset(["abc"]), base, "x", retrain=True
        )
    with pytest.raises(errors.InputError, match="both"):
        erasure.Retirement(base, base, "x", model="churn")
    with pytest.raises(errors.InputError, match="reason"):
        erasure.Deletion([people], base, " ")
    assert (people / "leaderboard.parquet").is_file()
    assert not (base / "receipts").exists()
