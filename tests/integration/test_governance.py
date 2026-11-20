"""Audit events and the data-access ledger."""

import dataclasses
import json
import pathlib

import numpy
import support

from scmle.contracts import tasks
from scmle.data.splitting import splitters
from scmle.governance import ledger


def read_events(run_dir):
    lines = (pathlib.Path(run_dir) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def run(frame, tmp_path, **kwargs):
    return support.make_project(
        frame, "y", budget=300, directory=tmp_path, **kwargs
    ).run()


def test_every_decision_that_shapes_the_run_is_an_audit_event(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path)
    events = read_events(result.directory)
    kinds = [e["kind"] for e in events]
    for expected in (
        "features.selected",
        "split.created",
        "selection.frozen",
        "test.opened",
    ):
        assert kinds.count(expected) == 1, expected
    assert (
        kinds.index("split.created")
        < kinds.index("selection.frozen")
        < kinds.index("test.opened")
    )
    split = next(e for e in events if e["kind"] == "split.created")
    assert split["data"]["fingerprint"] == result.evaluation["digests"]["split"]
    assert split["data"]["development_rows"] + split["data"][
        "test_rows"
    ] == len(binary_frame)
    features = next(e for e in events if e["kind"] == "features.selected")
    assert set(features["data"]) == {"numeric", "categorical", "dropped"}
    frozen = next(e for e in events if e["kind"] == "selection.frozen")
    assert frozen["data"]["trial_id"] == result.evaluation["selected"]["trial"]


def test_exclusions_are_audited_with_reasons(binary_frame, tmp_path):
    frame = binary_frame.assign(constant=1, rid=numpy.arange(len(binary_frame)))
    result = run(frame, tmp_path, identifiers=["rid"])
    features = next(
        e
        for e in read_events(result.directory)
        if e["kind"] == "features.selected"
    )
    assert features["data"]["dropped"]["constant"] == "constant"
    assert features["data"]["dropped"]["rid"] == "id column"


def test_data_access_ledger_is_written_and_verified(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    record = json.loads(
        (result.directory / "diagnostics" / "data_access.json").read_text()
    )
    assert all(record["verified"].values())
    partitions = record["partitions"]
    assert partitions["development_rows"] + partitions[
        "locked_test_rows"
    ] == len(binary_frame)
    operations = {o["operation"]: o for o in record["operations"]}
    assert operations["final_evaluation"]["opened"] == 1
    assert operations["cross_validation"]["fits_per_candidate"] == 5
    assert operations["threshold_optimization"]["used"] is False


def test_ledger_detects_violations(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    spec = support.make_project(binary_frame, "y").spec
    task = tasks.Task(result.plan.task)
    split = splitters.Splitter(spec, task).split(binary_frame)
    honest = ledger.compose(split, result.trials, 1, False)
    assert all(honest["verified"].values())
    reopened = ledger.compose(split, result.trials, 2, False)
    assert not reopened["verified"]["test_set_opened_exactly_once"]


def test_ledger_checks_what_trials_actually_received(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)
    spec = support.make_project(binary_frame, "y").spec
    task = tasks.Task(result.plan.task)
    split = splitters.Splitter(spec, task).split(binary_frame)
    observed = [t for t in result.trials if t.access]
    assert observed and all(
        len(t.access) == len(split.manifest.folds)
        for t in observed
        if t.status == "ok" and len(t.access) > 2
    )
    honest = ledger.compose(split, result.trials, 1, False)
    assert honest["observed"]["folds_checked"] > 0
    assert honest["verified"][
        "observed_fit_and_scoring_rows_match_the_manifest"
    ]
    forged = dataclasses.replace(
        observed[0],
        observed_access=[
            {
                **observed[0].access[0],
                "fit": {"rows": 1, "digest": "x"},
            },
            *observed[0].access[1:],
        ],
    )
    swapped = [forged if t is observed[0] else t for t in result.trials]
    caught = ledger.compose(split, swapped, 1, False)
    assert not caught["verified"][
        "observed_fit_and_scoring_rows_match_the_manifest"
    ]
    assert caught["observed"]["folds_mismatched"] == 1
