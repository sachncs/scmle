"""A charter run blocks a column declared known only after the outcome."""

import json
import pathlib

import support

from scmle.assurance.charter import sealing
from tests.unit.test_charter_contracts import make_charter

KEY = "integration-key"


def completed(run_dir):
    text = (pathlib.Path(run_dir) / "events.jsonl").read_text("utf-8")
    rows = [json.loads(line) for line in text.splitlines()]
    return next(
        r["data"] for r in rows if r["kind"] == "admissibility.completed"
    )


def test_a_declared_outcome_column_is_quarantined_in_a_charter_run(
    binary_frame, tmp_path
):
    path = tmp_path / "charter.json"
    sealing.inscribe(path, make_charter(effect=0.5, quota=3), signing_key=KEY)
    result = support.make_project(
        binary_frame,
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=path,
        known={"b": "outcome"},
    ).run()

    done = completed(result.run_dir)
    assert "b" in done["suspects"]
    assert "b" in done["quarantined"]
    plan = json.loads(
        (pathlib.Path(result.run_dir) / "dataset_profile.json").read_text()
    )
    assert "b" in plan["plan"]["dropped"]
