"""A charter run writes the descriptive framing link report."""

import json
import pathlib

from tests.integration.test_charter_run import event
from tests.integration.test_charter_run import run
from tests.integration.test_charter_run import write_charter


def test_a_charter_run_writes_a_descriptive_framing_link(
    binary_frame, tmp_path
):
    result = run(binary_frame, tmp_path, write_charter(tmp_path))
    folder = pathlib.Path(result.run_dir)

    link = json.loads((folder / "framing_link.json").read_text())
    assert link["gating"] is False
    assert set(link["fields"]) == {"target", "prediction_moment", "unit"}
    for field in link["fields"].values():
        assert field["status"] in {"resolved", "unresolved"}
    done = event(result.run_dir, "framing.completed")["data"]
    assert set(done) == set(link["fields"])


def test_a_run_without_a_charter_has_no_framing_link(binary_frame, tmp_path):
    result = run(binary_frame, tmp_path)

    assert not (pathlib.Path(result.run_dir) / "framing_link.json").exists()
