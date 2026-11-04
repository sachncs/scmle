"""A canary value planted in every row must not reach any run artifact.

The canary column holds a distinct value per row. After a full charter run,
no file in the run directory may contain the canary of a row of `C` (sealed
until the one confirmation) or of the final test set `F`. Exploration rows
may appear: they are the rows exploration is allowed to read.

Canary values are random, not an arithmetic progression: a quantile that
interpolates between two exploration rows would otherwise land exactly on a
sealed row and look like a leak.

Limit: the check is a text search of the files for the decimal spelling of
each value (repr and a few common roundings). A value written in an
unrelated encoding, or only a function of several rows, would not be found.
"""

import json
import pathlib

import numpy as np
import pandas as pd
import support

from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter

KEY = "canary-key"
CANARY = "canary"
LOW = 1_000_000.0
HIGH = 9_000_000.0


def canary_frame():
    rng = np.random.default_rng(0)
    rows = 500
    frame = pd.DataFrame(
        {
            "a": rng.normal(size=rows),
            "b": rng.normal(size=rows),
            CANARY: rng.uniform(LOW, HIGH, size=rows),
        }
    )
    frame["y"] = (frame.a + 0.5 * frame.b + rng.normal(size=rows) > 0).astype(
        int
    )
    return frame


def spellings(value):
    return {repr(float(value)), f"{value:.6f}", f"{value:.4f}", f"{value:.2f}"}


def contents(folder):
    return {
        path.name: path.read_text(encoding="utf-8", errors="ignore")
        for path in folder.rglob("*")
        if path.is_file() and path.suffix in {".json", ".jsonl", ".md", ".txt"}
    }


def test_no_artifact_holds_a_canary_of_c_or_the_final_test_set(tmp_path):
    frame = canary_frame()
    path = tmp_path / "charter.json"
    sealing.write_charter(
        path, make_charter(min_effect=0.5, k_max=3), signing_key=KEY
    )
    result = support.make_project(
        frame,
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=path,
    ).run()
    folder = pathlib.Path(result.run_dir)
    manifest = json.loads((folder / "split_manifest.json").read_text())
    values = frame[CANARY].to_numpy()
    sealed_ids = manifest["confirmation_ids"] + manifest["final_test_ids"]
    assert manifest["confirmation_ids"] and manifest["final_test_ids"]
    files = contents(folder)
    assert "custody_log.jsonl" in files and len(files) >= 10
    for name, text in files.items():
        for row in sealed_ids:
            for spelling in spellings(values[row]):
                assert spelling not in text, f"{name} holds a canary of {row}"


def test_the_search_finds_a_value_that_is_written(tmp_path):
    (tmp_path / "x.json").write_text(json.dumps({"v": 1234567.891234}))
    (tmp_path / "ignored.bin").write_text("1234567.891234")
    files = contents(tmp_path)
    assert list(files) == ["x.json"]
    assert any(s in files["x.json"] for s in spellings(1234567.891234))
