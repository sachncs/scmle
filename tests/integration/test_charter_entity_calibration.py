"""A charter with unit_of_prediction new_entity calibrates per entity."""

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
import support

from scmle.contracts import errors
from scmle.data.charter import contracts as c
from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter

KEY = "integration-key"


def panel(entities=120, per=6, seed=0):
    rng = np.random.default_rng(seed)
    level = rng.normal(0, 1.5, entities)
    rows = []
    for entity in range(entities):
        x = rng.normal(size=per)
        y = 2 * x + level[entity] + rng.normal(0, 0.5, per)
        rows.append(pd.DataFrame({"entity": f"e{entity}", "x": x, "y": y}))
    return pd.concat(rows, ignore_index=True)


def charter_file(tmp_path, **overrides):
    path = tmp_path / "charter.json"
    values = {"min_effect": 0.5, "k_max": 3, **overrides}
    sealing.write_charter(path, make_charter(**values), signing_key=KEY)
    return path


def run(tmp_path, charter, **kwargs):
    return support.make_project(
        panel(),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter,
        **kwargs,
    ).run()


def events(run_dir):
    lines = (pathlib.Path(run_dir) / "events.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines]


def test_a_new_entity_charter_calibrates_on_one_score_per_entity(tmp_path):
    path = charter_file(tmp_path, unit_of_prediction=c.NEW_ENTITY)
    result = run(tmp_path, path, group_column="entity")
    found = [
        e
        for e in events(result.run_dir)
        if e["kind"] == "conformal.entity_wise"
    ]
    assert found
    entities = found[0]["data"]["entities"]
    assert 20 <= entities <= 120
    calibrator = result.bundle.conformal
    assert calibrator is not None and len(calibrator.scores) == entities
    assert result.bundle.cv_plus is None


def test_without_the_field_calibration_stays_row_wise(tmp_path):
    path = charter_file(tmp_path)
    result = run(tmp_path, path, group_column="entity")
    assert all(
        e["kind"] != "conformal.entity_wise" for e in events(result.run_dir)
    )
    assert len(result.bundle.conformal.scores) > 120


def test_a_new_entity_charter_without_a_group_column_is_refused(tmp_path):
    path = charter_file(tmp_path, unit_of_prediction=c.NEW_ENTITY)
    with pytest.raises(errors.ConfigError, match="group_column"):
        run(tmp_path, path)
