"""A charter with unit_of_prediction known_entity_future does not make an
entity the unit: its rows sit on both sides, the final test rows come last."""

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

KEY = "known-entity-key"


def panel(entities=60, per=30, seed=0):
    rng = np.random.default_rng(seed)
    level = rng.normal(0, 1.5, entities)
    rows = []
    for entity in range(entities):
        x = rng.normal(size=per)
        y = 2 * x + level[entity] + rng.normal(0, 0.5, per)
        rows.append(
            pd.DataFrame(
                {"entity": f"e{entity}", "t": np.arange(per), "x": x, "y": y}
            )
        )
    return pd.concat(rows, ignore_index=True)


def charter_file(tmp_path, **overrides):
    path = tmp_path / "charter.json"
    values = {
        "min_effect": 0.5,
        "k_max": 3,
        "unit_of_prediction": c.KNOWN_ENTITY_FUTURE,
        **overrides,
    }
    sealing.write_charter(path, make_charter(**values), signing_key=KEY)
    return path


def run(tmp_path, charter, **kwargs):
    kwargs.setdefault("group_column", "entity")
    kwargs.setdefault("time_column", "t")
    return support.make_project(
        panel(),
        "y",
        time_budget=300,
        output_dir=tmp_path / "out",
        charter=charter,
        **kwargs,
    ).run()


def read(result, name):
    return json.loads((pathlib.Path(result.run_dir) / name).read_text())


def test_entities_are_on_both_sides_and_the_final_rows_come_last(tmp_path):
    result = run(tmp_path, charter_file(tmp_path))

    manifest = read(result, "split_manifest.json")
    frame = panel()
    explore = frame.iloc[manifest["exploration_ids"]]
    confirm = frame.iloc[manifest["confirmation_ids"]]
    assert len(confirm) > 0
    shared = set(explore.entity) & set(confirm.entity)
    assert len(shared) > 0.5 * confirm.entity.nunique()
    validation = read(result, "validation_manifest.json")
    final = frame.iloc[validation["test_ids"]]
    development = frame.iloc[validation["train_ids"]]
    assert final.t.min() >= development.t.quantile(0.5)
    assert set(final.entity) <= set(development.entity)
    events = (pathlib.Path(result.run_dir) / "events.jsonl").read_text()
    assert "unit.known_entity_future" in events


def test_without_the_field_whole_entities_are_the_units(tmp_path):
    path = charter_file(tmp_path, unit_of_prediction=None)
    result = run(tmp_path, path)
    manifest = read(result, "split_manifest.json")
    frame = panel()
    explore = frame.iloc[manifest["exploration_ids"]]
    confirm = frame.iloc[manifest["confirmation_ids"]]
    assert not (set(explore.entity) & set(confirm.entity)) or len(confirm) == 0


def test_a_known_entity_charter_needs_the_entity_and_time_columns(tmp_path):
    path = charter_file(tmp_path)
    with pytest.raises(errors.ConfigError, match="time_column"):
        run(tmp_path, path, time_column=None)
    with pytest.raises(errors.ConfigError, match="group_column"):
        run(tmp_path, path, group_column=None)


def test_a_known_entity_charter_refuses_an_entity_disjoint_validation(tmp_path):
    path = charter_file(tmp_path)
    with pytest.raises(errors.ConfigError, match="time-ordered"):
        run(tmp_path, path, validation="group")


def test_the_metric_interval_resamples_whole_entities(tmp_path):
    result = run(tmp_path, charter_file(tmp_path))

    uncertainty = read(result, "diagnostics/metric_uncertainty.json")
    assert uncertainty["resampling_unit"] == "groups"
    assert uncertainty["independent_units"] < uncertainty["evaluation_rows"]
