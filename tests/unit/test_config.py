"""Tests for project configuration files."""

import json

import pytest

from mml.contracts import errors
from mml.contracts import tasks
from mml.core import config


def make_spec():
    return tasks.ProjectSpec(
        data=tasks.DataSpec(uri="d.csv", target="y"),
        metric="roc_auc",
        validation=tasks.ValidationSpec(folds=3),
    )


def test_a_specification_round_trips_through_a_file(tmp_path):
    spec = make_spec()
    config.save_spec(spec, tmp_path / "spec.json")

    loaded = config.load_spec(tmp_path / "spec.json")

    assert loaded == spec and loaded.config_hash() == spec.config_hash()


def test_an_invalid_specification_is_a_config_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"data": {"uri": "x"}}))
    with pytest.raises(errors.ConfigError, match="invalid project"):
        config.load_spec(path)


def test_sections_build_typed_specs_and_reject_unknown_names(tmp_path):
    path = tmp_path / "sections.json"
    path.write_text(json.dumps({"search": {"top_models": 1}}))
    assert config.load_sections(path)["search"].top_models == 1
    assert config.load_sections(None) == {}
    path.write_text(json.dumps({"mystery": {}}))
    with pytest.raises(errors.ConfigError, match="unknown config sections"):
        config.load_sections(path)
