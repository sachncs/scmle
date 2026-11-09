"""Tests for the typed project specification."""

import pydantic
import pytest

from scmle.contracts import tasks


def make_spec(data_overrides=None, **kwargs):
    data = tasks.Dataset(uri="x.csv", target="y", **(data_overrides or {}))
    return tasks.Config(data=data, **kwargs)


def test_hash_is_stable_and_sensitive():
    assert make_spec().config_hash() == make_spec().config_hash()
    assert make_spec().config_hash() != make_spec(metric="r2").config_hash()


def test_json_roundtrip():
    spec = make_spec(task="classification", metric="roc_auc")
    assert tasks.Config.model_validate_json(spec.model_dump_json()) == spec


def test_group_strategy_requires_group_column():
    validation = tasks.Validation(strategy="group")
    with pytest.raises(pydantic.ValidationError, match="group_column"):
        make_spec(validation=validation)


def test_temporal_strategy_requires_time_column():
    validation = tasks.Validation(strategy="temporal")
    with pytest.raises(pydantic.ValidationError, match="time_column"):
        make_spec(validation=validation)


def test_stratified_regression_rejected():
    validation = tasks.Validation(strategy="stratified")
    with pytest.raises(pydantic.ValidationError):
        make_spec(task="regression", validation=validation)


@pytest.mark.parametrize(
    "overrides",
    [
        {"group_column": "y"},
        {"time_column": "y"},
        {"id_columns": ["y"]},
        {"excluded_columns": ["y"]},
    ],
)
def test_target_cannot_play_another_role(overrides):
    with pytest.raises(pydantic.ValidationError):
        make_spec(overrides)


def test_unknown_fields_rejected():
    payload = {"data": {"uri": "a", "target": "y"}, "bogus": 1}
    with pytest.raises(pydantic.ValidationError):
        tasks.Config.model_validate(payload)


def test_spec_is_immutable():
    with pytest.raises(pydantic.ValidationError):
        make_spec().metric = "r2"


def test_task_type_classification_flag():
    assert tasks.Task.BINARY.is_classification
    assert not tasks.Task.REGRESSION.is_classification
