import pytest
from pydantic import ValidationError

from mml.contracts.tasks import DataSpec, ProjectSpec, ValidationSpec


def spec(**kw):
    data = kw.pop("data", {})
    return ProjectSpec(data=DataSpec(uri="x.csv", target="y", **data), **kw)


def test_defaults_and_hash_is_stable():
    a, b = spec(), spec()
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != spec(metric="r2").config_hash()


def test_roundtrip_json():
    a = spec(task="classification", metric="roc_auc")
    assert ProjectSpec.model_validate_json(a.model_dump_json()) == a


def test_group_requires_group_column():
    with pytest.raises(ValidationError, match="group_column"):
        spec(validation=ValidationSpec(strategy="group"))


def test_temporal_requires_time_column():
    with pytest.raises(ValidationError, match="time_column"):
        spec(validation=ValidationSpec(strategy="temporal"))


def test_stratified_regression_rejected():
    with pytest.raises(ValidationError):
        spec(task="regression", validation=ValidationSpec(strategy="stratified"))


def test_target_cannot_be_group_or_id():
    with pytest.raises(ValidationError):
        spec(data={"group_column": "y"})
    with pytest.raises(ValidationError):
        spec(data={"id_columns": ["y"]})


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        ProjectSpec.model_validate({"data": {"uri": "a", "target": "y"}, "bogus": 1})
