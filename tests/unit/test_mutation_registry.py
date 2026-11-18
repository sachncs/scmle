"""The fault-plan registry and evidence refusals, word for word."""

import dataclasses

import pytest

from scmle.contracts import errors
from scmle.contracts import reliability
from tests.unit.test_reliability_contracts import make_evidence

VERSION_MESSAGE = r"^unsupported fault-injection plan schema version$"


@pytest.mark.parametrize("version", [True, "1", 1.0])
def test_only_an_integer_schema_version_is_read(version) -> None:
    with pytest.raises(errors.InputError, match=VERSION_MESSAGE):
        reliability.Registry.from_dict(
            {"schema_version": version, "scenarios": []}
        )


def test_each_scenario_must_be_a_mapping() -> None:
    with pytest.raises(
        errors.KindError, match=r"^each scenario must be a mapping$"
    ):
        reliability.Registry.from_dict(
            {"schema_version": reliability.VERSION, "scenarios": ["plan"]}
        )


def test_serialized_text_must_hold_an_object() -> None:
    with pytest.raises(
        errors.KindError,
        match=r"^serialized plan must contain a JSON object$",
    ):
        reliability.Registry.from_json("[]")


def test_only_a_contingency_can_be_logged() -> None:
    with pytest.raises(
        errors.KindError, match=r"^scenario must be Contingency$"
    ):
        reliability.Registry().log("crash", ())


@pytest.mark.parametrize("name", ["execution", "observation"])
def test_blank_evidence_text_is_refused_by_name(name) -> None:
    with pytest.raises(errors.InputError, match=rf"^{name} must not be empty$"):
        dataclasses.replace(make_evidence("crash"), **{name: " "})
