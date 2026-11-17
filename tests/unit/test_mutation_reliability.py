"""The fault-plan registry's refusals, word for word."""

import dataclasses

import pytest

from scmle.contracts import errors
from scmle.contracts import reliability
from tests.unit.test_reliability_contracts import make_evidence
from tests.unit.test_reliability_contracts import make_plan


def test_release_blocking_must_be_a_bool() -> None:
    with pytest.raises(
        errors.KindError, match=r"^release_blocking must be bool$"
    ):
        dataclasses.replace(make_plan(), release_blocking=1)


def test_an_unknown_schema_version_is_refused() -> None:
    with pytest.raises(
        errors.InputError,
        match=r"^unsupported fault-injection plan schema version$",
    ):
        reliability.Registry.from_dict({"schema_version": 2})


def test_only_a_plan_can_be_registered() -> None:
    with pytest.raises(errors.KindError, match=r"^plan must be Plan$"):
        reliability.Registry().register("not a plan")


def test_evidence_must_hold_evidence_records() -> None:
    registry = reliability.Registry.prerelease()
    evidence = (make_evidence("process_termination"), "not evidence")
    with pytest.raises(
        errors.KindError, match=r"^evidence must contain Evidence values$"
    ):
        registry.log(reliability.Contingency.PROCESS_DEATH, evidence)
