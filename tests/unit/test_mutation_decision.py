"""Exact-message and boundary tests for the charter decision block."""

import dataclasses
from fractions import Fraction
import math

import pytest

from scmle.assurance.charter import decision
from scmle.contracts import errors


def experiment(**changes):
    fields = {
        "unit": "user",
        "primary": "conversion",
        "assignment": 0.5,
        "delta": 0.05,
        "tolerance": 0.01,
        "limit": 1000,
        "promotion": Fraction(1, 20),
        "rollback": Fraction(1, 20),
    }
    fields.update(changes)
    return decision.Experiment(**fields)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"unit": " "}, "unit must not be empty"),
        ({"salt": " "}, "salt must not be empty"),
        ({"high": math.inf}, "high must be a finite number"),
        ({"srm": 1.5}, "srm must be strictly between 0 and 1"),
    ],
)
def test_experiment_names_the_field_it_rejects(changes, message):
    with pytest.raises(errors.InputError) as caught:
        experiment(**changes)

    assert str(caught.value) == message


def test_experiment_from_dict_names_a_bad_promotion_level():
    record = experiment().to_dict() | {"promotion": "one/two"}

    with pytest.raises(errors.InputError) as caught:
        decision.Experiment.from_dict(record)

    assert str(caught.value) == "promotion: invalid fraction 'one/two'"


def test_a_clip_of_exactly_one_is_allowed():
    assert decision.Gates(clip=1).clip == 1


def test_gates_name_a_bad_propensity():
    with pytest.raises(errors.InputError) as caught:
        decision.Gates(propensity=0.0)

    assert str(caught.value) == "propensity must be strictly between 0 and 1"


def test_guardrail_names_a_bad_margin():
    with pytest.raises(errors.InputError) as caught:
        decision.Guardrail(name="latency", margin=0.0)

    assert str(caught.value) == "margin must be greater than 0"


def test_policy_names_a_blank_unit():
    with pytest.raises(errors.InputError) as caught:
        decision.Policy(unit="  ")

    assert str(caught.value) == "unit must not be empty"


def test_policy_from_dict_keeps_the_declared_baselines():
    policy = decision.Policy(unit="EUR", baselines=("treat_none",))

    rebuilt = decision.Policy.from_dict(policy.to_dict())

    assert rebuilt.baselines == ("treat_none",)
    assert dataclasses.asdict(rebuilt) == dataclasses.asdict(policy)
