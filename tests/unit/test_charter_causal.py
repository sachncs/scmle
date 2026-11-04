"""The causal block of the charter: validation, hashing and versioning."""

import dataclasses
import json

import pytest

from scmle.contracts import errors
from scmle.data.charter import causal
from scmle.data.charter import contracts as c
from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_charter_rules import FROZEN_HASH

KEY = "causal-key"


def spec(**changes) -> causal.CausalSpec:
    values = {
        "treatment": "coupon",
        "outcome": "spend",
        "estimand": "ATE",
        "assumptions": (
            "unconfoundedness_given:age,region",
            "positivity",
            "no_interference",
        ),
        "strategies": ("backdoor_aipw",),
        "negative_controls": (causal.NegativeControl("outcome", "prior"),),
        "strata": ("region",),
        "sensitivity": causal.SensitivitySpec("strongest_observed", 1.0),
    }
    values.update(changes)
    return causal.CausalSpec(**values)


def causal_charter(**changes):
    base = make_charter()
    estimand = dataclasses.replace(
        base.estimand, kind="causal", causal=spec(**changes)
    )
    return make_charter(estimand=estimand)


def test_a_charter_without_the_block_keeps_its_hash():
    charter = make_charter()
    assert charter.charter_hash == FROZEN_HASH
    assert "causal" not in charter.estimand.to_dict()
    assert charter.format_version == 2


def test_the_block_round_trips_and_needs_format_four():
    charter = causal_charter()
    record = json.loads(json.dumps(charter.to_dict()))
    assert c.AnalysisCharter.from_dict(record) == charter
    assert charter.format_version == 4


@pytest.mark.parametrize(
    "changes",
    [
        {"treatment": "other"},
        {"estimand": "ATT"},
        {"cutoff": 3.0},
        {"trim": 0.05},
        {"assumptions": ("positivity",)},
        {"strategies": ("did", "backdoor_aipw")},
        {"strata": ("period",)},
        {"sensitivity": causal.SensitivitySpec("fixed", 0.1)},
        {"negative_controls": ()},
    ],
)
def test_the_hash_changes_with_any_causal_field(changes):
    assert (
        causal_charter(**changes).charter_hash != causal_charter().charter_hash
    )


def test_a_causal_block_needs_a_causal_kind():
    with pytest.raises(ValueError, match="kind='causal'"):
        dataclasses.replace(make_charter().estimand, causal=spec())


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"estimand": "ATX"}, "estimand"),
        ({"assumptions": ("magic",)}, "unknown assumption"),
        ({"assumptions": ("unconfoundedness_given:",)}, "needs covariates"),
        ({"assumptions": ("positivity", "positivity")}, "duplicate"),
        ({"strategies": ("did", "did")}, "duplicate"),
        ({"trim": 0.5}, "trim"),
        ({"min_share": 0.0}, "min_share"),
        ({"assumptions": ["positivity"]}, "tuple"),
    ],
)
def test_invalid_blocks_are_refused(changes, message):
    with pytest.raises(ValueError, match=message):
        spec(**changes)


def test_covariates_are_read_from_the_attestation():
    assert causal.covariates_of(["positivity"]) is None
    assert causal.covariates_of(["unconfoundedness_given:a, b"]) == ("a", "b")
    assert causal.covariates_of(["unconfoundedness_given:none"]) == ()


def test_m_max_is_part_of_the_hash_and_must_not_be_negative():
    one = causal_charter(sensitivity=causal.SensitivitySpec(m_max=1.0))
    two = causal_charter(sensitivity=causal.SensitivitySpec(m_max=2.0))
    assert one.charter_hash != two.charter_hash
    with pytest.raises(ValueError, match="m_max"):
        causal.SensitivitySpec(m_max=-1.0)


def test_a_fixed_benchmark_is_a_partial_r2():
    with pytest.raises(ValueError, match="below 1"):
        causal.SensitivitySpec("fixed", 1.5)
    with pytest.raises(ValueError, match="benchmark"):
        causal.SensitivitySpec("odd", 0.5)


def test_a_causal_block_in_an_older_format_is_refused(tmp_path):
    path = tmp_path / "charter.json"
    wrapper = sealing.envelope(causal_charter(), KEY)
    assert wrapper["format"] == 4
    wrapper["format"] = 3
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="causal"):
        sealing.read_verified(path, signing_key=KEY)
