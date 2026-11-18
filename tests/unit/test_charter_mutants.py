"""Tests added to kill surviving mutants in the charter and attestations.

Messages are compared whole: a mutant that wraps a message in extra
characters still contains the original as a substring.
"""

from functools import partial
import json

import pytest

from scmle.assurance.admissibility import attestations
from scmle.assurance.admissibility import provenance
from scmle.assurance.charter import contracts
from scmle.assurance.framing import costs
from tests.unit.test_admissibility_attestations import attest
from tests.unit.test_charter_contracts import make_charter


def message(call):
    with pytest.raises(ValueError) as caught:
        call()
    return str(caught.value)


HUMAN = (costs.Basis.HUMAN,) * 4


def cost_model():
    return costs.Tariff(0.0, 5.0, 1.0, 0.0, 0.05, 0.5, HUMAN, costs.Basis.HUMAN)


ESTIMAND = {
    "kind": "predictive",
    "population": "p",
    "target": "t",
    "moment": "m",
    "intercurrent": (contracts.Event("e", "s"),),
    "summary": "auc",
}


@pytest.mark.parametrize("field", ["event", "strategy"])
def test_intercurrent_event_fields_are_named(field):
    values = {"event": "e", "strategy": "s", field: " "}
    assert message(partial(contracts.Event, **values)) == (
        f"{field} must not be empty"
    )


@pytest.mark.parametrize("field", ["population", "target", "moment", "summary"])
def test_estimand_text_fields_are_named(field):
    values = {**ESTIMAND, field: ""}
    assert message(partial(contracts.AnalysisEstimand, **values)) == (
        f"{field} must not be empty"
    )


def test_estimand_events_must_be_a_non_empty_tuple():
    assert (
        message(
            partial(
                contracts.AnalysisEstimand,
                **{**ESTIMAND, "intercurrent": []},
            )
        )
        == "intercurrent must be a tuple"
    )
    assert message(
        partial(
            contracts.AnalysisEstimand,
            **{**ESTIMAND, "intercurrent": ()},
        )
    ) == (
        "intercurrent must not be empty: state a strategy "
        "for each event, or declare 'none' as not applicable"
    )


@pytest.mark.parametrize("field", ["check", "trigger"])
def test_decision_row_fields_are_named(field):
    values = {
        "check": "leakage.skill",
        "trigger": "t",
        "action": contracts.Edict.QUARANTINE_COLUMN,
        field: " ",
    }
    assert message(partial(contracts.Trigger, **values)) == (
        f"{field} must not be empty"
    )


def test_set_sizes_bounds_and_messages():
    contracts.SetSizes(confirmation=contracts.MAX_CONFIRMATION_FRACTION)
    bound = (
        "confirmation must be in (0, "
        f"{contracts.MAX_CONFIRMATION_FRACTION}] so exploration keeps a majority"
    )
    for bad in (0.0, contracts.MAX_CONFIRMATION_FRACTION + 0.01, -0.1):
        assert message(partial(contracts.SetSizes, confirmation=bad)) == bound
    assert message(partial(contracts.SetSizes, confirmation=float("nan"))) == (
        "confirmation must be a finite number"
    )
    assert message(partial(contracts.SetSizes, 0.25, -1)) == (
        "embargo must be non-negative"
    )


def test_choice_messages():
    assert message(partial(contracts.Choice, " ", ("a",), "a")) == (
        "name must not be empty"
    )
    assert message(partial(contracts.Choice, "n", ["a"], "a")) == (
        "options must be a tuple"
    )
    assert message(partial(contracts.Choice, "n", (), "a")) == (
        "options must be non-empty and distinct"
    )
    assert message(partial(contracts.Choice, "n", ("a", "a"), "a")) == (
        "options must be non-empty and distinct"
    )
    assert message(partial(contracts.Choice, "n", ("a", "b"), "z")) == (
        "default must be one of the options"
    )
    contracts.Choice("n", ("a", "b"), "b")


@pytest.mark.parametrize(
    ("field", "bad", "expected"),
    [
        ("question", " ", "question must not be empty"),
        ("unit", " ", "unit must not be empty"),
        (
            "population",
            " ",
            "population must not be empty",
        ),
        ("gates", [], "gates must be a tuple"),
        ("triggers", [], "triggers must be a tuple"),
        ("multiverse", [], "multiverse must be a tuple"),
        ("effect", 0, "effect must be greater than 0"),
        ("quota", 0, "quota must be at least 1"),
        ("power", 1.0, "power must be strictly between 0 and 1"),
        ("power", 0.0, "power must be strictly between 0 and 1"),
        ("seed", -1, "seed must be non-negative"),
    ],
)
def test_charter_fields_are_named_in_their_errors(field, bad, expected):
    assert message(partial(make_charter, **{field: bad})) == expected


def test_a_single_trial_and_a_seed_of_zero_are_allowed():
    charter = make_charter(quota=1, seed=0)
    assert charter.quota == 1 and charter.seed == 0


def test_the_template_is_pinned():
    template = contracts.template()
    assert json.loads(json.dumps(template)) == template
    assert template["estimand"]["kind"] == "predictive"
    assert template["sets"] == {
        "confirmation": 0.25,
        "embargo": 0,
    }
    assert (template["effect"], template["quota"]) == (0.2, 10)
    assert (template["power"], template["seed"]) == (0.8, 0)
    assert template["question"] == ""
    assert template["unit"] == ""
    assert template["population"] == ""
    assert template["gates"] == []
    assert template["triggers"] == []
    assert template["multiverse"] == []
    assert set(template) >= {
        "question",
        "estimand",
        "unit",
        "population",
        "gates",
        "triggers",
        "sets",
        "effect",
        "quota",
        "power",
        "budget",
        "multiverse",
        "seed",
    }
    assert set(template["estimand"]) == {
        "kind",
        "population",
        "target",
        "moment",
        "intercurrent",
        "summary",
    }
    assert template["estimand"]["population"] == ""
    assert template["estimand"]["intercurrent"] == []


def test_a_charter_with_and_without_a_cost_model_round_trips():
    plain = make_charter()
    assert plain.costs is None
    assert contracts.AnalysisCharter.from_dict(plain.to_dict()).costs is None
    priced = make_charter(costs=cost_model())
    again = contracts.AnalysisCharter.from_dict(priced.to_dict())
    assert again.costs == cost_model()
    assert again == priced


# attestations -----------------------------------------------------------


def test_an_attestation_needs_a_column_and_a_finite_lead():
    good = attest().attestation
    assert (
        message(partial(attestations.Affidavit, "", good, 1.0))
        == "column must be a non-empty string"
    )
    assert (
        message(partial(attestations.Affidavit, 3, good, 1.0))
        == "column must be a non-empty string"
    )
    assert (
        message(partial(attestations.Affidavit, "c", good, float("inf")))
        == "seconds must be a finite number"
    )


def test_the_provenance_record_of_an_attestation_is_exact():
    item = attest("copy")
    field = item.field()
    assert field.name == "copy"
    assert field.origin is provenance.Origin.DERIVED
    assert field.writer == attestations.WRITER
    assert field.stamp is provenance.Stamp.UNKNOWN
    assert field.overwritten is provenance.Overwritten.UNKNOWN
    assert field.definitions == ()
    assert field.lead == provenance.Lead(provenance.Cutoff.LAG, 3600.0)
    assert field.provenance is provenance.Provenance.ATTESTED
    assert field.availability is provenance.Availability.YES
    assert field.attestation == item.attestation


def test_a_missing_own_field_is_reported_by_name():
    record = attest().to_dict()
    del record["seconds"]
    assert message(partial(attestations.Affidavit.from_dict, record)) == (
        "missing field(s): ['seconds']"
    )


def test_apply_returns_valid_ones_and_a_reason_for_each_rejected_one():
    good = attest("a", scope="S")
    other = attest("b", scope="T")
    stale = attest("c", scope="S", expires="2026-02-01")
    applied = attestations.apply((good, other, stale), "S", "2026-06-01")
    assert applied.valid == {"a": good}
    assert applied.rejected == {
        "b": "scope 'T' is not this charter ('S')",
        "c": "valid 2026-01-01 to 2026-02-01, not on 2026-06-01",
    }
    assert message(partial(attestations.apply, (), "S", "not-a-date")) != ""


def test_load_errors_are_whole_messages(tmp_path):
    path = tmp_path / "a.json"
    path.write_text("[]")
    assert message(partial(attestations.load, path)) == (
        "attestations must be an object with schema_version 2"
    )
    path.write_text(json.dumps({"schema_version": 2, "attestations": {}}))
    assert message(partial(attestations.load, path)) == (
        "attestations must hold a list under 'attestations'"
    )
    one = attest("a").to_dict()
    path.write_text(
        json.dumps({"schema_version": 2, "attestations": [one, one]})
    )
    assert message(partial(attestations.load, path)) == (
        "at most one attestation per column"
    )
    assert message(
        partial(attestations.load, tmp_path / "none.json")
    ).startswith(f"cannot read attestations {tmp_path / 'none.json'}: ")


def test_every_text_slot_of_the_template_is_empty_and_named():
    estimand = contracts.template()["estimand"]
    for key in ("population", "target", "moment", "summary"):
        assert estimand[key] == "", key


def test_apply_names_the_date_argument_in_its_error():
    assert message(partial(attestations.apply, (), "S", "not-a-date")) == (
        "today must be an ISO date (YYYY-MM-DD)"
    )
