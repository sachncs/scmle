"""Tests added to kill surviving mutants in the charter and attestations.

Messages are compared whole: a mutant that wraps a message in extra
characters still contains the original as a substring.
"""

from functools import partial
import json

import pytest

from scmle.data.admissibility import attestations
from scmle.data.admissibility import provenance
from scmle.data.charter import contracts
from scmle.data.framing import costs
from tests.unit.test_admissibility_attestations import attest
from tests.unit.test_charter_contracts import make_charter


def message(call):
    with pytest.raises(ValueError) as caught:
        call()
    return str(caught.value)


HUMAN = (costs.CostSource.HUMAN,) * 4


def cost_model():
    return costs.DecisionCostModel(
        0.0, 5.0, 1.0, 0.0, 0.05, 0.5, HUMAN, costs.CostSource.HUMAN
    )


ESTIMAND = {
    "kind": "predictive",
    "population": "p",
    "target": "t",
    "prediction_moment": "m",
    "intercurrent_events": (contracts.IntercurrentEvent("e", "s"),),
    "summary_measure": "auc",
}


@pytest.mark.parametrize("field", ["event", "strategy"])
def test_intercurrent_event_fields_are_named(field):
    values = {"event": "e", "strategy": "s", field: " "}
    assert message(partial(contracts.IntercurrentEvent, **values)) == (
        f"{field} must not be empty"
    )


@pytest.mark.parametrize(
    "field", ["population", "target", "prediction_moment", "summary_measure"]
)
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
                **{**ESTIMAND, "intercurrent_events": []},
            )
        )
        == "intercurrent_events must be a tuple"
    )
    assert message(
        partial(
            contracts.AnalysisEstimand,
            **{**ESTIMAND, "intercurrent_events": ()},
        )
    ) == (
        "intercurrent_events must not be empty: state a strategy "
        "for each event, or declare 'none' as not applicable"
    )


@pytest.mark.parametrize("field", ["check_id", "trigger"])
def test_decision_row_fields_are_named(field):
    values = {
        "check_id": "leakage.skill",
        "trigger": "t",
        "action": contracts.DecisionAction.QUARANTINE_COLUMN,
        field: " ",
    }
    assert message(partial(contracts.DecisionRow, **values)) == (
        f"{field} must not be empty"
    )


def test_set_sizes_bounds_and_messages():
    contracts.SetSizes(
        confirmation_fraction=contracts.MAX_CONFIRMATION_FRACTION
    )
    bound = (
        "confirmation_fraction must be in (0, "
        f"{contracts.MAX_CONFIRMATION_FRACTION}] so exploration keeps a majority"
    )
    for bad in (0.0, contracts.MAX_CONFIRMATION_FRACTION + 0.01, -0.1):
        assert (
            message(partial(contracts.SetSizes, confirmation_fraction=bad))
            == bound
        )
    assert message(
        partial(contracts.SetSizes, confirmation_fraction=float("nan"))
    ) == ("confirmation_fraction must be a finite number")
    assert message(partial(contracts.SetSizes, 0.25, -1)) == (
        "embargo_rows must be non-negative"
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
            "deployment_population",
            " ",
            "deployment_population must not be empty",
        ),
        ("gating_checks", [], "gating_checks must be a tuple"),
        ("decision_map", [], "decision_map must be a tuple"),
        ("multiverse", [], "multiverse must be a tuple"),
        ("min_effect", 0, "min_effect must be greater than 0"),
        ("k_max", 0, "k_max must be at least 1"),
        ("power", 1.0, "power must be strictly between 0 and 1"),
        ("power", 0.0, "power must be strictly between 0 and 1"),
        ("seed", -1, "seed must be non-negative"),
    ],
)
def test_charter_fields_are_named_in_their_errors(field, bad, expected):
    assert message(partial(make_charter, **{field: bad})) == expected


def test_a_single_trial_and_a_seed_of_zero_are_allowed():
    charter = make_charter(k_max=1, seed=0)
    assert charter.k_max == 1 and charter.seed == 0


def test_the_template_is_pinned():
    template = contracts.template()
    assert json.loads(json.dumps(template)) == template
    assert template["estimand"]["kind"] == "predictive"
    assert template["sets"] == {
        "confirmation_fraction": 0.25,
        "embargo_rows": 0,
    }
    assert (template["min_effect"], template["k_max"]) == (0.2, 10)
    assert (template["power"], template["seed"]) == (0.8, 0)
    assert template["question"] == ""
    assert template["unit"] == ""
    assert template["deployment_population"] == ""
    assert template["gating_checks"] == []
    assert template["decision_map"] == []
    assert template["multiverse"] == []
    assert set(template) >= {
        "question",
        "estimand",
        "unit",
        "deployment_population",
        "gating_checks",
        "decision_map",
        "sets",
        "min_effect",
        "k_max",
        "power",
        "error_budget",
        "multiverse",
        "seed",
    }
    assert set(template["estimand"]) == {
        "kind",
        "population",
        "target",
        "prediction_moment",
        "intercurrent_events",
        "summary_measure",
    }
    assert template["estimand"]["population"] == ""
    assert template["estimand"]["intercurrent_events"] == []


def test_a_charter_with_and_without_a_cost_model_round_trips():
    plain = make_charter()
    assert plain.cost_model is None
    assert (
        contracts.AnalysisCharter.from_dict(plain.to_dict()).cost_model is None
    )
    priced = make_charter(cost_model=cost_model())
    again = contracts.AnalysisCharter.from_dict(priced.to_dict())
    assert again.cost_model == cost_model()
    assert again == priced


# attestations -----------------------------------------------------------


def test_an_attestation_needs_a_column_and_a_finite_lead():
    good = attest().attestation
    assert (
        message(partial(attestations.LeakageAttestation, "", good, 1.0))
        == "column must be a non-empty string"
    )
    assert (
        message(partial(attestations.LeakageAttestation, 3, good, 1.0))
        == "column must be a non-empty string"
    )
    assert (
        message(
            partial(attestations.LeakageAttestation, "c", good, float("inf"))
        )
        == "lead_seconds must be a finite number"
    )


def test_the_provenance_record_of_an_attestation_is_exact():
    item = attest("copy")
    field = item.field()
    assert field.name == "copy"
    assert field.origin is provenance.Origin.DERIVED
    assert field.writer == attestations.WRITER
    assert field.written_at is provenance.WrittenAt.UNKNOWN
    assert field.overwritten is provenance.Overwritten.UNKNOWN
    assert field.definition_history == ()
    assert field.as_of == provenance.AsOf(provenance.AsOfKind.LAG, 3600.0)
    assert field.as_of_source is provenance.AsOfSource.ATTESTED
    assert field.available_at_serving is provenance.Availability.YES
    assert field.attestation == item.attestation


def test_a_missing_own_field_is_reported_by_name():
    record = attest().to_dict()
    del record["lead_seconds"]
    assert message(
        partial(attestations.LeakageAttestation.from_dict, record)
    ) == ("missing field(s): ['lead_seconds']")


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
        "attestations must be an object with format_version 1"
    )
    path.write_text(json.dumps({"format_version": 1, "attestations": {}}))
    assert message(partial(attestations.load, path)) == (
        "attestations must hold a list under 'attestations'"
    )
    one = attest("a").to_dict()
    path.write_text(
        json.dumps({"format_version": 1, "attestations": [one, one]})
    )
    assert message(partial(attestations.load, path)) == (
        "at most one attestation per column"
    )
    assert message(
        partial(attestations.load, tmp_path / "none.json")
    ).startswith(f"cannot read attestations {tmp_path / 'none.json'}: ")


def test_every_text_slot_of_the_template_is_empty_and_named():
    estimand = contracts.template()["estimand"]
    for key in ("population", "target", "prediction_moment", "summary_measure"):
        assert estimand[key] == "", key


def test_apply_names_the_date_argument_in_its_error():
    assert message(partial(attestations.apply, (), "S", "not-a-date")) == (
        "today must be an ISO date (YYYY-MM-DD)"
    )
