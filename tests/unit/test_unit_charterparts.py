"""Behavioural tests of the charter parts and specification modules."""

import dataclasses
import json

import pytest

from scmle.assurance.charter import rules
from scmle.assurance.charter import uncertainty
from scmle.assurance.charter.contracts import parts
from scmle.assurance.charter.contracts import specification
from scmle.assurance.ledger import budget
from scmle.contracts import errors
from scmle.contracts import records


def make_estimand(**changes):
    values = {
        "kind": "predictive",
        "population": "active subscribers",
        "target": "churn in 30 days",
        "moment": "first of the month",
        "intercurrent": (parts.Event("closed", "treat as outcome"),),
        "summary": "ROC AUC",
    }
    values.update(changes)
    return parts.AnalysisEstimand(**values)


def make_trigger(check="leakage.skill", certificate="cert"):
    return parts.Trigger(
        check=check,
        trigger="lower bound above floor",
        action=parts.Edict.QUARANTINE_COLUMN,
        certificate=certificate,
    )


def make_charter(**changes):
    values = {
        "question": "Who churns?",
        "estimand": make_estimand(),
        "unit": "customer-month",
        "population": "all subscribers",
        "gates": ("leakage.skill",),
        "triggers": (make_trigger(),),
        "sets": parts.SetSizes(confirmation=0.25),
        "effect": 0.2,
        "quota": 10,
        "power": 0.8,
        "budget": budget.ErrorBudget.create(0.05, 0.1),
        "multiverse": (
            parts.Choice("imputation", ("median", "mice"), "median"),
        ),
        "seed": 7,
    }
    values.update(changes)
    return specification.AnalysisCharter(**values)


# ---- parts ------------------------------------------------------------------


def test_the_part_constants_hold_their_values():
    assert parts.FORMAT_VERSION == 11
    assert parts.OLDEST_READABLE_FORMAT == 1
    assert parts.ESTIMAND_KINDS == ("predictive", "causal")
    assert parts.MAX_CONFIRMATION_FRACTION == 0.5
    assert parts.UNITS_OF_PREDICTION == ("new_entity", "known_entity_future")
    assert parts.Edict("block_run") is parts.Edict.BLOCK_RUN
    assert len(parts.Edict) == 7


def test_texts_strips_each_value_and_refuses_a_blank():
    assert parts.texts([" a ", 3], "names") == ("a", "3")
    with pytest.raises(errors.InputError, match="names must not be empty"):
        parts.texts(["a", "  "], "names")


def test_an_event_round_trips_and_refuses_blanks():
    event = parts.Event("closed", "censor")
    assert event.to_dict() == {"event": "closed", "strategy": "censor"}
    assert parts.Event.from_dict(event.to_dict()) == event
    with pytest.raises(errors.InputError, match="event must not be empty"):
        parts.Event(" ", "censor")
    with pytest.raises(errors.InputError, match="strategy must not be empty"):
        parts.Event("closed", "")
    with pytest.raises(ValueError):
        parts.Event.from_dict({"event": "closed"})


def test_an_estimand_serialises_without_an_absent_causal_block():
    estimand = make_estimand()
    record = estimand.to_dict()
    assert record == {
        "kind": "predictive",
        "population": "active subscribers",
        "target": "churn in 30 days",
        "moment": "first of the month",
        "intercurrent": [{"event": "closed", "strategy": "treat as outcome"}],
        "summary": "ROC AUC",
    }
    assert parts.AnalysisEstimand.from_dict(record) == estimand


def test_an_estimand_reads_its_old_key_names():
    old = {
        "kind": "predictive",
        "population": "p",
        "target": "t",
        "prediction_moment": "m",
        "intercurrent_events": [{"event": "e", "strategy": "s"}],
        "summary_measure": "auc",
    }
    estimand = parts.AnalysisEstimand.from_dict(old)
    assert (estimand.moment, estimand.summary) == ("m", "auc")
    assert estimand.intercurrent == (parts.Event("e", "s"),)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"kind": "descriptive"}, "kind must be one of"),
        ({"population": " "}, "population must not be empty"),
        ({"target": ""}, "target must not be empty"),
        ({"moment": ""}, "moment must not be empty"),
        ({"summary": ""}, "summary must not be empty"),
        (
            {"intercurrent": [parts.Event("e", "s")]},
            "intercurrent must be a tuple",
        ),
        ({"intercurrent": ()}, "intercurrent must not be empty"),
        ({"causal": object()}, "a causal block needs kind='causal'"),
    ],
)
def test_an_invalid_estimand_is_refused(changes, message):
    with pytest.raises(errors.InputError, match=message):
        make_estimand(**changes)


def test_a_trigger_round_trips_with_its_action_value():
    trigger = make_trigger()
    assert trigger.to_dict() == {
        "check": "leakage.skill",
        "trigger": "lower bound above floor",
        "action": "quarantine_column",
        "certificate": "cert",
    }
    assert parts.Trigger.from_dict(trigger.to_dict()) == trigger
    assert parts.Trigger("c", "t", parts.Edict.REWEIGHT).certificate == ""


def test_a_trigger_reads_its_old_key_names():
    trigger = parts.Trigger.from_dict(
        {
            "check_id": "c",
            "trigger": "t",
            "action": "reweight",
            "certificate_id": "x",
        }
    )
    assert (trigger.check, trigger.certificate) == ("c", "x")


def test_an_invalid_trigger_is_refused():
    with pytest.raises(errors.InputError, match="check must not be empty"):
        parts.Trigger(" ", "t", parts.Edict.REWEIGHT)
    with pytest.raises(errors.InputError, match="trigger must not be empty"):
        parts.Trigger("c", " ", parts.Edict.REWEIGHT)
    with pytest.raises(errors.InputError, match="action must be an Edict"):
        parts.Trigger("c", "t", "reweight")
    with pytest.raises(ValueError):
        parts.Trigger.from_dict(
            {
                "check": "c",
                "trigger": "t",
                "action": "explode",
                "certificate": "",
            }
        )


def test_set_sizes_accept_the_half_and_refuse_beyond():
    assert parts.SetSizes(0.5).to_dict() == {"confirmation": 0.5, "embargo": 0}
    assert parts.SetSizes.from_dict(
        {"confirmation_fraction": 0.1, "embargo_rows": 3}
    ) == (parts.SetSizes(0.1, 3))
    for bad in (0.0, 0.5000001, -0.1):
        with pytest.raises(
            errors.InputError, match=r"confirmation must be in \(0, 0.5\]"
        ):
            parts.SetSizes(bad)
    with pytest.raises(errors.InputError, match="embargo must be non-negative"):
        parts.SetSizes(0.2, -1)
    with pytest.raises(errors.InputError, match="finite number"):
        parts.SetSizes(float("nan"))


def test_a_choice_round_trips_and_validates_its_options():
    choice = parts.Choice("imputation", ("median", "mice"), "mice")
    assert choice.to_dict() == {
        "name": "imputation",
        "options": ["median", "mice"],
        "default": "mice",
    }
    assert parts.Choice.from_dict(choice.to_dict()) == choice
    with pytest.raises(errors.InputError, match="name must not be empty"):
        parts.Choice("", ("a",), "a")
    with pytest.raises(errors.InputError, match="options must be a tuple"):
        parts.Choice("n", ["a"], "a")
    with pytest.raises(errors.InputError, match="non-empty and distinct"):
        parts.Choice("n", (), "a")
    with pytest.raises(errors.InputError, match="non-empty and distinct"):
        parts.Choice("n", ("a", "a"), "a")
    with pytest.raises(errors.InputError, match="default must be one of"):
        parts.Choice("n", ("a", "b"), "c")


def test_positive_accepts_only_finite_numbers_above_zero():
    assert parts.positive(2, "x") == 2.0
    assert parts.positive(1e-300, "x") == 1e-300
    with pytest.raises(errors.InputError, match="x must be greater than 0"):
        parts.positive(0, "x")
    with pytest.raises(errors.InputError, match="x must be greater than 0"):
        parts.positive(-1.5, "x")
    with pytest.raises(errors.InputError, match="x must be a finite number"):
        parts.positive(True, "x")


# ---- specification ----------------------------------------------------------


def test_a_charter_round_trips_through_json():
    charter = make_charter()
    record = json.loads(json.dumps(charter.to_dict()))
    assert record[records.KEY] == 2
    assert specification.AnalysisCharter.from_dict(record) == charter
    assert set(record) == {
        records.KEY,
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


def test_the_hash_covers_the_version_one_layout():
    charter = make_charter()
    anchor = charter.anchor()
    assert records.KEY not in anchor
    assert "gating_checks" in anchor and "gates" not in anchor
    assert anchor["decision_map"][0]["check_id"] == "leakage.skill"
    assert len(charter.charter_hash) == 64
    assert charter.charter_hash != make_charter(seed=8).charter_hash


def test_a_version_one_record_loads_and_keeps_its_hash():
    charter = make_charter()
    old = charter.anchor()
    rebuilt = specification.AnalysisCharter.from_dict(old)
    assert rebuilt == charter
    assert rebuilt.charter_hash == charter.charter_hash


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"question": " "}, "question must not be empty"),
        ({"unit": ""}, "unit must not be empty"),
        ({"population": ""}, "population must not be empty"),
        ({"gates": ["leakage.skill"]}, "gates must be a tuple"),
        ({"triggers": []}, "triggers must be a tuple"),
        ({"multiverse": []}, "multiverse must be a tuple"),
        ({"effect": 0.0}, "effect must be greater than 0"),
        ({"quota": 0}, "quota must be at least 1"),
        ({"power": 1.0}, "power must be strictly between 0 and 1"),
        ({"power": 0.0}, "power must be strictly between 0 and 1"),
        ({"seed": -1}, "seed must be non-negative"),
        ({"purpose": "everyone"}, "purpose must be one of"),
        ({"decision": object()}, "a decision layer needs costs"),
    ],
)
def test_an_invalid_charter_is_refused(changes, message):
    with pytest.raises(errors.InputError, match=message):
        make_charter(**changes)


def test_a_repeated_multiverse_name_is_refused():
    choice = parts.Choice("imputation", ("a",), "a")
    with pytest.raises(errors.InputError, match="duplicate name in multiverse"):
        make_charter(multiverse=(choice, choice))


def test_cover_requires_one_certified_row_per_gate():
    with pytest.raises(
        errors.InputError, match=r"duplicate decision row for leakage\.skill"
    ):
        make_charter(triggers=(make_trigger(), make_trigger()))
    with pytest.raises(
        errors.InputError, match="gating check other has no decision row"
    ):
        make_charter(gates=("other",))
    with pytest.raises(
        errors.InputError,
        match=r"gating check leakage\.skill needs a certificate",
    ):
        make_charter(triggers=(make_trigger(certificate="  "),))
    ungated = make_charter(gates=(), triggers=(make_trigger(certificate=""),))
    assert ungated.gates == ()


def test_lint_rejects_unknown_rows_and_gates():
    charter = make_charter()
    charter.lint(["leakage.skill", "other"])
    with pytest.raises(
        errors.InputError, match=r"unknown check leakage\.skill in the charter"
    ):
        charter.lint(["other"])


def test_the_format_version_follows_the_optional_blocks():
    assert make_charter().format_version == 2
    assert make_charter(rules=rules.Procedure()).format_version == 3
    assert make_charter(purpose=parts.NEW_ENTITY).format_version == 6
    assert (
        make_charter(uncertainty=uncertainty.Uncertainty()).format_version == 10
    )
    confirmatory = dataclasses.replace(
        uncertainty.Uncertainty(), confirmatory=True
    )
    assert make_charter(uncertainty=confirmatory).format_version == 11
    both = make_charter(
        rules=rules.Procedure(), purpose=parts.KNOWN_ENTITY_FUTURE
    )
    assert both.format_version == 6


def test_optional_blocks_serialise_only_when_set():
    charter = make_charter(
        rules=rules.Procedure(),
        purpose=parts.NEW_ENTITY,
        uncertainty=uncertainty.Uncertainty(),
    )
    record = charter.to_dict()
    assert record["purpose"] == "new_entity"
    assert "rules" in record and "uncertainty" in record
    assert "costs" not in record and "decision" not in record
    assert specification.AnalysisCharter.from_dict(record) == charter


def test_revive_skips_an_absent_block():
    assert specification.revive(None, int) is None
    assert specification.revive("3", int) == 3


def test_the_template_is_a_current_skeleton_that_fails_validation():
    record = specification.template()
    assert record[records.KEY] == 2
    assert record["sets"] == {"confirmation": 0.25, "embargo": 0}
    assert (
        record["effect"],
        record["quota"],
        record["power"],
        record["seed"],
    ) == (
        0.2,
        10,
        0.8,
        0,
    )
    assert record["estimand"]["kind"] == "predictive"
    assert record["budget"] == budget.ErrorBudget.create(0.05, 0.1).to_dict()
    with pytest.raises(errors.ConfigError, match="invalid analysis charter"):
        specification.parse(record)


def test_parse_wraps_missing_keys_and_type_errors():
    record = make_charter().to_dict()
    del record["seed"]
    with pytest.raises(errors.ConfigError, match="invalid analysis charter"):
        specification.parse(record)
    assert specification.parse(make_charter().to_dict()) == make_charter()


def test_deserialize_reads_an_object_and_refuses_anything_else(tmp_path):
    good = tmp_path / "good.json"
    good.write_text('{"a": 1}', encoding="utf-8")
    assert specification.deserialize(good) == {"a": 1}
    listed = tmp_path / "list.json"
    listed.write_text("[1]", encoding="utf-8")
    with pytest.raises(errors.ConfigError, match="is not a JSON object"):
        specification.deserialize(listed)
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(errors.ConfigError, match="cannot read charter"):
        specification.deserialize(broken)
    with pytest.raises(errors.ConfigError, match="cannot read charter"):
        specification.deserialize(tmp_path / "absent.json")


def test_peek_reads_a_bare_or_a_wrapped_charter(tmp_path):
    charter = make_charter()
    bare = tmp_path / "bare.json"
    bare.write_text(json.dumps(charter.to_dict()), encoding="utf-8")
    wrapped = tmp_path / "wrapped.json"
    wrapped.write_text(
        json.dumps({"charter": charter.to_dict(), "signature": "x"}),
        encoding="utf-8",
    )
    assert specification.peek(bare) == charter
    assert specification.peek(wrapped) == charter
