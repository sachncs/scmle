"""Analysis charter: validation, hashing, round trip and decision map."""

import dataclasses
import json

import pytest

from scmle.assurance.charter import contracts
from scmle.assurance.ledger import budget
from scmle.contracts import errors


def make_charter(**overrides):
    values = {
        "question": "Which customers churn within 30 days?",
        "estimand": contracts.Estimand(
            kind="predictive",
            population="active subscribers in 2024",
            target="churned within 30 days of the prediction moment",
            moment="first day of each billing month",
            intercurrent=(
                contracts.Event("account closed", "treat as outcome"),
            ),
            summary="ROC AUC",
        ),
        "unit": "customer-month",
        "population": "stated marginals: all active subscribers",
        "gates": ("leakage.skill",),
        "triggers": (
            contracts.Trigger(
                check="leakage.skill",
                trigger="lower bound above floor at the budgeted level",
                action=contracts.Edict.QUARANTINE_COLUMN,
                certificate="cert.leakage.skill",
            ),
        ),
        "sets": contracts.Sizes(confirmation=0.25, embargo=0),
        "effect": 0.2,
        "quota": 10,
        "power": 0.8,
        "budget": budget.Alpha.create(0.05, 0.1),
        "multiverse": (
            contracts.Choice("imputation", ("median", "mice"), "median"),
        ),
        "seed": 7,
    }
    values.update(overrides)
    return contracts.Charter(**values)


def test_valid_charter_round_trips_through_json():
    charter = make_charter()
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.Charter.from_dict(record) == charter


def test_hash_is_stable_and_sensitive_to_a_single_change():
    first = make_charter()
    assert first.digest == make_charter().digest
    assert len(first.digest) == 64
    changed = make_charter(question="Which customers churn within 31 days?")
    assert changed.digest != first.digest


def test_hash_ignores_dict_key_order():
    record = make_charter().to_dict()
    reordered = dict(reversed(list(record.items())))
    rebuilt = contracts.Charter.from_dict(reordered)
    assert rebuilt.digest == make_charter().digest


@pytest.mark.parametrize("field", ["population", "target", "moment", "summary"])
def test_estimand_rejects_a_blank_required_attribute(field):
    values = dataclasses.asdict(make_charter().estimand)
    values["intercurrent"] = make_charter().estimand.intercurrent
    values[field] = "  "
    with pytest.raises(ValueError, match=field):
        contracts.Estimand(**values)


def test_estimand_requires_an_intercurrent_event_strategy():
    base = make_charter().estimand
    with pytest.raises(ValueError, match="intercurrent"):
        dataclasses.replace(base, intercurrent=())


def test_estimand_kind_is_closed():
    base = make_charter().estimand
    with pytest.raises(ValueError, match="kind"):
        dataclasses.replace(base, kind="descriptive")


def test_gating_check_without_a_decision_row_is_rejected():
    with pytest.raises(ValueError, match="no decision row"):
        make_charter(gates=("leakage.skill", "missingness.mcar"))


def test_gating_row_needs_a_certificate_id():
    row = contracts.Trigger(
        check="leakage.skill",
        trigger="t",
        action=contracts.Edict.BLOCK_RUN,
        certificate="",
    )
    with pytest.raises(ValueError, match="certificate"):
        make_charter(triggers=(row,))


def test_duplicate_decision_rows_for_one_check_are_rejected():
    row = make_charter().triggers[0]
    with pytest.raises(ValueError, match="duplicate"):
        make_charter(triggers=(row, row))


def test_decision_row_for_unknown_check_is_rejected_by_the_registry_lint():
    charter = make_charter()
    with pytest.raises(ValueError, match="unknown check"):
        charter.lint(known_checks={"missingness.mcar"})
    charter.lint(known_checks={"leakage.skill"})


def test_decision_action_is_a_closed_set():
    assert {a.value for a in contracts.Edict} == {
        "drop_column",
        "quarantine_column",
        "reweight",
        "switch_split",
        "block_run",
        "downgrade_tier",
        "refuse_claim",
    }


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("effect", 0.0),
        ("effect", float("nan")),
        ("quota", 0),
        ("power", 1.0),
        ("power", 0.0),
        ("seed", -1),
    ],
)
def test_numeric_fields_are_range_checked(field, value):
    with pytest.raises(ValueError, match=field):
        make_charter(**{field: value})


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, float("nan"), 0.6, True])
def test_confirmation_fraction_must_leave_room_for_exploration(fraction):
    with pytest.raises(ValueError, match="confirmation"):
        contracts.Sizes(confirmation=fraction, embargo=0)


def test_from_dict_rejects_unknown_and_missing_fields():
    record = make_charter().to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        contracts.Charter.from_dict({**record, "extra": 1})
    del record["seed"]
    with pytest.raises(ValueError, match="missing"):
        contracts.Charter.from_dict(record)


def test_multiverse_default_must_be_one_of_the_options():
    with pytest.raises(ValueError, match="default"):
        contracts.Choice("imputation", ("median", "mice"), "knn")


def test_load_rejects_malformed_json_as_a_config_error(tmp_path):
    path = tmp_path / "charter.json"
    path.write_text("{not json")
    with pytest.raises(errors.ConfigError):
        contracts.peek(path)
