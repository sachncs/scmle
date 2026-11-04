"""Analysis charter: validation, hashing, round trip and decision map."""

import dataclasses
import json

import pytest

from scmle.contracts import errors
from scmle.data.charter import contracts as c
from scmle.data.ledger import budget as budget_module


def make_charter(**overrides):
    values = {
        "question": "Which customers churn within 30 days?",
        "estimand": c.AnalysisEstimand(
            kind="predictive",
            population="active subscribers in 2024",
            target="churned within 30 days of the prediction moment",
            prediction_moment="first day of each billing month",
            intercurrent_events=(
                c.IntercurrentEvent("account closed", "treat as outcome"),
            ),
            summary_measure="ROC AUC",
        ),
        "unit": "customer-month",
        "deployment_population": "stated marginals: all active subscribers",
        "gating_checks": ("leakage.skill",),
        "decision_map": (
            c.DecisionRow(
                check_id="leakage.skill",
                trigger="lower bound above floor at the budgeted level",
                action=c.DecisionAction.QUARANTINE_COLUMN,
                certificate_id="cert.leakage.skill",
            ),
        ),
        "sets": c.SetSizes(confirmation_fraction=0.25, embargo_rows=0),
        "min_effect": 0.2,
        "k_max": 10,
        "power": 0.8,
        "error_budget": budget_module.ErrorBudget.create(0.05, 0.1),
        "multiverse": (c.Choice("imputation", ("median", "mice"), "median"),),
        "seed": 7,
    }
    values.update(overrides)
    return c.AnalysisCharter(**values)


def test_valid_charter_round_trips_through_json():
    charter = make_charter()
    record = json.loads(json.dumps(charter.to_dict()))
    assert c.AnalysisCharter.from_dict(record) == charter


def test_hash_is_stable_and_sensitive_to_a_single_change():
    first = make_charter()
    assert first.charter_hash == make_charter().charter_hash
    assert len(first.charter_hash) == 64
    changed = make_charter(question="Which customers churn within 31 days?")
    assert changed.charter_hash != first.charter_hash


def test_hash_ignores_dict_key_order():
    record = make_charter().to_dict()
    reordered = dict(reversed(list(record.items())))
    rebuilt = c.AnalysisCharter.from_dict(reordered)
    assert rebuilt.charter_hash == make_charter().charter_hash


@pytest.mark.parametrize(
    "field", ["population", "target", "prediction_moment", "summary_measure"]
)
def test_estimand_rejects_a_blank_required_attribute(field):
    values = dataclasses.asdict(make_charter().estimand)
    values["intercurrent_events"] = make_charter().estimand.intercurrent_events
    values[field] = "  "
    with pytest.raises(ValueError, match=field):
        c.AnalysisEstimand(**values)


def test_estimand_requires_an_intercurrent_event_strategy():
    base = make_charter().estimand
    with pytest.raises(ValueError, match="intercurrent_events"):
        dataclasses.replace(base, intercurrent_events=())


def test_estimand_kind_is_closed():
    base = make_charter().estimand
    with pytest.raises(ValueError, match="kind"):
        dataclasses.replace(base, kind="descriptive")


def test_gating_check_without_a_decision_row_is_rejected():
    with pytest.raises(ValueError, match="no decision row"):
        make_charter(gating_checks=("leakage.skill", "missingness.mcar"))


def test_gating_row_needs_a_certificate_id():
    row = c.DecisionRow(
        check_id="leakage.skill",
        trigger="t",
        action=c.DecisionAction.BLOCK_RUN,
        certificate_id="",
    )
    with pytest.raises(ValueError, match="certificate_id"):
        make_charter(decision_map=(row,))


def test_duplicate_decision_rows_for_one_check_are_rejected():
    row = make_charter().decision_map[0]
    with pytest.raises(ValueError, match="duplicate"):
        make_charter(decision_map=(row, row))


def test_decision_row_for_unknown_check_is_rejected_by_the_registry_lint():
    charter = make_charter()
    with pytest.raises(ValueError, match="unknown check"):
        charter.lint(known_checks={"missingness.mcar"})
    charter.lint(known_checks={"leakage.skill"})


def test_decision_action_is_a_closed_set():
    assert {a.value for a in c.DecisionAction} == {
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
        ("min_effect", 0.0),
        ("min_effect", float("nan")),
        ("k_max", 0),
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
    with pytest.raises(ValueError, match="confirmation_fraction"):
        c.SetSizes(confirmation_fraction=fraction, embargo_rows=0)


def test_from_dict_rejects_unknown_and_missing_fields():
    record = make_charter().to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        c.AnalysisCharter.from_dict({**record, "extra": 1})
    del record["seed"]
    with pytest.raises(ValueError, match="missing"):
        c.AnalysisCharter.from_dict(record)


def test_multiverse_default_must_be_one_of_the_options():
    with pytest.raises(ValueError, match="default"):
        c.Choice("imputation", ("median", "mice"), "knn")


def test_load_rejects_malformed_json_as_a_config_error(tmp_path):
    path = tmp_path / "charter.json"
    path.write_text("{not json")
    with pytest.raises(errors.ConfigError):
        c.read_charter(path)
