"""Qualification types stay below the pinned top-level API."""

import json
from pathlib import Path

import scmle
from scmle.contracts import qualification
from scmle.core import randomness
from scmle.data.ledger import budget
from scripts import check_api_compatibility

SNAPSHOT = Path(__file__).parents[1] / "fixtures" / "public_api.json"
NEW_NAMES = {
    "Check",
    "Claim",
    "Certificate",
    "Tier",
    "Decision",
    "SetId",
    "ErrorBudget",
    "ExplorationBudget",
    "ConfirmationBudget",
    "MultiplicityRule",
    "component_rng",
}


def test_no_previously_pinned_export_changed():
    pinned = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["exports"]
    current = check_api_compatibility.current_snapshot()["exports"]

    assert pinned
    for name, record in pinned.items():
        assert current[name] == record, name


def test_new_types_are_not_top_level_exports_yet():
    assert NEW_NAMES.isdisjoint(scmle.__all__)
    assert all(not hasattr(scmle, name) for name in NEW_NAMES)


def test_new_modules_expose_the_documented_names():
    assert qualification.Check and qualification.Claim
    assert qualification.Certificate and qualification.Tier
    assert budget.ErrorBudget and randomness.component_rng
