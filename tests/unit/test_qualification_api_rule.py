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
    "Correction",
    "component_rng",
}


def test_no_previously_pinned_name_changed():
    pinned = json.loads(SNAPSHOT.read_text(encoding="utf-8"))["surfaces"]
    current = check_api_compatibility.current_snapshot()["surfaces"]

    assert pinned
    for module_name, names in pinned.items():
        for name, record in names.items():
            assert current[module_name][name] == record, (module_name, name)


def test_new_types_are_not_package_root_names():
    assert all(not hasattr(scmle, name) for name in NEW_NAMES)


def test_new_modules_expose_the_documented_names():
    assert qualification.Check and qualification.Claim
    assert qualification.Certificate and qualification.Tier
    assert budget.ErrorBudget and randomness.component_rng
