"""Pre-registered run rules in the charter, and format versioning."""

import dataclasses
import json

import pytest

from scmle.contracts import errors
from scmle.data.charter import contracts
from scmle.data.charter import rules as rules_module
from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter

KEY = "rules-key"
# Hash and signature of `make_charter()` computed before `rules` existed; an
# already signed charter must keep both.
FROZEN_HASH = "df685532bd1c21ef568e17997220f2b97da7c38d0a7a9752b9dcfecea49fa158"
FROZEN_SIGNATURE = (
    "f28886692a8912b73c0d02ed560b197b0041f29fb38bb5c2ab0343a56bf6406b"
)


def chartered(**changes):
    return make_charter(rules=rules_module.RunRules(**changes))


def test_a_charter_without_rules_keeps_its_hash_and_signature():
    charter = make_charter()
    assert charter.rules is None
    assert charter.charter_hash == FROZEN_HASH
    assert "rules" not in charter.to_dict()
    wrapper = sealing.envelope(charter, KEY)
    assert wrapper["format"] == 2
    assert sealing.envelope(charter, "k")["signature"] == FROZEN_SIGNATURE


def test_an_old_signed_file_still_loads_and_verifies(tmp_path):
    path = tmp_path / "old.json"
    wrapper = sealing.envelope(make_charter(), KEY)
    wrapper["format"] = 1
    path.write_text(json.dumps(wrapper))
    loaded = sealing.read_verified(path, signing_key=KEY)
    assert loaded.signed and loaded.charter.rules is None
    assert loaded.charter_hash == FROZEN_HASH
    del wrapper["format"]
    path.write_text(json.dumps(wrapper))
    assert sealing.read_verified(path, signing_key=KEY).signed


@pytest.mark.parametrize(
    "changes",
    [
        {"ledger_rule": "lord++"},
        {"confirmation_rule": "bh"},
        {"stability_filter": True},
        {"s_min": 0.9},
        {"band": 0.4},
        {"q_min": 0.7},
    ],
)
def test_changing_any_rule_changes_the_charter_hash(changes):
    base = chartered()
    changed = chartered(**changes)
    assert changed.charter_hash != base.charter_hash
    assert base.charter_hash != FROZEN_HASH


def test_rules_round_trip_through_a_signed_file(tmp_path):
    charter = chartered(stability_filter=True, s_min=0.9)
    path = sealing.write_charter(tmp_path / "c.json", charter, KEY)
    assert json.loads(path.read_text())["format"] == 3
    loaded = sealing.read_verified(path, signing_key=KEY)
    assert loaded.signed and loaded.charter == charter
    assert loaded.charter.rules.stability_filter is True


def test_editing_a_rule_in_a_signed_file_is_detected(tmp_path):
    path = sealing.write_charter(tmp_path / "c.json", chartered(), KEY)
    document = json.loads(path.read_text())
    document["charter"]["rules"]["stability_filter"] = True
    path.write_text(json.dumps(document))
    with pytest.raises(errors.ArtifactError):
        sealing.read_verified(path, signing_key=KEY)


def test_a_wrapper_from_a_newer_format_is_refused(tmp_path):
    path = sealing.write_charter(tmp_path / "c.json", chartered(), KEY)
    document = json.loads(path.read_text())
    document["format"] = contracts.FORMAT_VERSION + 1
    path.write_text(json.dumps(document))
    with pytest.raises(errors.ConfigError, match="format"):
        sealing.read_verified(path, signing_key=KEY)


def test_rules_in_an_older_format_are_refused(tmp_path):
    path = sealing.write_charter(tmp_path / "c.json", chartered(), KEY)
    document = json.loads(path.read_text())
    document["format"] = 2
    path.write_text(json.dumps(document))
    with pytest.raises(errors.ConfigError, match="need format 3"):
        sealing.read_verified(path, signing_key=KEY)


@pytest.mark.parametrize(
    "changes",
    [
        {"ledger_rule": "alpha-investing"},
        {"confirmation_rule": "fisher"},
        {"failure_policy": "thresholdout"},
        {"stability_filter": "yes"},
        {"s_min": 0.0},
        {"q_min": 1.5},
        {"band": 0.0},
        {"band": float("nan")},
    ],
)
def test_invalid_rules_are_rejected(changes):
    with pytest.raises(ValueError):
        rules_module.RunRules(**changes)


def test_rules_reject_unknown_and_missing_fields():
    record = rules_module.RunRules().to_dict()
    with pytest.raises(ValueError, match="unexpected"):
        rules_module.RunRules.from_dict({**record, "extra": 1})
    del record["band"]
    with pytest.raises(ValueError, match="missing"):
        rules_module.RunRules.from_dict(record)


def test_defaults_match_the_shipped_stability_rule():
    from scmle.data.robustness import stability

    defaults = rules_module.RunRules()
    assert dataclasses.asdict(stability.DEFAULT_RULE) == {
        "s_min": defaults.s_min,
        "band": defaults.band,
        "q_min": defaults.q_min,
    }
