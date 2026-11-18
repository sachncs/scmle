"""The charter's unit of prediction: format 6 and hash compatibility."""

import json

import pytest

from scmle.assurance.charter import contracts
from scmle.assurance.charter import sealing
from scmle.contracts import errors
from tests.unit.test_charter_contracts import make_charter


def test_a_charter_without_the_field_serialises_and_hashes_as_before():
    charter = make_charter()
    assert charter.purpose is None
    assert "purpose" not in charter.to_dict()
    assert charter.format_version == 2
    record = json.loads(json.dumps(charter.to_dict()))
    again = contracts.AnalysisCharter.from_dict(record)
    assert again.charter_hash == charter.charter_hash


def test_the_field_changes_the_hash_round_trips_and_needs_format_six():
    base = make_charter()
    charter = make_charter(purpose=contracts.NEW_ENTITY)
    assert charter.charter_hash != base.charter_hash
    assert charter.format_version == 6
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.AnalysisCharter.from_dict(record) == charter
    other = make_charter(purpose=contracts.KNOWN_ENTITY_FUTURE)
    assert other.charter_hash != charter.charter_hash


def test_an_unknown_unit_is_refused():
    with pytest.raises(ValueError, match="purpose"):
        make_charter(purpose="row")


def test_the_field_in_an_older_format_is_refused(tmp_path):
    charter = make_charter(purpose=contracts.NEW_ENTITY)
    wrapper = sealing.envelope(charter, "k")
    assert wrapper["format"] == 6
    wrapper["format"] = 5
    path = tmp_path / "charter.json"
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="purpose"):
        sealing.read_verified(path, signing_key="k")


def test_a_format_six_charter_is_read_back(tmp_path):
    charter = make_charter(purpose=contracts.KNOWN_ENTITY_FUTURE)
    path = sealing.inscribe(tmp_path / "charter.json", charter, "k")
    assert sealing.read_verified(path, signing_key="k").charter == charter
