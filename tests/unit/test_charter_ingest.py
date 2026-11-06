"""The charter's ingest block: format 7 and hash compatibility."""

import json

import pytest

from scmle.contracts import errors
from scmle.data.charter import contracts as c
from scmle.data.charter import ingest
from scmle.data.charter import sealing
from tests.unit.test_charter_contracts import make_charter

# Hash of make_charter() with no new field, pinned so that adding the
# ingest block can never change the hash or signature of an older charter.
PINNED_HASH = "df685532bd1c21ef568e17997220f2b97da7c38d0a7a9752b9dcfecea49fa158"


def test_a_charter_without_the_block_keeps_its_hash_and_format():
    charter = make_charter()
    assert charter.charter_hash == PINNED_HASH
    assert "ingest" not in charter.to_dict()
    assert charter.format_version == 2
    record = json.loads(json.dumps(charter.to_dict()))
    assert c.AnalysisCharter.from_dict(record).charter_hash == PINNED_HASH


def test_a_signed_older_charter_still_verifies(tmp_path):
    path = tmp_path / "charter.json"
    sealing.write_charter(path, make_charter(), signing_key="k")
    assert sealing.read_verified(path, signing_key="k").signed


def test_the_block_changes_the_hash_round_trips_and_needs_format_seven():
    charter = make_charter(ingest=ingest.IngestSpec(feature_budget=50, lag=1.0))
    assert charter.charter_hash != PINNED_HASH
    assert charter.format_version == 7
    record = json.loads(json.dumps(charter.to_dict()))
    assert c.AnalysisCharter.from_dict(record) == charter


def test_the_block_in_an_older_format_is_refused(tmp_path):
    charter = make_charter(ingest=ingest.IngestSpec())
    path = tmp_path / "charter.json"
    sealing.write_charter(path, charter, signing_key="k")
    wrapper = json.loads(path.read_text())
    wrapper["format"] = 6
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="ingest"):
        sealing.read_verified(path, signing_key="k")


@pytest.mark.parametrize(
    "bad",
    [
        {"feature_budget": 0},
        {"windows": ()},
        {"windows": (7.0, 7.0)},
        {"windows": (-1.0,)},
        {"lag": -1.0},
        {"eps_fk": -0.1},
        {"max_seconds": -1.0},
    ],
)
def test_invalid_blocks_are_refused(bad):
    with pytest.raises(ValueError):
        ingest.IngestSpec(**bad)


# Hash and signature (key "k") of an ingest charter written under format 7,
# before the entity block existed; both must survive the block.
FORMAT_SEVEN_HASH = (
    "43003effa1e4ec1473ee3aeccbe48aae65f8ddd11e06dd1b6f175e09c587ba6f"
)
FORMAT_SEVEN_SIGNATURE = (
    "1b1c634d860f0bd72c41b5f746699a15c528225a0102771f7a4d9141d57ec0d1"
)


def entity_block():
    return ingest.EntitySpec(
        fields=(ingest.EntityField("name"), ingest.EntityField("email", False)),
        passes=(
            ingest.EntityPass("email"),
            ingest.EntityPass("name", "soundex"),
        ),
    )


def test_a_format_seven_ingest_charter_keeps_hash_and_signature():
    charter = make_charter(ingest=ingest.IngestSpec(feature_budget=50, lag=1.0))
    assert charter.charter_hash == FORMAT_SEVEN_HASH
    assert charter.format_version == 7
    assert "entity" not in charter.to_dict()["ingest"]
    assert sealing.envelope(charter, "k")["signature"] == FORMAT_SEVEN_SIGNATURE


def test_the_entity_block_needs_format_eight_and_round_trips():
    spec = ingest.IngestSpec(feature_budget=50, lag=1.0, entity=entity_block())
    charter = make_charter(ingest=spec)
    assert charter.format_version == 8
    assert charter.charter_hash != FORMAT_SEVEN_HASH
    record = json.loads(json.dumps(charter.to_dict()))
    assert c.AnalysisCharter.from_dict(record) == charter


def test_the_entity_block_in_format_seven_is_refused(tmp_path):
    charter = make_charter(ingest=ingest.IngestSpec(entity=entity_block()))
    path = tmp_path / "charter.json"
    sealing.write_charter(path, charter, signing_key="k")
    wrapper = json.loads(path.read_text())
    wrapper["format"] = 7
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="entity"):
        sealing.read_verified(path, signing_key="k")


@pytest.mark.parametrize(
    "build",
    [
        lambda: ingest.EntitySpec((), (ingest.EntityPass("a"),)),
        lambda: ingest.EntitySpec((ingest.EntityField("a"),), ()),
        lambda: ingest.EntitySpec(
            (ingest.EntityField("a"),), (ingest.EntityPass("a"),), ""
        ),
        lambda: ingest.EntitySpec(
            (ingest.EntityField("a"), ingest.EntityField("a")),
            (ingest.EntityPass("a"),),
        ),
        lambda: ingest.EntityPass("a", "metaphone"),
    ],
)
def test_an_invalid_entity_block_is_refused(build):
    with pytest.raises(ValueError):
        build()
