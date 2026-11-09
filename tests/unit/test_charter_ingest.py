"""The charter's ingest block: format 7 and hash compatibility."""

import json

import pytest

from scmle.contracts import errors
from scmle.data.charter import contracts
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
    assert (
        contracts.AnalysisCharter.from_dict(record).charter_hash == PINNED_HASH
    )


def test_a_signed_older_charter_still_verifies(tmp_path):
    path = tmp_path / "charter.json"
    sealing.write_charter(path, make_charter(), signing_key="k")
    assert sealing.read_verified(path, signing_key="k").signed


def test_the_block_changes_the_hash_round_trips_and_needs_format_seven():
    charter = make_charter(ingest=ingest.Ingestion(feature_budget=50, lag=1.0))
    assert charter.charter_hash != PINNED_HASH
    assert charter.format_version == 7
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.AnalysisCharter.from_dict(record) == charter


def test_the_block_in_an_older_format_is_refused(tmp_path):
    charter = make_charter(ingest=ingest.Ingestion())
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
        ingest.Ingestion(**bad)


# Hash and signature (key "k") of an ingest charter written under format 7,
# before the entity block existed; both must survive the block.
FORMAT_SEVEN_HASH = (
    "43003effa1e4ec1473ee3aeccbe48aae65f8ddd11e06dd1b6f175e09c587ba6f"
)
FORMAT_SEVEN_SIGNATURE = (
    "1b1c634d860f0bd72c41b5f746699a15c528225a0102771f7a4d9141d57ec0d1"
)


def entity_block():
    return ingest.Entity(
        fields=(ingest.EntityField("name"), ingest.EntityField("email", False)),
        passes=(
            ingest.EntityPass("email"),
            ingest.EntityPass("name", "soundex"),
        ),
    )


def test_a_format_seven_ingest_charter_keeps_hash_and_signature():
    charter = make_charter(ingest=ingest.Ingestion(feature_budget=50, lag=1.0))
    assert charter.charter_hash == FORMAT_SEVEN_HASH
    assert charter.format_version == 7
    assert "entity" not in charter.to_dict()["ingest"]
    assert sealing.envelope(charter, "k")["signature"] == FORMAT_SEVEN_SIGNATURE


def test_the_entity_block_needs_format_eight_and_round_trips():
    spec = ingest.Ingestion(feature_budget=50, lag=1.0, entity=entity_block())
    charter = make_charter(ingest=spec)
    assert charter.format_version == 8
    assert charter.charter_hash != FORMAT_SEVEN_HASH
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.AnalysisCharter.from_dict(record) == charter


def test_the_entity_block_in_format_seven_is_refused(tmp_path):
    charter = make_charter(ingest=ingest.Ingestion(entity=entity_block()))
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
        lambda: ingest.Entity((), (ingest.EntityPass("a"),)),
        lambda: ingest.Entity((ingest.EntityField("a"),), ()),
        lambda: ingest.Entity(
            (ingest.EntityField("a"),), (ingest.EntityPass("a"),), ""
        ),
        lambda: ingest.Entity(
            (ingest.EntityField("a"), ingest.EntityField("a")),
            (ingest.EntityPass("a"),),
        ),
        lambda: ingest.EntityPass("a", "metaphone"),
    ],
)
def test_an_invalid_entity_block_is_refused(build):
    with pytest.raises(ValueError):
        build()


def reconcile_block():
    return ingest.Expectation(
        (
            ingest.TableTotals(
                "orders",
                rows=100,
                sums=(ingest.ColumnTotal("amount", 5000.0),),
                checksum_column="order_id",
                checksum="ab" * 32,
            ),
        )
    )


def test_the_reconcile_block_needs_format_eight_and_round_trips():
    spec = ingest.Ingestion(reconcile=reconcile_block())
    charter = make_charter(ingest=spec)
    assert charter.format_version == 8
    assert charter.charter_hash != FORMAT_SEVEN_HASH
    record = json.loads(json.dumps(charter.to_dict()))
    assert contracts.AnalysisCharter.from_dict(record) == charter
    plain = make_charter(ingest=ingest.Ingestion(feature_budget=50, lag=1.0))
    assert "reconcile" not in plain.to_dict()["ingest"]
    assert plain.charter_hash == FORMAT_SEVEN_HASH


def test_the_reconcile_block_in_format_seven_is_refused(tmp_path):
    charter = make_charter(ingest=ingest.Ingestion(reconcile=reconcile_block()))
    path = tmp_path / "charter.json"
    sealing.write_charter(path, charter, signing_key="k")
    wrapper = json.loads(path.read_text())
    wrapper["format"] = 7
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="reconcile"):
        sealing.read_verified(path, signing_key="k")


@pytest.mark.parametrize(
    "build",
    [
        lambda: ingest.Expectation(()),
        lambda: ingest.TableTotals("t"),
        lambda: ingest.TableTotals("t", rows=-1),
        lambda: ingest.TableTotals("t", checksum_column="k"),
        lambda: ingest.TableTotals("t", checksum="ab" * 32),
        lambda: ingest.TableTotals("t", checksum_column="k", checksum="zz"),
        lambda: ingest.ColumnTotal("a", float("nan")),
        lambda: ingest.Expectation(
            (ingest.TableTotals("t", rows=1), ingest.TableTotals("t", rows=2))
        ),
    ],
)
def test_an_invalid_reconcile_block_is_refused(build):
    with pytest.raises(ValueError):
        build()
