"""The charter's uncertainty block: format 10 and hash compatibility."""

import json

import pytest

from scmle.contracts import errors
from scmle.data.charter import contracts as c
from scmle.data.charter import sealing
from scmle.data.charter import uncertainty
from tests.unit.test_charter_contracts import make_charter
from tests.unit.test_charter_ingest import PINNED_HASH

# Hash and signature (key "k") of a charter that carries the block, pinned
# so that later optional fields cannot change them.
FORMAT_TEN_HASH = (
    "d7012fdf29cab370e2fb42854f187fbaa4ff3a7787fdb2c59a61e38c9bd732a9"
)
FORMAT_TEN_SIGNATURE = (
    "ff735ca118159d15a6bc714773770e610988d0aa06c0d343322868e7b5f36a2d"
)


def with_cap(size=5_000_000):
    return make_charter(uncertainty=uncertainty.UncertaintySpec(size))


def test_a_charter_without_the_block_keeps_its_hash_and_format():
    charter = make_charter()
    assert charter.charter_hash == PINNED_HASH
    assert "uncertainty" not in charter.to_dict()
    assert charter.format_version == 2
    assert c.FORMAT_VERSION == 10


def test_an_empty_block_is_an_empty_record_and_still_format_ten():
    charter = make_charter(uncertainty=uncertainty.UncertaintySpec())
    record = json.loads(json.dumps(charter.to_dict()))
    assert record["uncertainty"] == {}
    assert charter.format_version == 10
    assert c.AnalysisCharter.from_dict(record) == charter


def test_the_block_changes_the_hash_round_trips_and_needs_format_ten():
    charter = with_cap()
    assert charter.charter_hash != PINNED_HASH
    assert charter.format_version == 10
    record = json.loads(json.dumps(charter.to_dict()))
    assert record["uncertainty"] == {"cv_plus_max_bytes": 5_000_000}
    assert c.AnalysisCharter.from_dict(record) == charter


def test_a_format_ten_charter_keeps_its_hash_and_signature():
    charter = with_cap()
    assert charter.charter_hash == FORMAT_TEN_HASH
    assert sealing.envelope(charter, "k")["signature"] == FORMAT_TEN_SIGNATURE


def test_the_block_in_an_older_format_is_refused(tmp_path):
    path = tmp_path / "charter.json"
    sealing.write_charter(path, with_cap(), signing_key="k")
    wrapper = json.loads(path.read_text())
    wrapper["format"] = 9
    path.write_text(json.dumps(wrapper))
    with pytest.raises(errors.ConfigError, match="needs format 10"):
        sealing.read_verified(path, signing_key="k")


@pytest.mark.parametrize("bad", [0, -5, 1.5, True])
def test_a_cap_must_be_a_positive_integer(bad):
    with pytest.raises((ValueError, TypeError)):
        uncertainty.UncertaintySpec(bad)


def test_unknown_fields_are_refused():
    with pytest.raises(ValueError, match="unknown"):
        uncertainty.UncertaintySpec.from_dict({"cv_plus_max_bytes": 5, "x": 1})


def stage_context(path):
    from types import SimpleNamespace

    return SimpleNamespace(spec=SimpleNamespace(charter_path=path))


def test_the_stage_reads_the_cap_from_the_charter(tmp_path):
    from scmle.core import stages
    from scmle.evaluation import cv_plus

    path = tmp_path / "charter.json"
    sealing.write_charter(path, with_cap(1234), signing_key="k")
    plain = tmp_path / "plain.json"
    sealing.write_charter(plain, make_charter(), signing_key="k")
    package = stages.PackageStage()

    assert package.cv_plus_cap(stage_context(str(path))) == 1234
    assert package.cv_plus_cap(stage_context(str(plain))) == cv_plus.MAX_BYTES
    assert package.cv_plus_cap(stage_context(None)) == cv_plus.MAX_BYTES
