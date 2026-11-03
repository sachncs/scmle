"""Split manifest v2: full hash, round trip, corruption and v1 reading."""

import dataclasses
import json

import pytest

from scmle.contracts import errors
from scmle.data.custody import manifest as m
from scmle.data.splitting import splitters


def make(**overrides):
    values = {
        "strategy": "stratified",
        "seed": 1,
        "embargo_rows": 0,
        "charter_hash": "a" * 64,
        "data_hash": "b" * 64,
        "exploration_ids": (0, 1, 2, 3),
        "confirmation_ids": (4, 5),
        "embargo_ids": (),
        "final_test_ids": (6, 7),
        "folds": (((0, 1), (2, 3)), ((2, 3), (0, 1))),
        "isolation": (("C", "vault process"), ("final_test", "vault process")),
        "scope": "confirmatory",
        "notes": ("note",),
    }
    values.update(overrides)
    return m.CustodyManifest(**values)


def test_hash_is_a_full_sha256_and_deterministic():
    assert len(make().sha256) == 64
    assert make().sha256 == make().sha256


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 2},
        {"embargo_rows": 3},
        {"charter_hash": "c" * 64},
        {"data_hash": "d" * 64},
        {"exploration_ids": (0, 1, 2)},
        {"confirmation_ids": (4,)},
        {"final_test_ids": (6,)},
        {"folds": (((0, 1), (2, 3)),)},
        {"scope": "exploratory_unconfirmed"},
        {"isolation": (("C", "in-process protocol"),)},
    ],
)
def test_every_field_changes_the_hash(change):
    assert make(**change).sha256 != make().sha256


def test_round_trip_through_json():
    manifest = make()
    record = json.loads(json.dumps(manifest.to_dict()))
    assert record["manifest_version"] == 2
    assert m.load_manifest(record) == manifest


def test_corrupting_any_id_fails_the_hash_check():
    record = make().to_dict()
    record["confirmation_ids"][0] = 99
    with pytest.raises(errors.ArtifactError, match="hash"):
        m.load_manifest(record)


def test_overlapping_sets_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        make(confirmation_ids=(3, 4))
    with pytest.raises(ValueError, match="overlap"):
        make(final_test_ids=(0,))
    with pytest.raises(ValueError, match="overlap"):
        make(embargo_ids=(5,))


def test_unknown_version_is_refused():
    record = make().to_dict()
    record["manifest_version"] = 3
    with pytest.raises(errors.ArtifactError, match="version"):
        m.load_manifest(record)


def test_unknown_field_is_refused():
    record = make().to_dict()
    record["extra"] = 1
    with pytest.raises((errors.ArtifactError, ValueError)):
        m.load_manifest(record)


def test_v1_manifest_is_still_readable_and_verified():
    v1 = splitters.SplitManifest(
        strategy="random",
        seed=3,
        train_ids=(0, 1, 2),
        test_ids=(3,),
        folds=(((0, 1), (2,)),),
        notes=("n",),
    )
    loaded = m.load_manifest(json.loads(json.dumps(v1.to_dict())))
    assert isinstance(loaded, splitters.SplitManifest)
    assert loaded == v1
    assert loaded.fingerprint == v1.fingerprint


def test_v1_manifest_with_a_bad_fingerprint_is_refused():
    v1 = splitters.SplitManifest("random", 3, (0, 1), (2,), (((0,), (1,)),))
    record = v1.to_dict()
    record["fingerprint"] = "0" * 16
    with pytest.raises(errors.ArtifactError, match="fingerprint"):
        m.load_manifest(record)


def test_manifest_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().seed = 5


def test_scope_is_a_closed_set():
    with pytest.raises(ValueError, match="scope"):
        make(scope="whatever")
