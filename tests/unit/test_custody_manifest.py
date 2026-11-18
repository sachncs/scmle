"""Split manifest v2: full hash, round trip, corruption and v1 reading."""

import dataclasses
import json

import pandas
import pytest

from scmle.assurance.custody import merkle
import scmle.assurance.custody.manifest
from scmle.contracts import errors
from scmle.data.splitting import splitters

DIGESTS = {"charter": "a" * 64, "data": "b" * 64}
SETS = {
    "exploration": (0, 1, 2, 3),
    "confirmation": (4, 5),
    "embargo": (),
    "final": (6, 7),
}


def make(digests=None, sets=None, **overrides):
    """A manifest; `digests` and `sets` override single nested fields."""
    values = {
        "strategy": "stratified",
        "seed": 1,
        "embargo": 0,
        "digests": scmle.assurance.custody.manifest.Digests(
            **{**DIGESTS, **(digests or {})}
        ),
        "sets": scmle.assurance.custody.manifest.Sets(
            **{**SETS, **(sets or {})}
        ),
        "folds": (((0, 1), (2, 3)), ((2, 3), (0, 1))),
        "isolation": (("C", "vault process"), ("final_test", "vault process")),
        "scope": "confirmatory",
        "notes": ("note",),
    }
    values.update(overrides)
    return scmle.assurance.custody.manifest.Manifest(**values)


def test_hash_is_a_full_sha256_and_deterministic():
    assert len(make().sha256) == 64
    assert make().sha256 == make().sha256


@pytest.mark.parametrize(
    "change",
    [
        {"seed": 2},
        {"embargo": 3},
        {"digests": {"charter": "c" * 64}},
        {"digests": {"data": "d" * 64}},
        {"sets": {"exploration": (0, 1, 2)}},
        {"sets": {"confirmation": (4,)}},
        {"sets": {"final": (6,)}},
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
    assert scmle.assurance.custody.manifest.restore(record) == manifest


def test_corrupting_any_id_fails_the_hash_check():
    record = make().to_dict()
    record["sets"]["confirmation"][0] = 99
    with pytest.raises(errors.ArtifactError, match="hash"):
        scmle.assurance.custody.manifest.restore(record)


def test_overlapping_sets_are_rejected():
    with pytest.raises(ValueError, match="overlap"):
        make(sets={"confirmation": (3, 4)})
    with pytest.raises(ValueError, match="overlap"):
        make(sets={"final": (0,)})
    with pytest.raises(ValueError, match="overlap"):
        make(sets={"embargo": (5,)})


def test_unknown_version_is_refused():
    record = make().to_dict()
    record["manifest_version"] = 3
    with pytest.raises(errors.ArtifactError, match="version"):
        scmle.assurance.custody.manifest.restore(record)


def test_unknown_field_is_refused():
    record = make().to_dict()
    record["extra"] = 1
    with pytest.raises((errors.ArtifactError, ValueError)):
        scmle.assurance.custody.manifest.restore(record)


def test_v1_manifest_is_still_readable_and_verified():
    v1 = splitters.Manifest(
        strategy="random",
        seed=3,
        train_ids=(0, 1, 2),
        test_ids=(3,),
        folds=(((0, 1), (2,)),),
        notes=("n",),
    )
    loaded = scmle.assurance.custody.manifest.restore(
        json.loads(json.dumps(v1.to_dict()))
    )
    assert isinstance(loaded, splitters.Manifest)
    assert loaded == v1
    assert loaded.fingerprint == v1.fingerprint


def test_v1_manifest_with_a_bad_fingerprint_is_refused():
    v1 = splitters.Manifest("random", 3, (0, 1), (2,), (((0,), (1,)),))
    record = v1.to_dict()
    record["fingerprint"] = "0" * 16
    with pytest.raises(errors.ArtifactError, match="fingerprint"):
        scmle.assurance.custody.manifest.restore(record)


def test_manifest_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make().seed = 5


def test_scope_is_a_closed_set():
    with pytest.raises(ValueError, match="scope"):
        make(scope="whatever")


def table():
    return pandas.DataFrame(
        {"a": [float(i) for i in range(8)], "y": [0, 1] * 4}
    )


def rooted(frame=None):
    frame = table() if frame is None else frame
    manifest = make()
    roots = tuple(
        (name, merkle.merkleize(frame, members))
        for name, members in manifest.rowids().items()
    )
    return make(roots=roots)


def test_a_manifest_without_roots_keeps_its_hash_and_body():
    assert "roots" not in make().body()
    assert make().sha256 == make(roots=()).sha256


def test_the_roots_are_hashed_and_round_trip():
    manifest = rooted()
    assert manifest.sha256 != make().sha256
    record = json.loads(json.dumps(manifest.to_dict()))
    assert scmle.assurance.custody.manifest.restore(record) == manifest
    record["roots"][0][1] = "0" * 64
    with pytest.raises(errors.ArtifactError, match="hash"):
        scmle.assurance.custody.manifest.restore(record)


def test_verify_rows_passes_on_the_original_rows():
    rooted().verify_rows(table())


@pytest.mark.parametrize(
    ("row", "name"), [(1, "exploration"), (4, "confirmation")]
)
def test_a_changed_value_inside_a_row_is_caught_though_ids_are_equal(row, name):
    manifest = rooted()
    edited = table()
    edited.loc[row, "a"] += 0.001
    assert manifest.sha256 == rooted().sha256
    with pytest.raises(errors.ArtifactError, match=name):
        manifest.verify_rows(edited)


def test_a_change_in_the_embargo_rows_is_not_covered():
    # Embargo rows belong to no set and carry no root: stated limit.
    manifest = make(sets={"embargo": (), "confirmation": (4, 5)})
    assert "embargo" not in manifest.rowids()


def test_verify_rows_refuses_a_manifest_with_no_roots():
    with pytest.raises(errors.ArtifactError, match="no row content"):
        make().verify_rows(table())
