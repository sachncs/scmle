"""The bundle digest's canonical bytes and its refusals, word for word."""

import hashlib
import json

import pytest

from scmle.contracts import errors
from scmle.deploy import provenance


def expected(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("ascii")).hexdigest()


def test_the_manifest_digest_hashes_sorted_compact_ascii_json() -> None:
    manifest = {"b": [1, 2], "a": "é"}
    assert provenance.manifest_digest(manifest) == expected(
        '{"a":"\\u00e9","b":[1,2]}'
    )


def test_key_order_does_not_change_the_manifest_digest() -> None:
    first = provenance.manifest_digest({"a": 1, "b": 2})
    assert first == provenance.manifest_digest({"b": 2, "a": 1})
    assert first == expected('{"a":1,"b":2}')


def test_the_bundle_digest_is_the_digest_of_the_stored_manifest(
    tmp_path,
) -> None:
    manifest = {"z": 1, "a": "é"}
    (tmp_path / provenance.MANIFEST_FILE).write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    assert provenance.digest(tmp_path) == expected('{"a":"\\u00e9","z":1}')


def test_a_manifest_that_is_not_an_object_is_refused_word_for_word(
    tmp_path,
) -> None:
    (tmp_path / provenance.MANIFEST_FILE).write_text("[1]", encoding="utf-8")
    with pytest.raises(
        errors.ArtifactError, match=r"^the bundle manifest is not an object$"
    ):
        provenance.digest(tmp_path)


def test_a_file_larger_than_one_chunk_digests_all_its_bytes(tmp_path) -> None:
    data = bytes(range(256)) * 9000
    path = tmp_path / "blob.bin"
    path.write_bytes(data)
    assert provenance.file_digest(path) == (
        "sha256:" + hashlib.sha256(data).hexdigest()
    )
