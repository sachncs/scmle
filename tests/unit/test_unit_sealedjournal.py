"""Unit tests for scmle.assurance.custody.sealed.Journal loading."""

import pytest

from scmle.assurance.custody import sealed
from scmle.contracts import errors


@pytest.mark.parametrize("line", ["not json", "[1, 2]", '"text"', "7"])
def test_a_log_line_that_is_not_a_json_object_is_an_artifact_error(
    tmp_path, line
):
    path = tmp_path / "custody.jsonl"
    path.write_text(line + "\n", encoding="utf-8")
    with pytest.raises(errors.ArtifactError):
        sealed.Journal(path, signing_key=b"key")


def test_a_log_that_is_not_utf8_is_an_artifact_error(tmp_path):
    path = tmp_path / "custody.jsonl"
    path.write_bytes(b"\xff\xfe\x00\n")
    with pytest.raises(errors.ArtifactError):
        sealed.Journal(path, signing_key=b"key")
