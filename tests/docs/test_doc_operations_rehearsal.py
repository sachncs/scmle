"""Guarantees of docs/OPERATIONS_REHEARSAL.md that no other test enforced."""

import pytest

import scmle.artifacts.bundle
from scmle.contracts import errors
from tests.unit.test_bundle import trained  # noqa: F401 (pytest fixture)


def test_rotation_rejects_the_old_key_and_leaves_no_temporary_file(
    trained,  # noqa: F811 (the imported pytest fixture)
    tmp_path,
):
    """Enforces docs/OPERATIONS_REHEARSAL.md:64.

    Also enforces website/src/pages/docs/rehearsal.astro:35.
    """
    bundle, features = trained
    directory = bundle.save(tmp_path / "m", signing_key="old-key")
    before = sorted(p.name for p in directory.rglob("*"))
    scmle.artifacts.bundle.Bundle.rotate_signing_key(
        directory, "old-key", "new-key"
    )
    assert sorted(p.name for p in directory.rglob("*")) == before
    with pytest.raises(errors.ArtifactError, match="signature is invalid"):
        scmle.artifacts.bundle.Bundle.load(directory, signing_key="old-key")
    loaded = scmle.artifacts.bundle.Bundle.load(
        directory, signing_key="new-key"
    )
    assert len(loaded.predict(features)) == len(features)
