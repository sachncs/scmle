"""Guarantees of docs/SECURITY.md that no other test enforced."""

import json
import pathlib
import platform

import pytest
import sklearn

from scmle.artifacts import bundle
from scmle.contracts import errors
from scmle.inference import parity

ROOT = pathlib.Path(__file__).resolve().parents[2]


def test_a_python_version_mismatch_is_refused_unless_allowed(tmp_path):
    """Enforces docs/SECURITY.md:27 (the Python half; scikit-learn is in
    tests/unit/test_bundle.py)."""
    golden = parity.goldens(ROOT / "deploy" / "local" / "golden")
    directory = parity.handwrite(golden.spec).save(tmp_path / "m")
    path = directory / bundle.MANIFEST_FILE
    manifest = json.loads(path.read_text())
    assert manifest["python"] == platform.python_version()
    assert manifest["sklearn"] == sklearn.__version__
    manifest["python"] = "2.7.18"
    path.write_text(json.dumps(manifest))
    with pytest.raises(errors.ArtifactError, match=r"python 2\.7\.18"):
        bundle.Bundle.load(directory, trusted=True)
    relaxed = bundle.Bundle.load(
        directory, trusted=True, allow_version_mismatch=True
    )
    assert relaxed.info.python == "2.7.18"
